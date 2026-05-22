from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

from model import PolicyOutput
from model_grouped_planet import GroupedPlanetOrbitWarsPolicy
from obs_builder import (
    EDGE_FEATURE_DIM,
    F_MAX_DEFAULT,
    GLOBAL_FEATURE_DIM,
    P_MAX_DEFAULT,
    PLANET_FEATURE_DIM,
)

PF_IDX_OWNER_SELF = 0
PF_IDX_OWNER_ENEMY_START = 1
PF_IDX_OWNER_ENEMY_END = 4
PF_IDX_OWNER_NEUTRAL = 4
PF_IDX_SHIPS = 5
PF_IDX_PRODUCTION = 6
PF_IDX_RADIUS = 7
PF_IDX_R_SUN = 8
PF_IDX_COS_THETA = 9
PF_IDX_SIN_THETA = 10
PF_IDX_IS_ROTATING = 12
PF_IDX_OMEGA_DT = 14
PF_IDX_INCOMING_FRIENDLY_TOTAL = 24
PF_IDX_INCOMING_ENEMY_TOTAL = 25
EB_IDX_DISTANCE = 0
EB_IDX_CROSSES_SUN = 5
GRAPH_NODE_METRIC_DIM = 8
# When flux features ([friendly_in, friendly_out, enemy_in, enemy_out]) are
# appended to the base node metrics (see ``rl/flux.py``), the per-node
# metric vector grows to this many channels.  Phase 1 only uses this in
# the reward summary path; the encoder still consumes the base 8 channels.
GRAPH_NODE_FLUX_DIM = 4
GRAPH_NODE_METRIC_DIM_FLUX = GRAPH_NODE_METRIC_DIM + GRAPH_NODE_FLUX_DIM  # = 12
PYG_PLANET_FEATURE_DIM = PLANET_FEATURE_DIM + GRAPH_NODE_METRIC_DIM
PYG_PLANET_FEATURE_DIM_FLUX = PLANET_FEATURE_DIM + GRAPH_NODE_METRIC_DIM_FLUX
GRAPH_EDGE_METRIC_DIM = 9
PYG_EDGE_FEATURE_DIM = EDGE_FEATURE_DIM + GRAPH_EDGE_METRIC_DIM
BOARD_SIZE = 100.0
SUN_X = 50.0
SUN_Y = 50.0
DEFAULT_MAX_SHIP_SPEED = 6.0
REFERENCE_SHIPS_FOR_ETA = 50
ETA_NORM = 50.0
INFLUENCE_TAU_TURNS = 12.0

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool
except ImportError:
    Batch = None
    Data = None
    GINEConv = None
    global_max_pool = None
    global_mean_pool = None


def _require_pyg() -> None:
    if Data is None or Batch is None or GINEConv is None:
        raise ImportError(
            "model_graph_pyg requires torch_geometric. Install a torch-compatible "
            "PyTorch Geometric build before using PygPlanetGraphPolicy."
        )


def _planet_edge_index(num_nodes: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    idx = torch.arange(num_nodes, dtype=torch.long, device=device)
    src = idx.repeat_interleave(num_nodes)
    dst = idx.repeat(num_nodes)
    keep = src != dst
    return torch.stack([src[keep], dst[keep]], dim=0)


def _fleet_speed_from_ships(ships: torch.Tensor) -> torch.Tensor:
    ships_safe = ships.clamp_min(1.0)
    speed = 1.0 + (DEFAULT_MAX_SHIP_SPEED - 1.0) * (torch.log(ships_safe) / math.log(1000.0)).clamp_min(0.0).pow(1.5)
    return torch.where(ships <= 1.0, torch.ones_like(speed), speed)


def _estimated_intercept_eta_matrix(planet_feats: torch.Tensor, edge_bias: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if num_nodes <= 0:
        return planet_feats.new_zeros((0, 0))
    feats = planet_feats[:num_nodes]
    r = feats[:, PF_IDX_R_SUN].clamp_min(0.0) * 50.0
    cos_t = feats[:, PF_IDX_COS_THETA]
    sin_t = feats[:, PF_IDX_SIN_THETA]
    theta = torch.atan2(sin_t, cos_t)
    x = SUN_X + r * cos_t
    y = SUN_Y + r * sin_t
    radius = feats[:, PF_IDX_RADIUS].clamp_min(0.0) * 5.0
    omega = feats[:, PF_IDX_OMEGA_DT] * feats[:, PF_IDX_IS_ROTATING].clamp(0.0, 1.0)
    # 1.0-turn resolution: 51 grid points instead of 101; halves memory/FLOPs
    # with negligible RL accuracy loss (intercept search granularity).
    t_grid = torch.arange(0.0, ETA_NORM + 1.0, 1.0, dtype=planet_feats.dtype, device=planet_feats.device)
    tgt_theta = theta[None, :, None] + omega[None, :, None] * t_grid[None, None, :]
    tgt_x = SUN_X + r[None, :, None] * torch.cos(tgt_theta)
    tgt_y = SUN_Y + r[None, :, None] * torch.sin(tgt_theta)
    src_x = x[:, None, None]
    src_y = y[:, None, None]
    spawn_offset = radius[:, None, None] + 0.1
    speed = _fleet_speed_from_ships(planet_feats.new_full((num_nodes,), float(REFERENCE_SHIPS_FOR_ETA)))
    dist = torch.sqrt((tgt_x - src_x).pow(2) + (tgt_y - src_y).pow(2)).sub(spawn_offset).clamp_min(0.0)
    reachable = dist <= speed[:, None, None] * t_grid[None, None, :]
    has_hit = reachable.any(dim=-1)
    first_idx = reachable.to(torch.long).argmax(dim=-1)
    eta = t_grid[first_idx]
    fallback = (edge_bias[:num_nodes, :num_nodes, EB_IDX_DISTANCE].clamp_min(0.0) * BOARD_SIZE / speed[:, None]).clamp(0.0, ETA_NORM)
    eta = torch.where(has_hit, eta, fallback)
    diag = torch.arange(num_nodes, device=planet_feats.device)
    eta[diag, diag] = 0.0
    return eta


def _batched_eta_matrix(planet_feats: torch.Tensor, edge_bias: torch.Tensor, planet_mask: torch.Tensor) -> torch.Tensor:
    """Vectorized intercept-ETA matrix over a batch.

    Args
    ----
    planet_feats : (B, P, F_pl)
    edge_bias    : (B, T, T, F_eb) where T >= P. Sliced to (B, P, P, F_eb).
    planet_mask  : (B, P) bool

    Returns
    -------
    eta : (B, P, P) where eta[b, i, j] is intercept ETA from src i to tgt j.
          Padded slots are zeroed out via planet_mask.
    """
    B, P, _ = planet_feats.shape
    # edge_bias may be (B, T, T, F_eb) with T = P_max + F_max + 1 (planet+fleet+global tokens).
    # Slice to the planet-only block for graph computation.
    if edge_bias.shape[1] != P:
        edge_bias = edge_bias[:, :P, :P, :]
    r = planet_feats[..., PF_IDX_R_SUN].clamp_min(0.0) * 50.0
    cos_t = planet_feats[..., PF_IDX_COS_THETA]
    sin_t = planet_feats[..., PF_IDX_SIN_THETA]
    theta = torch.atan2(sin_t, cos_t)
    x = SUN_X + r * cos_t
    y = SUN_Y + r * sin_t
    radius = planet_feats[..., PF_IDX_RADIUS].clamp_min(0.0) * 5.0
    omega = planet_feats[..., PF_IDX_OMEGA_DT] * planet_feats[..., PF_IDX_IS_ROTATING].clamp(0.0, 1.0)
    t_grid = torch.arange(0.0, ETA_NORM + 1.0, 1.0, dtype=planet_feats.dtype, device=planet_feats.device)
    # tgt positions at each time: (B, 1, P_tgt, T)
    tgt_theta = theta[:, None, :, None] + omega[:, None, :, None] * t_grid[None, None, None, :]
    tgt_x = SUN_X + r[:, None, :, None] * torch.cos(tgt_theta)
    tgt_y = SUN_Y + r[:, None, :, None] * torch.sin(tgt_theta)
    # src positions: (B, P_src, 1, 1)
    src_x = x[:, :, None, None]
    src_y = y[:, :, None, None]
    spawn_offset = radius[:, :, None, None] + 0.1
    speed = _fleet_speed_from_ships(planet_feats.new_full((B, P), float(REFERENCE_SHIPS_FOR_ETA)))
    dist = torch.sqrt((tgt_x - src_x).pow(2) + (tgt_y - src_y).pow(2)).sub(spawn_offset).clamp_min(0.0)
    reachable = dist <= speed[:, :, None, None] * t_grid[None, None, None, :]
    has_hit = reachable.any(dim=-1)
    first_idx = reachable.to(torch.long).argmax(dim=-1)
    eta = t_grid[first_idx]
    fallback = (edge_bias[..., EB_IDX_DISTANCE].clamp_min(0.0) * BOARD_SIZE / speed[:, :, None]).clamp(0.0, ETA_NORM)
    eta = torch.where(has_hit, eta, fallback)
    diag = torch.eye(P, dtype=torch.bool, device=eta.device)
    eta = eta.masked_fill(diag[None], 0.0)
    valid = planet_mask[:, :, None].bool() & planet_mask[:, None, :].bool()
    eta = eta * valid.to(eta.dtype)
    return eta


def _owner_masks(feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_self = feats[:, PF_IDX_OWNER_SELF].clamp(0.0, 1.0)
    is_enemy = feats[:, PF_IDX_OWNER_ENEMY_START:PF_IDX_OWNER_ENEMY_END].sum(dim=-1).clamp(0.0, 1.0)
    is_neutral = feats[:, PF_IDX_OWNER_NEUTRAL].clamp(0.0, 1.0)
    return is_self, is_enemy, is_neutral


def _graph_node_metrics(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    num_nodes: int,
    *,
    tau: float = INFLUENCE_TAU_TURNS,
    eta: Optional[torch.Tensor] = None,
    flux_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-node graph metrics for a single-graph slice.

    When ``flux_features`` is provided (shape ``(num_nodes, GRAPH_NODE_FLUX_DIM)``)
    the four flux channels are appended after the base 8 channels and the
    return dimension grows to :data:`GRAPH_NODE_METRIC_DIM_FLUX`.  This is
    used by the reward summary path (Phase 1) and, opt-in, by the encoder
    in Phase 3.
    """
    out_dim = GRAPH_NODE_METRIC_DIM + (
        GRAPH_NODE_FLUX_DIM if flux_features is not None else 0
    )
    if num_nodes <= 0:
        return planet_feats.new_zeros((0, out_dim))
    feats = planet_feats[:num_nodes]
    if eta is None:
        eta = _estimated_intercept_eta_matrix(planet_feats, edge_bias, num_nodes)
    kernel = torch.exp(-eta / max(float(tau), 1e-6))
    support_kernel = kernel.clone()
    support_kernel.fill_diagonal_(0.0)
    is_self, is_enemy, _ = _owner_masks(feats)
    ships = feats[:, PF_IDX_SHIPS].clamp_min(0.0) * 100.0
    production = feats[:, PF_IDX_PRODUCTION].clamp_min(0.0) * 5.0
    incoming_friendly = feats[:, PF_IDX_INCOMING_FRIENDLY_TOTAL].clamp_min(0.0) * 100.0
    incoming_enemy = feats[:, PF_IDX_INCOMING_ENEMY_TOTAL].clamp_min(0.0) * 100.0
    friendly_sources = ships * is_self
    enemy_sources = ships * is_enemy
    friendly_influence = kernel.transpose(0, 1).matmul(friendly_sources)
    enemy_influence = kernel.transpose(0, 1).matmul(enemy_sources)
    friendly_support = support_kernel.transpose(0, 1).matmul(friendly_sources)
    supportability = (friendly_support / (enemy_influence + incoming_enemy + 1.0)).clamp(0.0, 10.0) / 10.0
    vulnerability = (enemy_influence / (ships + incoming_friendly + 1.0)).clamp(0.0, 10.0) / 10.0
    production_potential = kernel.transpose(0, 1).matmul(production)
    friendly_scaled = friendly_influence / 100.0
    enemy_scaled = enemy_influence / 100.0
    production_scaled = production_potential / 25.0
    strategic_potential = production_scaled + friendly_scaled - enemy_scaled
    base = torch.stack([
        friendly_scaled,
        enemy_scaled,
        friendly_scaled - enemy_scaled,
        friendly_support / 100.0,
        supportability,
        vulnerability,
        production_scaled,
        strategic_potential,
    ], dim=-1)
    if flux_features is None:
        return base
    flux = flux_features[:num_nodes].to(dtype=base.dtype, device=base.device)
    if flux.shape != (num_nodes, GRAPH_NODE_FLUX_DIM):
        raise ValueError(
            f"flux_features shape must be ({num_nodes}, {GRAPH_NODE_FLUX_DIM}); got {tuple(flux.shape)}"
        )
    return torch.cat([base, flux], dim=-1)


def _graph_edge_metric_matrix(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    num_nodes: int,
    *,
    eta: Optional[torch.Tensor] = None,
    node_metrics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if num_nodes <= 0:
        return planet_feats.new_zeros((0, 0, GRAPH_EDGE_METRIC_DIM))
    feats = planet_feats[:num_nodes]
    is_self, is_enemy, is_neutral = _owner_masks(feats)
    if eta is None:
        eta = _estimated_intercept_eta_matrix(planet_feats, edge_bias, num_nodes)
    eta_norm = eta.clamp(0.0, ETA_NORM) / ETA_NORM
    crosses_sun = edge_bias[:num_nodes, :num_nodes, EB_IDX_CROSSES_SUN].clamp(0.0, 1.0)
    movement_resistance = (eta_norm * (1.0 + crosses_sun)).clamp(0.0, 2.0) / 2.0
    owner_code = is_self + 2.0 * is_enemy + 3.0 * is_neutral
    same_owner = ((owner_code[:, None] == owner_code[None, :]) & (owner_code[:, None] > 0.0)).to(dtype=planet_feats.dtype)
    if node_metrics is None:
        node_metrics = _graph_node_metrics(planet_feats, edge_bias, num_nodes, eta=eta)
    # Index 7 = ``strategic_potential`` in the base 8 channels; using -1 would
    # break when flux features are appended (Phase 1+) and shift the last
    # column to ``enemy_out``.
    strategic = node_metrics[:, GRAPH_NODE_METRIC_DIM - 1]
    strategic_gradient = (strategic[None, :] - strategic[:, None]).clamp(-5.0, 5.0) / 5.0
    flow_opportunity = torch.relu(strategic[None, :] - strategic[:, None]) / (eta + 1.0)
    return torch.stack([
        eta_norm,
        movement_resistance,
        same_owner,
        is_self[:, None].expand(-1, num_nodes),
        is_self[None, :].expand(num_nodes, -1),
        is_enemy[None, :].expand(num_nodes, -1),
        is_neutral[None, :].expand(num_nodes, -1),
        strategic_gradient,
        flow_opportunity,
    ], dim=-1)


def _planet_pair_edge_features(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    num_nodes: int,
    *,
    eta: Optional[torch.Tensor] = None,
    node_metrics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base = edge_bias[:num_nodes, :num_nodes, :]
    metrics = _graph_edge_metric_matrix(
        planet_feats, edge_bias, num_nodes,
        eta=eta, node_metrics=node_metrics,
    )
    return torch.cat([base, metrics.to(dtype=base.dtype)], dim=-1)


def _padded_planet_pair_edge_features(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    num_nodes: int,
    *,
    eta: Optional[torch.Tensor] = None,
    node_metrics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = edge_bias.new_zeros((planet_feats.shape[0], planet_feats.shape[0], PYG_EDGE_FEATURE_DIM))
    if num_nodes <= 0:
        return out
    out[:num_nodes, :num_nodes] = _planet_pair_edge_features(
        planet_feats, edge_bias, num_nodes,
        eta=eta, node_metrics=node_metrics,
    )
    return out


def _planet_edge_attr(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    num_nodes: int,
    edge_index: torch.Tensor,
    *,
    eta: Optional[torch.Tensor] = None,
    node_metrics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_bias.new_zeros((0, PYG_EDGE_FEATURE_DIM))
    pair = _planet_pair_edge_features(
        planet_feats, edge_bias, num_nodes,
        eta=eta, node_metrics=node_metrics,
    )
    return pair[edge_index[0], edge_index[1]]


def _append_graph_node_metrics(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    num_nodes: int,
    *,
    eta: Optional[torch.Tensor] = None,
    node_metrics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    base = planet_feats[:num_nodes]
    if node_metrics is None:
        node_metrics = _graph_node_metrics(planet_feats, edge_bias, num_nodes, eta=eta)
    return torch.cat([base, node_metrics.to(dtype=base.dtype)], dim=-1)


def observation_to_pyg_data(
    obs: Any,
    *,
    device: Optional[torch.device] = None,
    flux_features: Optional[torch.Tensor] = None,
) -> Any:
    """Convert an :class:`obs_builder.Observation` to a PyG ``Data``.

    When ``flux_features`` (shape ``(P_max, GRAPH_NODE_FLUX_DIM)`` or
    ``(num_real, GRAPH_NODE_FLUX_DIM)``) is provided, the node feature
    vector is widened to :data:`PYG_PLANET_FEATURE_DIM_FLUX` (= 46).
    Callers opting in must use an encoder built with
    ``enable_flux_features=True``.
    """
    _require_pyg()
    planet_feats = torch.as_tensor(obs.planet_feats, dtype=torch.float32, device=device)
    edge_bias = torch.as_tensor(obs.edge_bias, dtype=torch.float32, device=device)
    planet_mask = torch.as_tensor(obs.planet_mask, dtype=torch.bool, device=device)
    source_mask = torch.as_tensor(obs.source_mask, dtype=torch.bool, device=device)
    global_feats = torch.as_tensor(obs.global_feats, dtype=torch.float32, device=device)
    n = int(planet_mask.sum().item())
    edge_index = _planet_edge_index(n, device=device)
    # Compute ETA + node metrics once, share with edge attr and node features.
    eta = _estimated_intercept_eta_matrix(planet_feats, edge_bias, n) if n > 0 else None
    flux_slice = None
    if flux_features is not None and n > 0:
        flux_t = torch.as_tensor(flux_features, dtype=torch.float32, device=device)
        if flux_t.shape[0] >= n:
            flux_slice = flux_t[:n]
        else:
            raise ValueError(
                f"flux_features first dim must be >= num_real_planets ({n}); got {flux_t.shape[0]}"
            )
    node_metrics = (
        _graph_node_metrics(planet_feats, edge_bias, n, eta=eta, flux_features=flux_slice)
        if n > 0 else None
    )
    return Data(
        x=_append_graph_node_metrics(planet_feats, edge_bias, n, node_metrics=node_metrics),
        edge_index=edge_index,
        edge_attr=_planet_edge_attr(
            planet_feats, edge_bias, n, edge_index,
            eta=eta, node_metrics=node_metrics,
        ),
        global_feats=global_feats.unsqueeze(0),
        planet_slots=torch.arange(n, dtype=torch.long, device=device),
        planet_mask=planet_mask,
        source_mask=source_mask,
        p_max=torch.tensor([planet_feats.shape[0]], dtype=torch.long, device=device),
    )


def observations_to_pyg_batch(
    obs_list: Iterable[Any],
    *,
    device: Optional[torch.device] = None,
    flux_features_list: Optional[Iterable[torch.Tensor]] = None,
) -> Any:
    """Batch a list of observations into a single PyG ``Batch``.

    ``flux_features_list``, if given, must be aligned with ``obs_list``
    and supplies per-observation flux feature tensors (shape
    ``(P_max, GRAPH_NODE_FLUX_DIM)``).  ``None`` entries fall back to
    the legacy 8-channel node metric path.
    """
    _require_pyg()
    if flux_features_list is None:
        return Batch.from_data_list([observation_to_pyg_data(obs, device=device) for obs in obs_list])
    return Batch.from_data_list([
        observation_to_pyg_data(obs, device=device, flux_features=flux)
        for obs, flux in zip(obs_list, flux_features_list)
    ])


def _batched_graph_node_metrics(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    planet_mask: torch.Tensor,
    *,
    tau: float = INFLUENCE_TAU_TURNS,
    eta: Optional[torch.Tensor] = None,
    flux_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized graph node metrics over a padded batch.

    Returns ``(B, P, GRAPH_NODE_METRIC_DIM)`` by default, or
    ``(B, P, GRAPH_NODE_METRIC_DIM_FLUX)`` when ``flux_features`` (shape
    ``(B, P, GRAPH_NODE_FLUX_DIM)``) is provided.  Padded slots are zeroed.
    """
    B, P, _ = planet_feats.shape
    if eta is None:
        eta = _batched_eta_matrix(planet_feats, edge_bias, planet_mask)
    kernel = torch.exp(-eta / max(float(tau), 1e-6))
    valid = planet_mask[:, :, None].bool() & planet_mask[:, None, :].bool()
    kernel = kernel * valid.to(kernel.dtype)
    diag = torch.eye(P, dtype=torch.bool, device=kernel.device)
    support_kernel = kernel.masked_fill(diag[None], 0.0)
    is_self = planet_feats[..., PF_IDX_OWNER_SELF].clamp(0.0, 1.0)
    is_enemy = planet_feats[..., PF_IDX_OWNER_ENEMY_START:PF_IDX_OWNER_ENEMY_END].sum(dim=-1).clamp(0.0, 1.0)
    ships = planet_feats[..., PF_IDX_SHIPS].clamp_min(0.0) * 100.0
    production = planet_feats[..., PF_IDX_PRODUCTION].clamp_min(0.0) * 5.0
    incoming_friendly = planet_feats[..., PF_IDX_INCOMING_FRIENDLY_TOTAL].clamp_min(0.0) * 100.0
    incoming_enemy = planet_feats[..., PF_IDX_INCOMING_ENEMY_TOTAL].clamp_min(0.0) * 100.0
    friendly_sources = (ships * is_self).unsqueeze(-1)
    enemy_sources = (ships * is_enemy).unsqueeze(-1)
    # kernel.T @ sources broadcast over batch: (B, P, P) -> (B, P_tgt, P_src) @ (B, P_src, 1)
    kT = kernel.transpose(-1, -2)
    friendly_influence = torch.bmm(kT, friendly_sources).squeeze(-1)
    enemy_influence = torch.bmm(kT, enemy_sources).squeeze(-1)
    friendly_support = torch.bmm(support_kernel.transpose(-1, -2), friendly_sources).squeeze(-1)
    supportability = (friendly_support / (enemy_influence + incoming_enemy + 1.0)).clamp(0.0, 10.0) / 10.0
    vulnerability = (enemy_influence / (ships + incoming_friendly + 1.0)).clamp(0.0, 10.0) / 10.0
    production_potential = torch.bmm(kT, production.unsqueeze(-1)).squeeze(-1)
    friendly_scaled = friendly_influence / 100.0
    enemy_scaled = enemy_influence / 100.0
    production_scaled = production_potential / 25.0
    strategic_potential = production_scaled + friendly_scaled - enemy_scaled
    metrics = torch.stack([
        friendly_scaled,
        enemy_scaled,
        friendly_scaled - enemy_scaled,
        friendly_support / 100.0,
        supportability,
        vulnerability,
        production_scaled,
        strategic_potential,
    ], dim=-1)
    metrics = metrics * planet_mask.to(metrics.dtype).unsqueeze(-1)
    if flux_features is not None:
        flux = flux_features.to(dtype=metrics.dtype, device=metrics.device)
        if flux.shape != (B, P, GRAPH_NODE_FLUX_DIM):
            raise ValueError(
                f"flux_features shape must be ({B}, {P}, {GRAPH_NODE_FLUX_DIM}); got {tuple(flux.shape)}"
            )
        flux = flux * planet_mask.to(flux.dtype).unsqueeze(-1)
        metrics = torch.cat([metrics, flux], dim=-1)
    return metrics


def _batched_graph_edge_metrics(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    planet_mask: torch.Tensor,
    *,
    eta: Optional[torch.Tensor] = None,
    node_metrics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized edge metrics over a padded batch.

    Returns (B, P, P, GRAPH_EDGE_METRIC_DIM). Padded pairs are zeroed.
    """
    B, P, _ = planet_feats.shape
    if edge_bias.shape[1] != P:
        edge_bias = edge_bias[:, :P, :P, :]
    if eta is None:
        eta = _batched_eta_matrix(planet_feats, edge_bias, planet_mask)
    if node_metrics is None:
        node_metrics = _batched_graph_node_metrics(planet_feats, edge_bias, planet_mask, eta=eta)
    is_self = planet_feats[..., PF_IDX_OWNER_SELF].clamp(0.0, 1.0)
    is_enemy = planet_feats[..., PF_IDX_OWNER_ENEMY_START:PF_IDX_OWNER_ENEMY_END].sum(dim=-1).clamp(0.0, 1.0)
    is_neutral = planet_feats[..., PF_IDX_OWNER_NEUTRAL].clamp(0.0, 1.0)
    eta_norm = eta.clamp(0.0, ETA_NORM) / ETA_NORM
    crosses_sun = edge_bias[..., EB_IDX_CROSSES_SUN].clamp(0.0, 1.0)
    movement_resistance = (eta_norm * (1.0 + crosses_sun)).clamp(0.0, 2.0) / 2.0
    owner_code = is_self + 2.0 * is_enemy + 3.0 * is_neutral
    same_owner = ((owner_code[:, :, None] == owner_code[:, None, :]) & (owner_code[:, :, None] > 0.0)).to(planet_feats.dtype)
    # Fixed index instead of -1 so flux-augmented node metrics (Phase 1+)
    # don't shift the strategic_potential column.
    strategic = node_metrics[..., GRAPH_NODE_METRIC_DIM - 1]
    strategic_gradient = (strategic[:, None, :] - strategic[:, :, None]).clamp(-5.0, 5.0) / 5.0
    flow_opportunity = torch.relu(strategic[:, None, :] - strategic[:, :, None]) / (eta + 1.0)
    is_self_src = is_self[:, :, None].expand(B, P, P)
    is_self_tgt = is_self[:, None, :].expand(B, P, P)
    is_enemy_tgt = is_enemy[:, None, :].expand(B, P, P)
    is_neutral_tgt = is_neutral[:, None, :].expand(B, P, P)
    metrics = torch.stack([
        eta_norm,
        movement_resistance,
        same_owner,
        is_self_src,
        is_self_tgt,
        is_enemy_tgt,
        is_neutral_tgt,
        strategic_gradient,
        flow_opportunity,
    ], dim=-1)
    valid = planet_mask[:, :, None].bool() & planet_mask[:, None, :].bool()
    metrics = metrics * valid.to(metrics.dtype).unsqueeze(-1)
    return metrics


def _batched_pair_edge_features(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    planet_mask: torch.Tensor,
    *,
    eta: Optional[torch.Tensor] = None,
    node_metrics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized concatenation of base edge_bias and graph edge metrics.

    Returns (B, P, P, PYG_EDGE_FEATURE_DIM). Padded pairs are zeroed.
    """
    B, P, _ = planet_feats.shape
    if edge_bias.shape[1] != P:
        edge_bias = edge_bias[:, :P, :P, :]
    if eta is None:
        eta = _batched_eta_matrix(planet_feats, edge_bias, planet_mask)
    if node_metrics is None:
        node_metrics = _batched_graph_node_metrics(planet_feats, edge_bias, planet_mask, eta=eta)
    edge_metrics = _batched_graph_edge_metrics(
        planet_feats, edge_bias, planet_mask, eta=eta, node_metrics=node_metrics,
    )
    valid = planet_mask[:, :, None].bool() & planet_mask[:, None, :].bool()
    base = edge_bias * valid.to(edge_bias.dtype).unsqueeze(-1)
    return torch.cat([base, edge_metrics.to(base.dtype)], dim=-1)


def packed_observations_to_pyg_batch(
    batch: dict[str, torch.Tensor],
    *,
    flux_features: Optional[torch.Tensor] = None,
) -> Any:
    """Convert a packed observation batch (output of ``pack_observations``)
    into a PyG ``Batch``.

    ``flux_features`` is an optional ``(B, P_max, GRAPH_NODE_FLUX_DIM)``
    tensor.  When provided the resulting ``Data.x`` has
    :data:`PYG_PLANET_FEATURE_DIM_FLUX` (= 46) columns instead of 42; the
    consumer encoder must be built with ``enable_flux_features=True``.
    The tensor may also be passed via ``batch["flux_features"]`` for
    convenience.
    """
    _require_pyg()
    planet_feats = batch["planet_feats"]
    edge_bias = batch["edge_bias"]
    planet_mask = batch["planet_mask"].bool()
    source_mask = batch["source_mask"].bool()
    global_feats = batch["global_feats"]
    if flux_features is None:
        flux_features = batch.get("flux_features", None)
    B, P_max, _ = planet_feats.shape
    if flux_features is not None:
        flux_features = torch.as_tensor(
            flux_features, dtype=planet_feats.dtype, device=planet_feats.device,
        )
        if flux_features.shape != (B, P_max, GRAPH_NODE_FLUX_DIM):
            raise ValueError(
                f"flux_features shape must be ({B}, {P_max}, {GRAPH_NODE_FLUX_DIM}); "
                f"got {tuple(flux_features.shape)}"
            )
    # Vectorized ETA + node-metric computation (one pass over batch).
    eta_b = _batched_eta_matrix(planet_feats, edge_bias, planet_mask)
    node_metrics_b = _batched_graph_node_metrics(
        planet_feats, edge_bias, planet_mask, eta=eta_b, flux_features=flux_features,
    )
    data_list = []
    for b in range(B):
        n = int(planet_mask[b].sum().item())
        edge_index = _planet_edge_index(n, device=planet_feats.device)
        eta_b_slice = eta_b[b, :n, :n] if n > 0 else None
        nm_slice = node_metrics_b[b, :n] if n > 0 else None
        data_list.append(Data(
            x=_append_graph_node_metrics(
                planet_feats[b], edge_bias[b], n,
                eta=eta_b_slice, node_metrics=nm_slice,
            ),
            edge_index=edge_index,
            edge_attr=_planet_edge_attr(
                planet_feats[b], edge_bias[b], n, edge_index,
                eta=eta_b_slice, node_metrics=nm_slice,
            ),
            global_feats=global_feats[b].unsqueeze(0),
            planet_slots=torch.arange(n, dtype=torch.long, device=planet_feats.device),
            planet_mask=planet_mask[b],
            source_mask=source_mask[b],
            p_max=torch.tensor([P_max], dtype=torch.long, device=planet_feats.device),
        ))
    return Batch.from_data_list(data_list)


class BilinearEdgeHead(nn.Module):
    """Edge-scoring head that fuses src/tgt/edge/global via additive projection.

    Replaces ``MLP(concat([src, tgt, edge, global]))`` (4*d_model -> d_model)
    with parallel ``Linear(d_model -> d_model)`` projections, summed and
    fed through a small fuse MLP.  Same parameter count, **far fewer FLOPs**
    on the (B, P, P, _) tensor because the heavy projection is done at the
    per-node level (B*P) before pairwise broadcasting.
    """

    def __init__(self, d_model: int, output_dim: int = 1):
        super().__init__()
        self.W_src = nn.Linear(d_model, d_model)
        self.W_tgt = nn.Linear(d_model, d_model)
        self.W_edge = nn.Linear(d_model, d_model)
        self.W_global = nn.Linear(d_model, d_model)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self,
        src_tok: torch.Tensor,
        tgt_tok: torch.Tensor,
        edge_tok: torch.Tensor,
        global_tok: torch.Tensor,
    ) -> torch.Tensor:
        """Returns (B, P, P, output_dim).

        Shapes
        ------
        src_tok    : (B, P, d)
        tgt_tok    : (B, P, d)
        edge_tok   : (B, P, P, d)
        global_tok : (B, d)
        """
        s = self.W_src(src_tok).unsqueeze(2)        # (B, P, 1, d)
        t = self.W_tgt(tgt_tok).unsqueeze(1)        # (B, 1, P, d)
        e = self.W_edge(edge_tok)                   # (B, P, P, d)
        g = self.W_global(global_tok).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, d)
        h = s + t + e + g                            # broadcast -> (B, P, P, d)
        return self.fuse(h)


class SinkhornEdgeHead(nn.Module):
    """Edge-scoring head with log-domain Sinkhorn iterations.

    Replaces a plain :class:`BilinearEdgeHead(d_model, output_dim=1)` with
    the same scorer followed by *K* alternating row / column
    ``logsumexp`` normalization passes.  The result is an approximately
    doubly-stochastic edge logit tensor: in exp-space each row sums to
    one and each column sums to one (modulo masked cells).

    Why this matters for multi-launch coordination:

        * Plain BilinearEdgeHead can let a single attractive *target*
          dominate every source's logit row, encouraging all empires to
          pile onto the same planet.
        * Sinkhorn enforces that, on average, every source spreads its
          probability mass across the target axis, **and** every target
          absorbs only one source's worth of mass.  This is the right
          inductive bias for a "transport plan" over ship flow.

    PPO compatibility: the head returns ``(B, P, P, 1)`` logits with the
    *same* shape as :class:`BilinearEdgeHead(output_dim=1)`, so
    :class:`PygEdgeNativePolicy`'s downstream ``Categorical`` sampling
    and ``evaluate_actions`` paths are unchanged.

    Parameters
    ----------
    d_model:
        Hidden width passed to the underlying scorer.
    num_iterations:
        Alternating row / col normalization passes (``K``).  ``K=0``
        bypasses Sinkhorn and behaves like the bilinear baseline.
    tau:
        Temperature applied to the raw scores before normalization;
        ``tau < 1`` sharpens the resulting plan, ``tau > 1`` softens it.
        Must be strictly positive.
    """

    def __init__(self, d_model: int, *, num_iterations: int = 5, tau: float = 1.0):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"sinkhorn tau must be > 0; got {tau}")
        if num_iterations < 0:
            raise ValueError(f"num_iterations must be >= 0; got {num_iterations}")
        self.scorer = BilinearEdgeHead(d_model, output_dim=1)
        self.num_iterations = int(num_iterations)
        self.tau = float(tau)

    @staticmethod
    def sinkhorn_log(
        log_scores: torch.Tensor,
        num_iterations: int,
        *,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ``num_iterations`` log-domain Sinkhorn passes.

        ``log_scores`` shape: ``(B, P, P)``.  ``edge_mask`` (optional)
        is a boolean tensor of the same shape with ``True`` marking
        valid edges; masked cells are pushed to ``-inf`` and stay there
        across iterations.
        """
        L = log_scores
        very_neg = torch.finfo(L.dtype).min
        if edge_mask is not None:
            L = L.masked_fill(~edge_mask, very_neg)
        for _ in range(num_iterations):
            L = L - torch.logsumexp(L, dim=2, keepdim=True)
            L = L - torch.logsumexp(L, dim=1, keepdim=True)
        return L

    def forward(
        self,
        src_tok: torch.Tensor,
        tgt_tok: torch.Tensor,
        edge_tok: torch.Tensor,
        global_tok: torch.Tensor,
        *,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns ``(B, P, P, 1)`` log-domain (approximately) doubly-stochastic logits."""
        raw = self.scorer(src_tok, tgt_tok, edge_tok, global_tok).squeeze(-1)
        L = raw / self.tau
        if self.num_iterations > 0:
            L = self.sinkhorn_log(L, self.num_iterations, edge_mask=edge_mask)
        return L.unsqueeze(-1)


class PygPlanetGraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        planet_dim: int = PYG_PLANET_FEATURE_DIM,
        edge_dim: int = PYG_EDGE_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.0,
        enable_flux_features: bool = False,
    ):
        """GNN encoder over the (planets-only) Orbit Wars graph.

        Set ``enable_flux_features=True`` to expand the node input to
        :data:`PYG_PLANET_FEATURE_DIM_FLUX` (= 46) so the four
        ``[friendly_in, friendly_out, enemy_in, enemy_out]`` flux
        channels can be consumed.  Callers must then ensure the
        downstream ``Data.x`` actually carries 46 columns -- the
        ``observation_to_pyg_data`` / ``packed_observations_to_pyg_batch``
        helpers take a ``flux_features`` argument for this.
        """
        super().__init__()
        _require_pyg()
        if enable_flux_features and planet_dim == PYG_PLANET_FEATURE_DIM:
            planet_dim = PYG_PLANET_FEATURE_DIM_FLUX
        self.enable_flux_features = bool(enable_flux_features)
        self.planet_dim = planet_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.d_model = d_model
        self.n_layers = n_layers
        # Input LayerNorm normalizes the heterogeneous graph metric scales
        # (ships vs. production vs. normalized 0-1 metrics) before the projection.
        self.node_proj = nn.Sequential(
            nn.LayerNorm(planet_dim),
            nn.Linear(planet_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.edge_proj = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.convs.append(GINEConv(mlp, edge_dim=d_model))
            self.norms.append(nn.LayerNorm(d_model))
        self.summary = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, graph_batch: Any) -> dict[str, torch.Tensor]:
        x = self.node_proj(graph_batch.x)
        edge_attr = self.edge_proj(graph_batch.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            msg = conv(x, graph_batch.edge_index, edge_attr)
            x = norm(x + self.dropout(msg))
        mean_pool = global_mean_pool(x, graph_batch.batch)
        max_pool = global_max_pool(x, graph_batch.batch)
        global_tok = self.global_proj(graph_batch.global_feats)
        graph_summary = self.summary(torch.cat([mean_pool, max_pool, global_tok], dim=-1))
        return {"node_tokens": x, "graph_summary": graph_summary}


def pyg_nodes_to_padded(node_tokens: torch.Tensor, graph_batch: Any, *, p_max: int) -> torch.Tensor:
    B = int(graph_batch.num_graphs)
    out = node_tokens.new_zeros((B, p_max, node_tokens.shape[-1]))
    out[graph_batch.batch, graph_batch.planet_slots] = node_tokens
    return out


class PygPlanetGraphPolicy(GroupedPlanetOrbitWarsPolicy):
    def __init__(
        self,
        *,
        p_max: int = P_MAX_DEFAULT,
        planet_dim: int = PYG_PLANET_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        edge_dim: int = PYG_EDGE_FEATURE_DIM,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.0,
        enable_flux_features: bool = False,
    ):
        nn.Module.__init__(self)
        _require_pyg()
        self.p_max = p_max
        # ``planet_dim`` is exposed as the *raw* feature dim so that
        # ``build_observation`` (used by the rollout) produces tensors of
        # the right shape.  The encoder receives augmented features
        # (PYG_PLANET_FEATURE_DIM = raw + GRAPH_NODE_METRIC_DIM) appended
        # internally by ``packed_observations_to_pyg_batch``.
        self.planet_dim = PLANET_FEATURE_DIM
        # When flux features are enabled the encoder ingests
        # PYG_PLANET_FEATURE_DIM_FLUX (= 46) columns, otherwise 42.
        if enable_flux_features and planet_dim == PYG_PLANET_FEATURE_DIM:
            planet_dim = PYG_PLANET_FEATURE_DIM_FLUX
        self.enable_flux_features = bool(enable_flux_features)
        self.encoder_planet_dim = planet_dim
        self.global_dim = global_dim
        self.edge_dim = edge_dim
        self.d_model = d_model
        self.f_max = F_MAX_DEFAULT
        self.model_variant = "pyg_planet_graph"
        self.encoder = PygPlanetGraphEncoder(
            planet_dim=planet_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            enable_flux_features=enable_flux_features,
        )
        self.src_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.noop_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # Edge projection is shared with the encoder: same input (raw edge
        # features) projected to the same d_model space.  Saves parameters
        # and keeps representations consistent across message-passing and
        # action scoring.
        # Bilinear heads: fewer FLOPs than 4*d_model concat MLP for (B, P, P, _).
        self.pair_score_head = BilinearEdgeHead(d_model, output_dim=1)
        self.ship_head = BilinearEdgeHead(d_model, output_dim=2)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.aux_owner_head = nn.Linear(d_model, 5)

    def forward(
        self,
        planet_feats: torch.Tensor,
        fleet_feats: torch.Tensor,
        global_feats: torch.Tensor,
        edge_bias: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
        source_mask: torch.Tensor,
        flux_features: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        # When the encoder was built with flux features, callers must
        # supply a ``(B, P_max, 4)`` tensor; otherwise we fail loudly
        # rather than silently broadcasting zeros (which would mask bugs
        # in the rollout / buffer plumbing).
        if self.enable_flux_features and flux_features is None:
            raise ValueError(
                "PygPlanetGraphPolicy was built with enable_flux_features=True "
                "but `flux_features` was not passed to forward()."
            )
        batch_dict: dict[str, torch.Tensor] = {
            "planet_feats": planet_feats,
            "global_feats": global_feats,
            "edge_bias": edge_bias,
            "planet_mask": planet_mask,
            "source_mask": source_mask,
        }
        if self.enable_flux_features:
            batch_dict["flux_features"] = flux_features  # type: ignore[assignment]
        graph_batch = packed_observations_to_pyg_batch(batch_dict)
        enc = self.encoder(graph_batch)
        planet_tok = pyg_nodes_to_padded(enc["node_tokens"], graph_batch, p_max=self.p_max)
        graph_summary = enc["graph_summary"]
        global_per_planet = graph_summary[:, None, :].expand(-1, self.p_max, -1)
        src_input = torch.cat([planet_tok, global_per_planet], dim=-1)
        source_launch = self.src_head(src_input).squeeze(-1)
        noop_raw = self.noop_head(src_input).squeeze(-1)
        very_neg = torch.finfo(source_launch.dtype).min
        source_launch = source_launch.masked_fill(~source_mask, very_neg)
        noop_raw = noop_raw.masked_fill(~source_mask, very_neg)
        noop = noop_raw.max(dim=1, keepdim=True).values
        src_logits = torch.cat([noop_raw, noop], dim=1)

        # Vectorized pairwise edge features (one batched ETA pass).
        edge_pair = _batched_pair_edge_features(planet_feats, edge_bias, planet_mask)
        edge_tok = self.encoder.edge_proj(edge_pair)

        tgt_logits = self.pair_score_head(planet_tok, planet_tok, edge_tok, graph_summary).squeeze(-1)
        tgt_logits = tgt_logits + source_launch[:, :, None]
        eye = torch.eye(self.p_max, dtype=torch.bool, device=planet_feats.device)
        tgt_logits = tgt_logits.masked_fill(eye[None], very_neg)
        tgt_logits = tgt_logits.masked_fill((~planet_mask)[:, None, :], very_neg)
        tgt_logits = tgt_logits.masked_fill((~source_mask)[:, :, None], very_neg)

        ship_params = self.ship_head(planet_tok, planet_tok, edge_tok, graph_summary)
        value = self.value_head(graph_summary).squeeze(-1)
        aux_owner_logits = self.aux_owner_head(planet_tok)
        return PolicyOutput(
            src_logits=src_logits,
            tgt_logits=tgt_logits,
            ship_params=ship_params,
            value=value,
            tokens=planet_tok,
            aux_owner_logits=aux_owner_logits,
        )
