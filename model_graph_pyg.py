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
GRAPH_NODE_METRIC_DIM = 10
# Index of selected base channels (stable when new channels are appended).
GRAPH_NODE_METRIC_IDX_STRATEGIC = 8
GRAPH_NODE_METRIC_IDX_HELD_DISCOUNTED_PRODUCTION = 9
# When flux features ([friendly_in, friendly_out, enemy_in, enemy_out]) are
# appended to the base node metrics (see ``rl/flux.py``), the per-node
# metric vector grows to this many channels.  Phase 1 only uses this in
# the reward summary path; the encoder still consumes the base channels.
GRAPH_NODE_FLUX_DIM = 4
GRAPH_NODE_METRIC_DIM_FLUX = GRAPH_NODE_METRIC_DIM + GRAPH_NODE_FLUX_DIM  # = 14
PYG_PLANET_FEATURE_DIM = PLANET_FEATURE_DIM + GRAPH_NODE_METRIC_DIM
PYG_PLANET_FEATURE_DIM_FLUX = PLANET_FEATURE_DIM + GRAPH_NODE_METRIC_DIM_FLUX
GRAPH_EDGE_METRIC_DIM = 9
PYG_EDGE_FEATURE_DIM = EDGE_FEATURE_DIM + GRAPH_EDGE_METRIC_DIM
# Number of directed flux channels appended to the per-edge feature
# vector when ``enable_flux_features=True`` is wired through the encoder:
# ``[friendly_flux_ij, enemy_flux_ij]`` (ETA-weighted, normalised by
# ``ship_norm`` to share the scale with the node flux channels).
GRAPH_EDGE_FLUX_DIM = 2
PYG_EDGE_FEATURE_DIM_FLUX = PYG_EDGE_FEATURE_DIM + GRAPH_EDGE_FLUX_DIM  # = 17
BOARD_SIZE = 100.0
SUN_X = 50.0
SUN_Y = 50.0
DEFAULT_MAX_SHIP_SPEED = 6.0
REFERENCE_SHIPS_FOR_ETA = 50
ETA_NORM = 50.0
INFLUENCE_TAU_TURNS = 12.0
DEFAULT_MAX_TURNS_FOR_GRAPH = 500
DEFAULT_HELD_PRODUCTION_NORM = 500.0
DEFAULT_HOLD_SURVIVAL_BASE = 0.99
DEFAULT_HOLD_SURVIVAL_MIN = 0.5
DEFAULT_HOLD_SURVIVAL_MAX = 0.99


def _hold_metric_kwargs_from_obs(obs: Any) -> dict[str, Any]:
    """Resolve held-production kwargs for graph node metrics from an Observation."""
    meta = getattr(obs, "meta", None) or {}
    remaining = meta.get("remaining_turns")
    if remaining is None:
        gf = getattr(obs, "global_feats", None)
        turn = int(round(float(gf[0]) * 200.0)) if gf is not None and len(gf) > 0 else 0
        max_turns = int(meta.get("max_turns", DEFAULT_MAX_TURNS_FOR_GRAPH))
        remaining = max(0, max_turns - turn)
    return {
        "remaining_turns": int(remaining),
        "held_production_norm": float(
            meta.get("held_production_norm", DEFAULT_HELD_PRODUCTION_NORM)
        ),
        "hold_survival_base": float(
            meta.get("hold_survival_base", DEFAULT_HOLD_SURVIVAL_BASE)
        ),
        "hold_survival_from_vulnerability": bool(
            meta.get("hold_survival_from_vulnerability", True)
        ),
        "hold_survival_min": float(
            meta.get("hold_survival_min", DEFAULT_HOLD_SURVIVAL_MIN)
        ),
        "hold_survival_max": float(
            meta.get("hold_survival_max", DEFAULT_HOLD_SURVIVAL_MAX)
        ),
    }


def _remaining_turns_from_batch(batch: dict[str, Any]) -> Optional[torch.Tensor]:
    """Per-batch remaining turns ``(B,)`` from global feats (turn / 200)."""
    gf = batch.get("global_feats")
    if gf is None:
        return None
    turns = (gf[:, 0] * 200.0).round().long().clamp(min=0)
    max_turns = int(batch.get("max_turns", DEFAULT_MAX_TURNS_FOR_GRAPH))
    return (max_turns - turns).clamp(min=0)

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
    """Ensure PyG symbols are importable (retries after a mid-session pip install)."""
    global Batch, Data, GINEConv, global_max_pool, global_mean_pool
    if Data is not None and Batch is not None and GINEConv is not None:
        return
    try:
        from torch_geometric.data import Batch as _Batch, Data as _Data
        from torch_geometric.nn import (
            GINEConv as _GINEConv,
            global_max_pool as _global_max_pool,
            global_mean_pool as _global_mean_pool,
        )
    except ImportError as exc:
        raise ImportError(
            "model_graph_pyg requires torch_geometric. Install a torch-compatible "
            "PyTorch Geometric build before using PygPlanetGraphPolicy."
        ) from exc
    Batch, Data = _Batch, _Data
    GINEConv = _GINEConv
    global_max_pool = _global_max_pool
    global_mean_pool = _global_mean_pool


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
    remaining_turns: int = 0,
    held_production_norm: float = 500.0,
    hold_survival_base: float = 0.99,
    hold_survival_from_vulnerability: bool = True,
    hold_survival_min: float = 0.5,
    hold_survival_max: float = 0.99,
) -> torch.Tensor:
    """Compute per-node graph metrics for a single-graph slice.

    When ``flux_features`` is provided (shape ``(num_nodes, GRAPH_NODE_FLUX_DIM)``)
    the four flux channels are appended after the base 10 channels and the
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
    friendly_production_sources = production * is_self
    friendly_influence = kernel.transpose(0, 1).matmul(friendly_sources)
    enemy_influence = kernel.transpose(0, 1).matmul(enemy_sources)
    friendly_support = support_kernel.transpose(0, 1).matmul(friendly_sources)
    supportability = (friendly_support / (enemy_influence + incoming_enemy + 1.0)).clamp(0.0, 10.0) / 10.0
    vulnerability = (enemy_influence / (ships + incoming_friendly + 1.0)).clamp(0.0, 10.0) / 10.0
    production_potential = kernel.transpose(0, 1).matmul(production)
    friendly_production_potential = kernel.transpose(0, 1).matmul(friendly_production_sources)
    friendly_scaled = friendly_influence / 100.0
    enemy_scaled = enemy_influence / 100.0
    production_scaled = production_potential / 25.0
    friendly_production_scaled = friendly_production_potential / 25.0
    strategic_potential = production_scaled + friendly_scaled - enemy_scaled
    from rl.held_production import held_discounted_production_value

    held_discounted_production = held_discounted_production_value(
        production,
        is_self,
        vulnerability,
        remaining_turns=int(max(0, remaining_turns)),
        norm=float(held_production_norm),
        base=float(hold_survival_base),
        from_vulnerability=bool(hold_survival_from_vulnerability),
        survival_min=float(hold_survival_min),
        survival_max=float(hold_survival_max),
    )
    base = torch.stack([
        friendly_scaled,
        enemy_scaled,
        friendly_scaled - enemy_scaled,
        friendly_support / 100.0,
        supportability,
        vulnerability,
        production_scaled,
        friendly_production_scaled,
        strategic_potential,
        held_discounted_production,
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
    strategic = node_metrics[:, GRAPH_NODE_METRIC_IDX_STRATEGIC]
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
    edge_flux_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pair-wise edge feature stack of shape ``(n, n, D)``.

    When ``edge_flux_features`` (shape ``(n, n, GRAPH_EDGE_FLUX_DIM)``)
    is provided the two flux channels are appended after the metrics
    block so ``D == PYG_EDGE_FEATURE_DIM_FLUX`` (= 17); otherwise
    ``D == PYG_EDGE_FEATURE_DIM`` (= 15).
    """
    base = edge_bias[:num_nodes, :num_nodes, :]
    metrics = _graph_edge_metric_matrix(
        planet_feats, edge_bias, num_nodes,
        eta=eta, node_metrics=node_metrics,
    )
    parts = [base, metrics.to(dtype=base.dtype)]
    if edge_flux_features is not None:
        flux = edge_flux_features[:num_nodes, :num_nodes]
        if flux.shape != (num_nodes, num_nodes, GRAPH_EDGE_FLUX_DIM):
            raise ValueError(
                f"edge_flux_features shape must be ({num_nodes}, {num_nodes}, "
                f"{GRAPH_EDGE_FLUX_DIM}); got {tuple(flux.shape)}"
            )
        parts.append(flux.to(dtype=base.dtype))
    return torch.cat(parts, dim=-1)


def _padded_planet_pair_edge_features(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    num_nodes: int,
    *,
    eta: Optional[torch.Tensor] = None,
    node_metrics: Optional[torch.Tensor] = None,
    edge_flux_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out_dim = (
        PYG_EDGE_FEATURE_DIM_FLUX if edge_flux_features is not None else PYG_EDGE_FEATURE_DIM
    )
    out = edge_bias.new_zeros((planet_feats.shape[0], planet_feats.shape[0], out_dim))
    if num_nodes <= 0:
        return out
    out[:num_nodes, :num_nodes] = _planet_pair_edge_features(
        planet_feats, edge_bias, num_nodes,
        eta=eta, node_metrics=node_metrics,
        edge_flux_features=edge_flux_features,
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
    edge_flux_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out_dim = (
        PYG_EDGE_FEATURE_DIM_FLUX if edge_flux_features is not None else PYG_EDGE_FEATURE_DIM
    )
    if edge_index.numel() == 0:
        return edge_bias.new_zeros((0, out_dim))
    pair = _planet_pair_edge_features(
        planet_feats, edge_bias, num_nodes,
        eta=eta, node_metrics=node_metrics,
        edge_flux_features=edge_flux_features,
    )
    return pair[edge_index[0], edge_index[1]]


def _append_graph_node_metrics(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    num_nodes: int,
    *,
    eta: Optional[torch.Tensor] = None,
    node_metrics: Optional[torch.Tensor] = None,
    hold_metric_kwargs: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    base = planet_feats[:num_nodes]
    if node_metrics is None:
        kwargs = dict(hold_metric_kwargs or {})
        node_metrics = _graph_node_metrics(
            planet_feats, edge_bias, num_nodes, eta=eta, **kwargs,
        )
    return torch.cat([base, node_metrics.to(dtype=base.dtype)], dim=-1)


def observation_to_pyg_data(
    obs: Any,
    *,
    device: Optional[torch.device] = None,
    flux_features: Optional[torch.Tensor] = None,
    edge_flux_features: Optional[torch.Tensor] = None,
) -> Any:
    """Convert an :class:`obs_builder.Observation` to a PyG ``Data``.

    When ``flux_features`` (shape ``(P_max, GRAPH_NODE_FLUX_DIM)`` or
    ``(num_real, GRAPH_NODE_FLUX_DIM)``) is provided, the node feature
    vector is widened to :data:`PYG_PLANET_FEATURE_DIM_FLUX` (= 47).
    When ``edge_flux_features`` (shape ``(P_max, P_max, GRAPH_EDGE_FLUX_DIM)``
    or ``(num_real, num_real, GRAPH_EDGE_FLUX_DIM)``) is provided the
    edge feature vector is widened to :data:`PYG_EDGE_FEATURE_DIM_FLUX`
    (= 17).  Callers opting in must use an encoder built with
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
    edge_flux_slice = None
    if edge_flux_features is not None and n > 0:
        ef_t = torch.as_tensor(edge_flux_features, dtype=torch.float32, device=device)
        if ef_t.ndim != 3 or ef_t.shape[-1] != GRAPH_EDGE_FLUX_DIM:
            raise ValueError(
                f"edge_flux_features must have shape (P, P, {GRAPH_EDGE_FLUX_DIM}); "
                f"got {tuple(ef_t.shape)}"
            )
        if ef_t.shape[0] < n or ef_t.shape[1] < n:
            raise ValueError(
                f"edge_flux_features first two dims must be >= num_real_planets ({n}); "
                f"got {tuple(ef_t.shape[:2])}"
            )
        edge_flux_slice = ef_t[:n, :n]
    hold_kwargs = _hold_metric_kwargs_from_obs(obs)
    node_metrics = (
        _graph_node_metrics(
            planet_feats, edge_bias, n, eta=eta, flux_features=flux_slice, **hold_kwargs,
        )
        if n > 0 else None
    )
    return Data(
        x=_append_graph_node_metrics(planet_feats, edge_bias, n, node_metrics=node_metrics),
        edge_index=edge_index,
        edge_attr=_planet_edge_attr(
            planet_feats, edge_bias, n, edge_index,
            eta=eta, node_metrics=node_metrics,
            edge_flux_features=edge_flux_slice,
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
    edge_flux_features_list: Optional[Iterable[torch.Tensor]] = None,
) -> Any:
    """Batch a list of observations into a single PyG ``Batch``.

    ``flux_features_list`` / ``edge_flux_features_list``, if given, must
    be aligned with ``obs_list`` and supply per-observation flux feature
    tensors (shapes ``(P_max, GRAPH_NODE_FLUX_DIM)`` and
    ``(P_max, P_max, GRAPH_EDGE_FLUX_DIM)`` respectively).  ``None``
    entries fall back to the legacy non-flux node / edge metric paths.
    """
    _require_pyg()
    if flux_features_list is None and edge_flux_features_list is None:
        return Batch.from_data_list([observation_to_pyg_data(obs, device=device) for obs in obs_list])
    obs_list = list(obs_list)
    n_obs = len(obs_list)
    node_flux_iter = list(flux_features_list) if flux_features_list is not None else [None] * n_obs
    edge_flux_iter = list(edge_flux_features_list) if edge_flux_features_list is not None else [None] * n_obs
    if len(node_flux_iter) != n_obs or len(edge_flux_iter) != n_obs:
        raise ValueError(
            "flux_features_list and edge_flux_features_list must be aligned with obs_list"
        )
    return Batch.from_data_list([
        observation_to_pyg_data(
            obs, device=device, flux_features=nf, edge_flux_features=ef,
        )
        for obs, nf, ef in zip(obs_list, node_flux_iter, edge_flux_iter)
    ])


def _batched_graph_node_metrics(
    planet_feats: torch.Tensor,
    edge_bias: torch.Tensor,
    planet_mask: torch.Tensor,
    *,
    tau: float = INFLUENCE_TAU_TURNS,
    eta: Optional[torch.Tensor] = None,
    flux_features: Optional[torch.Tensor] = None,
    remaining_turns: Optional[torch.Tensor] = None,
    held_production_norm: float = DEFAULT_HELD_PRODUCTION_NORM,
    hold_survival_base: float = DEFAULT_HOLD_SURVIVAL_BASE,
    hold_survival_from_vulnerability: bool = True,
    hold_survival_min: float = DEFAULT_HOLD_SURVIVAL_MIN,
    hold_survival_max: float = DEFAULT_HOLD_SURVIVAL_MAX,
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
    friendly_production_sources = (production * is_self).unsqueeze(-1)
    # kernel.T @ sources broadcast over batch: (B, P, P) -> (B, P_tgt, P_src) @ (B, P_src, 1)
    kT = kernel.transpose(-1, -2)
    friendly_influence = torch.bmm(kT, friendly_sources).squeeze(-1)
    enemy_influence = torch.bmm(kT, enemy_sources).squeeze(-1)
    friendly_support = torch.bmm(support_kernel.transpose(-1, -2), friendly_sources).squeeze(-1)
    supportability = (friendly_support / (enemy_influence + incoming_enemy + 1.0)).clamp(0.0, 10.0) / 10.0
    vulnerability = (enemy_influence / (ships + incoming_friendly + 1.0)).clamp(0.0, 10.0) / 10.0
    production_potential = torch.bmm(kT, production.unsqueeze(-1)).squeeze(-1)
    friendly_production_potential = torch.bmm(kT, friendly_production_sources).squeeze(-1)
    friendly_scaled = friendly_influence / 100.0
    enemy_scaled = enemy_influence / 100.0
    production_scaled = production_potential / 25.0
    friendly_production_scaled = friendly_production_potential / 25.0
    strategic_potential = production_scaled + friendly_scaled - enemy_scaled
    from rl.held_production import held_discounted_production_value

    if remaining_turns is None:
        rem = 0
    elif isinstance(remaining_turns, torch.Tensor):
        rem = remaining_turns.to(dtype=planet_feats.dtype, device=planet_feats.device)
    else:
        rem = int(max(0, remaining_turns))
    held_discounted_production = held_discounted_production_value(
        production,
        is_self,
        vulnerability,
        remaining_turns=rem,
        norm=float(held_production_norm),
        base=float(hold_survival_base),
        from_vulnerability=bool(hold_survival_from_vulnerability),
        survival_min=float(hold_survival_min),
        survival_max=float(hold_survival_max),
    )
    metrics = torch.stack([
        friendly_scaled,
        enemy_scaled,
        friendly_scaled - enemy_scaled,
        friendly_support / 100.0,
        supportability,
        vulnerability,
        production_scaled,
        friendly_production_scaled,
        strategic_potential,
        held_discounted_production,
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
    strategic = node_metrics[..., GRAPH_NODE_METRIC_IDX_STRATEGIC]
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
    edge_flux_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Vectorized concatenation of base edge_bias, graph edge metrics, and
    (optionally) the two directed flux channels.

    Returns ``(B, P, P, PYG_EDGE_FEATURE_DIM)`` by default or
    ``(B, P, P, PYG_EDGE_FEATURE_DIM_FLUX)`` when
    ``edge_flux_features`` (shape ``(B, P, P, GRAPH_EDGE_FLUX_DIM)``) is
    provided.  Padded pairs are zeroed.
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
    parts = [base, edge_metrics.to(base.dtype)]
    if edge_flux_features is not None:
        ef = edge_flux_features.to(dtype=base.dtype, device=base.device)
        if ef.shape != (B, P, P, GRAPH_EDGE_FLUX_DIM):
            raise ValueError(
                f"edge_flux_features shape must be ({B}, {P}, {P}, "
                f"{GRAPH_EDGE_FLUX_DIM}); got {tuple(ef.shape)}"
            )
        ef = ef * valid.to(ef.dtype).unsqueeze(-1)
        parts.append(ef)
    return torch.cat(parts, dim=-1)


def packed_observations_to_pyg_batch(
    batch: dict[str, torch.Tensor],
    *,
    flux_features: Optional[torch.Tensor] = None,
    edge_flux_features: Optional[torch.Tensor] = None,
) -> Any:
    """Convert a packed observation batch (output of ``pack_observations``)
    into a PyG ``Batch``.

    ``flux_features`` is an optional ``(B, P_max, GRAPH_NODE_FLUX_DIM)``
    tensor.  When provided the resulting ``Data.x`` has
    :data:`PYG_PLANET_FEATURE_DIM_FLUX` (= 47) columns instead of 43.

    ``edge_flux_features`` is an optional
    ``(B, P_max, P_max, GRAPH_EDGE_FLUX_DIM)`` tensor.  When provided
    the resulting ``Data.edge_attr`` has
    :data:`PYG_EDGE_FEATURE_DIM_FLUX` (= 17) columns instead of 15.

    Both tensors may also be supplied via ``batch["flux_features"]`` /
    ``batch["edge_flux_features"]``.  The consumer encoder must be built
    with ``enable_flux_features=True`` whenever either is enabled.
    """
    _require_pyg()
    planet_feats = batch["planet_feats"]
    edge_bias = batch["edge_bias"]
    planet_mask = batch["planet_mask"].bool()
    source_mask = batch["source_mask"].bool()
    global_feats = batch["global_feats"]
    if flux_features is None:
        flux_features = batch.get("flux_features", None)
    if edge_flux_features is None:
        edge_flux_features = batch.get("edge_flux_features", None)
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
    if edge_flux_features is not None:
        edge_flux_features = torch.as_tensor(
            edge_flux_features, dtype=planet_feats.dtype, device=planet_feats.device,
        )
        if edge_flux_features.shape != (B, P_max, P_max, GRAPH_EDGE_FLUX_DIM):
            raise ValueError(
                f"edge_flux_features shape must be ({B}, {P_max}, {P_max}, "
                f"{GRAPH_EDGE_FLUX_DIM}); got {tuple(edge_flux_features.shape)}"
            )
    # Vectorized ETA + node-metric computation (one pass over batch).
    eta_b = _batched_eta_matrix(planet_feats, edge_bias, planet_mask)
    rem_b = _remaining_turns_from_batch(batch)
    node_metrics_b = _batched_graph_node_metrics(
        planet_feats,
        edge_bias,
        planet_mask,
        eta=eta_b,
        flux_features=flux_features,
        remaining_turns=rem_b,
    )
    data_list = []
    for b in range(B):
        n = int(planet_mask[b].sum().item())
        edge_index = _planet_edge_index(n, device=planet_feats.device)
        eta_b_slice = eta_b[b, :n, :n] if n > 0 else None
        nm_slice = node_metrics_b[b, :n] if n > 0 else None
        edge_flux_slice = (
            edge_flux_features[b, :n, :n] if (edge_flux_features is not None and n > 0) else None
        )
        data_list.append(Data(
            x=_append_graph_node_metrics(
                planet_feats[b], edge_bias[b], n,
                eta=eta_b_slice, node_metrics=nm_slice,
            ),
            edge_index=edge_index,
            edge_attr=_planet_edge_attr(
                planet_feats[b], edge_bias[b], n, edge_index,
                eta=eta_b_slice, node_metrics=nm_slice,
                edge_flux_features=edge_flux_slice,
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
        use_input_embed: bool = True,
    ):
        """GNN encoder over the (planets-only) Orbit Wars graph.

        Parameters
        ----------
        enable_flux_features:
            ``True`` expands the node input to
            :data:`PYG_PLANET_FEATURE_DIM_FLUX` (= 47) and the edge input
            to :data:`PYG_EDGE_FEATURE_DIM_FLUX` (= 17) so the four node
            ``[friendly_in, friendly_out, enemy_in, enemy_out]`` and two
            directed edge ``[friendly_flux_ij, enemy_flux_ij]`` channels
            can be consumed.  Callers must then ensure the downstream
            ``Data.x`` / ``Data.edge_attr`` actually carry those columns
            -- the ``observation_to_pyg_data`` /
            ``packed_observations_to_pyg_batch`` helpers take
            ``flux_features`` and ``edge_flux_features`` arguments for
            this.
        use_input_embed:
            ``True`` (default) projects raw node / edge features through
            ``LayerNorm + Linear + GELU + LayerNorm`` blocks before they
            enter the GINE convs.  ``False`` skips the node projection
            entirely and feeds the raw 43/47-D node and 15/17-D edge
            features directly into the first GINE layer's MLP
            (NodeEmbed / EdgeEmbed-free pipeline; matches the
            ``DirectFcPlanetGnn`` prototype in
            ``docs/new_graph_policy_design.ipynb``).  ``edge_proj`` is
            kept either way because the action heads always score on
            ``d_model``-wide edge tokens.
        """
        super().__init__()
        _require_pyg()
        if enable_flux_features and planet_dim == PYG_PLANET_FEATURE_DIM:
            planet_dim = PYG_PLANET_FEATURE_DIM_FLUX
        if enable_flux_features and edge_dim == PYG_EDGE_FEATURE_DIM:
            edge_dim = PYG_EDGE_FEATURE_DIM_FLUX
        self.enable_flux_features = bool(enable_flux_features)
        self.use_input_embed = bool(use_input_embed)
        self.planet_dim = planet_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.d_model = d_model
        self.n_layers = n_layers
        # ``edge_proj`` always projects raw edge features to ``d_model`` so
        # the bilinear action heads (BilinearEdgeHead) can score on
        # consistent edge tokens whether or not the convs receive the
        # projected version.
        self.edge_proj = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        # ``node_proj`` is gated by ``use_input_embed``.  When disabled,
        # the GINE first-layer MLP absorbs the projection by accepting
        # the raw node dim directly.
        if self.use_input_embed:
            self.node_proj = nn.Sequential(
                nn.LayerNorm(planet_dim),
                nn.Linear(planet_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            conv_node_in = d_model
            conv_edge_in = d_model
        else:
            self.node_proj = nn.Identity()
            conv_node_in = planet_dim
            conv_edge_in = edge_dim
        self.conv_node_in = conv_node_in
        self.conv_edge_in = conv_edge_in
        self.global_proj = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(n_layers):
            in_dim = conv_node_in if i == 0 else d_model
            mlp = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.convs.append(GINEConv(mlp, edge_dim=conv_edge_in))
            self.norms.append(nn.LayerNorm(d_model))
        self.summary = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, graph_batch: Any) -> dict[str, torch.Tensor]:
        x = self.node_proj(graph_batch.x)
        if self.use_input_embed:
            edge_attr = self.edge_proj(graph_batch.edge_attr)
        else:
            edge_attr = graph_batch.edge_attr
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            msg = conv(x, graph_batch.edge_index, edge_attr)
            if not self.use_input_embed and i == 0:
                # First layer's MLP just projected raw -> d_model; from
                # here on we have d_model node states, so residual would
                # require an aligned projection.  Drop the residual on
                # this first layer to keep the no-embed path well-formed.
                x = norm(self.dropout(msg))
            else:
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
        use_input_embed: bool = True,
    ):
        """Planet graph policy.

        Parameters
        ----------
        enable_flux_features:
            When ``True`` the encoder ingests the four node and two
            edge flux channels in addition to the base graph metrics
            (node dim 43 -> 47, edge dim 15 -> 17).  The rollout is
            responsible for pre-computing
            :func:`rl.flux.compute_node_flux_features` and
            :func:`rl.flux.compute_flux_matrices` for every step.
        use_input_embed:
            Forwarded to :class:`PygPlanetGraphEncoder`.  ``False``
            skips the NodeEmbed/EdgeEmbed projections and feeds the raw
            graph features directly into the GINE convs (embed-free
            pipeline matching the ``DirectFcPlanetGnn`` prototype in
            the design notebook).
        """
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
        # PYG_PLANET_FEATURE_DIM_FLUX (= 47) node columns and
        # PYG_EDGE_FEATURE_DIM_FLUX (= 17) edge columns.
        if enable_flux_features and planet_dim == PYG_PLANET_FEATURE_DIM:
            planet_dim = PYG_PLANET_FEATURE_DIM_FLUX
        if enable_flux_features and edge_dim == PYG_EDGE_FEATURE_DIM:
            edge_dim = PYG_EDGE_FEATURE_DIM_FLUX
        self.enable_flux_features = bool(enable_flux_features)
        self.use_input_embed = bool(use_input_embed)
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
            use_input_embed=use_input_embed,
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
        edge_flux_features: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        # When the encoder was built with flux features, callers must
        # supply both the node ``(B, P_max, 4)`` and edge
        # ``(B, P_max, P_max, 2)`` tensors.  We fail loudly rather than
        # silently broadcasting zeros, which would mask bugs in the
        # rollout / buffer plumbing.
        if self.enable_flux_features and flux_features is None:
            raise ValueError(
                "PygPlanetGraphPolicy was built with enable_flux_features=True "
                "but `flux_features` was not passed to forward()."
            )
        if self.enable_flux_features and edge_flux_features is None:
            raise ValueError(
                "PygPlanetGraphPolicy was built with enable_flux_features=True "
                "but `edge_flux_features` was not passed to forward()."
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
            batch_dict["edge_flux_features"] = edge_flux_features  # type: ignore[assignment]
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
        edge_pair = _batched_pair_edge_features(
            planet_feats, edge_bias, planet_mask,
            edge_flux_features=edge_flux_features if self.enable_flux_features else None,
        )
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
