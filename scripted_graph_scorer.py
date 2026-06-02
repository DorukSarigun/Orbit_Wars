"""Graph-metric scoring helpers for scripted agents (Phase A).

Reuses the same node / edge structural metrics as the PyG encoder node
features (``model_graph_pyg._graph_node_metrics``) and the ``graph_native``
reward / arrival-credit channel weights.  No GNN forward pass is required.

Scripted ranking uses the same weighted node vector at each slot::

    dot(node_metrics[slot], channel_weights)

Empire objective sums that dot product over self-owned planets only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from obs_builder import PF_IDX_OWNER_SELF

# Mirrors ``model_graph_pyg._graph_node_metrics`` stack order (base 10).
IDX_FRIENDLY_INFLUENCE = 0
IDX_ENEMY_INFLUENCE = 1
IDX_INFLUENCE_BALANCE = 2
IDX_FRIENDLY_SUPPORT = 3
IDX_SUPPORTABILITY = 4
IDX_VULNERABILITY = 5
IDX_PRODUCTION_POTENTIAL = 6
IDX_FRIENDLY_PRODUCTION_POTENTIAL = 7
IDX_STRATEGIC_POTENTIAL = 8
IDX_HELD_DISCOUNTED_PRODUCTION = 9

# ``flow_opportunity`` column in ``_graph_edge_metric_matrix``.
IDX_FLOW_OPPORTUNITY = 8

# Default independent channel weights (mirrors ``graph_native`` preset).
DEFAULT_GRAPH_CHANNEL_WEIGHTS: tuple[float, ...] = (
    1.0,   # friendly_influence
    -1.0,  # enemy_influence
    0.0,   # influence_balance (redundant combo)
    1.0,   # friendly_support
    1.0,   # supportability
    -1.0,  # vulnerability
    1.0,   # production_potential
    1.0,   # friendly_production_potential
    0.0,   # strategic_potential (redundant combo)
    1.0,   # held_discounted_production
)


@dataclass(frozen=True)
class ScriptedGraphScores:
    """Per-slot node metrics and pair-wise flow opportunity."""

    node_by_slot: np.ndarray
    flow_opportunity: np.ndarray
    n_real: int

    def weighted_node_score(
        self,
        slot: int,
        channel_weights: Sequence[float],
    ) -> float:
        """Same weighted node vector as ``empire_node_objective`` at one slot."""
        return weighted_node_objective(self, int(slot), channel_weights)

    def target_score(
        self,
        slot: int,
        channel_weights: Sequence[float] = DEFAULT_GRAPH_CHANNEL_WEIGHTS,
    ) -> float:
        return self.weighted_node_score(slot, channel_weights)

    def source_score(
        self,
        slot: int,
        channel_weights: Sequence[float] = DEFAULT_GRAPH_CHANNEL_WEIGHTS,
    ) -> float:
        return self.weighted_node_score(slot, channel_weights)

    def pair_score(self, src_slot: int, tgt_slot: int) -> float:
        return float(self.flow_opportunity[int(src_slot), int(tgt_slot)])

    def slot_channel_vector(self, slot: int) -> np.ndarray:
        return self.node_by_slot[int(slot)]


def weighted_node_objective(
    scores: ScriptedGraphScores,
    slot: int,
    channel_weights: Sequence[float],
) -> float:
    """Weighted dot product of the node metric vector at ``slot``."""
    weights = np.asarray(channel_weights, dtype=np.float32).reshape(-1)
    vec = scores.slot_channel_vector(int(slot))
    n = min(vec.shape[0], weights.shape[0])
    return float(np.dot(vec[:n], weights[:n]))


def empire_node_objective(
    scores: ScriptedGraphScores,
    obs,
    channel_weights: Sequence[float],
) -> float:
    """Weighted sum of node metrics over planets owned by ``player``."""
    weights = np.asarray(channel_weights, dtype=np.float32).reshape(-1)
    total = 0.0
    owner_self = np.asarray(obs.planet_feats[:, PF_IDX_OWNER_SELF], dtype=np.float32)
    for slot in range(int(scores.n_real)):
        if owner_self[slot] <= 0.5:
            continue
        vec = scores.slot_channel_vector(slot)
        n = min(vec.shape[0], weights.shape[0])
        total += float(np.dot(vec[:n], weights[:n]))
    return total


def local_target_objective(
    scores: ScriptedGraphScores,
    tgt_slot: int,
    channel_weights: Sequence[float],
) -> float:
    """Weighted node objective at a single target slot (capture locus)."""
    return weighted_node_objective(scores, int(tgt_slot), channel_weights)


def build_scripted_graph_scores(
    obs,
    *,
    reward_cfg=None,
    turn: int = 0,
) -> Optional[ScriptedGraphScores]:
    """Compute graph ranking tensors from a policy ``Observation``.

    Uses the same ``_graph_node_metrics`` path as the PyG encoder (including
    ``held_discounted_production`` with ``remaining_turns`` from ``obs.meta``).

    Returns ``None`` when torch / PyG metric helpers are unavailable or the
    observation has no real planets.
    """
    try:
        import torch
        from model_graph_pyg import (
            _estimated_intercept_eta_matrix,
            _graph_edge_metric_matrix,
            _graph_node_metrics,
        )
        from rl.rewards import _hold_metric_kwargs_from_cfg
    except Exception:
        return None

    with torch.no_grad():
        planet_feats = torch.as_tensor(obs.planet_feats, dtype=torch.float32)
        edge_bias = torch.as_tensor(obs.edge_bias, dtype=torch.float32)
        planet_mask = torch.as_tensor(obs.planet_mask, dtype=torch.bool)
        p_max = int(planet_feats.shape[0])
        n_real = int(planet_mask.sum().item())
        if n_real <= 0:
            return None

        eta = _estimated_intercept_eta_matrix(planet_feats, edge_bias, n_real)
        hold_kwargs = _hold_metric_kwargs_from_cfg(reward_cfg, turn=int(turn), obs=obs)
        node_metrics = _graph_node_metrics(
            planet_feats, edge_bias, n_real, eta=eta, **hold_kwargs,
        )
        edge_metrics = _graph_edge_metric_matrix(
            planet_feats, edge_bias, n_real, eta=eta, node_metrics=node_metrics,
        )

        node_np = np.zeros((p_max, node_metrics.shape[-1]), dtype=np.float32)
        node_np[:n_real] = node_metrics.detach().cpu().numpy()

        flow_np = np.zeros((p_max, p_max), dtype=np.float32)
        flow_slice = edge_metrics[..., IDX_FLOW_OPPORTUNITY].detach().cpu().numpy()
        flow_np[:n_real, :n_real] = flow_slice

        return ScriptedGraphScores(
            node_by_slot=node_np,
            flow_opportunity=flow_np,
            n_real=n_real,
        )
