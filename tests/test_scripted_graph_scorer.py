"""Tests for scripted graph-metric ranking helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scripted_graph_scorer import (
    DEFAULT_GRAPH_CHANNEL_WEIGHTS,
    IDX_FLOW_OPPORTUNITY,
    IDX_HELD_DISCOUNTED_PRODUCTION,
    IDX_PRODUCTION_POTENTIAL,
    IDX_STRATEGIC_POTENTIAL,
    IDX_SUPPORTABILITY,
    IDX_VULNERABILITY,
    ScriptedGraphScores,
    build_scripted_graph_scores,
    weighted_node_objective,
)
from obs_builder import build_observation
from rl.rewards import RewardConfig, compute_planet_metric_vectors


def _minimal_obs(*, p_max: int = 4):
    env_obs = {
        "planets": [
            [0, 0, 20.0, 50.0, 1.0, 30, 2],
            [1, -1, 80.0, 50.0, 1.0, 5, 3],
            [2, 0, 50.0, 20.0, 1.0, 10, 1],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 50.0, 1.0, 30, 2],
            [1, -1, 80.0, 50.0, 1.0, 5, 3],
            [2, 0, 50.0, 20.0, 1.0, 10, 1],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "step": 1,
    }
    return build_observation(
        env_obs,
        player=0,
        angular_velocity=0.0,
        p_max=p_max,
        f_max=8,
        max_turns=250,
        turn=10,
    )


def test_scripted_graph_scores_ranking_helpers():
    node = np.zeros((4, 10), dtype=np.float32)
    node[0] = [0, 0, 0, 0, 0.8, 0.1, 0.5, 0.2, 0.7, 0.3]
    node[1] = [0, 0, 0, 0, 0.2, 0.9, 0.9, 0.1, 0.4, 0.0]
    flow = np.zeros((4, 4), dtype=np.float32)
    flow[0, 1] = 0.6
    flow[2, 1] = 0.2
    scores = ScriptedGraphScores(node_by_slot=node, flow_opportunity=flow, n_real=3)
    weights = DEFAULT_GRAPH_CHANNEL_WEIGHTS

    assert scores.target_score(1, weights) == pytest.approx(
        weighted_node_objective(scores, 1, weights)
    )
    assert scores.source_score(0, weights) == pytest.approx(
        weighted_node_objective(scores, 0, weights)
    )
    assert scores.pair_score(0, 1) == pytest.approx(0.6)


def test_build_scripted_graph_scores_from_observation():
    obs = _minimal_obs()
    cfg = RewardConfig()
    scores = build_scripted_graph_scores(obs, reward_cfg=cfg, turn=10)
    if scores is None:
        pytest.skip("torch / model_graph_pyg unavailable")

    assert scores.n_real == 3
    assert scores.node_by_slot.shape[1] == 10
    assert scores.node_by_slot.shape[0] == obs.planet_feats.shape[0]
    assert scores.flow_opportunity.shape == (
        obs.planet_feats.shape[0],
        obs.planet_feats.shape[0],
    )
    assert np.isfinite(scores.node_by_slot[: scores.n_real]).all()
    assert np.isfinite(scores.flow_opportunity[: scores.n_real, : scores.n_real]).all()
    assert scores.node_by_slot[scores.n_real :].sum() == 0.0


def test_scripted_scores_match_reward_metric_vectors():
    """Scripted scorer must reuse the same node metrics as arrival-credit / GNN."""
    obs = _minimal_obs()
    cfg = RewardConfig()
    scores = build_scripted_graph_scores(obs, reward_cfg=cfg, turn=10)
    reward_vecs = compute_planet_metric_vectors(obs, cfg=cfg, turn=10)
    if scores is None or reward_vecs is None:
        pytest.skip("torch / model_graph_pyg unavailable")

    np.testing.assert_allclose(
        scores.node_by_slot[: scores.n_real],
        reward_vecs[: scores.n_real],
        rtol=1e-5,
        atol=1e-6,
    )


def test_held_discounted_production_nonzero_for_owned_planet():
    obs = _minimal_obs()
    cfg = RewardConfig()
    scores = build_scripted_graph_scores(obs, reward_cfg=cfg, turn=10)
    if scores is None:
        pytest.skip("torch / model_graph_pyg unavailable")

    owned = scores.node_by_slot[0, IDX_HELD_DISCOUNTED_PRODUCTION]
    neutral = scores.node_by_slot[1, IDX_HELD_DISCOUNTED_PRODUCTION]
    assert owned > 0.0
    assert neutral == pytest.approx(0.0)
