"""Unit tests for the graph_native reward path.

Covers:
- ``compute_graph_metrics_summary`` returns finite scalars for typical states
- ``compute_graph_native_reward`` accepts both Observation and pre-computed
  summary inputs and yields identical results
- Potential-based shaping formula:  r = γ * φ(s') - φ(s)  per channel
- Edge-level reward channel fires only when the weight is non-zero
- Redundant channels (influence_balance / strategic_potential) are 0 by default
- ``RunningMeanStd`` warmup + variance behavior
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from obs_builder import build_observation
from rl.rewards import (
    RewardConfig,
    RunningMeanStd,
    build_reward_config,
    compute_graph_metrics_summary,
    compute_graph_native_reward,
)


def _toy_obs():
    return {
        "planets": [
            [0, 0, 70.0, 70.0, 3.0, 100, 3],
            [1, 0, 80.0, 70.0, 2.0, 50, 2],
            [2, 1, 30.0, 30.0, 2.0, 80, 2],
            [3, -1, 20.0, 80.0, 2.0, 10, 1],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 70.0, 70.0, 3.0, 100, 3],
            [1, 0, 80.0, 70.0, 2.0, 50, 2],
            [2, 1, 30.0, 30.0, 2.0, 80, 2],
            [3, -1, 20.0, 80.0, 2.0, 10, 1],
        ],
        "comet_planet_ids": [],
        "comets": [],
    }


def _build_obs(player: int = 0):
    return build_observation(
        _toy_obs(), player=player, angular_velocity=0.05, p_max=8, f_max=4,
    )


# ---------------------------------------------------------------------------
# compute_graph_metrics_summary
# ---------------------------------------------------------------------------

def test_summary_shape_and_finiteness():
    obs = _build_obs()
    summary = compute_graph_metrics_summary(obs)
    assert "node_sum" in summary and "edge_sum" in summary
    assert len(summary["node_sum"]) == 8  # GRAPH_NODE_METRIC_DIM
    for v in summary["node_sum"]:
        assert v == v and abs(v) < 1e6  # not NaN, bounded
    assert summary["edge_sum"] == summary["edge_sum"]


def test_summary_edge_signal_selection():
    obs = _build_obs()
    flow = compute_graph_metrics_summary(obs, edge_signal="flow_opportunity")
    same_owner = compute_graph_metrics_summary(obs, edge_signal="same_owner")
    # same_owner is binary indicator -> sum equals number of same-owner pairs.
    # In this toy state we have two player-0 planets, so same_owner counts at
    # least the (0,1) and (1,0) directed pairs (= 2).
    assert same_owner["edge_sum"] >= 2.0
    # flow_opportunity is a different scalar
    assert abs(flow["edge_sum"] - same_owner["edge_sum"]) > 1e-6


# ---------------------------------------------------------------------------
# compute_graph_native_reward (potential-based shaping)
# ---------------------------------------------------------------------------

def test_reward_zero_on_identical_states():
    obs = _build_obs()
    cfg = build_reward_config("graph_native")
    cfg = RewardConfig(**{**cfg.__dict__, "graph_potential_discount": 1.0})
    # γ=1 + identical state -> γφ(s')-φ(s) = 0 for every channel.
    r = compute_graph_native_reward(obs, obs, player=0, cfg=cfg, turn=0)
    assert r == 0.0


def test_reward_accepts_summary_dict():
    obs = _build_obs()
    cfg = build_reward_config("graph_native")
    summary = compute_graph_metrics_summary(obs, edge_signal=cfg.graph_edge_efficiency_signal)
    # Passing identical pre-computed summaries should match passing observations.
    r_from_summary = compute_graph_native_reward(summary, summary, player=0, cfg=cfg, turn=0)
    r_from_obs = compute_graph_native_reward(obs, obs, player=0, cfg=cfg, turn=0)
    assert abs(r_from_summary - r_from_obs) < 1e-9


def test_potential_shaping_uses_discount():
    """For γ < 1 and an unchanging state, r = (γ-1) · weighted_φ != 0 in general."""
    obs = _build_obs()
    cfg = RewardConfig(
        reward_system="graph_native",
        graph_friendly_influence_weight=1.0,
        graph_enemy_influence_weight=0.0,
        graph_friendly_support_weight=0.0,
        graph_supportability_weight=0.0,
        graph_vulnerability_weight=0.0,
        graph_production_potential_weight=0.0,
        graph_potential_discount=0.5,
        graph_edge_efficiency_weight=0.0,
        reward_scale=1.0,
        max_turns=1000,  # disable time decay
        time_decay_floor=1.0,
        capture_time_curvature=0.0,
    )
    summary = compute_graph_metrics_summary(obs)
    friendly_phi = summary["node_sum"][0]
    # r = γ * curr - prev = (0.5 - 1) * φ = -0.5 * φ
    expected = -0.5 * friendly_phi
    r = compute_graph_native_reward(summary, summary, player=0, cfg=cfg, turn=0)
    assert abs(r - expected) < 1e-6


def test_edge_weight_disabled_by_default():
    """A non-default edge_sum delta should not influence the reward when
    ``graph_edge_efficiency_weight == 0``."""
    cfg = RewardConfig(
        reward_system="graph_native",
        graph_friendly_influence_weight=1.0,
        graph_potential_discount=1.0,
        graph_edge_efficiency_weight=0.0,
        max_turns=1000, time_decay_floor=1.0, capture_time_curvature=0.0,
        reward_scale=1.0,
    )
    prev = {"node_sum": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "edge_sum": 0.0}
    curr = {"node_sum": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "edge_sum": 100.0}
    r = compute_graph_native_reward(prev, curr, player=0, cfg=cfg, turn=0)
    assert r == 0.0


def test_edge_weight_active():
    cfg = RewardConfig(
        reward_system="graph_native",
        graph_potential_discount=1.0,
        graph_edge_efficiency_weight=2.0,
        max_turns=1000, time_decay_floor=1.0, capture_time_curvature=0.0,
        reward_scale=1.0,
    )
    prev = {"node_sum": [0.0] * 8, "edge_sum": 5.0}
    curr = {"node_sum": [0.0] * 8, "edge_sum": 8.0}
    # r = 2.0 * (1.0 * 8 - 5) = 6.0
    r = compute_graph_native_reward(prev, curr, player=0, cfg=cfg, turn=0)
    assert abs(r - 6.0) < 1e-9


def test_redundant_channels_default_zero():
    """``influence_balance`` and ``strategic_potential`` are linear combos of
    other channels; the graph_native preset must default them to 0."""
    cfg = build_reward_config("graph_native")
    assert cfg.graph_influence_balance_weight == 0.0
    assert cfg.graph_strategic_potential_weight == 0.0


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------

def test_running_mean_std_warmup_passthrough():
    rms = RunningMeanStd(warmup=5)
    # Within warmup, normalize is the identity.
    for x in [1.0, 2.0, 3.0]:
        rms.update(x)
        assert rms.normalize(x) == x


def test_running_mean_std_variance_estimate():
    rms = RunningMeanStd(warmup=0)
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    for x in samples:
        rms.update(x)
    # Population unbiased variance of [1..5] = 2.5
    assert abs(rms.variance - 2.5) < 1e-9
    assert abs(rms.std - 2.5 ** 0.5) < 1e-9


def test_running_mean_std_normalize_post_warmup():
    rms = RunningMeanStd(warmup=2)
    rms.update(2.0)
    rms.update(-2.0)  # std = 2 sqrt(2) approx 2.828
    out = rms.normalize(4.0)
    assert abs(out - 4.0 / rms.std) < 1e-9
