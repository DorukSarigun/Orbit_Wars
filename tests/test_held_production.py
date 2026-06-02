"""Tests for held discounted production graph node metric."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model_graph_pyg import (
    GRAPH_NODE_METRIC_DIM,
    GRAPH_NODE_METRIC_IDX_HELD_DISCOUNTED_PRODUCTION,
    PYG_PLANET_FEATURE_DIM,
    observation_to_pyg_data,
)
from obs_builder import PLANET_FEATURE_DIM, build_observation
from rl.held_production import geometric_hold_multiplier, held_discounted_production_value
from rl.rewards import RewardConfig, build_reward_config, sum_planet_metric_channels


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


def test_geometric_series_closed_form():
    geom = geometric_hold_multiplier(3, np.array([0.5], dtype=np.float32))
    assert geom.shape == (1,)
    assert abs(float(geom[0]) - 1.75) < 1e-5  # 1 + 0.5 + 0.25


def test_geometric_series_x_equals_one():
    geom = geometric_hold_multiplier(4, np.array([1.0], dtype=np.float32))
    assert abs(float(geom[0]) - 4.0) < 1e-5


def test_held_value_zero_for_non_self():
    import torch

    prod = torch.tensor([10.0, 10.0])
    is_self = torch.tensor([1.0, 0.0])
    vuln = torch.tensor([0.0, 0.0])
    out = held_discounted_production_value(
        prod, is_self, vuln,
        remaining_turns=100,
        norm=1.0,
        from_vulnerability=False,
        base=0.99,
    )
    assert out[1].item() == 0.0
    assert out[0].item() > out[1].item()


def test_capture_turn_independent_same_state():
    obs_early = build_observation(
        _toy_obs(), player=0, angular_velocity=0.0, p_max=8, f_max=4,
        turn=10, max_turns=250,
    )
    obs_late = build_observation(
        _toy_obs(), player=0, angular_velocity=0.0, p_max=8, f_max=4,
        turn=200, max_turns=250,
    )
    cfg = build_reward_config("graph_native")
    s_early = sum_planet_metric_channels(obs_early, cfg=cfg, turn=10)
    s_late = sum_planet_metric_channels(obs_late, cfg=cfg, turn=200)
    held_early = s_early[GRAPH_NODE_METRIC_IDX_HELD_DISCOUNTED_PRODUCTION]
    held_late = s_late[GRAPH_NODE_METRIC_IDX_HELD_DISCOUNTED_PRODUCTION]
    assert held_early > held_late > 0.0


def test_loss_drops_held_metric():
    env = _toy_obs()
    obs_full = build_observation(
        env, player=0, angular_velocity=0.0, p_max=8, f_max=4, turn=5, max_turns=250,
    )
    reduced = dict(env)
    reduced["planets"] = [
        [0, 0, 70.0, 70.0, 3.0, 100, 3],
        [1, 1, 80.0, 70.0, 2.0, 50, 2],
        [2, 1, 30.0, 30.0, 2.0, 80, 2],
        [3, -1, 20.0, 80.0, 2.0, 10, 1],
    ]
    obs_reduced = build_observation(
        reduced, player=0, angular_velocity=0.0, p_max=8, f_max=4, turn=5, max_turns=250,
    )
    cfg = RewardConfig(max_turns=250)
    idx = GRAPH_NODE_METRIC_IDX_HELD_DISCOUNTED_PRODUCTION
    held_full = sum_planet_metric_channels(obs_full, cfg=cfg, turn=5)[idx]
    held_reduced = sum_planet_metric_channels(obs_reduced, cfg=cfg, turn=5)[idx]
    assert held_full > held_reduced > 0.0


def test_pyg_node_feature_includes_held_channel():
    obs = build_observation(
        _toy_obs(), player=0, angular_velocity=0.05, p_max=8, f_max=4,
        turn=20, max_turns=250,
    )
    data = observation_to_pyg_data(obs)
    assert data.x.shape[1] == PYG_PLANET_FEATURE_DIM == PLANET_FEATURE_DIM + GRAPH_NODE_METRIC_DIM
    held = data.x[:, PLANET_FEATURE_DIM + GRAPH_NODE_METRIC_IDX_HELD_DISCOUNTED_PRODUCTION]
    assert held.isfinite().all()
    assert float(held.max()) > 0.0
