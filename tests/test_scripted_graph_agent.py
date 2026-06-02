"""Tests for graph-maximizing scripted agent."""

from __future__ import annotations

import pytest

from scripted_graph_agent import GraphMaximizingAgent, GraphMaximizingConfig
from scripted_graph_scorer import (
    DEFAULT_GRAPH_CHANNEL_WEIGHTS,
    ScriptedGraphScores,
    build_scripted_graph_scores,
    empire_node_objective,
)
from obs_builder import build_observation, PF_IDX_OWNER_SELF


def _env_obs():
    return {
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


def test_empire_node_objective_counts_self_owned_only():
    obs = build_observation(
        _env_obs(), player=0, angular_velocity=0.0, p_max=8, f_max=8, max_turns=250,
    )
    scores = build_scripted_graph_scores(obs)
    if scores is None:
        pytest.skip("torch / model_graph_pyg unavailable")

    all_slots = empire_node_objective(scores, obs, DEFAULT_GRAPH_CHANNEL_WEIGHTS)
    obs.planet_feats[0, PF_IDX_OWNER_SELF] = 0.0
    reduced = empire_node_objective(scores, obs, DEFAULT_GRAPH_CHANNEL_WEIGHTS)
    assert reduced <= all_slots


def test_graph_maximizing_agent_returns_env_actions():
    agent = GraphMaximizingAgent(GraphMaximizingConfig(max_launches=2))
    actions = agent.act(_env_obs(), player=0, current_turn=1, max_turns=250)
    assert isinstance(actions, list)
    for row in actions:
        assert len(row) == 3
        assert int(row[2]) > 0
