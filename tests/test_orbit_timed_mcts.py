"""Tests for time-aware abstract capture MCTS."""

from __future__ import annotations

import random

from orbit_graph import build_orbit_graph
from orbit_timed_mcts import TimedCaptureMCTS, TimedMCTSConfig
from orbit_timed_mcts_agent import OrbitTimedMCTSAgent


def _gateway_env():
    planets = [
        [0, 0, 20.0, 50.0, 1.0, 60, 2],
        [1, -1, 20.0, 74.0, 1.0, 6, 2],
        [2, -1, 20.0, 96.0, 1.0, 5, 9],
        [3, -1, 80.0, 50.0, 1.0, 6, 1],
    ]
    return {
        "planets": planets,
        "fleets": [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def _early_vs_late_env():
    planets = [
        [0, 0, 10.0, 10.0, 1.0, 120, 1],
        [1, -1, 10.0, 20.0, 1.0, 3, 3],
        [2, -1, 20.0, 45.0, 1.0, 3, 4],
    ]
    return {
        "planets": planets,
        "fleets": [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_timed_mcts_action_order_values_early_production_income():
    og = build_orbit_graph(_early_vs_late_env(), player=0)
    mcts = TimedCaptureMCTS(
        og,
        TimedMCTSConfig(
            simulations=8,
            horizon_turns=24,
            eta_horizon=60.0,
            source_reserve_frac=0.0,
            gamma=1.0,
            ship_cost_weight=0.0,
            eta_cost_weight=0.0,
        ),
        root_player=0,
        rng=random.Random(0),
    )
    ranked = mcts._rank_actions(mcts._initial_state)
    assert 1 in ranked and 2 in ranked, ranked
    assert ranked[0] == 1, ranked


def test_timed_mcts_prefers_gateway_when_horizon_can_use_it():
    og = build_orbit_graph(_gateway_env(), player=0)
    mcts = TimedCaptureMCTS(
        og,
        TimedMCTSConfig(
            simulations=80,
            horizon_turns=24,
            eta_horizon=40.0,
            source_reserve_frac=0.0,
        ),
        root_player=0,
        rng=random.Random(0),
    )
    bonus = mcts.search()
    assert bonus.get(1, 0.0) > 0.0, bonus
    assert bonus.get(1, 0.0) >= bonus.get(3, 0.0), bonus


def test_timed_mcts_zero_simulations_returns_empty_bonus():
    og = build_orbit_graph(_gateway_env(), player=0)
    mcts = TimedCaptureMCTS(
        og,
        TimedMCTSConfig(simulations=0),
        root_player=0,
    )
    assert mcts.search() == {}


def test_timed_mcts_agent_act_runs():
    agent = OrbitTimedMCTSAgent(
        timed_mcts_cfg=TimedMCTSConfig(
            simulations=16,
            horizon_turns=20,
            source_reserve_frac=0.0,
        )
    )
    actions = agent.act(_gateway_env(), player=0, current_turn=0, max_turns=300)
    assert isinstance(actions, list)
