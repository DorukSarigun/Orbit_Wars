"""Tests for abstract capture MCTS."""

from __future__ import annotations

from orbit_graph import build_orbit_graph
from orbit_mcts import CaptureMCTS, MCTSConfig
from orbit_mcts_agent import OrbitMCTSAgent


def _gateway_env():
    planets = [
        [0, 0, 20.0, 50.0, 1.0, 40, 2],
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


def test_mcts_prefers_gateway_over_unreachable_high_prod():
    og = build_orbit_graph(_gateway_env(), player=0)
    mcts = CaptureMCTS(
        og,
        MCTSConfig(simulations=64, rollout_depth=3, eta_horizon=40.0),
        root_player=0,
        rng=__import__("random").Random(0),
    )
    bonus = mcts.search()
    assert bonus.get(1, 0.0) > 0.0, bonus
    # Planet 2 is not directly reachable; gateway 1 should dominate.
    assert bonus.get(1, 0.0) >= bonus.get(3, 0.0), bonus


def test_zero_simulations_returns_empty_bonus():
    og = build_orbit_graph(_gateway_env(), player=0)
    mcts = CaptureMCTS(og, MCTSConfig(simulations=0), root_player=0)
    assert mcts.search() == {}


def test_mcts_agent_act_runs():
    agent = OrbitMCTSAgent(mcts_cfg=MCTSConfig(simulations=16))
    actions = agent.act(_gateway_env(), player=0, current_turn=0, max_turns=300)
    assert isinstance(actions, list)


def test_mcts_agent_disabled_matches_base_hook():
    agent = OrbitMCTSAgent(mcts_cfg=MCTSConfig(simulations=0))
    og = build_orbit_graph(_gateway_env(), player=0)
    assert agent._attack_priority_bonus(
        og, None, [], omega=0.0, current_turn=0, max_turns=300,
    ) is None
