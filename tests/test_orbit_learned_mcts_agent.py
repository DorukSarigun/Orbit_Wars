"""Tests for learned-prior timed MCTS agent."""

from __future__ import annotations

import random

from orbit_graph import build_orbit_graph
from orbit_learned_mcts_agent import (
    BranchPriorRequest,
    LearnedBranchTimedMCTS,
    LearnedMCTSConfig,
    LinearBranchPrior,
    OrbitLearnedMCTSAgent,
)
from orbit_timed_mcts import TimedMCTSConfig


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


def _favor_target_one(request: BranchPriorRequest):
    return {int(t): (10.0 if int(t) == 1 else 0.0) for t in request.candidate_targets}


def test_learned_prior_can_override_root_branch_order():
    og = build_orbit_graph(_early_vs_late_env(), player=0)
    mcts = LearnedBranchTimedMCTS(
        og,
        LearnedMCTSConfig(
            timed_mcts=TimedMCTSConfig(
                simulations=8,
                horizon_turns=24,
                eta_horizon=60.0,
                source_reserve_frac=0.0,
                gamma=1.0,
                ship_cost_weight=0.0,
                eta_cost_weight=0.0,
            ),
            prior_weight=1.0,
            heuristic_logit_weight=0.0,
        ),
        root_player=0,
        prior_provider=_favor_target_one,
        rng=random.Random(0),
    )
    ranked = mcts._rank_actions(mcts._initial_state)
    assert ranked[0] == 1, ranked


def test_learned_mcts_agent_defaults_to_lp_logistics_and_runs():
    agent = OrbitLearnedMCTSAgent(
        learned_mcts_cfg=LearnedMCTSConfig(
            timed_mcts=TimedMCTSConfig(
                simulations=4,
                horizon_turns=12,
                source_reserve_frac=0.0,
            )
        )
    )
    assert agent.cfg.attack_allocator == "lp"
    assert agent.cfg.reinforcement_allocator == "lp"
    actions = agent.act(_early_vs_late_env(), player=0, current_turn=0, max_turns=300)
    assert isinstance(actions, list)


def test_linear_branch_prior_json_roundtrip(tmp_path):
    path = tmp_path / "prior.json"
    prior = LinearBranchPrior({"target_prod_norm": 2.0, "arrival_norm": -1.0})
    prior.to_json(path)
    loaded = LinearBranchPrior.from_json(path)
    assert dict(loaded.weights) == dict(prior.weights)
