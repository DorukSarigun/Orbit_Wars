"""Tests for the stepping-stone lookahead layer."""

from __future__ import annotations

from orbit_graph import build_orbit_graph
from orbit_attack_allocate import rank_capture_targets
from orbit_attack_targets import attack_config_from_env, plan_attack_targets
from orbit_lookahead_agent import (
    LookaheadConfig,
    OrbitLookaheadAgent,
    _StepStonePlanner,
    _owned_ids,
    _candidate_target_ids,
    _reachable_etas,
)


def _gateway_env():
    # P0 at (20,50). Gateway neutral 1 directly reachable; high-prod neutral 2
    # only reachable AFTER owning 1; neutral 3 across the sun (unreachable now).
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


def test_capturing_gateway_unlocks_far_planet():
    og = build_orbit_graph(_gateway_env(), player=0)
    owned = _owned_ids(og)
    cand = _candidate_target_ids(og)
    reach0 = _reachable_etas(og, owned, cand, eta_horizon=40.0)
    reach1 = _reachable_etas(og, owned | {1}, cand, eta_horizon=40.0)
    assert 1 in reach0          # gateway reachable now
    assert 2 not in reach0      # far world not reachable yet
    assert 2 in reach1          # ... but unlocked once we own the gateway


def test_gateway_gets_positive_bonus():
    og = build_orbit_graph(_gateway_env(), player=0)
    planner = _StepStonePlanner(
        og, LookaheadConfig(eta_horizon=40.0, max_depth=2),
        current_turn=0, max_turns=300)
    bonus = planner.bonus()
    assert bonus.get(1, 0.0) > 0.0, bonus


def test_zero_weight_disables_planner():
    agent = OrbitLookaheadAgent(lookahead_cfg=LookaheadConfig(lookahead_weight=0.0))
    og = build_orbit_graph(_gateway_env(), player=0)
    bonus = agent._attack_priority_bonus(
        og, None, [], omega=0.0, current_turn=0, max_turns=300)
    assert bonus is None


def test_priority_bonus_promotes_target_in_ranking():
    env = _gateway_env()
    og = build_orbit_graph(env, player=0)
    acfg = attack_config_from_env(env, current_turn=0, max_turns=300)
    plan = plan_attack_targets(og, cfg=acfg)

    base = rank_capture_targets(og, plan, attack_cfg=acfg)
    boosted = rank_capture_targets(
        og, plan, attack_cfg=acfg, priority_bonus={1: 1000.0})

    base_pri = {r.planet_id: r.priority for r in base}
    boost_pri = {r.planet_id: r.priority for r in boosted}
    if 1 in base_pri:
        assert boost_pri[1] == base_pri[1] + 1000.0
        # planet 1 should now sort first among reachable targets.
        assert boosted[0].planet_id == 1


def test_empty_bonus_identical_to_none():
    env = _gateway_env()
    og = build_orbit_graph(env, player=0)
    acfg = attack_config_from_env(env, current_turn=0, max_turns=300)
    plan = plan_attack_targets(og, cfg=acfg)
    a = rank_capture_targets(og, plan, attack_cfg=acfg, priority_bonus=None)
    b = rank_capture_targets(og, plan, attack_cfg=acfg, priority_bonus={})
    assert [r.sort_key for r in a] == [r.sort_key for r in b]


def test_lookahead_agent_act_runs():
    env = _gateway_env()
    agent = OrbitLookaheadAgent()
    actions = agent.act(env, player=0, current_turn=0, max_turns=300)
    assert isinstance(actions, list)
    for a in actions:
        assert len(a) == 3
