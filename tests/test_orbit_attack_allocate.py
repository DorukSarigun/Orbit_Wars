"""Tests for greedy nearest-source capture allocation."""

from __future__ import annotations

from orbit_attack_allocate import (
    allocate_lp_captures,
    allocate_greedy_captures,
    friendly_influence_on_target,
    rank_capture_targets,
)
from orbit_attack_targets import AttackTargetConfig, plan_attack_targets
from orbit_graph import build_orbit_graph


def _env_vertical():
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 40, 2],
            [1, -1, 20.0, 70.0, 1.0, 5, 3],
            [2, 1, 80.0, 20.0, 1.0, 3, 1],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 40, 2],
            [1, -1, 20.0, 70.0, 1.0, 5, 3],
            [2, 1, 80.0, 20.0, 1.0, 3, 1],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def _env_two_sources_one_hard_neutral():
    """Two owned planets must both contribute to capture neutral 2."""
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 12, 2],
            [3, 0, 80.0, 20.0, 1.0, 12, 2],
            [2, -1, 50.0, 20.0, 1.0, 20, 4],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 12, 2],
            [3, 0, 80.0, 20.0, 1.0, 12, 2],
            [2, -1, 50.0, 20.0, 1.0, 20, 4],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_rank_capture_targets_includes_enemy_influence():
    og = build_orbit_graph(_env_vertical(), player=0)
    plan = plan_attack_targets(
        og, cfg=AttackTargetConfig(current_turn=10, max_turns=250),
    )
    ranked = rank_capture_targets(og, plan, attack_cfg=AttackTargetConfig())
    assert ranked
    assert all(r.friendly_influence >= 0.0 for r in ranked)


def test_greedy_neutral_capture_from_nearest_source():
    env = _env_vertical()
    og = build_orbit_graph(env, player=0)
    plan = plan_attack_targets(
        og, cfg=AttackTargetConfig(current_turn=10, max_turns=250),
    )
    remaining = dict(plan.launch_budgets)
    used: set[int] = set()
    actions = allocate_greedy_captures(
        og, plan, remaining, used,
        attack_cfg=AttackTargetConfig(current_turn=10, max_turns=250),
        max_launches=8,
    )
    assert actions
    assert int(actions[0][0]) == 0


def test_greedy_can_use_multiple_sources_for_one_target():
    env = _env_two_sources_one_hard_neutral()
    og = build_orbit_graph(env, player=0)
    plan = plan_attack_targets(
        og, cfg=AttackTargetConfig(current_turn=10, max_turns=250),
    )
    remaining = dict(plan.launch_budgets)
    used: set[int] = set()
    actions = allocate_greedy_captures(
        og, plan, remaining, used,
        attack_cfg=AttackTargetConfig(current_turn=10, max_turns=250),
        max_launches=8,
    )
    assert len(actions) >= 2
    src_ids = {int(a[0]) for a in actions}
    assert src_ids == {0, 3}
    assert friendly_influence_on_target(og, 2) >= 0.0


def test_lp_allocator_can_use_multiple_sources_for_one_target():
    env = _env_two_sources_one_hard_neutral()
    og = build_orbit_graph(env, player=0)
    cfg = AttackTargetConfig(current_turn=10, max_turns=250)
    plan = plan_attack_targets(og, cfg=cfg)
    remaining = dict(plan.launch_budgets)
    used: set[int] = set()
    actions = allocate_lp_captures(
        og,
        plan,
        remaining,
        used,
        attack_cfg=cfg,
        max_launches=8,
        lp_top_k_targets=2,
    )
    assert len(actions) >= 2
    src_ids = {int(a[0]) for a in actions}
    assert src_ids == {0, 3}
    assert sum(int(a[2]) for a in actions) >= 21
    assert used == {0, 3}
