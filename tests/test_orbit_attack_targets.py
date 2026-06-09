"""Tests for launch budgets and node attack metrics."""

from __future__ import annotations

from orbit_attack_targets import (
    AttackTargetConfig,
    node_enemy_attack,
    node_neutral_attack,
    plan_attack_targets,
)
from orbit_defense import compute_launch_budgets
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


def test_launch_budget_from_surplus():
    og = build_orbit_graph(_env_vertical(), player=0)
    budgets = compute_launch_budgets(og)
    b = budgets[0]
    assert b.launchable_ships >= 0
    assert b.launchable_ships <= b.garrison


def test_neutral_metrics_on_node():
    og = build_orbit_graph(_env_vertical(), player=0)
    plan = plan_attack_targets(
        og, cfg=AttackTargetConfig(current_turn=10, max_turns=250),
    )
    m = node_neutral_attack(og, 1)
    assert m is not None
    assert m.feasible
    assert m.planet_id == 1
    assert m.best_src_id == 0
    assert m.capture_eta < 60.0
    assert m.ships_lost >= m.ships_required >= 6
    assert m.production_value > 0.0
    assert m.efficiency == m.production_value / m.ships_lost
    assert plan.best_neutral is not None
    assert plan.best_neutral.planet_id == 1


def test_enemy_metrics_on_node_with_denial():
    og = build_orbit_graph(_env_vertical(), player=0)
    plan_attack_targets(og, cfg=AttackTargetConfig(current_turn=10, max_turns=250))
    m = node_enemy_attack(og, 2)
    assert m is not None
    assert m.feasible
    assert m.combined_value == m.acquisition_value + m.denial_value
    assert m.production == 1
    assert m.ships_lost > 0


def test_owned_planet_has_no_attack_metrics():
    og = build_orbit_graph(_env_vertical(), player=0)
    plan_attack_targets(og, cfg=AttackTargetConfig(current_turn=10, max_turns=250))
    assert node_neutral_attack(og, 0) is None
    assert node_enemy_attack(og, 0) is None
