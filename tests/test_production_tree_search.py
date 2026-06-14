"""Tests for production-potential capture tree search."""

from __future__ import annotations

import pytest

from production_tree_search import (
    ProductionTreeConfig,
    format_capture_plan,
    plan_capture_tree,
    production_value_until_end,
)


def _env_vertical(*, step=10):
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 80, 2],
            [1, -1, 20.0, 70.0, 1.0, 5, 5],
            [2, -1, 80.0, 20.0, 1.0, 5, 1],
            [3, 1, 80.0, 70.0, 1.0, 12, 4],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 80, 2],
            [1, -1, 20.0, 70.0, 1.0, 5, 5],
            [2, -1, 80.0, 20.0, 1.0, 5, 1],
            [3, 1, 80.0, 70.0, 1.0, 12, 4],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "step": step,
    }


def test_production_value_until_end_is_raw_prod_times_hold_by_default():
    assert production_value_until_end(3, capture_turn=20.0, max_turns=100) == 240.0
    assert production_value_until_end(3, capture_turn=90.0, max_turns=100) == 30.0
    assert production_value_until_end(3, capture_turn=100.0, max_turns=100) == 0.0


def test_tree_prefers_high_production_neutral_first():
    node = plan_capture_tree(
        _env_vertical(),
        0,
        cfg=ProductionTreeConfig(max_turns=200, max_depth=2, beam_width=8),
    )
    assert node.steps
    assert node.steps[0].target_id == 1
    assert node.captured_value > 0.0


def test_tree_can_chain_multiple_captures():
    node = plan_capture_tree(
        _env_vertical(),
        0,
        cfg=ProductionTreeConfig(max_turns=200, max_depth=3, beam_width=16),
    )
    captured = [s.target_id for s in node.steps]
    assert 1 in captured
    assert len(captured) >= 2
    assert node.production_rate > 2.0
    assert "total_value=" in format_capture_plan(node)


def test_tree_tracks_launch_path_and_per_planet_ship_counts():
    node = plan_capture_tree(
        _env_vertical(),
        0,
        cfg=ProductionTreeConfig(max_turns=200, max_depth=1, beam_width=8),
    )
    step = node.steps[0]
    assert step.flight.src_id == step.source_id
    assert step.flight.tgt_id == step.target_id
    assert step.flight.eta == pytest.approx(step.eta)
    assert step.flight.arrival_turn == pytest.approx(step.capture_turn)
    assert (
        (step.flight.start_x - step.flight.intercept_x) ** 2
        + (step.flight.start_y - step.flight.intercept_y) ** 2
    ) > 1.0
    assert step.source_ships_after_launch < step.source_ships_before
    assert node.planet_states[step.target_id].owner == 0
    assert node.planet_states[step.target_id].ships == pytest.approx(
        step.target_ships_after_capture,
    )


def test_enemy_target_produces_while_waiting_and_fleet_travels():
    env = {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 500, 0],
            [1, 1, 20.0, 70.0, 1.0, 5, 5],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 500, 0],
            [1, 1, 20.0, 70.0, 1.0, 5, 5],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
        "step": 0,
    }
    node = plan_capture_tree(
        env,
        0,
        cfg=ProductionTreeConfig(max_turns=200, max_depth=1, beam_width=4),
    )
    assert node.steps
    step = node.steps[0]
    assert step.target_id == 1
    assert step.ships_required > 6
    assert step.target_ships_before_combat > 5.0


def test_tree_returns_root_when_no_owned_planets():
    env = _env_vertical()
    for row in env["planets"]:
        row[1] = 1
    node = plan_capture_tree(env, 0)
    assert node.steps == ()
    assert node.captured_value == pytest.approx(0.0)
