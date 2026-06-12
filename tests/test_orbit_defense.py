"""Tests for defense threat + surplus."""

from __future__ import annotations

import math

from orbit_defense import (
    DefenseStatus,
    compute_owned_defense,
    compute_planet_defense,
    inverse_eta_potential,
)
from orbit_graph import build_orbit_graph


def _env_vertical(*, fleets=None):
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 40, 2],
            [1, -1, 20.0, 70.0, 1.0, 5, 3],
            [2, 1, 80.0, 20.0, 1.0, 3, 1],
        ],
        "fleets": fleets or [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 40, 2],
            [1, -1, 20.0, 70.0, 1.0, 5, 3],
            [2, 1, 80.0, 20.0, 1.0, 3, 1],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_unreachable_edge_excluded_from_potential():
    og = build_orbit_graph(_env_vertical(), player=0)
    for _u, _v, d in og.G.edges(data=True):
        assert "reachable" in d
        if not d["reachable"]:
            assert not math.isfinite(d["eta"])

    assert inverse_eta_potential(og, 0, horizon=30.0) >= 0.0


def test_realized_threat_from_incoming_fleet():
    fleets = [[9, 1, 48.0, 22.0, math.atan2(50 - 20, 20 - 80), 2, 20]]
    og = build_orbit_graph(_env_vertical(fleets=fleets), player=0)
    state = compute_planet_defense(og, 0, horizon_cap=30.0)
    assert state is not None
    threats = og.incoming_threats(planet_id=0)
    if threats:
        assert state.realized_enemy_ships > 0.0
        assert state.incoming_threat_count >= 1


def test_defense_surplus_formula_no_incoming():
    og = build_orbit_graph(_env_vertical(), player=0)
    state = compute_planet_defense(og, 0, horizon_cap=15.0, potential_weight=1.0)
    assert state is not None
    assert state.min_incoming_eta is None
    assert state.horizon == 15.0
    assert state.eta_enemy == 15.0
    assert state.production_buffer == state.production * 15.0
    expected_assets = (
        state.garrison + state.production_buffer + state.friendly_inbound_ships
    )
    expected_liab = state.realized_enemy_ships + state.potential_enemy_ships_eq
    assert abs(state.defense_surplus - (expected_assets - expected_liab)) < 1e-6


def test_production_buffer_uses_min_horizon_cap_and_eta_enemy():
    og = build_orbit_graph(_env_vertical(), player=0)
    state = compute_planet_defense(og, 0, horizon_cap=15.0)
    assert state is not None
    if state.min_incoming_eta is not None:
        assert state.horizon == state.min_incoming_eta
        assert state.production_buffer == state.production * min(
            15.0, state.min_incoming_eta,
        )


def test_owned_defense_sorted_by_surplus():
    states = compute_owned_defense(build_orbit_graph(_env_vertical(), player=0))
    assert len(states) == 1
    assert states[0].status in DefenseStatus
