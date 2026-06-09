"""Tests for expansion target efficiency scoring."""

from __future__ import annotations

import math

from orbit_expansion import (
    ExpansionConfig,
    enumerate_expansion_candidates,
    pick_best_expansion,
    production_value_until_end,
    score_expansion_launch,
)
from orbit_graph import build_orbit_graph


def _minimal_env(*, fleets=None, step=10):
    # Vertical / offset layout so launches do not cross the sun at (50, 50).
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
        "step": step,
    }


def test_production_value_decreases_with_later_eta():
    early_val, early_hold = production_value_until_end(
        3, eta=5.0, current_turn=10, max_turns=250,
    )
    late_val, late_hold = production_value_until_end(
        3, eta=40.0, current_turn=10, max_turns=250,
    )
    assert early_hold > late_hold
    assert early_val > late_val > 0.0


def test_efficiency_favours_high_production_neutral():
    env = _minimal_env()
    og = build_orbit_graph(env, player=0)
    planets = og.planets
    src = og.planet_by_id[0]

    from navigation import compute_launch
    launch_hi = compute_launch(src, og.planet_by_id[1], 10, planets, 0.0)
    assert launch_hi is not None
    cand_hi = score_expansion_launch(
        src,
        og.planet_by_id[1],
        10,
        eta=float(launch_hi["eta"]),
        angle=float(launch_hi["angle"]),
        og=og,
        player=0,
        current_turn=10,
        max_turns=250,
    )
    assert cand_hi is not None
    assert cand_hi.production_value > 0.0
    assert cand_hi.efficiency == cand_hi.production_value / cand_hi.fleet_cost
    assert cand_hi.ships_required <= cand_hi.ships_sent


def test_pick_best_expansion_prefers_prod3_over_prod1():
    best = pick_best_expansion(_minimal_env(), player=0, current_turn=10, max_turns=250)
    assert best is not None
    # Planet 1 has production 3 vs enemy planet 2 production 1.
    assert best.tgt_id == 1
    assert best.production_value > 0.0
    assert best.efficiency > 0.0
    assert best.eta >= 0.0


def test_enumerate_returns_sorted_by_efficiency():
    cands = enumerate_expansion_candidates(
        build_orbit_graph(_minimal_env(), player=0),
        _minimal_env(),
        player=0,
        current_turn=10,
        max_turns=250,
        cfg=ExpansionConfig(target_enemies=True),
    )
    assert len(cands) >= 2
    assert cands[0].tgt_id == 1  # prod-3 neutral beats prod-1 enemy on efficiency
    for a, b in zip(cands, cands[1:]):
        assert a.score_tuple >= b.score_tuple
