"""Tests for production diffusion potential on the orbit graph."""

from __future__ import annotations

import math

from orbit_graph import build_orbit_graph
from orbit_production_pressure import (
    diffusion_potential_balance,
    friendly_diffusion_potential_on_target,
    node_diffusion_potential,
)


def _minimal_env(fleets=None):
    return {
        "planets": [
            [0, 0, 20.0, 50.0, 1.0, 30, 5],
            [1, -1, 80.0, 50.0, 1.0, 5, 1],
            [2, 1, 50.0, 20.0, 1.0, 3, 8],
        ],
        "fleets": fleets or [],
        "initial_planets": [
            [0, 0, 20.0, 50.0, 1.0, 30, 5],
            [1, -1, 80.0, 50.0, 1.0, 5, 1],
            [2, 1, 50.0, 20.0, 1.0, 3, 8],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_high_prod_owned_planet_raises_friendly_production_pressure_on_enemy():
    og = build_orbit_graph(_minimal_env(), player=0)
    # Planet 0: prod 5, large garrison; planet 2 enemy prod 8.
    p_on_enemy = friendly_diffusion_potential_on_target(og, 2)
    assert p_on_enemy > 0.0
    ch = node_diffusion_potential(og, 2)
    assert ch["potential_friendly_diffusion"] > 0.0
    assert ch["realized_friendly_diffusion"] == 0.0


def test_higher_owned_production_raises_potential_pressure_on_enemy():
    base = _minimal_env()
    env_low = dict(base)
    env_high = dict(base)
    env_low["planets"] = [
        [0, 0, 20.0, 50.0, 1.0, 30, 2],
        [1, -1, 80.0, 50.0, 1.0, 5, 1],
        [2, 1, 50.0, 20.0, 1.0, 3, 8],
    ]
    env_high["planets"] = [
        [0, 0, 20.0, 50.0, 1.0, 30, 10],
        [1, -1, 80.0, 50.0, 1.0, 5, 1],
        [2, 1, 50.0, 20.0, 1.0, 3, 8],
    ]
    env_low["initial_planets"] = env_low["planets"]
    env_high["initial_planets"] = env_high["planets"]
    og_low = build_orbit_graph(env_low, player=0)
    og_high = build_orbit_graph(env_high, player=0)
    p_low = friendly_diffusion_potential_on_target(og_low, 2)
    p_high = friendly_diffusion_potential_on_target(og_high, 2)
    assert p_high > p_low > 0.0


def test_production_pressure_ignores_garrison_when_prod_equal():
    """Two layouts with same production but different garrisons → similar prod pressure."""
    env_a = {
        "planets": [
            [0, 0, 20.0, 50.0, 1.0, 100, 4],
            [1, 1, 80.0, 50.0, 1.0, 3, 1],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 50.0, 1.0, 100, 4],
            [1, 1, 80.0, 50.0, 1.0, 3, 1],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }
    env_b = dict(env_a)
    env_b["planets"] = [
        [0, 0, 20.0, 50.0, 1.0, 5, 4],
        [1, 1, 80.0, 50.0, 1.0, 3, 1],
    ]
    env_b["initial_planets"] = env_b["planets"]
    og_a = build_orbit_graph(env_a, player=0)
    og_b = build_orbit_graph(env_b, player=0)
    pa = friendly_diffusion_potential_on_target(og_a, 1)
    pb = friendly_diffusion_potential_on_target(og_b, 1)
    assert math.isclose(pa, pb, rel_tol=1e-5)


def test_production_pressure_balance_on_enemy_node():
    og = build_orbit_graph(_minimal_env(), player=0)
    bal = diffusion_potential_balance(og, 2)
    assert isinstance(bal, float)
