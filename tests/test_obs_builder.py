"""Tests for obs_builder.

Validates shapes, masking semantics, and feature normalization for the
RL observation pipeline.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from obs_builder import (
    EDGE_FEATURE_DIM,
    FLEET_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    PLANET_FEATURE_DIM,
    Observation,
    build_observation,
    decode_action,
)


# -----------------------------------------------------------------------------
# Mini fake observation matching the env schema we consume
# -----------------------------------------------------------------------------
def make_obs(
    *,
    planets: list[list],
    fleets: list[list] | None = None,
    initial_planets: list[list] | None = None,
    comet_ids: list[int] | None = None,
):
    """Build a dict-style env observation. `planets` rows are
    [id, owner, x, y, radius, ships, production]; `fleets` rows are
    [id, owner, x, y, angle, from_planet_id, ships]."""
    return {
        "planets": planets,
        "fleets": fleets or [],
        "initial_planets": initial_planets if initial_planets is not None else planets,
        "comet_planet_ids": comet_ids or [],
        "comets": [],
    }


# -----------------------------------------------------------------------------
# Basic shape / dtype invariants
# -----------------------------------------------------------------------------
def test_shapes_and_dtypes():
    obs = make_obs(planets=[
        [0, 0, 70.0, 70.0, 3.0, 100, 3],   # mine, far from sun -> static
        [1, 1, 30.0, 30.0, 2.0, 50, 2],    # enemy
        [2, -1, 80.0, 20.0, 2.0, 20, 1],   # neutral
    ])
    o = build_observation(obs, player=0, angular_velocity=0.05, p_max=8, f_max=4)

    assert o.planet_feats.shape == (8, PLANET_FEATURE_DIM)
    assert o.planet_feats.dtype == np.float32
    assert o.planet_mask.shape == (8,) and o.planet_mask.dtype == bool
    assert o.fleet_feats.shape == (4, FLEET_FEATURE_DIM)
    assert o.global_feats.shape == (GLOBAL_FEATURE_DIM,)

    T = 8 + 4 + 1
    assert o.edge_bias.shape == (T, T, EDGE_FEATURE_DIM)
    assert o.source_mask.shape == (8,)
    assert o.planet_ids.shape == (8,)


def test_planet_mask_marks_real_slots_only():
    obs = make_obs(planets=[
        [0, 0, 70.0, 70.0, 3.0, 100, 3],
        [1, 1, 30.0, 30.0, 2.0, 50, 2],
    ])
    o = build_observation(obs, player=0, angular_velocity=0.0, p_max=6)
    assert o.planet_mask[:2].all()
    assert not o.planet_mask[2:].any()
    # Padding slots have id == -1.
    assert (o.planet_ids[2:] == -1).all()


def test_source_mask_only_own_planets_with_ships():
    obs = make_obs(planets=[
        [0, 0, 70.0, 70.0, 3.0, 100, 3],   # mine, plenty of ships
        [1, 0, 80.0, 30.0, 2.0, 0, 1],     # mine, but empty
        [2, 1, 30.0, 30.0, 2.0, 50, 2],    # enemy
        [3, -1, 20.0, 80.0, 2.0, 10, 1],   # neutral
    ])
    o = build_observation(obs, player=0, angular_velocity=0.0, min_ships_to_launch=1)
    assert o.source_mask[0]  # owned + ships
    assert not o.source_mask[1]  # owned but empty
    assert not o.source_mask[2]  # enemy
    assert not o.source_mask[3]  # neutral


def test_too_many_planets_raises():
    rows = [[i, 0, 60.0 + 0.1 * i, 60.0, 2.0, 10, 1] for i in range(5)]
    obs = make_obs(planets=rows)
    with pytest.raises(ValueError):
        build_observation(obs, player=0, angular_velocity=0.0, p_max=3)


# -----------------------------------------------------------------------------
# Owner one-hot consistency
# -----------------------------------------------------------------------------
def test_owner_one_hot_in_planet_features():
    obs = make_obs(planets=[
        [0, 0, 70.0, 70.0, 3.0, 100, 3],   # mine
        [1, 1, 30.0, 30.0, 2.0, 50, 2],    # enemy
        [2, -1, 80.0, 20.0, 2.0, 20, 1],   # neutral
    ])
    o = build_observation(obs, player=0, angular_velocity=0.0)
    # Cols 0,1,2 = (is_self, is_enemy, is_neutral); exactly one set per real planet.
    one_hot = o.planet_feats[:3, :3]
    assert np.allclose(one_hot.sum(axis=1), 1.0)
    assert one_hot[0, 0] == 1.0   # planet 0 -> self
    assert one_hot[1, 1] == 1.0   # planet 1 -> enemy
    assert one_hot[2, 2] == 1.0   # planet 2 -> neutral


# -----------------------------------------------------------------------------
# Edge bias geometry
# -----------------------------------------------------------------------------
def test_edge_bias_diagonal_is_zero():
    obs = make_obs(planets=[
        [0, 0, 70.0, 70.0, 3.0, 100, 3],
        [1, 1, 30.0, 30.0, 2.0, 50, 2],
    ])
    o = build_observation(obs, player=0, angular_velocity=0.0, p_max=4, f_max=2)
    T = o.edge_bias.shape[0]
    diag = np.array([o.edge_bias[i, i] for i in range(T)])
    assert np.allclose(diag, 0.0)


def test_edge_bias_distance_matches_geometry():
    # Two planets at known distance -> first edge feature is dist / BOARD_SIZE.
    obs = make_obs(planets=[
        [0, 0, 60.0, 60.0, 2.0, 10, 1],
        [1, 1, 60.0, 90.0, 2.0, 10, 1],
    ])
    o = build_observation(obs, player=0, angular_velocity=0.0, p_max=4, f_max=2)
    expected = 30.0 / 100.0
    assert math.isclose(o.edge_bias[0, 1, 0], expected, abs_tol=1e-5)
    assert math.isclose(o.edge_bias[1, 0, 0], expected, abs_tol=1e-5)


def test_edge_bias_angular_features_match_geometry():
    # Planet 0 at (sun + (10, 0)) -> phi = 0
    # Planet 1 at (sun + (0, 10)) -> phi = pi/2
    # dphi (0 -> 1) = pi/2  =>  sin = 1, cos = 0
    obs = make_obs(planets=[
        [0, 0, 60.0, 50.0, 2.0, 10, 1],
        [1, 0, 50.0, 60.0, 2.0, 10, 1],
    ])
    o = build_observation(obs, player=0, angular_velocity=0.0, p_max=4, f_max=2)
    sin_dphi = o.edge_bias[0, 1, 3]
    cos_dphi = o.edge_bias[0, 1, 4]
    assert math.isclose(sin_dphi, 1.0, abs_tol=1e-5)
    assert math.isclose(cos_dphi, 0.0, abs_tol=1e-5)
    # Anti-symmetric (pair flip): dphi(1->0) = -pi/2
    assert math.isclose(o.edge_bias[1, 0, 3], -1.0, abs_tol=1e-5)
    assert math.isclose(o.edge_bias[1, 0, 4], 0.0, abs_tol=1e-5)


def test_edge_bias_dphi_is_rotation_invariant():
    # Two planets orbiting together: rotating the whole board around the
    # sun must leave (sin dphi, cos dphi) unchanged. This is the whole
    # reason the angular edge features were added.
    import numpy as np

    def rotate(x, y, theta):
        cx, cy = 50.0, 50.0
        dx, dy = x - cx, y - cy
        return cx + dx * math.cos(theta) - dy * math.sin(theta), \
               cy + dx * math.sin(theta) + dy * math.cos(theta)

    base = [
        [0, 0, 70.0, 50.0, 2.0, 10, 1],   # east of sun
        [1, 0, 50.0, 80.0, 2.0, 10, 1],   # north of sun
    ]
    o0 = build_observation(make_obs(planets=base), player=0,
                           angular_velocity=0.0, p_max=4, f_max=2)

    rot = []
    for pid, owner, x, y, r, s, prod in base:
        nx, ny = rotate(x, y, 0.7)
        rot.append([pid, owner, nx, ny, r, s, prod])
    o1 = build_observation(make_obs(planets=rot), player=0,
                           angular_velocity=0.0, p_max=4, f_max=2)

    # Invariance holds only for planet-planet pairs (both ends co-rotate
    # with the board). The global token sits at the sun and padding slots
    # are fixed at (0, 0), so their phi is rotation-dependent and must be
    # excluded from this check.
    n = 2
    assert np.allclose(
        o0.edge_bias[:n, :n, 3:5], o1.edge_bias[:n, :n, 3:5], atol=1e-5
    )


# -----------------------------------------------------------------------------
# Fleets and global features
# -----------------------------------------------------------------------------
def test_fleet_features_populated_and_threats_counted():
    # Source planet at (60,60), fleet flying straight toward my own planet (90,60).
    planets = [
        [0, 0, 60.0, 60.0, 2.0, 100, 2],   # mine, source
        [1, 0, 90.0, 60.0, 2.0, 50, 2],    # mine, target
    ]
    fleets = [
        # enemy fleet headed toward planet 1
        [10, 1, 70.0, 60.0, 0.0, 99, 30],  # angle=0 -> +x
    ]
    obs = make_obs(planets=planets, fleets=fleets)
    o = build_observation(obs, player=0, angular_velocity=0.0)

    # Fleet slot 0 is real, rest padded.
    assert o.fleet_mask[0]
    assert not o.fleet_mask[1:].any()
    # is_enemy == 1
    assert o.fleet_feats[0, 1] == 1.0
    # ships > 0 -> log_ships > 0
    assert o.fleet_feats[0, 3] > 0.0
    # incoming_other (enemy ships heading to my planets) counted in meta + globals
    assert o.meta["incoming_other"] == 30
    assert o.global_feats[10] > 0.0   # incoming_other slot in global feats


def test_planet_features_track_multi_enemy_incoming_slots():
    planets = [
        [0, 0, 60.0, 60.0, 2.0, 100, 2],
        [1, 0, 90.0, 60.0, 2.0, 50, 2],
    ]
    fleets = [
        [10, 0, 70.0, 60.0, 0.0, 0, 20],
        [11, 1, 72.0, 60.0, 0.0, 0, 30],
        [12, 2, 74.0, 60.0, 0.0, 0, 40],
        [13, 3, 76.0, 60.0, 0.0, 0, 50],
        [14, 4, 78.0, 60.0, 0.0, 0, 60],
    ]
    obs = make_obs(planets=planets, fleets=fleets)
    o = build_observation(obs, player=0, angular_velocity=0.0, f_max=8)

    target_feats = o.planet_feats[1]
    assert math.isclose(target_feats[15], 20 / 500.0, abs_tol=1e-6)
    assert target_feats[16] > 0.0
    assert math.isclose(target_feats[17], 30 / 500.0, abs_tol=1e-6)
    assert target_feats[18] > 0.0
    assert math.isclose(target_feats[19], 40 / 500.0, abs_tol=1e-6)
    assert target_feats[20] > 0.0
    assert math.isclose(target_feats[21], 50 / 500.0, abs_tol=1e-6)
    assert target_feats[22] > 0.0


def test_global_features_player_aggregates():
    obs = make_obs(planets=[
        [0, 0, 70.0, 70.0, 2.0, 200, 4],   # mine
        [1, 0, 80.0, 70.0, 2.0, 100, 2],   # mine
        [2, 1, 30.0, 30.0, 2.0, 150, 3],   # enemy
        [3, -1, 20.0, 80.0, 2.0, 10, 1],   # neutral
    ])
    o = build_observation(obs, player=0, angular_velocity=0.0)
    # my_planets / 25
    assert math.isclose(o.global_feats[2], 2 / 25.0, abs_tol=1e-6)
    # enemy_planets / 25
    assert math.isclose(o.global_feats[3], 1 / 25.0, abs_tol=1e-6)
    # neutral / 25
    assert math.isclose(o.global_feats[4], 1 / 25.0, abs_tol=1e-6)
    # my_total_ships / 1000 = 300/1000
    assert math.isclose(o.global_feats[5], 0.3, abs_tol=1e-6)


# -----------------------------------------------------------------------------
# decode_action
# -----------------------------------------------------------------------------
def test_decode_action_no_op_returns_none():
    obs = make_obs(planets=[
        [0, 0, 70.0, 70.0, 2.0, 100, 2],
        [1, 1, 30.0, 30.0, 2.0, 50, 2],
    ])
    o = build_observation(obs, player=0, angular_velocity=0.0)
    assert decode_action(o, 0, 1, 0, no_op=True) is None


def test_decode_action_invalid_source_returns_none():
    obs = make_obs(planets=[
        [0, 0, 70.0, 70.0, 2.0, 100, 2],   # mine
        [1, 1, 30.0, 30.0, 2.0, 50, 2],    # enemy (invalid as source)
    ])
    o = build_observation(obs, player=0, angular_velocity=0.0)
    from feature_extraction import parse_planets
    planets = parse_planets(obs)
    # Try launching from the enemy planet -> should be rejected.
    assert decode_action(o, 1, 0, 0, planets=planets, angular_velocity=0.0) is None


def test_decode_action_valid_returns_env_format():
    # Both planets on the same side of the board so the path doesn't cross the sun.
    obs = make_obs(planets=[
        [0, 0, 70.0, 70.0, 2.0, 100, 2],
        [1, -1, 90.0, 80.0, 2.0, 20, 1],
    ])
    o = build_observation(obs, player=0, angular_velocity=0.0)
    from feature_extraction import parse_planets
    planets = parse_planets(obs)
    action = decode_action(o, 0, 1, 4, planets=planets, angular_velocity=0.0)
    assert action is not None
    assert len(action) == 3
    assert action[0] == 0  # source planet id
    assert isinstance(action[1], float)  # angle
    assert action[2] >= 1  # at least 1 ship
