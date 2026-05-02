"""Tests for orbit_wars navigation module.

Run from repo root:
    pytest tests/ -v
"""
from __future__ import annotations

import math
import sys
import os

import pytest

# Make repo root importable regardless of where pytest is invoked from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from feature_extraction import Planet, Fleet, fleet_speed, SUN_X, SUN_Y, SUN_RADIUS
from navigation import (
    segment_min_dist_to_point,
    segment_stays_in_bounds,
    crosses_sun,
    find_intercepts,
    find_intercepts_from,
    path_collision,
    path_collision_from,
    compute_launch,
    compute_launch_at,
    earliest_reachable,
    reachability_scan,
    predict_fleet_target,
    predict_all_fleet_targets,
    incoming_threats,
)


# =============================================================================
# Planet factories for tests
# =============================================================================
def make_static(pid: int, x: float, y: float, *,
                owner: int = -1, radius: float = 2.0,
                ships: int = 10, prod: int = 1) -> Planet:
    """Build a static planet. Forces is_rotating=False so that tests can
    place the planet anywhere on the board (even close to the sun) without
    accidentally getting a rotating body.
    """
    p = Planet.from_obs_row([pid, owner, x, y, radius, ships, prod])
    p.is_rotating = False
    p.initial_theta = None
    return p


def make_fleet(fid: int, owner: int, x: float, y: float, angle: float,
               from_pid: int, ships: int) -> Fleet:
    return Fleet.from_obs_row([fid, owner, x, y, angle, from_pid, ships])


def make_rotating(pid: int, r: float, theta0: float, *,
                  owner: int = -1, radius: float = 1.5,
                  ships: int = 10, prod: int = 1) -> Planet:
    """Build a rotating planet at polar (r, theta0) around the sun.

    r + radius must be < 50 (ROTATION_RADIUS_LIMIT) for classification.
    """
    x = SUN_X + r * math.cos(theta0)
    y = SUN_Y + r * math.sin(theta0)
    # initial_row = current row -> "step 0" placement
    row = [pid, owner, x, y, radius, ships, prod]
    return Planet.from_obs_row(row, initial_row=row)


# =============================================================================
# fleet_speed
# =============================================================================
class TestFleetSpeed:
    def test_zero_ships_is_zero(self):
        assert fleet_speed(0) == 0.0

    def test_negative_is_zero(self):
        assert fleet_speed(-5) == 0.0

    def test_one_ship_is_one(self):
        assert fleet_speed(1) == 1.0

    def test_1000_is_max(self):
        assert fleet_speed(1000) == pytest.approx(6.0, abs=1e-9)

    def test_monotonic_increasing(self):
        speeds = [fleet_speed(s) for s in range(1, 1001, 10)]
        for a, b in zip(speeds, speeds[1:]):
            assert b >= a

    def test_spot_values_reasonable(self):
        # Between known bounds (v(10) ~= 1.96, v(100) ~= 3.80)
        assert 1.5 < fleet_speed(10) < 2.5
        assert 3.0 < fleet_speed(100) < 4.5

    def test_max_speed_parameter(self):
        # Doubling max changes only the asymptote, not the shape
        assert fleet_speed(1000, max_speed=10) == pytest.approx(10.0)


# =============================================================================
# segment geometry
# =============================================================================
class TestSegmentGeometry:
    def test_point_on_line_midway(self):
        # Point (5, 3) projects onto segment (0,0)-(10,0) at (5, 0)
        d = segment_min_dist_to_point(0, 0, 10, 0, 5, 3)
        assert d == pytest.approx(3.0)

    def test_point_past_endpoint_clamps(self):
        # Projection is past B=(10,0); clamps to B; dist = 10
        d = segment_min_dist_to_point(0, 0, 10, 0, 20, 0)
        assert d == pytest.approx(10.0)

    def test_point_before_start_clamps(self):
        d = segment_min_dist_to_point(0, 0, 10, 0, -5, 0)
        assert d == pytest.approx(5.0)

    def test_degenerate_segment(self):
        # Zero-length segment falls back to point distance
        d = segment_min_dist_to_point(5, 5, 5, 5, 8, 9)
        assert d == pytest.approx(math.hypot(3, 4))

    def test_segment_bounds_rejects_outside_endpoint(self):
        assert segment_stays_in_bounds(10, 10, 90, 90)
        assert not segment_stays_in_bounds(10, 10, 101, 90)


# =============================================================================
# Sun crossing
# =============================================================================
class TestCrossesSun:
    def test_horizontal_through_center_crosses(self):
        # (10,50) -> (90,50) passes through (50,50)
        assert crosses_sun(10, 50, 90, 50)

    def test_vertical_through_center_crosses(self):
        assert crosses_sun(50, 10, 50, 90)

    def test_edge_miss_clears(self):
        # Passes far from sun
        assert not crosses_sun(0, 0, 100, 5)

    def test_tangent_case_with_margin(self):
        # Exactly grazing the sun -- should be flagged due to margin
        # Line y = SUN_Y + R; sun center to line distance = R
        assert crosses_sun(0, SUN_Y + SUN_RADIUS, 100, SUN_Y + SUN_RADIUS)

    def test_just_outside_is_clear(self):
        # Well clear of the sun
        assert not crosses_sun(0, 80, 100, 80)


# =============================================================================
# find_intercepts (static target)
# =============================================================================
class TestFindInterceptsStatic:
    def test_closed_form(self):
        src = make_static(0, 10, 10)
        tgt = make_static(1, 40, 10)
        ships = 100
        v = fleet_speed(ships)
        cands = find_intercepts(src, tgt, ships, angular_velocity=0.0)
        assert len(cands) == 1
        t_star, ix, iy = cands[0]
        assert t_star == pytest.approx(30.0 / v)
        assert (ix, iy) == (40.0, 10.0)

    def test_zero_ships_empty(self):
        src = make_static(0, 10, 10)
        tgt = make_static(1, 40, 10)
        assert find_intercepts(src, tgt, 0, 0.0) == []


# =============================================================================
# find_intercepts (rotating target) -- lead shot
# =============================================================================
class TestFindInterceptsRotating:
    def test_intercept_equation_holds(self):
        # Src far; target rotating -- first root must satisfy
        # ||src - tgt(t*)|| == v * t*
        src = make_static(0, 90, 10)  # far corner (not across sun)
        tgt = make_rotating(1, r=30, theta0=math.pi * 0.8)
        ships = 50
        omega = 0.04
        v = fleet_speed(ships)

        cands = find_intercepts(src, tgt, ships, omega, max_results=4)
        assert len(cands) >= 1
        t_star, ix, iy = cands[0]
        d = math.hypot(ix - src.x, iy - src.y)
        assert d == pytest.approx(v * t_star, rel=1e-3)

    def test_candidates_ascending(self):
        src = make_static(0, 90, 10)
        tgt = make_rotating(1, r=30, theta0=math.pi * 0.8)
        cands = find_intercepts(src, tgt, 50, 0.05, max_results=4)
        ts = [c[0] for c in cands]
        assert ts == sorted(ts)
        assert all(a != b for a, b in zip(ts, ts[1:]))  # strictly increasing

    def test_unreachable_fleet_too_slow(self):
        # If fleet slower than target's tangential speed, f(t) may never
        # reach zero within t_max -> fewer (or zero) roots.
        src = make_static(0, 90, 10)
        tgt = make_rotating(1, r=40, theta0=0.0)
        # Big omega, minimum ship count -> very hard
        cands = find_intercepts(src, tgt, 1, 0.5, t_max=50, max_results=4)
        # May find 0 or 1; we just assert the answer (if any) is sane
        for t_star, ix, iy in cands:
            v = fleet_speed(1)
            d = math.hypot(ix - src.x, iy - src.y)
            assert d == pytest.approx(v * t_star, rel=1e-3)


# =============================================================================
# find_intercepts_from (generalized source position)
# =============================================================================
class TestFindInterceptsFrom:
    def test_matches_find_intercepts_when_t_offset_zero(self):
        src = make_static(0, 10, 10)
        tgt = make_rotating(1, r=25, theta0=math.pi / 3)
        ships = 30
        omega = 0.03

        a = find_intercepts(src, tgt, ships, omega, max_results=3)
        b = find_intercepts_from(src.x, src.y, tgt, ships, omega,
                                 t_offset=0.0, max_results=3)
        assert len(a) == len(b)
        for (ta, xa, ya), (tb, xb, yb) in zip(a, b):
            assert ta == pytest.approx(tb, rel=1e-4)
            assert xa == pytest.approx(xb, abs=1e-3)
            assert ya == pytest.approx(yb, abs=1e-3)

    def test_t_offset_shifts_target(self):
        # Launching later means the rotating target is at a different spot.
        tgt = make_rotating(1, r=25, theta0=0.0)
        omega = 0.1
        # From origin-ish corner
        c0 = find_intercepts_from(5, 5, tgt, 40, omega, t_offset=0.0)
        c10 = find_intercepts_from(5, 5, tgt, 40, omega, t_offset=10.0)
        # At least one intercept in each; the intercept x,y should differ
        assert c0 and c10
        _, ix0, iy0 = c0[0]
        _, ix10, iy10 = c10[0]
        # Position at t_offset=10 is rotated by omega*10 = 1 rad => definitely different
        assert math.hypot(ix0 - ix10, iy0 - iy10) > 1.0


# =============================================================================
# path_collision
# =============================================================================
class TestPathCollision:
    def test_clear_path(self):
        src = make_static(0, 10, 10)
        # Only a far-away planet; fleet flies east; no obstacles
        others = [make_static(2, 99, 99)]
        v = fleet_speed(100)
        t_arrival = 80 / v
        assert path_collision(src, angle=0.0, ships=100,
                              t_arrival=t_arrival,
                              other_planets=others,
                              angular_velocity=0.0) is None

    def test_static_blocker_on_line(self):
        src = make_static(0, 10, 10)
        blocker = make_static(2, 50, 10, radius=3.0)
        others = [blocker]
        v = fleet_speed(100)
        t_arrival = 80 / v  # reach x=90
        blocker_id = path_collision(src, 0.0, 100, t_arrival, others, 0.0)
        assert blocker_id == blocker.id

    def test_blocker_off_path_ignored(self):
        src = make_static(0, 10, 10)
        # Blocker far above the east-bound path
        off_path = make_static(2, 50, 90, radius=2.0)
        v = fleet_speed(100)
        t_arrival = 80 / v
        assert path_collision(src, 0.0, 100, t_arrival, [off_path], 0.0) is None

    def test_rotating_blocker_start_of_interval(self):
        """Regression for bug where a rotating blocker slipped through
        because path_collision only checked the END of each sub-interval.

        Scenario reproduced from user report (step 69 launch from planet 0
        at (68.6, 95.1) toward planet 17). Rotating planet 15 crosses
        the trajectory at turn ~80, at position_at(~11).
        """
        # Src at the actual coords from the bug report
        src = Planet.from_obs_row(
            [0, 0, 68.60863828631827, 95.09783068169907, 2.386, 298, 4],
        )
        src.is_rotating = False
        src.initial_theta = None

        # Rotating blocker matching planet 15: r_sun=26.2, theta_launch=0.148
        r15 = 26.2
        theta15_launch = 0.148
        blocker = make_rotating(15, r=r15, theta0=theta15_launch, radius=1.0)

        # Fake target placeholder (far away, off-path) -- compute_launch's
        # intercept was (77.9, 28.7). We only care that path_collision flags
        # the rotating blocker on the way there.
        v = fleet_speed(20)
        angle = -1.4309822458870551  # from history
        # Fly ~27 turns at this speed, matching the original arrival.
        t_arrival = 27.6

        blocker_id = path_collision(
            src, angle=angle, ships=20, t_arrival=t_arrival,
            other_planets=[blocker],
            angular_velocity=0.036833916252796396,
            sample_step=1.0, extra_margin=0.1,
        )
        # THE fix: this MUST now return 15 (previously silently passed).
        assert blocker_id == 15


# =============================================================================
# compute_launch (top-level)
# =============================================================================
class TestComputeLaunch:
    def test_simple_success(self):
        src = make_static(0, 10, 10)
        tgt = make_static(1, 80, 10)
        result = compute_launch(src, tgt, 20, [src, tgt], 0.0)
        assert result is not None
        assert result["ships"] == 20
        # Pointing east
        assert result["angle"] == pytest.approx(0.0, abs=1e-3)

    def test_src_eq_tgt_is_none(self):
        src = make_static(0, 10, 10)
        assert compute_launch(src, src, 20, [src], 0.0) is None

    def test_zero_ships_is_none(self):
        src = make_static(0, 10, 10)
        tgt = make_static(1, 80, 10)
        assert compute_launch(src, tgt, 0, [src, tgt], 0.0) is None

    def test_sun_blocks_when_no_alternative(self):
        # Two static planets across the sun, no other path option.
        src = make_static(0, 15, 50)
        tgt = make_static(1, 85, 50)
        assert compute_launch(src, tgt, 20, [src, tgt], 0.0) is None

    def test_eta_equals_distance_over_speed_static(self):
        src = make_static(0, 10, 10)
        tgt = make_static(1, 80, 10)
        ships = 30
        result = compute_launch(src, tgt, ships, [src, tgt], 0.0)
        assert result is not None
        d = 70.0
        assert result["eta"] == pytest.approx(d / fleet_speed(ships), rel=1e-3)


# =============================================================================
# Regression: compute_launch vs compute_launch_at(t_offset=0)
# =============================================================================
class TestComputeLaunchRegression:
    def test_equivalent_at_t_offset_zero_static(self):
        # Both on the top row, no sun in between
        src = make_static(0, 10, 10)
        tgt = make_static(1, 80, 20)
        planets = [src, tgt]
        a = compute_launch(src, tgt, 30, planets, 0.0)
        b = compute_launch_at(src, tgt, 30, planets, 0.0, t_offset=0.0)
        assert a is not None and b is not None
        for key in ("ships", "angle", "eta", "intercept_x", "intercept_y"):
            assert a[key] == pytest.approx(b[key], rel=1e-4)

    def test_equivalent_at_t_offset_zero_rotating(self):
        # Src static, tgt rotating -- same equivalence
        src = make_static(0, 10, 10)
        tgt = make_rotating(1, r=20, theta0=1.0)
        planets = [src, tgt]
        omega = 0.03
        a = compute_launch(src, tgt, 40, planets, omega)
        b = compute_launch_at(src, tgt, 40, planets, omega, t_offset=0.0)
        assert a is not None and b is not None
        for key in ("angle", "eta", "intercept_x", "intercept_y"):
            assert a[key] == pytest.approx(b[key], rel=1e-3)


# =============================================================================
# compute_launch_at with non-zero t_offset
# =============================================================================
class TestComputeLaunchAt:
    def test_rotating_target_shifts_with_delay(self):
        src = make_static(0, 90, 90)
        tgt = make_rotating(1, r=30, theta0=0.0)
        planets = [src, tgt]
        omega = 0.05

        a = compute_launch_at(src, tgt, 30, planets, omega, t_offset=0.0)
        b = compute_launch_at(src, tgt, 30, planets, omega, t_offset=8.0)
        assert a is not None and b is not None
        # The intercept points should differ: rotating target is at different spot
        dx = b["intercept_x"] - a["intercept_x"]
        dy = b["intercept_y"] - a["intercept_y"]
        assert math.hypot(dx, dy) > 1.0


# =============================================================================
# earliest_reachable
# =============================================================================
class TestEarliestReachable:
    def test_feasible_k_zero(self):
        src = make_static(0, 10, 10)
        tgt = make_static(1, 80, 10)
        info = earliest_reachable(src, tgt, 20, [src, tgt], 0.0, horizon=5)
        assert info is not None
        assert info["k"] == 0
        assert info["total_time"] == pytest.approx(info["eta"])

    def test_infeasible_across_sun_is_none(self):
        # Two static planets across the sun: no amount of waiting helps
        src = make_static(0, 15, 50)
        tgt = make_static(1, 85, 50)
        info = earliest_reachable(src, tgt, 20, [src, tgt], 0.0, horizon=10)
        assert info is None

    def test_ship_accumulation_gate(self):
        # src has 5 ships, production=1; we need 20 -> k = 15 minimum
        src_row = [0, 0, 10, 10, 2.0, 5, 1]
        src = Planet.from_obs_row(src_row)
        tgt = make_static(1, 80, 10)
        info = earliest_reachable(
            src, tgt, 20, [src, tgt], 0.0,
            horizon=30, consider_production=True,
        )
        assert info is not None
        assert info["k"] >= 15
        assert info["ships_available_at_k"] >= 20

    def test_total_time_is_k_plus_eta(self):
        src = make_static(0, 10, 10)
        tgt = make_static(1, 80, 20)
        info = earliest_reachable(src, tgt, 20, [src, tgt], 0.0, horizon=5)
        assert info is not None
        assert info["total_time"] == pytest.approx(info["k"] + info["eta"])


# =============================================================================
# reachability_scan
# =============================================================================
class TestReachabilityScan:
    def test_sorted_by_total_time(self):
        src = make_static(0, 5, 5)
        t_near = make_static(1, 20, 5)   # close
        t_far = make_static(2, 95, 5)    # further, same axis
        rows = reachability_scan(src, [t_near, t_far], 50,
                                 [src, t_near, t_far], 0.0, horizon=20)
        assert len(rows) == 2
        assert rows[0]["target_id"] == t_near.id
        assert rows[1]["target_id"] == t_far.id
        assert rows[0]["total_time"] < rows[1]["total_time"]

    def test_infeasible_at_end(self):
        src = make_static(0, 15, 50)
        ok = make_static(1, 15, 10)       # reachable
        blocked = make_static(2, 85, 50)  # across sun
        rows = reachability_scan(src, [ok, blocked], 30,
                                 [src, ok, blocked], 0.0, horizon=10)
        # feasible first, infeasible last
        assert rows[0]["target_id"] == ok.id
        assert rows[0]["feasible"] is True
        assert rows[-1]["target_id"] == blocked.id
        assert rows[-1]["feasible"] is False
        assert rows[-1]["total_time"] == float("inf")

    def test_skips_self(self):
        src = make_static(0, 10, 10)
        other = make_static(1, 80, 10)
        rows = reachability_scan(src, [src, other], 30, [src, other], 0.0)
        ids = [r["target_id"] for r in rows]
        assert src.id not in ids


# =============================================================================
# predict_fleet_target
# =============================================================================
class TestPredictFleetTarget:
    def test_head_on_static_hit(self):
        # Fleet at (10, 10) flying east toward static planet at (50, 10).
        # y=10 keeps us well away from the sun at (50, 50).
        target = make_static(7, 50, 10, radius=2.0)
        fleet = make_fleet(0, owner=0, x=10, y=10, angle=0.0,
                           from_pid=99, ships=20)
        pred = predict_fleet_target(fleet, [target], 0.0)
        assert pred is not None
        assert pred["target_id"] == 7
        v = fleet_speed(20)
        assert pred["eta"] == pytest.approx(38.0 / v, rel=0.1)

    def test_perpendicular_miss_returns_none(self):
        # Fleet flying south; planet far to the east
        target = make_static(7, 80, 50, radius=2.0)
        fleet = make_fleet(0, 0, 10, 10, math.pi / 2, 99, 20)
        pred = predict_fleet_target(fleet, [target], 0.0, t_max=100.0)
        assert pred is None

    def test_source_planet_excluded(self):
        # Fleet just spawned next to its source. Without exclusion it
        # would 'predict' itself hitting the source planet.
        source = make_static(5, 10, 10, radius=2.0)
        far_target = make_static(7, 80, 10, radius=2.0)
        fleet = make_fleet(0, 0, 12.5, 10, 0.0, from_pid=5, ships=20)
        pred = predict_fleet_target(fleet, [source, far_target], 0.0)
        assert pred is not None
        assert pred["target_id"] == far_target.id

    def test_out_of_bounds_returns_none(self):
        # Fleet at (95, 50) flying east -> leaves board with no planets nearby
        fleet = make_fleet(0, 0, 95, 50, 0.0, from_pid=99, ships=20)
        # Nothing in the way
        pred = predict_fleet_target(fleet, [], 0.0)
        assert pred is None

    def test_sun_destroys_fleet_returns_none(self):
        # Fleet at (10, 50) flying east into the sun, with planet only on
        # the FAR side -> fleet dies before reaching planet
        target = make_static(7, 90, 50, radius=2.0)
        fleet = make_fleet(0, 0, 10, 50, 0.0, from_pid=99, ships=20)
        pred = predict_fleet_target(fleet, [target], 0.0)
        assert pred is None

    def test_rotating_target_intercept(self):
        # Rotating planet at theta=-pi/2 (above sun, at (50, 20)).
        # Fleet flies east along y=20 -> hits it.
        tgt = make_rotating(7, r=30, theta0=-math.pi / 2)  # at (50, 20)
        fleet = make_fleet(0, 0, 10, 20, 0.0, from_pid=99, ships=30)
        pred = predict_fleet_target(fleet, [tgt], 0.0)  # omega=0
        assert pred is not None
        assert pred["target_id"] == 7


# =============================================================================
# predict_all_fleet_targets + incoming_threats
# =============================================================================
class TestThreats:
    def test_predict_all_keys_by_fleet_id(self):
        target = make_static(7, 50, 10)
        f1 = make_fleet(11, 0, 10, 10, 0.0, 99, 20)        # east -> hits
        f2 = make_fleet(22, 1, 90, 10, math.pi, 99, 20)    # west -> hits
        out = predict_all_fleet_targets([f1, f2], [target], 0.0)
        assert set(out.keys()) == {11, 22}
        assert out[11]["target_id"] == 7
        assert out[22]["target_id"] == 7

    def test_incoming_threats_filters_my_planets(self):
        # Player 0 owns planet 7. Enemy fleet heading to it = threat.
        # Friendly fleet heading to enemy planet 8 = NOT a threat.
        my_planet = make_static(7, 50, 10, owner=0)
        enemy_planet = make_static(8, 50, 90, owner=1)
        enemy_attacker = make_fleet(11, 1, 10, 10, 0.0, 99, 25)  # -> my 7
        my_outgoing = make_fleet(22, 0, 10, 90, 0.0, 99, 20)     # -> enemy 8

        threats = incoming_threats(
            player=0,
            fleets=[enemy_attacker, my_outgoing],
            planets=[my_planet, enemy_planet],
            angular_velocity=0.0,
        )
        ids = [t["fleet_id"] for t in threats]
        assert 11 in ids
        assert 22 not in ids

    def test_incoming_threats_sorted_by_eta(self):
        my_planet = make_static(7, 50, 10, owner=0)
        # Two enemies, one closer than the other
        near_enemy = make_fleet(11, 1, 30, 10, 0.0, 99, 20)
        far_enemy = make_fleet(22, 1, 5, 10, 0.0, 99, 20)
        threats = incoming_threats(0, [far_enemy, near_enemy],
                                   [my_planet], 0.0)
        assert len(threats) == 2
        # Closer enemy should arrive first
        assert threats[0]["fleet_id"] == 11
        assert threats[0]["eta"] < threats[1]["eta"]
