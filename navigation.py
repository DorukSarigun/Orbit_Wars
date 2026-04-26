"""
Orbit Wars - Navigation / launch geometry.

Given two planets and a fleet size, compute the launch angle that sends
the fleet from src to target, accounting for:
  - target motion (rotating planets, comets) -> lead shot
  - the sun in the middle of the board (deadly to fleets)
  - other planets in the way (cause unwanted combat)

The agent only chooses (from_planet, to_planet, ships); this module turns
that into a concrete (angle, ships) launch order, or returns None if no
valid trajectory exists.

Math overview
-------------
A fleet flies in a straight line at constant speed = fleet_speed(ships).
We need to find the smallest arrival time t* > 0 such that, if we point
the fleet at where the target WILL BE at time t*, the fleet covers
exactly that distance in t* turns:

    dist(src, target(t*)) == speed * t*                    (intercept)

Equivalently, we want the smallest positive root of

    f(t) = dist(src, target(t)) - speed * t

For static targets f is monotone -> closed form.
For rotating / comet targets f oscillates and can have many roots; we
scan coarsely, then bisect each sign change.

Once t* is known, the launch direction is:

    angle = atan2(target_y(t*) - src_y, target_x(t*) - src_x)
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from feature_extraction import (
    Planet,
    Fleet,
    SUN_X,
    SUN_Y,
    SUN_RADIUS,
    DEFAULT_MAX_SHIP_SPEED,
    fleet_speed,
)


# =============================================================================
# 2) Geometry helpers
# =============================================================================
def segment_min_dist_to_point(x1, y1, x2, y2, cx, cy) -> float:
    """Shortest distance from point (cx,cy) to the closed segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return math.hypot(cx - x1, cy - y1)
    # Project (cx,cy) onto the line, clamp t to [0,1] for segment.
    t = ((cx - x1) * dx + (cy - y1) * dy) / L2
    t = max(0.0, min(1.0, t))
    px, py = x1 + t * dx, y1 + t * dy
    return math.hypot(cx - px, cy - py)


def crosses_sun(x1, y1, x2, y2, margin: float = 0.5) -> bool:
    """True iff the straight segment passes within SUN_RADIUS+margin of the sun center.

    The env destroys any fleet whose path comes within SUN_RADIUS of (50,50);
    we add a small margin because the env checks a polyline of per-turn
    sub-segments, which can graze where a single straight segment doesn't.
    """
    return segment_min_dist_to_point(x1, y1, x2, y2, SUN_X, SUN_Y) < (SUN_RADIUS + margin)


# =============================================================================
# 3) Intercept solver
# =============================================================================
def find_intercepts(
    src: Planet,
    target: Planet,
    ships: int,
    angular_velocity: float,
    *,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    t_max: float = 500.0,
    coarse_step: float = 0.5,
    bisect_iters: int = 40,
    bisect_tol: float = 1e-4,
    max_results: int = 8,
) -> list[tuple[float, float, float]]:
    """Find intercept times t* in [0, t_max] sorted ascending.

    Returns list of tuples (t_star, intercept_x, intercept_y).

    For a static target there is exactly one solution (closed form).
    For a moving target there can be multiple "lap" solutions; we return
    up to `max_results` of them in ascending order.

    The smallest t* is the desired one if the corresponding straight path
    is geometrically valid (sun + collision); otherwise the caller can
    consider the next candidate.
    """
    speed = fleet_speed(ships, max_speed)
    if speed <= 0.0:
        return []

    # ---- Static target: single closed-form solution ----
    if target.is_static:
        d = math.hypot(target.x - src.x, target.y - src.y)
        return [(d / speed, target.x, target.y)]

    # ---- Moving target: scan f(t) for sign changes, bisect each ----
    def f(t: float) -> float:
        tx, ty = target.position_at(t, angular_velocity=angular_velocity)
        return math.hypot(tx - src.x, ty - src.y) - speed * t

    results: list[tuple[float, float, float]] = []

    t_prev = 0.0
    f_prev = f(t_prev)

    # Edge case: already converged at t=0 (shouldn't really happen since
    # src and target are distinct planets, but be defensive).
    if f_prev <= 0.0:
        tx, ty = target.position_at(0.0, angular_velocity=angular_velocity)
        results.append((0.0, tx, ty))

    t = coarse_step
    while t <= t_max and len(results) < max_results:
        f_t = f(t)
        # Sign change from positive to non-positive => a root in (t_prev, t].
        if f_prev > 0.0 and f_t <= 0.0:
            lo, hi = t_prev, t
            for _ in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                if f(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
                if (hi - lo) < bisect_tol:
                    break
            t_star = hi
            tx, ty = target.position_at(t_star, angular_velocity=angular_velocity)
            results.append((t_star, tx, ty))
        t_prev, f_prev = t, f_t
        t += coarse_step

    return results


# =============================================================================
# 4) Path collision against other planets
# =============================================================================
def path_collision(
    src: Planet,
    angle: float,
    ships: int,
    t_arrival: float,
    other_planets: Iterable[Planet],
    angular_velocity: float,
    *,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    sample_step: float = 1.0,
    extra_margin: float = 0.1,
) -> Optional[int]:
    """Walk the fleet's straight-line path in `sample_step` increments;
    at each interval, test whether the fleet's motion segment passes
    within (planet.radius + extra_margin) of any planet (at that planet's
    position at the END of the interval, since the env rotates planets
    AFTER fleet movement each turn).

    Returns the id of the first blocking planet, or None if the path is
    clear for the entire interval [0, t_arrival].

    Notes:
      - Sun is not considered here; use `crosses_sun` separately.
      - The fleet spawns at src.x + cos(angle) * (radius + 0.1) (matches env).
      - Static planets are checked once with their fixed position; rotating
        and comet planets are evaluated at each sub-step.
    """
    speed = fleet_speed(ships, max_speed)
    if speed <= 0.0:
        return None

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    spawn_offset = src.radius + 0.1  # where the fleet actually starts

    def fleet_pos(t: float) -> tuple[float, float]:
        d = spawn_offset + speed * t
        return src.x + cos_a * d, src.y + sin_a * d

    # Walk integer-ish turns; final partial turn handled by clamp.
    t = 0.0
    fx, fy = fleet_pos(0.0)
    while t < t_arrival - 1e-9:
        t_next = min(t + sample_step, t_arrival)
        nx, ny = fleet_pos(t_next)
        for p in other_planets:
            if p.id == src.id:
                continue
            # Planet position at the end of the sub-interval (env rotates
            # after fleet movement; see README turn order).
            px, py = p.position_at(t_next, angular_velocity=angular_velocity)
            if segment_min_dist_to_point(fx, fy, nx, ny, px, py) < (p.radius + extra_margin):
                return p.id
        t, fx, fy = t_next, nx, ny

    return None


# =============================================================================
# 5) Top-level: compute_launch
# =============================================================================
def compute_launch(
    src: Planet,
    target: Planet,
    ships: int,
    planets: Iterable[Planet],
    angular_velocity: float,
    *,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    t_max: float = 500.0,
    sun_margin: float = 0.5,
    planet_margin: float = 0.1,
    max_candidates: int = 8,
) -> Optional[dict]:
    """Compute the soonest valid (angle, eta) that sends `ships` ships from
    `src` to `target`, avoiding the sun and any third-party planet collision.

    Returns:
        None if no valid trajectory found within t_max.
        Otherwise a dict:
            {
                "angle":       float,  # radians, ready for env action
                "ships":       int,
                "eta":         float,  # arrival time in turns
                "intercept_x": float,
                "intercept_y": float,
                "speed":       float,
                "candidate_index": int,   # 0 = soonest possible
            }

    Algorithm:
        1. Enumerate intercept candidates (smallest -> largest t*).
        2. For each candidate:
             - Compute aim angle.
             - Reject if straight segment crosses the sun.
             - Reject if any other planet would intercept the fleet first.
        3. Return the first surviving candidate.
    """
    if src.id == target.id or ships <= 0:
        return None

    speed = fleet_speed(ships, max_speed)
    if speed <= 0.0:
        return None

    # We exclude src itself (we spawn outside its radius) and target
    # (we WANT to hit target). Keep everything else as a potential blocker.
    other = [p for p in planets if p.id != src.id and p.id != target.id]

    candidates = find_intercepts(
        src, target, ships, angular_velocity,
        max_speed=max_speed, t_max=t_max, max_results=max_candidates,
    )

    for idx, (t_star, ix, iy) in enumerate(candidates):
        angle = math.atan2(iy - src.y, ix - src.x)

        # Spawn point of the fleet (matches env).
        sx = src.x + math.cos(angle) * (src.radius + 0.1)
        sy = src.y + math.sin(angle) * (src.radius + 0.1)

        # 1) Sun check on the actual flight segment.
        if crosses_sun(sx, sy, ix, iy, margin=sun_margin):
            continue

        # 2) Other-planet collision check, simulated turn-by-turn.
        blocker = path_collision(
            src, angle, ships, t_star, other, angular_velocity,
            max_speed=max_speed, extra_margin=planet_margin,
        )
        if blocker is not None:
            continue

        return {
            "angle": angle,
            "ships": int(ships),
            "eta": t_star,
            "intercept_x": ix,
            "intercept_y": iy,
            "speed": speed,
            "candidate_index": idx,
        }

    return None


# =============================================================================
# 6) Convenience: build the env action triple
# =============================================================================
def to_move(launch: dict, src: Planet) -> list:
    """Convert a compute_launch() result into the env's move format
    [from_planet_id, angle, ships]."""
    return [src.id, launch["angle"], launch["ships"]]


# =============================================================================
# 7) Time-shifted (delayed-launch) variants
# =============================================================================
#
# The functions below answer the question:
#
#     "If I wait k turns and THEN launch from src toward target, is there
#      a valid trajectory? When does the fleet arrive?"
#
# This matters because:
#   - rotating planets keep moving while we wait, so the geometry changes
#   - sometimes a path that's currently blocked (sun, other planet) opens
#     up after a few turns of orbital rotation
#   - the agent may want to defer a launch (e.g. accumulate ships) and
#     needs to know when that becomes feasible
#
# Convention: t_offset = absolute turns from "now" until the launch moment.
#             t (inside f(t)) = turns AFTER launch.
#             So a planet's position when the fleet is at travel-time t is
#             evaluated at absolute time (t_offset + t).
# =============================================================================
def find_intercepts_from(
    src_x: float,
    src_y: float,
    target: Planet,
    ships: int,
    angular_velocity: float,
    *,
    t_offset: float = 0.0,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    t_max: float = 500.0,
    coarse_step: float = 0.5,
    bisect_iters: int = 40,
    bisect_tol: float = 1e-4,
    max_results: int = 8,
) -> list[tuple[float, float, float]]:
    """Same as `find_intercepts` but the source position is given as a
    free point and the target is sampled at absolute time `t_offset + t`.

    Use case: you'll launch in `t_offset` turns from a known location;
    where will you be able to hit the moving target?
    """
    speed = fleet_speed(ships, max_speed)
    if speed <= 0.0:
        return []

    # Static targets: closed form, regardless of t_offset.
    if target.is_static:
        d = math.hypot(target.x - src_x, target.y - src_y)
        return [(d / speed, target.x, target.y)]

    def f(t: float) -> float:
        tx, ty = target.position_at(t_offset + t, angular_velocity=angular_velocity)
        return math.hypot(tx - src_x, ty - src_y) - speed * t

    results: list[tuple[float, float, float]] = []
    t_prev, f_prev = 0.0, f(0.0)
    if f_prev <= 0.0:
        tx, ty = target.position_at(t_offset, angular_velocity=angular_velocity)
        results.append((0.0, tx, ty))

    t = coarse_step
    while t <= t_max and len(results) < max_results:
        f_t = f(t)
        if f_prev > 0.0 and f_t <= 0.0:
            lo, hi = t_prev, t
            for _ in range(bisect_iters):
                mid = 0.5 * (lo + hi)
                if f(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
                if (hi - lo) < bisect_tol:
                    break
            t_star = hi
            tx, ty = target.position_at(t_offset + t_star,
                                        angular_velocity=angular_velocity)
            results.append((t_star, tx, ty))
        t_prev, f_prev = t, f_t
        t += coarse_step

    return results


def path_collision_from(
    src_x: float,
    src_y: float,
    src_radius: float,
    angle: float,
    ships: int,
    t_arrival: float,
    other_planets: Iterable[Planet],
    angular_velocity: float,
    *,
    t_offset: float = 0.0,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    sample_step: float = 1.0,
    extra_margin: float = 0.1,
) -> Optional[int]:
    """Generalized path_collision: launch from arbitrary (src_x, src_y) at
    absolute time t_offset; check planet positions at (t_offset + t)."""
    speed = fleet_speed(ships, max_speed)
    if speed <= 0.0:
        return None

    cos_a, sin_a = math.cos(angle), math.sin(angle)
    spawn_offset = src_radius + 0.1

    def fleet_pos(t: float) -> tuple[float, float]:
        d = spawn_offset + speed * t
        return src_x + cos_a * d, src_y + sin_a * d

    t = 0.0
    fx, fy = fleet_pos(0.0)
    while t < t_arrival - 1e-9:
        t_next = min(t + sample_step, t_arrival)
        nx, ny = fleet_pos(t_next)
        for p in other_planets:
            px, py = p.position_at(t_offset + t_next,
                                   angular_velocity=angular_velocity)
            if segment_min_dist_to_point(fx, fy, nx, ny, px, py) < (p.radius + extra_margin):
                return p.id
        t, fx, fy = t_next, nx, ny

    return None


def compute_launch_at(
    src: Planet,
    target: Planet,
    ships: int,
    planets: Iterable[Planet],
    angular_velocity: float,
    *,
    t_offset: float = 0.0,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    t_max: float = 500.0,
    sun_margin: float = 0.5,
    planet_margin: float = 0.1,
    max_candidates: int = 8,
) -> Optional[dict]:
    """Like `compute_launch`, but assumes the launch happens `t_offset`
    turns from now. Both src and other planets are evaluated at their
    predicted positions at absolute time (t_offset + t).

    Returns the same shape dict as compute_launch, plus a `t_offset` key.
    """
    if src.id == target.id or ships <= 0:
        return None
    speed = fleet_speed(ships, max_speed)
    if speed <= 0.0:
        return None

    sx_launch, sy_launch = src.position_at(t_offset, angular_velocity=angular_velocity)
    other = [p for p in planets if p.id != src.id and p.id != target.id]

    candidates = find_intercepts_from(
        sx_launch, sy_launch, target, ships, angular_velocity,
        t_offset=t_offset, max_speed=max_speed, t_max=t_max,
        max_results=max_candidates,
    )

    for idx, (t_star, ix, iy) in enumerate(candidates):
        angle = math.atan2(iy - sy_launch, ix - sx_launch)

        # Spawn point at launch moment.
        spawn_x = sx_launch + math.cos(angle) * (src.radius + 0.1)
        spawn_y = sy_launch + math.sin(angle) * (src.radius + 0.1)
        if crosses_sun(spawn_x, spawn_y, ix, iy, margin=sun_margin):
            continue

        blocker = path_collision_from(
            sx_launch, sy_launch, src.radius, angle, ships, t_star,
            other, angular_velocity,
            t_offset=t_offset, max_speed=max_speed,
            extra_margin=planet_margin,
        )
        if blocker is not None:
            continue

        return {
            "angle": angle,
            "ships": int(ships),
            "eta": t_star,
            "intercept_x": ix,
            "intercept_y": iy,
            "speed": speed,
            "candidate_index": idx,
            "t_offset": float(t_offset),
            "src_pos_at_launch": (sx_launch, sy_launch),
        }

    return None


# =============================================================================
# 8) Earliest reachable: when does this target become launch-feasible?
# =============================================================================
def earliest_reachable(
    src: Planet,
    target: Planet,
    ships: int,
    planets: Iterable[Planet],
    angular_velocity: float,
    *,
    horizon: int = 30,
    step: int = 1,
    consider_production: bool = False,
    src_production: Optional[int] = None,
    src_current_ships: Optional[int] = None,
    **kwargs,
) -> Optional[dict]:
    """For k = 0, 1, ..., horizon, find the SMALLEST k such that launching
    `ships` from `src` at step (now + k) gives a feasible trajectory.

    Args:
        ships: how many ships we want to send (must be available at launch).
        horizon: max delay to search.
        step: granularity of the delay scan (1 = check every turn).

        consider_production: if True, ALSO require that we'll have enough
            ships at launch time given the source planet's production rate.
            Disabled by default because:
              - the env adds production at the END of the turn,
              - and the "available ships" depends on what other launches
                we've done meanwhile (caller usually handles bookkeeping).

    Returns:
        Dict with all `compute_launch_at` fields plus:
            "k":           the launch delay (turns)
            "total_time":  k + eta (turns until capture from now)
            "ships_available_at_k":  if consider_production was used
        Or None if no feasible launch in [0, horizon].

    Typical usage:
        info = earliest_reachable(src, tgt, ships=20, planets=ps, omega=ω)
        if info is None:
            print("Unreachable in next 30 turns")
        else:
            print(f"Wait {info['k']} turns, fleet arrives in "
                  f"{info['eta']:.1f} more (total {info['total_time']:.1f}).")
    """
    plist = list(planets)

    base_ships = (src_current_ships if src_current_ships is not None
                  else getattr(src, "ships", 0))
    prod = (src_production if src_production is not None
            else getattr(src, "production", 0))

    k = 0
    while k <= horizon:
        # Optional: ship-availability gate at this delay.
        ships_available = base_ships + (prod * k if consider_production else 0)
        enough_ships = (not consider_production) or (ships_available >= ships)

        if enough_ships:
            launch = compute_launch_at(
                src, target, ships, plist, angular_velocity,
                t_offset=float(k), **kwargs,
            )
            if launch is not None:
                launch["k"] = k
                launch["total_time"] = k + launch["eta"]
                if consider_production:
                    launch["ships_available_at_k"] = ships_available
                return launch

        k += step

    return None


# =============================================================================
# 9) Fleet target prediction
# =============================================================================
#
# The env's fleet observation row is [id, owner, x, y, angle, from_planet_id,
# ships]. Notably it does NOT include a target planet id -- destination must
# be inferred from the fleet's straight-line trajectory.
#
# Because a fleet flies in a straight line at constant speed, prediction is
# deterministic: step time forward, compute fleet position and every planet's
# position at that time, and report the first planet whose capture radius
# the fleet enters.
# =============================================================================
BOARD_SIZE = 100.0


def predict_fleet_target(
    fleet: Fleet,
    planets: Iterable[Planet],
    angular_velocity: float,
    *,
    t_max: float = 300.0,
    step: float = 0.5,
    margin: float = 0.1,
) -> Optional[dict]:
    """Predict which planet an in-flight fleet will hit, and when.

    Walks the fleet's trajectory forward in `step`-turn increments, checking
    each sub-segment against every planet's time-shifted position. Returns
    the first hit, or None if the fleet leaves the board / enters the sun
    / reaches t_max without contact.

    Args:
        fleet: active Fleet (position, angle, speed already resolved).
        planets: all planets (source planet is auto-excluded).
        angular_velocity: env's omega for rotating planets.
        t_max: max turns to simulate forward.
        step: time granularity for the scan.
        margin: extra slack added to each planet's radius (matches env's
            per-turn collision tolerance).

    Returns dict with:
        target_id, eta, intercept_x, intercept_y
    or None if no target predicted.
    """
    v = fleet.speed
    if v <= 0.0:
        return None
    cos_a, sin_a = math.cos(fleet.angle), math.sin(fleet.angle)

    # Exclude source planet: fleet is just leaving it, so distance is ~radius.
    others = [p for p in planets if p.id != fleet.from_planet_id]

    t_prev = 0.0
    fx_prev, fy_prev = fleet.x, fleet.y

    t = step
    while t <= t_max + 1e-9:
        fx = fleet.x + cos_a * v * t
        fy = fleet.y + sin_a * v * t

        # Fleet destroyed by sun? (segment passes through it)
        if segment_min_dist_to_point(fx_prev, fy_prev, fx, fy,
                                     SUN_X, SUN_Y) < SUN_RADIUS:
            return None
        # Fleet out of bounds?
        if fx < 0 or fx > BOARD_SIZE or fy < 0 or fy > BOARD_SIZE:
            return None

        # Check each planet at the END of this sub-interval.
        best: Optional[tuple[float, Planet, float, float]] = None
        for p in others:
            px, py = p.position_at(t, angular_velocity=angular_velocity)
            # Minimum distance between the fleet's motion segment
            # (fx_prev, fy_prev) -> (fx, fy) and the planet's end position.
            d = segment_min_dist_to_point(fx_prev, fy_prev, fx, fy, px, py)
            if d <= p.radius + margin:
                if best is None or d < best[0]:
                    best = (d, p, px, py)

        if best is not None:
            _, p, px, py = best
            return {
                "target_id": p.id,
                "eta": t,
                "intercept_x": px,
                "intercept_y": py,
                "fleet_id": fleet.id,
            }

        t_prev, fx_prev, fy_prev = t, fx, fy
        t += step

    return None


def predict_all_fleet_targets(
    fleets: Iterable[Fleet],
    planets: Iterable[Planet],
    angular_velocity: float,
    **kwargs,
) -> dict[int, Optional[dict]]:
    """Batch predict_fleet_target for every fleet; keyed by fleet.id."""
    plist = list(planets)
    out: dict[int, Optional[dict]] = {}
    for f in fleets:
        out[f.id] = predict_fleet_target(f, plist, angular_velocity, **kwargs)
    return out


def incoming_threats(
    player: int,
    fleets: Iterable[Fleet],
    planets: Iterable[Planet],
    angular_velocity: float,
    **kwargs,
) -> list[dict]:
    """Enemy fleets predicted to hit any of `player`'s planets, sorted by
    ETA ascending. Each entry:

        {
            "fleet_id":    int,
            "owner":       int,
            "ships":       int,
            "target_id":   int,
            "eta":         float,
            "intercept":   (x, y),
        }

    Useful for defensive planning: pairs each incoming threat with its
    target so the agent can decide whether to reinforce.
    """
    plist = list(planets)
    my_planet_ids = {p.id for p in plist if p.owner == player}
    rows: list[dict] = []
    for f in fleets:
        if f.owner == player:
            continue
        pred = predict_fleet_target(f, plist, angular_velocity, **kwargs)
        if pred is None:
            continue
        if pred["target_id"] not in my_planet_ids:
            continue
        rows.append({
            "fleet_id": f.id,
            "owner": f.owner,
            "ships": f.ships,
            "target_id": pred["target_id"],
            "eta": pred["eta"],
            "intercept": (pred["intercept_x"], pred["intercept_y"]),
        })
    rows.sort(key=lambda r: r["eta"])
    return rows


def reachability_scan(
    src: Planet,
    targets: Iterable[Planet],
    ships: int,
    planets: Iterable[Planet],
    angular_velocity: float,
    *,
    horizon: int = 30,
    **kwargs,
) -> list[dict]:
    """Run earliest_reachable over many targets, sorted by total_time
    ascending (fastest reachable first). Targets that aren't feasible
    within `horizon` turns are returned with `total_time = inf` at the end.

    Use this as a "what should I attack" oracle.
    """
    plist = list(planets)
    rows: list[dict] = []
    for tgt in targets:
        if tgt.id == src.id:
            continue
        info = earliest_reachable(
            src, tgt, ships, plist, angular_velocity,
            horizon=horizon, **kwargs,
        )
        if info is None:
            rows.append({
                "target_id": tgt.id, "k": None, "eta": None,
                "total_time": float("inf"), "feasible": False,
            })
        else:
            rows.append({
                "target_id": tgt.id, "k": info["k"], "eta": info["eta"],
                "total_time": info["total_time"], "feasible": True,
                "angle": info["angle"], "intercept": (info["intercept_x"],
                                                       info["intercept_y"]),
            })
    rows.sort(key=lambda r: r["total_time"])
    return rows
