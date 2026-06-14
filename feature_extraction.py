"""
Orbit Wars - Planet abstraction.

Raw observation gives us planets as 7-element lists:
    [id, owner, x, y, radius, ships, production]

That's painful to work with. This module wraps each planet in a class that:
  - exposes named fields (no more p[5] vs p[6] confusion)
  - converts cartesian -> sun-centered polar coordinates
  - classifies planet type (static / rotating / comet)
  - predicts future position (lead shot)
  - knows about ownership relative to the agent

The sun is at (50, 50) with radius 10. The board is 100x100.
A planet is "rotating" iff orbital_radius + planet_radius < 50 and it's
not a comet. Comets follow precomputed elliptical paths (group["paths"]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ----- Board constants (mirrors orbit_wars.py defaults) ----------------------
BOARD_SIZE = 100.0
SUN_X, SUN_Y = 50.0, 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0  # orbital_r + planet_r < this -> rotating
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
DEFAULT_MAX_SHIP_SPEED = 6.0


# ----- Fleet kinematics ------------------------------------------------------
def fleet_speed(ships: int, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> float:
    """Speed of a fleet (units per turn) given its ship count.

    Mirrors orbit_wars.py exactly:
        speed = 1 + (max_speed - 1) * (ln(ships) / ln(1000)) ^ 1.5
    Edge cases:
        ships <= 0 -> 0    (no fleet)
        ships == 1 -> 1.0  (avoids log(1) blowup, matches env)
    """
    if ships <= 0:
        return 0.0
    if ships == 1:
        return 1.0
    return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


# ----- Planet ----------------------------------------------------------------
@dataclass
class Planet:
    """Rich wrapper around a single planet observation row.

    Construct via `Planet.from_obs_row(row, ...)` rather than the raw
    constructor when parsing observations -- the factory fills in the
    derived fields (polar coords, type flags, orbital phase).
    """

    # --- Raw fields straight from observation ---
    id: int
    owner: int          # -1 = neutral, 0..3 = player ID
    x: float
    y: float
    radius: float
    ships: int
    production: int

    # --- Derived (filled by from_obs_row) ---
    r_sun: float = 0.0          # distance from sun center
    theta: float = 0.0          # current angle around sun (radians, atan2)
    is_comet: bool = False
    is_rotating: bool = False   # orbits the sun each turn
    # For rotating planets only: angle at game start (step 0).
    # Together with angular_velocity this lets us extrapolate.
    initial_theta: Optional[float] = None
    # For comets only: index into the precomputed path + reference to it.
    comet_path: Optional[Sequence[Sequence[float]]] = None
    comet_path_index: int = -1

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_obs_row(
        cls,
        row: Sequence,
        *,
        comet_ids: Optional[set] = None,
        initial_row: Optional[Sequence] = None,
        comet_path: Optional[Sequence[Sequence[float]]] = None,
        comet_path_index: int = -1,
    ) -> "Planet":
        """Build a Planet from a raw observation row plus optional context.

        Args:
            row: 7-element list from obs["planets"].
            comet_ids: set of planet IDs that are comets (obs["comet_planet_ids"]).
            initial_row: matching row from obs["initial_planets"] (same id).
                Used to recover the orbital starting angle for rotating planets.
            comet_path: this comet's path (list of [x, y] points).
            comet_path_index: current index along the comet path.
        """
        pid, owner, x, y, r, ships, prod = row
        p = cls(
            id=int(pid),
            owner=int(owner),
            x=float(x),
            y=float(y),
            radius=float(r),
            ships=int(ships),
            production=int(prod),
        )

        # Polar (sun-centered) coordinates of the *current* position.
        dx, dy = p.x - SUN_X, p.y - SUN_Y
        p.r_sun = math.hypot(dx, dy)
        p.theta = math.atan2(dy, dx)

        # Classification
        p.is_comet = comet_ids is not None and pid in comet_ids
        if p.is_comet:
            p.is_rotating = False  # comets follow paths, not circular orbit
            p.comet_path = comet_path
            p.comet_path_index = comet_path_index
        else:
            # Use the initial position to decide rotating vs static.
            # This is the source of truth: the env decides "rotating" once
            # at game start based on initial orbital_radius + planet_radius.
            if initial_row is not None:
                ix, iy = float(initial_row[2]) - SUN_X, float(initial_row[3]) - SUN_Y
                init_r = math.hypot(ix, iy)
                p.is_rotating = (init_r + p.radius) < ROTATION_RADIUS_LIMIT
                p.initial_theta = math.atan2(iy, ix)
            else:
                # Fallback: use current position. Less reliable for rotating
                # planets mid-game (theta has drifted), but r_sun is invariant.
                p.is_rotating = (p.r_sun + p.radius) < ROTATION_RADIUS_LIMIT
                p.initial_theta = p.theta

        return p

    # ---------------------------------------------------------------- queries
    @property
    def is_static(self) -> bool:
        """True iff this body never moves."""
        return not self.is_rotating and not self.is_comet

    def is_mine(self, player: int) -> bool:
        return self.owner == player

    def is_neutral(self) -> bool:
        return self.owner == -1

    def is_enemy(self, player: int) -> bool:
        return self.owner != player and self.owner != -1

    # ---------------------------------------------------------- position math
    def position_at(
        self,
        steps_ahead: float,
        *,
        angular_velocity: float = 0.0,
    ) -> tuple[float, float]:
        """Predict (x, y) `steps_ahead` turns from now.

        - Static planets: returns current (x, y).
        - Rotating planets: rotates current theta by omega * steps_ahead at
          fixed r_sun.
        - Comets: follows the precomputed path (linearly interpolated between
          integer indices). Returns current position if no path available or
          if the predicted index exceeds the path length (comet will expire).
        """
        # Hot path: check the cheap instance-attr branches directly and skip
        # the ``is_static`` property (recomputed millions of times during graph
        # builds). Behaviour is identical: a body is static iff it is neither
        # rotating nor a comet, which is exactly the fall-through return.
        if self.is_rotating:
            # r_sun is invariant for orbiting bodies; only theta changes.
            theta_future = self.theta + angular_velocity * steps_ahead
            return (
                SUN_X + self.r_sun * math.cos(theta_future),
                SUN_Y + self.r_sun * math.sin(theta_future),
            )

        if self.is_comet and self.comet_path is not None:
            target_idx = self.comet_path_index + steps_ahead
            n = len(self.comet_path)
            if target_idx < 0 or target_idx >= n - 1:
                # Off-path: comet not yet on board or already expired.
                return self.x, self.y
            i_lo = int(math.floor(target_idx))
            frac = target_idx - i_lo
            x_lo, y_lo = self.comet_path[i_lo][0], self.comet_path[i_lo][1]
            x_hi, y_hi = self.comet_path[i_lo + 1][0], self.comet_path[i_lo + 1][1]
            return x_lo + frac * (x_hi - x_lo), y_lo + frac * (y_hi - y_lo)

        return self.x, self.y

    def ships_at(self, steps_ahead: float) -> float:
        """Naive ship count projection (ignores incoming fleets / combat).

        Used as a quick estimate for "how many ships will the target have
        when my fleet arrives". Production accrues only while owned.
        """
        if self.owner == -1:
            return self.ships  # neutrals don't produce
        return self.ships + self.production * max(0.0, steps_ahead)

    # ------------------------------------------------------------------ misc
    def to_obs_row(self) -> list:
        """Convert back to the raw 7-element row (for env round-trips)."""
        return [self.id, self.owner, self.x, self.y,
                self.radius, self.ships, self.production]

    def __repr__(self) -> str:
        kind = "comet" if self.is_comet else ("rot" if self.is_rotating else "static")
        return (f"Planet(id={self.id}, owner={self.owner}, {kind}, "
                f"r={self.r_sun:.1f}, theta={math.degrees(self.theta):+.0f}deg, "
                f"ships={self.ships}, prod={self.production})")


# ----- Convenience: parse a full observation ---------------------------------
def parse_planets(obs) -> list[Planet]:
    """Build a list[Planet] from an observation (dict or AttrDict).

    Wires up `initial_planets`, `comet_planet_ids`, and `comets` paths so
    that derived fields (initial_theta, comet_path, etc.) are populated.
    """
    get = (lambda k, d=None: obs.get(k, d)) if isinstance(obs, dict) \
        else (lambda k, d=None: getattr(obs, k, d))

    raw_planets = get("planets", []) or []
    raw_initial = get("initial_planets", []) or []
    comet_ids = set(get("comet_planet_ids", []) or [])
    comet_groups = get("comets", []) or []

    initial_by_id = {int(p[0]): p for p in raw_initial}

    # Map each comet id -> (path, current path_index)
    comet_path_by_id: dict[int, tuple[list, int]] = {}
    for group in comet_groups:
        idx = group.get("path_index", -1) if isinstance(group, dict) \
            else getattr(group, "path_index", -1)
        pids = group["planet_ids"] if isinstance(group, dict) else group.planet_ids
        paths = group["paths"] if isinstance(group, dict) else group.paths
        for pid, path in zip(pids, paths):
            comet_path_by_id[int(pid)] = (path, idx)

    planets = []
    for row in raw_planets:
        pid = int(row[0])
        path, idx = comet_path_by_id.get(pid, (None, -1))
        planets.append(
            Planet.from_obs_row(
                row,
                comet_ids=comet_ids,
                initial_row=initial_by_id.get(pid),
                comet_path=path,
                comet_path_index=idx,
            )
        )
    return planets


# =============================================================================
# Fleet
# =============================================================================
@dataclass
class Fleet:
    """A fleet currently in flight.

    Raw observation row:
        [id, owner, x, y, angle, from_planet_id, ships]

    Once launched, a fleet has fixed angle, ships, and speed. Only its
    (x, y) advances each turn by `speed` units along `angle`. The env
    will destroy it if it crosses the sun, leaves the board, or arrives
    at any planet (triggering combat).
    """

    # --- Raw fields ---
    id: int
    owner: int
    x: float
    y: float
    angle: float            # radians, 0 = +x, pi/2 = +y (down)
    from_planet_id: int
    ships: int

    # --- Derived ---
    speed: float = 0.0      # filled by from_obs_row

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_obs_row(cls, row: Sequence,
                     *, max_speed: float = DEFAULT_MAX_SHIP_SPEED) -> "Fleet":
        fid, fown, fx, fy, fang, ffrom, fships = row
        f = cls(
            id=int(fid),
            owner=int(fown),
            x=float(fx),
            y=float(fy),
            angle=float(fang),
            from_planet_id=int(ffrom),
            ships=int(fships),
        )
        f.speed = fleet_speed(f.ships, max_speed)
        return f

    # ---------------------------------------------------------------- queries
    def is_mine(self, player: int) -> bool:
        return self.owner == player

    def is_enemy(self, player: int) -> bool:
        return self.owner != player and self.owner != -1

    # --------------------------------------------------------- position math
    def position_at(self, steps_ahead: float) -> tuple[float, float]:
        """Where this fleet will be `steps_ahead` turns from now (linear)."""
        return (
            self.x + math.cos(self.angle) * self.speed * steps_ahead,
            self.y + math.sin(self.angle) * self.speed * steps_ahead,
        )

    @property
    def heading_vector(self) -> tuple[float, float]:
        """Unit vector along the fleet's heading."""
        return math.cos(self.angle), math.sin(self.angle)

    # ---------------------------------------------------------------- pretty
    def to_obs_row(self) -> list:
        return [self.id, self.owner, self.x, self.y, self.angle,
                self.from_planet_id, self.ships]

    def __repr__(self) -> str:
        return (f"Fleet(id={self.id}, owner={self.owner}, "
                f"pos=({self.x:.1f},{self.y:.1f}), "
                f"angle={math.degrees(self.angle):+.0f}deg, "
                f"ships={self.ships}, speed={self.speed:.2f})")


def parse_fleets(obs) -> list[Fleet]:
    """Build a list[Fleet] from an observation."""
    get = (lambda k, d=None: obs.get(k, d)) if isinstance(obs, dict) \
        else (lambda k, d=None: getattr(obs, k, d))
    raw = get("fleets", []) or []
    return [Fleet.from_obs_row(row) for row in raw]
