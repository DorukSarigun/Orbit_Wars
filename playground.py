"""
Orbit Wars - Interactive Playground.

A thin wrapper around the kaggle env that lets you:
  - launch fleets by (src_id, tgt_id, ships) [angle is auto-computed]
  - step the game forward
  - render the board with matplotlib

Designed for notebook use:

    pg = Playground(num_agents=2)
    pg.render()                       # see the initial board
    pg.planet_table()                 # pick a src and tgt by id
    pg.launch(src_id=0, tgt_id=7, ships=8)
    pg.step()                         # advance one turn (opponent does nothing)
    pg.render()                       # see the fleet in flight
"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from kaggle_environments import make

from feature_extraction import (
    parse_planets,
    parse_fleets,
    SUN_X, SUN_Y, SUN_RADIUS,
)
from navigation import compute_launch, earliest_reachable, reachability_scan


# ----- Color palette ---------------------------------------------------------
OWNER_COLORS = {
    -1: "#888888",   # neutral
    0:  "#5b9bff",   # player 0 (you, by default)
    1:  "#ff5b5b",   # player 1
    2:  "#5bff8b",   # player 2
    3:  "#ffaa5b",   # player 3
}
COMET_FILL  = "#ffd700"   # bright gold so comets pop against any owner color
COMET_EDGE  = "#fff5b0"
SUN_COLOR   = "#ffb000"
BG_COLOR    = "#0c0c1c"
ORBIT_COLOR = "#3a3a5a"


# ----- Renderer --------------------------------------------------------------
def render_board(
    obs,
    *,
    recent_launches: Optional[list[dict]] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (8.5, 8.5),
    show_orbits: bool = True,
    show_fleet_arrows: bool = True,
):
    """Draw the full board state for a given observation."""
    planets = parse_planets(obs)
    fleets = parse_fleets(obs)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # env: y increases downward (angle pi/2 = down)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    # ---- Orbit rings (rotating planets) and comet paths ----
    if show_orbits:
        seen_radii = set()
        for p in planets:
            if p.is_rotating:
                key = round(p.r_sun, 2)
                if key in seen_radii:
                    continue
                seen_radii.add(key)
                ring = Circle((SUN_X, SUN_Y), p.r_sun, fill=False,
                              ec=ORBIT_COLOR, ls="--", lw=0.6, zorder=1)
                ax.add_patch(ring)
            if p.is_comet and p.comet_path:
                xs = [pt[0] for pt in p.comet_path]
                ys = [pt[1] for pt in p.comet_path]
                ax.plot(xs, ys, color=COMET_EDGE, lw=0.4, ls=":", alpha=0.5, zorder=1)

    # ---- Sun ----
    ax.add_patch(Circle((SUN_X, SUN_Y), SUN_RADIUS,
                        color=SUN_COLOR, alpha=0.9, zorder=2))
    ax.add_patch(Circle((SUN_X, SUN_Y), SUN_RADIUS + 1.0,
                        fill=False, ec=SUN_COLOR, alpha=0.3, zorder=2))

    # ---- Planets ----
    for p in planets:
        if p.is_comet:
            # Comets: gold star marker, ALWAYS distinct from any planet.
            ax.plot(p.x, p.y, marker="*", markersize=14,
                    markerfacecolor=COMET_FILL, markeredgecolor=COMET_EDGE,
                    markeredgewidth=1.0, zorder=5)
            ax.text(p.x, p.y - 2.2, f"C{p.id}\n{p.ships}",
                    ha="center", va="center", color=COMET_FILL,
                    fontsize=6.5, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        else:
            c = OWNER_COLORS.get(p.owner, "#888")
            ax.add_patch(Circle((p.x, p.y), p.radius,
                                color=c, ec="black", lw=0.8, zorder=4))
            ax.text(p.x, p.y, f"{p.id}\n{p.ships}",
                    ha="center", va="center", color="white",
                    fontsize=7, fontweight="bold", zorder=6)

    # ---- Fleets (triangle ship icons) ----
    if show_fleet_arrows:
        # Outline effect for labels so they read on any background.
        outline = [pe.withStroke(linewidth=2.5, foreground="black")]

        for f in fleets:
            c = OWNER_COLORS.get(f.owner, "#fff")
            cos_a, sin_a = math.cos(f.angle), math.sin(f.angle)

            # Triangle size scales (mildly) with ship count: log10
            # so 10 ships ~ 1.0, 100 ~ 2.0, 1000 ~ 3.0.
            base = 1.4
            size = base + 0.55 * math.log10(max(1, f.ships))

            # Triangle vertices: tip points along angle, two back corners
            # offset perpendicular to angle.
            tip   = (f.x + cos_a * size,
                     f.y + sin_a * size)
            back  = (f.x - cos_a * size * 0.7,
                     f.y - sin_a * size * 0.7)
            left  = (back[0] - sin_a * size * 0.6,
                     back[1] + cos_a * size * 0.6)
            right = (back[0] + sin_a * size * 0.6,
                     back[1] - cos_a * size * 0.6)

            ship_icon = Polygon(
                [tip, left, right],
                closed=True, facecolor=c, edgecolor="white",
                linewidth=1.2, zorder=7,
            )
            ax.add_patch(ship_icon)

            # Heading line (where it's going next ~5 turns)
            look_ahead = 5.0 * f.speed
            ax.plot(
                [f.x, f.x + cos_a * look_ahead],
                [f.y, f.y + sin_a * look_ahead],
                color=c, lw=0.7, ls="-", alpha=0.45, zorder=2,
            )

            # Ship count label, offset perpendicular so it doesn't overlap
            label_off = size + 1.2
            lx = f.x - sin_a * label_off
            ly = f.y + cos_a * label_off
            ax.text(
                lx, ly, f"{f.ships}",
                color=c, fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=8,
                path_effects=outline,
            )

    # ---- Pending/recent launch trajectories (dotted preview) ----
    if recent_launches:
        for L in recent_launches:
            sx, sy = L["src_pos"]
            ix, iy = L["intercept"]
            color = OWNER_COLORS.get(L.get("player", 0), "white")
            ax.plot([sx, ix], [sy, iy],
                    ls=":", color=color, lw=1.0, alpha=0.55, zorder=2)
            ax.plot(ix, iy, marker="x", color=color, ms=8,
                    mew=2, zorder=6)

    step = (obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)) or 0
    n_comets = sum(1 for p in planets if p.is_comet)
    n_fleets = len(fleets)
    suffix = f"  |  fleets: {n_fleets}   comets: {n_comets}"
    ax.set_title((title or f"Step {step}") + suffix, color="white", fontsize=11)

    # Tiny banner when no fleets exist (common cause of "I don't see fleets!")
    if n_fleets == 0:
        ax.text(50, 4, "no fleets in flight",
                ha="center", va="top", color="#888", fontsize=9,
                style="italic", zorder=10)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#444")
    plt.tight_layout()
    plt.show()


# ----- Playground controller -------------------------------------------------
class Playground:
    """Interactive game controller. One instance == one game."""

    def __init__(self, num_agents: int = 2):
        self.env = make("orbit_wars", debug=True)
        self.env.reset(num_agents=num_agents)
        self.num_agents = num_agents
        # First call to step() actually generates the world; do an empty
        # step so render() and launch() have planets to work with.
        self.env.step([[] for _ in range(num_agents)])
        self.pending: list[list] = [[] for _ in range(num_agents)]
        self.history: list[dict] = []   # every launch ever recorded
        self.recent:  list[dict] = []   # launches still in flight (for render)

    # ----- properties --------------------------------------------------------
    @property
    def state(self):
        return self.env.state

    @property
    def obs(self):
        return self.env.state[0]["observation"]

    @property
    def step_num(self) -> int:
        return self.obs.get("step", 0) or 0

    def planets(self):
        return parse_planets(self.obs)

    def fleets(self):
        return parse_fleets(self.obs)

    def reach_table(self, src_id: int, ships: int = 20,
                    horizon: int = 30, player: int = 0):
        """Print 'when can I reach each target' table for a given source."""
        ps = self.planets()
        src = next((p for p in ps if p.id == src_id), None)
        if src is None:
            print(f"  invalid src id {src_id}")
            return
        omega = self.obs["angular_velocity"]
        targets = [p for p in ps if p.id != src_id]
        rows = reachability_scan(src, targets, ships, ps, omega, horizon=horizon)

        print(f"From planet {src_id} with {ships} ships, horizon={horizon} turns:")
        print(f"{'tgt':>3}  {'kind':<6}  {'owner':<5}  "
              f"{'wait k':>6}  {'eta':>6}  {'total':>6}")
        print("-" * 50)
        kind_of = {p.id: ("comet" if p.is_comet else
                          ("rot" if p.is_rotating else "static"))
                   for p in ps}
        owner_of = {p.id: ("ME" if p.owner == player else
                           ("Neut" if p.owner == -1 else f"P{p.owner}"))
                    for p in ps}
        for r in rows:
            tid = r["target_id"]
            if not r["feasible"]:
                print(f"{tid:>3}  {kind_of[tid]:<6}  {owner_of[tid]:<5}  "
                      f"{'--':>6}  {'--':>6}  {'∞':>6}")
            else:
                print(f"{tid:>3}  {kind_of[tid]:<6}  {owner_of[tid]:<5}  "
                      f"{r['k']:>6}  {r['eta']:>6.2f}  {r['total_time']:>6.2f}")

    def fleet_table(self, player: int = 0):
        """Print all fleets currently in flight."""
        flts = self.fleets()
        if not flts:
            print("(no fleets in flight)")
            return
        print(f"{'ID':>3}  {'Owner':<5}  {'X':>5}  {'Y':>5}  "
              f"{'Angle':>6}  {'Ships':>5}  {'Speed':>5}  {'From':>4}")
        print("-" * 60)
        for f in flts:
            owner = "ME" if f.owner == player else f"P{f.owner}"
            print(f"{f.id:>3}  {owner:<5}  {f.x:>5.1f}  {f.y:>5.1f}  "
                  f"{math.degrees(f.angle):>+6.1f}  {f.ships:>5}  "
                  f"{f.speed:>5.2f}  {f.from_planet_id:>4}")

    def status(self) -> list[str]:
        return [s["status"] for s in self.env.state]

    # ----- inspection --------------------------------------------------------
    def planet_table(self, player: int = 0):
        """Print all planets sorted by (mine first, then kind, then id)."""
        rows = []
        for p in self.planets():
            kind = "comet" if p.is_comet else ("rot" if p.is_rotating else "static")
            if p.owner == player:
                owner = "ME"
            elif p.owner == -1:
                owner = "Neut"
            else:
                owner = f"P{p.owner}"
            rows.append((p.id, owner, kind, p.x, p.y, p.ships,
                         p.production, p.r_sun))

        print(f"{'ID':>3}  {'Owner':<5}  {'Kind':<6}  "
              f"{'X':>6}  {'Y':>6}  {'Ships':>5}  {'Prod':>4}  {'rSun':>5}")
        print("-" * 60)
        rows.sort(key=lambda r: (r[1] != "ME", r[2], r[0]))
        for r in rows:
            print(f"{r[0]:>3}  {r[1]:<5}  {r[2]:<6}  "
                  f"{r[3]:>6.1f}  {r[4]:>6.1f}  {r[5]:>5}  "
                  f"{r[6]:>4}  {r[7]:>5.1f}")

    # ----- actions -----------------------------------------------------------
    def launch(self, src_id: int, tgt_id: int, ships: int, player: int = 0):
        """Queue a launch for `player` this turn. Returns the launch record
        (or None if the launch is infeasible)."""
        planets = self.planets()
        src = next((p for p in planets if p.id == src_id), None)
        tgt = next((p for p in planets if p.id == tgt_id), None)

        if src is None or tgt is None:
            print(f"  invalid id (src={src_id}, tgt={tgt_id})")
            return None
        if src.owner != player:
            print(f"  planet {src_id} not owned by P{player} (owner={src.owner})")
            return None
        if src.ships < ships:
            print(f"  planet {src_id} has only {src.ships} ships, asked {ships}")
            return None

        omega = self.obs["angular_velocity"]
        result = compute_launch(src, tgt, ships, planets, omega)
        if result is None:
            print(f"  no valid trajectory: {src_id} -> {tgt_id}, {ships} ships "
                  f"(sun and/or other planets block all candidates)")
            return None

        move = [src.id, result["angle"], int(ships)]
        self.pending[player].append(move)

        record = {
            "step":           self.step_num,
            "player":         player,
            "src_id":         src.id,
            "tgt_id":         tgt.id,
            "ships":          int(ships),
            "src_pos":        (src.x, src.y),
            "intercept":      (result["intercept_x"], result["intercept_y"]),
            "angle":          result["angle"],
            "eta":            result["eta"],
            "arrival_step":   self.step_num + int(math.ceil(result["eta"])),
        }
        self.history.append(record)
        self.recent.append(record)

        print(f"P{player} step {self.step_num}: "
              f"planet {src.id} -> {tgt.id}, {ships} ships, "
              f"angle={math.degrees(result['angle']):+.1f}°, "
              f"ETA={result['eta']:.2f} turns "
              f"(expected arrival ~ step {record['arrival_step']})")
        return record

    def step(self, opponent_action: Optional[list] = None):
        """Advance one turn. If `opponent_action` is given (2-player game),
        it overrides the queued action for player 1."""
        actions = [list(a) for a in self.pending]
        if self.num_agents == 2 and opponent_action is not None:
            actions[1] = opponent_action

        new_state = self.env.step(actions)
        self.pending = [[] for _ in range(self.num_agents)]

        # Drop launches whose arrival has passed.
        now = self.step_num
        self.recent = [r for r in self.recent if r["arrival_step"] >= now]
        return new_state

    def render(self, **kwargs):
        render_board(self.obs, recent_launches=self.recent, **kwargs)

    # ----- summary ----------------------------------------------------------
    def summary(self, player: int = 0):
        ps = self.planets()
        my_p   = [p for p in ps if p.owner == player]
        en_p   = [p for p in ps if p.is_enemy(player)]
        my_sh  = sum(p.ships for p in my_p)
        en_sh  = sum(p.ships for p in en_p)
        my_pr  = sum(p.production for p in my_p)
        en_pr  = sum(p.production for p in en_p)
        flts = parse_fleets(self.obs)
        in_air_mine  = sum(f.ships for f in flts if f.is_mine(player))
        in_air_enemy = sum(f.ships for f in flts if f.is_enemy(player))
        print(f"Step {self.step_num} | status={self.status()}")
        print(f"  ME (P{player}): planets={len(my_p):>2}  ships(garr)={my_sh:>4}  "
              f"ships(flight)={in_air_mine:>4}  prod={my_pr}/turn")
        print(f"  Enemies      : planets={len(en_p):>2}  ships(garr)={en_sh:>4}  "
              f"ships(flight)={in_air_enemy:>4}  prod={en_pr}/turn")
