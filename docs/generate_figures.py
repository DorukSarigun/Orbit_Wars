"""
Generate the figures embedded in docs/math.md.

Run once from repo root:
    python docs/generate_figures.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Arc, Wedge

# Make parent package importable when running this script directly.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from feature_extraction import fleet_speed, SUN_X, SUN_Y, SUN_RADIUS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

BG = "#0c0c1c"
FG = "#eaeaea"
MUTED = "#888"
BLUE = "#5b9bff"
RED = "#ff5b5b"
GOLD = "#ffd700"
SUN = "#ffb000"
GREEN = "#6be37a"


def _board_ax(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(MUTED)


def _sun(ax):
    ax.add_patch(Circle((SUN_X, SUN_Y), SUN_RADIUS, color=SUN, alpha=0.9, zorder=2))


# ============================================================================
# 1. Board overview (coordinate system + planet types)
# ============================================================================
def fig_board_overview():
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
    _board_ax(ax)
    _sun(ax)

    # Orbit rings
    for r in (20, 32, 44):
        ax.add_patch(Circle((SUN_X, SUN_Y), r, fill=False,
                            ec="#3a3a5a", ls="--", lw=0.6))

    # Static planets (outer)
    for (x, y, lbl) in [(12, 15, "stat"), (88, 10, "stat"), (10, 85, "stat"), (90, 88, "stat")]:
        ax.add_patch(Circle((x, y), 2.4, color=MUTED, ec="black", zorder=4))
        ax.text(x, y, lbl, ha="center", va="center", color="white",
                fontsize=6, fontweight="bold")

    # Rotating planets on rings
    for (r, theta, lbl) in [(20, 0.8, "rot"), (32, 2.2, "rot"), (44, 4.0, "rot")]:
        x, y = 50 + r*math.cos(theta), 50 + r*math.sin(theta)
        ax.add_patch(Circle((x, y), 1.6, color=BLUE, ec="black", zorder=4))
        ax.text(x, y, lbl, ha="center", va="center", color="white", fontsize=5.5, fontweight="bold")
        # Motion arrow along tangent
        tx, ty = -math.sin(theta), math.cos(theta)
        ax.annotate("", xytext=(x, y), xy=(x + tx*4, y + ty*4),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))

    # Comet on path
    pts = [(85, 70), (72, 60), (60, 55), (50, 52), (40, 52), (30, 55), (18, 62)]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, ":", color=GOLD, lw=0.8, alpha=0.7)
    cx, cy = pts[2]
    ax.plot(cx, cy, marker="*", markersize=16,
            markerfacecolor=GOLD, markeredgecolor="#fff5b0", mew=1, zorder=5)
    ax.text(cx, cy - 3, "comet", ha="center", color=GOLD, fontsize=6.5, fontweight="bold")

    # Fleet triangle
    fx, fy = 70, 75
    ang = math.radians(200)
    ca, sa = math.cos(ang), math.sin(ang)
    tip = (fx + ca*2.2, fy + sa*2.2)
    left = (fx - ca*1.4 - sa*1.4, fy - sa*1.4 + ca*1.4)
    right = (fx - ca*1.4 + sa*1.4, fy - sa*1.4 - ca*1.4)
    ax.add_patch(Polygon([tip, left, right], facecolor=BLUE, edgecolor="white", lw=1))
    ax.text(fx + 2, fy + 3, "fleet", color=BLUE, fontsize=6.5, fontweight="bold")

    # Axis legend
    ax.set_title("Board: sun (50,50, R=10), orbit rings, three planet types",
                 color=FG, fontsize=10)
    ax.annotate("", xytext=(2, 98), xy=(2, 92),
                arrowprops=dict(arrowstyle="->", color=FG))
    ax.text(4, 95, "+y (angle=π/2)", color=FG, fontsize=7)
    ax.annotate("", xytext=(2, 98), xy=(10, 98),
                arrowprops=dict(arrowstyle="->", color=FG))
    ax.text(5.5, 97, "+x (angle=0)", color=FG, fontsize=7, va="bottom")

    plt.tight_layout()
    fig.savefig(f"{OUT}/board_overview.png", dpi=120, facecolor=BG)
    plt.close(fig)


# ============================================================================
# 2. Fleet speed curve
# ============================================================================
def fig_speed_curve():
    ships = np.arange(1, 2001)
    v = np.array([fleet_speed(int(s)) for s in ships])

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(ships, v, color=BLUE, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("ships (log scale)", color=FG)
    ax.set_ylabel("speed (units / turn)", color=FG)
    ax.set_title("Fleet speed  v(s) = 1 + 5 · (ln s / ln 1000)^1.5",
                 color=FG, fontsize=10)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color="#222", ls="--", lw=0.5)
    for spine in ax.spines.values():
        spine.set_color(MUTED)

    # Annotations at common ship counts
    for s in (1, 10, 100, 1000):
        vs = fleet_speed(s)
        ax.plot(s, vs, "o", color=GOLD, markersize=5)
        ax.annotate(f"s={s}\nv={vs:.2f}",
                    xy=(s, vs), xytext=(5, 5), textcoords="offset points",
                    color=GOLD, fontsize=7)

    plt.tight_layout()
    fig.savefig(f"{OUT}/speed_curve.png", dpi=120, facecolor=BG)
    plt.close(fig)


# ============================================================================
# 3. Intercept f(t) for a rotating target: shows multiple roots
# ============================================================================
def fig_intercept_fn():
    # Setup: src static, target rotates around sun
    sx, sy = 85, 90
    r = 30
    theta0 = math.radians(180)  # starts on far side
    omega = 0.05                 # rad per turn (fast rotation for clear picture)
    ships = 15
    v = fleet_speed(ships)

    ts = np.linspace(0, 200, 2000)
    xs = 50 + r*np.cos(theta0 + omega*ts)
    ys = 50 + r*np.sin(theta0 + omega*ts)
    dist = np.hypot(xs - sx, ys - sy)
    ft = dist - v*ts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)

    # Left: f(t)
    ax1.set_facecolor(BG)
    ax1.plot(ts, ft, color=BLUE, lw=1.5, label="f(t) = ‖src − tgt(t)‖ − v·t")
    ax1.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax1.set_xlabel("t (turns)", color=FG)
    ax1.set_ylabel("f(t)", color=FG)
    ax1.set_title(f"Intercept function for a rotating target  (ships={ships}, v={v:.2f})",
                  color=FG, fontsize=10)
    # Mark zero crossings (sign changes from + to 0 or -)
    sign = np.sign(ft)
    cross = np.where((sign[:-1] > 0) & (sign[1:] <= 0))[0]
    for i, idx in enumerate(cross[:4]):
        t_star = ts[idx]
        ax1.plot(t_star, 0, "o", color=GOLD, ms=8, zorder=5)
        ax1.annotate(f"t*_{i}={t_star:.1f}",
                     xy=(t_star, 0), xytext=(4, -12 if i % 2 else 10),
                     textcoords="offset points", color=GOLD, fontsize=8)
    ax1.tick_params(colors=MUTED)
    ax1.grid(True, color="#222", ls="--", lw=0.5)
    for s in ax1.spines.values():
        s.set_color(MUTED)
    ax1.legend(facecolor=BG, labelcolor=FG, loc="upper right")

    # Right: board with intercept points
    _board_ax(ax2)
    _sun(ax2)
    # orbit
    ax2.add_patch(Circle((50, 50), r, fill=False, ec="#3a3a5a", ls="--"))
    # src
    ax2.add_patch(Circle((sx, sy), 2, color=BLUE, ec="white", zorder=4))
    ax2.text(sx, sy, "src", ha="center", va="center", color="white", fontsize=6, fontweight="bold")
    # target now
    tx0 = 50 + r*math.cos(theta0); ty0 = 50 + r*math.sin(theta0)
    ax2.add_patch(Circle((tx0, ty0), 1.6, color=RED, ec="white", zorder=4))
    ax2.text(tx0, ty0 - 2.5, "tgt(0)", color=RED, fontsize=6.5, ha="center", fontweight="bold")

    # intercept points and fleet paths
    for i, idx in enumerate(cross[:3]):
        t_star = ts[idx]
        ix = 50 + r*math.cos(theta0 + omega*t_star)
        iy = 50 + r*math.sin(theta0 + omega*t_star)
        ax2.plot([sx, ix], [sy, iy], ":", color=GOLD, lw=1.2, alpha=0.8)
        ax2.plot(ix, iy, "x", color=GOLD, ms=10, mew=2)
        ax2.text(ix + 2, iy + 2, f"t*_{i}", color=GOLD, fontsize=7, fontweight="bold")

    ax2.set_title("Intercept points on the board (lap solutions)",
                  color=FG, fontsize=10)

    plt.tight_layout()
    fig.savefig(f"{OUT}/intercept_fn.png", dpi=120, facecolor=BG)
    plt.close(fig)


# ============================================================================
# 4. Sun avoidance: segment-circle distance
# ============================================================================
def fig_sun_avoidance():
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
    _board_ax(ax)
    _sun(ax)

    # Case 1: segment that crosses sun (red)
    ax.plot([15, 90], [85, 15], color=RED, lw=2.0, alpha=0.9, label="crosses sun (BAD)")
    ax.plot(15, 85, "o", color=RED, ms=7); ax.text(12, 83, "A₁", color=RED, fontsize=9)
    ax.plot(90, 15, "o", color=RED, ms=7); ax.text(91, 14, "B₁", color=RED, fontsize=9)

    # Case 2: segment that clears sun (green)
    ax.plot([15, 85], [15, 85], color=GREEN, lw=2.0, alpha=0.9, label="clear (OK)")
    ax.plot(15, 15, "o", color=GREEN, ms=7); ax.text(10, 13, "A₂", color=GREEN, fontsize=9)
    ax.plot(85, 85, "o", color=GREEN, ms=7); ax.text(86, 87, "B₂", color=GREEN, fontsize=9)

    # perpendicular from sun to segment illustration (for the green case)
    # project (50,50) onto segment (15,15)-(85,85)
    dx, dy = 70, 70
    L2 = dx*dx + dy*dy
    t = ((50-15)*dx + (50-15)*dy) / L2
    px, py = 15 + t*dx, 15 + t*dy
    ax.plot([50, px], [50, py], ":", color=GOLD, lw=1.5)
    dist = math.hypot(50-px, 50-py)
    ax.annotate(f"d = {dist:.1f} > R = {SUN_RADIUS}",
                xy=((50+px)/2, (50+py)/2), xytext=(55, 40),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.8),
                color=GOLD, fontsize=9)

    # min dist for the bad one
    dx2, dy2 = 75, -70
    L22 = dx2*dx2 + dy2*dy2
    t2 = ((50-15)*dx2 + (50-85)*dy2) / L22
    t2 = max(0.0, min(1.0, t2))
    px2, py2 = 15 + t2*dx2, 85 + t2*dy2
    ax.plot([50, px2], [50, py2], ":", color=RED, lw=1.5)
    dist2 = math.hypot(50-px2, 50-py2)
    ax.annotate(f"d = {dist2:.1f} < R = {SUN_RADIUS}",
                xy=((50+px2)/2, (50+py2)/2), xytext=(18, 55),
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
                color=RED, fontsize=9)

    ax.set_title("Sun avoidance: segment-point min distance vs R_sun",
                 color=FG, fontsize=10)
    ax.legend(facecolor=BG, labelcolor=FG, loc="lower left")

    plt.tight_layout()
    fig.savefig(f"{OUT}/sun_avoidance.png", dpi=120, facecolor=BG)
    plt.close(fig)


# ============================================================================
# 5. Tangent path around sun (theoretical shortest path)
# ============================================================================
def fig_tangent_path():
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
    _board_ax(ax)
    _sun(ax)

    A = np.array([15, 85])
    B = np.array([90, 15])

    # Direct (blocked)
    ax.plot([A[0], B[0]], [A[1], B[1]], color=RED, lw=1.2, ls="--", alpha=0.6,
            label="direct (blocked)")
    ax.plot(*A, "o", color=BLUE, ms=7); ax.text(A[0]-4, A[1]-2, "A", color=BLUE, fontsize=10, fontweight="bold")
    ax.plot(*B, "o", color=BLUE, ms=7); ax.text(B[0]+1, B[1], "B", color=BLUE, fontsize=10, fontweight="bold")

    O = np.array([SUN_X, SUN_Y])
    R = SUN_RADIUS

    def tangent_points(P):
        """Both tangent points from external point P to circle (O, R)."""
        OP = P - O
        d = np.linalg.norm(OP)
        base_ang = math.atan2(OP[1], OP[0])
        alpha = math.acos(R / d)
        return [O + R * np.array([math.cos(base_ang + s*alpha), math.sin(base_ang + s*alpha)])
                for s in (+1, -1)]

    TA_pair = tangent_points(A)
    TB_pair = tangent_points(B)

    # Pick the combination yielding the shorter path; both go on the same side.
    best = None
    for ta in TA_pair:
        for tb in TB_pair:
            # arc angle from ta to tb along the circle, short way
            ang_a = math.atan2(ta[1]-O[1], ta[0]-O[0])
            ang_b = math.atan2(tb[1]-O[1], tb[0]-O[0])
            d_ang = abs((ang_b - ang_a + math.pi) % (2*math.pi) - math.pi)
            L = np.linalg.norm(A - ta) + R*d_ang + np.linalg.norm(B - tb)
            if best is None or L < best[0]:
                best = (L, ta, tb, d_ang, ang_a, ang_b)

    L, ta, tb, d_ang, ang_a, ang_b = best
    ax.plot([A[0], ta[0]], [A[1], ta[1]], color=GREEN, lw=2)
    ax.plot([B[0], tb[0]], [B[1], tb[1]], color=GREEN, lw=2)
    # arc
    # Determine sweep direction for the short arc
    a0 = math.degrees(ang_a)
    a1 = math.degrees(ang_b)
    # go short way
    delta = ((a1 - a0 + 180) % 360) - 180
    theta1, theta2 = (a0, a0 + delta) if delta > 0 else (a0 + delta, a0)
    ax.add_patch(Arc((O[0], O[1]), 2*R, 2*R, theta1=theta1, theta2=theta2,
                     edgecolor=GREEN, lw=2, zorder=3))

    # Tangent points
    ax.plot(*ta, "o", color=GOLD, ms=6); ax.text(ta[0]-2, ta[1]+2, "T_A", color=GOLD, fontsize=9)
    ax.plot(*tb, "o", color=GOLD, ms=6); ax.text(tb[0]+1, tb[1], "T_B", color=GOLD, fontsize=9)

    # Lengths
    ax.text(50, 92,
            f"L* = √(|OA|²−R²) + R·Δφ + √(|OB|²−R²)\n"
            f"   = {np.linalg.norm(A-ta):.2f} + {R*d_ang:.2f} + {np.linalg.norm(B-tb):.2f} "
            f"= {L:.2f}",
            color=FG, ha="center", fontsize=9,
            bbox=dict(facecolor="#111", edgecolor=MUTED, boxstyle="round,pad=0.3"))

    ax.set_title("Tangent + arc + tangent path around the sun (theoretical lower bound)",
                 color=FG, fontsize=10)
    ax.legend(facecolor=BG, labelcolor=FG, loc="lower right")

    plt.tight_layout()
    fig.savefig(f"{OUT}/tangent_path.png", dpi=120, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    fig_board_overview()
    fig_speed_curve()
    fig_intercept_fn()
    fig_sun_avoidance()
    fig_tangent_path()
    print(f"Figures written to {OUT}")
