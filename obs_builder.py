"""
Orbit Wars - Observation builder for RL agents.

Converts a raw env observation into fixed-shape numpy arrays suitable for
a Set Transformer policy.

Design choices
--------------
- Permutation invariant: planets and fleets are presented as sets, padded
  to a max size. Masks tell the model (and the loss) which slots are real.
- Polar features around the sun are preferred over raw (x, y), since the
  game's symmetry is rotational. We still expose sin/cos of theta to
  preserve full information.
- Edge bias matrix encodes pairwise geometry between every token (planet,
  fleet, global). The Set Transformer consumes it as additive attention
  bias so it doesn't have to re-learn distances from scratch.
- Action masking: only own planets with enough ships can be sources; all
  planets are valid targets. A NO-OP "do nothing" choice is exposed to
  the policy via an extra source slot (handled in the model, not here).

Outputs an `Observation` dataclass holding numpy arrays. The agent code
is responsible for batching and tensor conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from feature_extraction import (
    BOARD_SIZE,
    DEFAULT_MAX_SHIP_SPEED,
    Planet,
    Fleet,
    SUN_X,
    SUN_Y,
    fleet_speed,
    parse_planets,
    parse_fleets,
)
from navigation import predict_all_fleet_targets


# -----------------------------------------------------------------------------
# Sizing / feature dims (edit here if you change features below).
# -----------------------------------------------------------------------------
P_MAX_DEFAULT = 48   # env: 5..10 planet groups x 4 mirrors = 20..40 base,
                     # + up to ~8 active comets => 48 covers worst case.
F_MAX_DEFAULT = 64   # no env hard cap on in-flight fleets; 64 is a safe
                     # ceiling for both players' typical/peak load.

MAX_ENEMY_INCOMING_SLOTS = 3
PLANET_FEATURE_DIM = 23
FLEET_FEATURE_DIM = 11
GLOBAL_FEATURE_DIM = 12
EDGE_FEATURE_DIM = 5  # (dist_norm, dx_norm, dy_norm, sin(dphi_sun), cos(dphi_sun))
# The last two channels encode the angular difference between the two tokens
# as seen from the sun (the orbital center). For two planets that share the
# same angular velocity this difference is *time-invariant*, so the model
# gets a stable orbital-relationship signal independent of board rotation.
# For fleets / global token the angle is still well-defined geometrically
# but no longer time-invariant; the model can learn to weight it via the
# per-head edge_proj mixing.


# -----------------------------------------------------------------------------
# Output container
# -----------------------------------------------------------------------------
@dataclass
class Observation:
    """All tensors are numpy arrays; convert with torch.from_numpy at use site."""
    planet_feats: np.ndarray   # (P_max, PLANET_FEATURE_DIM) float32
    planet_mask: np.ndarray    # (P_max,) bool, True = real planet
    fleet_feats: np.ndarray    # (F_max, FLEET_FEATURE_DIM) float32
    fleet_mask: np.ndarray     # (F_max,) bool
    global_feats: np.ndarray   # (GLOBAL_FEATURE_DIM,) float32
    edge_bias: np.ndarray      # (T, T, EDGE_FEATURE_DIM) float32, T = P_max + F_max + 1
    source_mask: np.ndarray    # (P_max,) bool, planets we may launch from
    planet_ids: np.ndarray     # (P_max,) int32, env planet ids (-1 for padding)
    fleet_ids: np.ndarray      # (F_max,) int32
    # Bookkeeping (not a tensor, but useful):
    meta: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Feature extraction helpers
# -----------------------------------------------------------------------------
def _planet_features(
    p: Planet,
    player: int,
    omega: float,
    incoming: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """23-dim feature vector for a single planet."""
    is_self = float(p.owner == player)
    is_enemy = float(p.owner != player and p.owner != -1)
    is_neutral = float(p.owner == -1)

    ships = float(p.ships)
    prod = float(p.production)
    log_ships = math.log(ships + 1.0) / math.log(1000.0)

    # Polar features (sun-centered).
    r_norm = p.r_sun / 50.0
    cos_t = math.cos(p.theta)
    sin_t = math.sin(p.theta)

    omega_dt = omega if p.is_rotating else 0.0

    # "How many ships do I need to capture this?" - rough lower bound.
    ships_to_cap = ships + 1.0 if not is_self else 0.0
    incoming = incoming or {}
    self_ships = float(incoming.get("self_ships", 0.0))
    self_eta = float(incoming.get("self_eta_min", 0.0))
    enemy_ships = list(incoming.get("enemy_ships", [0.0] * MAX_ENEMY_INCOMING_SLOTS))
    enemy_eta = list(incoming.get("enemy_eta_min", [0.0] * MAX_ENEMY_INCOMING_SLOTS))
    enemy_ships = (enemy_ships + [0.0] * MAX_ENEMY_INCOMING_SLOTS)[:MAX_ENEMY_INCOMING_SLOTS]
    enemy_eta = (enemy_eta + [0.0] * MAX_ENEMY_INCOMING_SLOTS)[:MAX_ENEMY_INCOMING_SLOTS]

    return np.asarray([
        is_self,
        is_enemy,
        is_neutral,
        ships / 100.0,
        prod / 5.0,
        p.radius / 5.0,
        r_norm,
        cos_t,
        sin_t,
        float(p.is_static),
        float(p.is_rotating),
        float(p.is_comet),
        omega_dt,
        ships_to_cap / 100.0,
        log_ships,
        self_ships / 500.0,
        min(self_eta, 50.0) / 50.0 if self_eta > 0.0 else 0.0,
        enemy_ships[0] / 500.0,
        min(enemy_eta[0], 50.0) / 50.0 if enemy_eta[0] > 0.0 else 0.0,
        enemy_ships[1] / 500.0,
        min(enemy_eta[1], 50.0) / 50.0 if enemy_eta[1] > 0.0 else 0.0,
        enemy_ships[2] / 500.0,
        min(enemy_eta[2], 50.0) / 50.0 if enemy_eta[2] > 0.0 else 0.0,
    ], dtype=np.float32)


def _fleet_features(
    f: Fleet,
    player: int,
    pred: Optional[dict],
    max_speed: float,
) -> np.ndarray:
    """11-dim feature vector for a single in-flight fleet."""
    is_self = float(f.owner == player)
    is_enemy = float(f.owner != player and f.owner != -1)

    ships = float(f.ships)
    log_ships = math.log(ships + 1.0) / math.log(1000.0)
    speed_norm = f.speed / max_speed if max_speed > 0 else 0.0

    if pred is None:
        eta_norm = 0.0
        has_pred = 0.0
    else:
        eta_norm = min(pred["eta"], 50.0) / 50.0
        has_pred = 1.0

    return np.asarray([
        is_self,
        is_enemy,
        ships / 100.0,
        log_ships,
        speed_norm,
        math.cos(f.angle),
        math.sin(f.angle),
        f.x / BOARD_SIZE - 0.5,
        f.y / BOARD_SIZE - 0.5,
        eta_norm,
        has_pred,
    ], dtype=np.float32)


def _global_features(
    planets: list[Planet],
    fleets: list[Fleet],
    player: int,
    omega: float,
    turn: int,
    incoming_self: int,
    incoming_other: int,
) -> np.ndarray:
    """12-dim global state summary."""
    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
    neutral = [p for p in planets if p.owner == -1]

    my_ships = sum(p.ships for p in my_planets) + sum(f.ships for f in fleets if f.owner == player)
    enemy_ships = sum(p.ships for p in enemy_planets) + sum(
        f.ships for f in fleets if f.owner != player and f.owner != -1
    )
    my_prod = sum(p.production for p in my_planets)
    enemy_prod = sum(p.production for p in enemy_planets)

    return np.asarray([
        turn / 200.0,
        omega,
        len(my_planets) / 25.0,
        len(enemy_planets) / 25.0,
        len(neutral) / 25.0,
        my_ships / 1000.0,
        enemy_ships / 1000.0,
        my_prod / 50.0,
        enemy_prod / 50.0,
        len(fleets) / 25.0,
        incoming_other / 500.0,  # enemy fleets heading to my planets
        incoming_self / 500.0,   # friendly fleets heading to my planets
    ], dtype=np.float32)


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def build_observation(
    env_obs: Any,
    player: int,
    angular_velocity: float,
    *,
    turn: Optional[int] = None,
    p_max: int = P_MAX_DEFAULT,
    f_max: int = F_MAX_DEFAULT,
    min_ships_to_launch: int = 1,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    predict_targets: bool = True,
) -> Observation:
    """Build a fixed-shape Observation from a raw env observation dict.

    Args:
        env_obs: dict-like observation from `env.step(...)` for `player`.
        player: agent's player id (0..3).
        angular_velocity: env's omega; if you don't have it, pass 0.0 and
            rotating planets will be treated as static for the lookahead.
            (Most consumers cache omega from the first observation.)
        turn: current turn index (defaults to 0; pass actual turn for
            better global features).
        p_max, f_max: padding sizes. Real game has <= 25 planets / fleets.
        min_ships_to_launch: source planets with strictly fewer ships are
            masked out (caller can override for stricter policies).
        predict_targets: if True, run `predict_all_fleet_targets` to fill
            in fleet ETA features. Disable for speed-critical paths.

    Returns:
        Observation with all arrays allocated to fixed shapes.
    """
    planets = parse_planets(env_obs)
    fleets = parse_fleets(env_obs)
    if turn is None:
        turn = int(env_obs.get("step", 0) if isinstance(env_obs, dict) else getattr(env_obs, "step", 0) or 0)

    if len(planets) > p_max:
        # Defensive: keep this loud rather than silently truncating planets
        # (they're crucial for action masking).
        raise ValueError(
            f"Observation has {len(planets)} planets but p_max={p_max}. "
            f"Bump P_MAX_DEFAULT or pass a larger p_max."
        )
    if len(fleets) > f_max:
        # Fleet truncation is annoying but recoverable; warn via meta.
        fleets = fleets[:f_max]

    # Predict fleet targets (needed for fleet features and threat counting).
    if predict_targets and fleets:
        preds = predict_all_fleet_targets(fleets, planets, angular_velocity)
    else:
        preds = {}

    enemy_owners = sorted({f.owner for f in fleets if f.owner not in (player, -1)})
    enemy_owner_to_slot = {
        owner: i for i, owner in enumerate(enemy_owners[:MAX_ENEMY_INCOMING_SLOTS])
    }
    incoming_by_planet: dict[int, dict[str, Any]] = {
        p.id: {
            "self_ships": 0.0,
            "self_eta_min": 0.0,
            "enemy_ships": [0.0] * MAX_ENEMY_INCOMING_SLOTS,
            "enemy_eta_min": [0.0] * MAX_ENEMY_INCOMING_SLOTS,
        }
        for p in planets
    }

    # Threat aggregation for global features.
    my_ids = {p.id for p in planets if p.owner == player}
    incoming_other = 0  # enemy ships en route to my planets
    incoming_self = 0   # friendly ships en route to my planets
    for f in fleets:
        pred = preds.get(f.id)
        if pred is None:
            continue
        target_id = pred["target_id"]
        eta = float(pred["eta"])
        planet_incoming = incoming_by_planet.get(target_id)
        if planet_incoming is not None:
            if f.owner == player:
                planet_incoming["self_ships"] += f.ships
                old_eta = float(planet_incoming["self_eta_min"])
                planet_incoming["self_eta_min"] = eta if old_eta <= 0.0 else min(old_eta, eta)
            elif f.owner != -1 and f.owner in enemy_owner_to_slot:
                slot = enemy_owner_to_slot[f.owner]
                planet_incoming["enemy_ships"][slot] += f.ships
                old_eta = float(planet_incoming["enemy_eta_min"][slot])
                planet_incoming["enemy_eta_min"][slot] = eta if old_eta <= 0.0 else min(old_eta, eta)
        if target_id not in my_ids:
            continue
        if f.owner == player:
            incoming_self += f.ships
        elif f.owner != -1:
            incoming_other += f.ships

    # ------- Planet tensor (padded) -------
    planet_feats = np.zeros((p_max, PLANET_FEATURE_DIM), dtype=np.float32)
    planet_mask = np.zeros(p_max, dtype=bool)
    source_mask = np.zeros(p_max, dtype=bool)
    planet_ids = np.full(p_max, -1, dtype=np.int32)
    planet_pos = np.zeros((p_max, 2), dtype=np.float32)  # for edge bias

    for i, p in enumerate(planets):
        planet_feats[i] = _planet_features(
            p, player, angular_velocity, incoming=incoming_by_planet.get(p.id)
        )
        planet_mask[i] = True
        planet_ids[i] = p.id
        planet_pos[i] = (p.x, p.y)
        if p.owner == player and p.ships >= min_ships_to_launch:
            source_mask[i] = True

    # ------- Fleet tensor (padded) -------
    fleet_feats = np.zeros((f_max, FLEET_FEATURE_DIM), dtype=np.float32)
    fleet_mask = np.zeros(f_max, dtype=bool)
    fleet_ids = np.full(f_max, -1, dtype=np.int32)
    fleet_pos = np.zeros((f_max, 2), dtype=np.float32)

    for j, f in enumerate(fleets):
        fleet_feats[j] = _fleet_features(f, player, preds.get(f.id), max_speed)
        fleet_mask[j] = True
        fleet_ids[j] = f.id
        fleet_pos[j] = (f.x, f.y)

    # ------- Global features -------
    global_feats = _global_features(
        planets, fleets, player, angular_velocity, turn,
        incoming_self=incoming_self, incoming_other=incoming_other,
    )

    # ------- Edge bias (T x T x EDGE_FEATURE_DIM) -------
    # Token order: [planets..., fleets..., global]. Global token sits at
    # the sun (50, 50) so its bias to other tokens is well-defined.
    T = p_max + f_max + 1
    positions = np.zeros((T, 2), dtype=np.float32)
    positions[:p_max] = planet_pos
    positions[p_max:p_max + f_max] = fleet_pos
    positions[-1] = (SUN_X, SUN_Y)

    diffs = positions[None, :, :] - positions[:, None, :]   # (T, T, 2)
    dists = np.linalg.norm(diffs, axis=-1)                  # (T, T)

    # Polar angle of every token around the sun. For tokens that orbit at
    # the global angular_velocity (i.e. real planets), phi shifts uniformly
    # each turn, so the *difference* phi_j - phi_i is time-invariant.
    # The global token sits exactly on the sun (radius 0); its phi is
    # ill-defined and falls back to atan2(0, 0) == 0, which is a harmless
    # fixed reference.
    phi = np.arctan2(positions[:, 1] - SUN_Y, positions[:, 0] - SUN_X)  # (T,)
    dphi = phi[None, :] - phi[:, None]                                  # (T, T)

    edge_bias = np.zeros((T, T, EDGE_FEATURE_DIM), dtype=np.float32)
    edge_bias[..., 0] = dists / BOARD_SIZE
    edge_bias[..., 1] = diffs[..., 0] / BOARD_SIZE
    edge_bias[..., 2] = diffs[..., 1] / BOARD_SIZE
    edge_bias[..., 3] = np.sin(dphi)
    edge_bias[..., 4] = np.cos(dphi)

    # Diagonal must remain exactly zero (token paired with itself carries
    # no edge signal). cos(0) would otherwise leak a constant 1.0 onto the
    # diagonal and bias self-attention.
    diag_idx = np.arange(T)
    edge_bias[diag_idx, diag_idx, :] = 0.0

    return Observation(
        planet_feats=planet_feats,
        planet_mask=planet_mask,
        fleet_feats=fleet_feats,
        fleet_mask=fleet_mask,
        global_feats=global_feats,
        edge_bias=edge_bias,
        source_mask=source_mask,
        planet_ids=planet_ids,
        fleet_ids=fleet_ids,
        meta={
            "n_planets": len(planets),
            "n_fleets": len(fleets),
            "incoming_self": incoming_self,
            "incoming_other": incoming_other,
            "p_max": p_max,
            "f_max": f_max,
        },
    )


# -----------------------------------------------------------------------------
# Action decoding
# -----------------------------------------------------------------------------
DEFAULT_SHIP_BINS = (0.1, 0.25, 0.5, 0.75, 1.0)


def decode_action(
    obs: Observation,
    src_slot: int,
    tgt_slot: int,
    ship_bin: int,
    *,
    ship_bins: tuple = DEFAULT_SHIP_BINS,
    planets: Optional[list[Planet]] = None,
    angular_velocity: float = 0.0,
    no_op: bool = False,
) -> Optional[list]:
    """Convert a (src_slot, tgt_slot, ship_bin) policy choice into the
    env's action format `[from_id, angle, ships]`.

    Returns None if the action is a no-op or invalid (e.g., target == source,
    no valid trajectory). The actual launch geometry is resolved by
    `navigation.compute_launch`; the caller passes the parsed planets to
    avoid re-parsing.

    The model picks slots (0..p_max-1); we map them back to env planet ids
    via `obs.planet_ids`.
    """
    if no_op:
        return None
    src_id = int(obs.planet_ids[src_slot])
    tgt_id = int(obs.planet_ids[tgt_slot])
    if src_id < 0 or tgt_id < 0 or src_id == tgt_id:
        return None
    if not obs.source_mask[src_slot]:
        return None

    # Lazy import to avoid pulling navigation into hot training loops if
    # the caller wants to resolve actions in a separate process.
    from navigation import compute_launch

    if planets is None:
        # Caller didn't pass parsed planets; we can't resolve angle.
        return None

    src = next((p for p in planets if p.id == src_id), None)
    tgt = next((p for p in planets if p.id == tgt_id), None)
    if src is None or tgt is None:
        return None

    available = src.ships
    fraction = ship_bins[ship_bin]
    ships = max(1, int(available * fraction))

    launch = compute_launch(src, tgt, ships, planets, angular_velocity)
    if launch is None:
        return None
    return [src.id, launch["angle"], launch["ships"]]
