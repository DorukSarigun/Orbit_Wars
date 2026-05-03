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
PLANET_FEATURE_DIM = 31
INCOMING_ETA_BUCKETS = (5.0, 15.0, 30.0, float("inf"))

# Planet feature index constants (derived from _planet_features order).
# Use these instead of hard-coded ints to survive feature reordering.
PF_IDX_OWNER_SELF = 0
PF_IDX_OWNER_ENEMY = 1
PF_IDX_OWNER_NEUTRAL = 2
PF_IDX_SHIPS = 3          # ships / 100.0
PF_IDX_PRODUCTION = 4     # production / 5.0
PF_IDX_RADIUS = 5         # radius / 5.0
PF_IDX_R_SUN = 6          # r_sun / 50.0
PF_IDX_COS_THETA = 7
PF_IDX_SIN_THETA = 8
PF_IDX_IS_STATIC = 9
PF_IDX_IS_ROTATING = 10
PF_IDX_IS_COMET = 11
PF_IDX_OMEGA = 12         # omega if rotating else 0
PF_IDX_SHIPS_TO_CAP = 13  # ships + 1 (if not self) / 100.0
PF_IDX_LOG_SHIPS = 14
PF_IDX_INCOMING_SELF_B0 = 15        # self ships arriving in [0, 5] turns / 500.0
PF_IDX_INCOMING_SELF_B1 = 16        # self ships arriving in (5, 15] turns / 500.0
PF_IDX_INCOMING_SELF_B2 = 17        # self ships arriving in (15, 30] turns / 500.0
PF_IDX_INCOMING_SELF_B3 = 18        # self ships arriving after 30 turns / 500.0
PF_IDX_INCOMING_ENEMY_B0 = 19       # enemy ships arriving in [0, 5] turns / 500.0
PF_IDX_INCOMING_ENEMY_B1 = 20       # enemy ships arriving in (5, 15] turns / 500.0
PF_IDX_INCOMING_ENEMY_B2 = 21       # enemy ships arriving in (15, 30] turns / 500.0
PF_IDX_INCOMING_ENEMY_B3 = 22       # enemy ships arriving after 30 turns / 500.0
PF_IDX_INCOMING_SELF_TOTAL = 23
PF_IDX_INCOMING_ENEMY_TOTAL = 24
PF_IDX_INCOMING_SELF_MIN_ETA = 25
PF_IDX_INCOMING_ENEMY_MIN_ETA = 26
PF_IDX_INCOMING_ENEMY_SOONEST_SHIPS = 27
PF_IDX_INCOMING_ENEMY_LARGEST_SHIPS = 28
PF_IDX_INCOMING_ENEMY_LARGEST_ETA = 29
PF_IDX_INCOMING_NET_THREAT = 30
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
    planet_feature_dim: int = PLANET_FEATURE_DIM,
) -> np.ndarray:
    """Feature vector for a single planet."""
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
    self_buckets = list(incoming.get("self_buckets", [0.0] * len(INCOMING_ETA_BUCKETS)))
    enemy_buckets = list(incoming.get("enemy_buckets", [0.0] * len(INCOMING_ETA_BUCKETS)))
    self_buckets = (self_buckets + [0.0] * len(INCOMING_ETA_BUCKETS))[:len(INCOMING_ETA_BUCKETS)]
    enemy_buckets = (enemy_buckets + [0.0] * len(INCOMING_ETA_BUCKETS))[:len(INCOMING_ETA_BUCKETS)]
    self_total = float(incoming.get("self_total", sum(self_buckets)))
    enemy_total = float(incoming.get("enemy_total", sum(enemy_buckets)))
    self_min_eta = float(incoming.get("self_min_eta", 0.0))
    enemy_min_eta = float(incoming.get("enemy_min_eta", 0.0))
    enemy_soonest_ships = float(incoming.get("enemy_soonest_ships", 0.0))
    enemy_largest_ships = float(incoming.get("enemy_largest_ships", 0.0))
    enemy_largest_eta = float(incoming.get("enemy_largest_eta", 0.0))
    net_threat = enemy_total - ships - self_total if is_self else enemy_total - self_total

    feats = np.asarray([
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
        self_buckets[0] / 500.0,
        self_buckets[1] / 500.0,
        self_buckets[2] / 500.0,
        self_buckets[3] / 500.0,
        enemy_buckets[0] / 500.0,
        enemy_buckets[1] / 500.0,
        enemy_buckets[2] / 500.0,
        enemy_buckets[3] / 500.0,
        self_total / 500.0,
        enemy_total / 500.0,
        min(self_min_eta, 50.0) / 50.0 if self_min_eta > 0.0 else 0.0,
        min(enemy_min_eta, 50.0) / 50.0 if enemy_min_eta > 0.0 else 0.0,
        enemy_soonest_ships / 500.0,
        enemy_largest_ships / 500.0,
        min(enemy_largest_eta, 50.0) / 50.0 if enemy_largest_eta > 0.0 else 0.0,
        net_threat / 500.0,
    ], dtype=np.float32)
    return feats[:planet_feature_dim]


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


def _incoming_bucket_idx(eta: float) -> int:
    for i, upper in enumerate(INCOMING_ETA_BUCKETS):
        if eta <= upper:
            return i
    return len(INCOMING_ETA_BUCKETS) - 1


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
    planet_feature_dim: int = PLANET_FEATURE_DIM,
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
    all_fleets = parse_fleets(env_obs)
    if turn is None:
        turn = int(env_obs.get("step", 0) if isinstance(env_obs, dict) else getattr(env_obs, "step", 0) or 0)

    if len(planets) > p_max:
        # Defensive: keep this loud rather than silently truncating planets
        # (they're crucial for action masking).
        raise ValueError(
            f"Observation has {len(planets)} planets but p_max={p_max}. "
            f"Bump P_MAX_DEFAULT or pass a larger p_max."
        )
    # Predict fleet targets (needed for fleet features and threat counting).
    if predict_targets and all_fleets:
        preds = predict_all_fleet_targets(all_fleets, planets, angular_velocity)
    else:
        preds = {}

    my_ids = {p.id for p in planets if p.owner == player}
    if len(all_fleets) > f_max:
        def fleet_priority(f):
            pred = preds.get(f.id)
            target_id = pred.get("target_id") if pred else None
            eta = float(pred.get("eta", 999.0)) if pred else 999.0
            incoming_to_mine = target_id in my_ids
            mine = f.owner == player
            enemy = f.owner not in (player, -1)
            return (
                0 if incoming_to_mine and enemy else
                1 if incoming_to_mine and mine else
                2 if enemy else
                3 if mine else
                4,
                eta,
                -f.ships,
                f.id,
            )
        fleets = sorted(all_fleets, key=fleet_priority)[:f_max]
    else:
        fleets = all_fleets

    enemy_owners = sorted({f.owner for f in all_fleets if f.owner not in (player, -1)})
    enemy_owner_to_slot = {
        owner: i for i, owner in enumerate(enemy_owners[:MAX_ENEMY_INCOMING_SLOTS])
    }
    incoming_by_planet: dict[int, dict[str, Any]] = {
        p.id: {
            "self_buckets": [0.0] * len(INCOMING_ETA_BUCKETS),
            "enemy_buckets": [0.0] * len(INCOMING_ETA_BUCKETS),
            "self_total": 0.0,
            "enemy_total": 0.0,
            "self_min_eta": 0.0,
            "enemy_min_eta": 0.0,
            "enemy_soonest_ships": 0.0,
            "enemy_largest_ships": 0.0,
            "enemy_largest_eta": 0.0,
        }
        for p in planets
    }

    # Threat aggregation for global features.
    incoming_other = 0  # enemy ships en route to my planets
    incoming_self = 0   # friendly ships en route to my planets
    for f in all_fleets:
        pred = preds.get(f.id)
        if pred is None:
            continue
        target_id = pred["target_id"]
        eta = float(pred["eta"])
        planet_incoming = incoming_by_planet.get(target_id)
        if planet_incoming is not None:
            bucket = _incoming_bucket_idx(eta)
            if f.owner == player:
                planet_incoming["self_buckets"][bucket] += f.ships
                planet_incoming["self_total"] += f.ships
                if planet_incoming["self_min_eta"] == 0.0 or eta < planet_incoming["self_min_eta"]:
                    planet_incoming["self_min_eta"] = eta
            elif f.owner != -1:
                planet_incoming["enemy_buckets"][bucket] += f.ships
                planet_incoming["enemy_total"] += f.ships
                if planet_incoming["enemy_min_eta"] == 0.0 or eta < planet_incoming["enemy_min_eta"]:
                    planet_incoming["enemy_min_eta"] = eta
                    planet_incoming["enemy_soonest_ships"] = f.ships
                if f.ships > planet_incoming["enemy_largest_ships"]:
                    planet_incoming["enemy_largest_ships"] = f.ships
                    planet_incoming["enemy_largest_eta"] = eta
        if target_id not in my_ids:
            continue
        if f.owner == player:
            incoming_self += f.ships
        elif f.owner != -1:
            incoming_other += f.ships

    # ------- Planet tensor (padded) -------
    planet_feats = np.zeros((p_max, planet_feature_dim), dtype=np.float32)
    planet_mask = np.zeros(p_max, dtype=bool)
    source_mask = np.zeros(p_max, dtype=bool)
    planet_ids = np.full(p_max, -1, dtype=np.int32)
    planet_pos = np.zeros((p_max, 2), dtype=np.float32)  # for edge bias

    for i, p in enumerate(planets):
        planet_feats[i] = _planet_features(
            p, player, angular_velocity, incoming=incoming_by_planet.get(p.id),
            planet_feature_dim=planet_feature_dim,
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
        planets, all_fleets, player, angular_velocity, turn,
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


def decode_action_with_launch(
    obs: Observation,
    src_slot: int,
    tgt_slot: int,
    ship_bin: int,
    *,
    ship_bins: tuple = DEFAULT_SHIP_BINS,
    planets: Optional[list[Planet]] = None,
    angular_velocity: float = 0.0,
    no_op: bool = False,
) -> Optional[tuple[list, dict]]:
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
    return [src.id, launch["angle"], launch["ships"]], launch


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
    decoded = decode_action_with_launch(
        obs,
        src_slot,
        tgt_slot,
        ship_bin,
        ship_bins=ship_bins,
        planets=planets,
        angular_velocity=angular_velocity,
        no_op=no_op,
    )
    if decoded is None:
        return None
    return decoded[0]
