"""Observation builder for the final neural node policy.

The policy consumes a padded planet table.  It does not depend on the older
``reworked_*`` feature stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from feature_extraction import BOARD_SIZE, parse_fleets, parse_planets
from navigation import compute_launch, predict_all_fleet_targets


P_MAX = 44
NODE_FEATURE_DIM = 18
EDGE_FEATURE_DIM = 13
GLOBAL_FEATURE_DIM = 8


@dataclass(frozen=True)
class FinalObservation:
    node: np.ndarray          # (P_MAX, NODE_FEATURE_DIM)
    edge: np.ndarray          # (P_MAX, P_MAX, EDGE_FEATURE_DIM)
    global_features: np.ndarray  # (GLOBAL_FEATURE_DIM,)
    valid_mask: np.ndarray    # (P_MAX,)
    edge_mask: np.ndarray     # valid directed planet pairs
    source_mask: np.ndarray   # owned planets with ships
    target_mask: np.ndarray   # selectable targets (non-owned; +owned if enabled)
    planet_ids: np.ndarray    # slot -> planet id, -1 padded


def _owner_features(owner: int, player: int) -> tuple[float, float, float]:
    return (
        1.0 if int(owner) == int(player) else 0.0,
        1.0 if int(owner) >= 0 and int(owner) != int(player) else 0.0,
        1.0 if int(owner) == -1 else 0.0,
    )


def _env_omega(env_obs: Any) -> float:
    if isinstance(env_obs, dict):
        return float(env_obs.get("angular_velocity", 0.0) or 0.0)
    return float(getattr(env_obs, "angular_velocity", 0.0) or 0.0)


def build_final_observation(
    env_obs: Any,
    player: int,
    *,
    p_max: int = P_MAX,
    include_owned_targets: bool = False,
) -> FinalObservation:
    """Build the padded planet observation.

    ``include_owned_targets`` controls the ``target_mask`` semantics:

    * ``False`` (default, legacy): only non-owned planets are selectable as
      targets (attack / expansion).  Preserves the original behaviour.
    * ``True``: owned planets are also selectable, so the policy can issue
      reinforcement / defense / support launches toward its own worlds.
    """
    planets = parse_planets(env_obs)
    fleets = parse_fleets(env_obs)
    n = min(len(planets), int(p_max))
    omega = _env_omega(env_obs)

    node = np.zeros((p_max, NODE_FEATURE_DIM), dtype=np.float32)
    edge = np.zeros((p_max, p_max, EDGE_FEATURE_DIM), dtype=np.float32)
    valid = np.zeros(p_max, dtype=bool)
    edge_mask = np.zeros((p_max, p_max), dtype=bool)
    source = np.zeros(p_max, dtype=bool)
    target = np.zeros(p_max, dtype=bool)
    ids = np.full(p_max, -1, dtype=np.int64)

    incoming_self: dict[int, float] = {}
    incoming_enemy: dict[int, float] = {}
    self_fleet_ships = 0.0
    enemy_fleet_ships = 0.0
    predictions = predict_all_fleet_targets(fleets, planets, omega, t_max=180.0, step=1.0)
    for fl in fleets:
        if int(fl.owner) == int(player):
            self_fleet_ships += float(fl.ships)
        else:
            enemy_fleet_ships += float(fl.ships)
        pred = predictions.get(int(fl.id))
        if not pred:
            continue
        target_id = int(pred["target_id"])
        if int(fl.owner) == int(player):
            incoming_self[target_id] = incoming_self.get(target_id, 0.0) + float(fl.ships)
        else:
            incoming_enemy[target_id] = incoming_enemy.get(target_id, 0.0) + float(fl.ships)

    total_self_prod = sum(float(p.production) for p in planets[:n] if int(p.owner) == int(player))
    total_enemy_prod = sum(
        float(p.production)
        for p in planets[:n]
        if int(p.owner) >= 0 and int(p.owner) != int(player)
    )
    total_neutral_prod = sum(float(p.production) for p in planets[:n] if int(p.owner) == -1)
    total_self_ships = self_fleet_ships + sum(float(p.ships) for p in planets[:n] if int(p.owner) == int(player))
    total_enemy_ships = enemy_fleet_ships + sum(
        float(p.ships)
        for p in planets[:n]
        if int(p.owner) >= 0 and int(p.owner) != int(player)
    )

    for i, p in enumerate(planets[:n]):
        own, enemy, neutral = _owner_features(p.owner, player)
        valid[i] = True
        source[i] = bool(own and int(p.ships) > 0)
        target[i] = True if include_owned_targets else bool(not own)
        ids[i] = int(p.id)

        node[i] = np.asarray([
            own,
            enemy,
            neutral,
            float(p.ships) / 100.0,
            float(p.production) / 5.0,
            float(p.x) / BOARD_SIZE,
            float(p.y) / BOARD_SIZE,
            float(p.radius) / 10.0,
            float(p.r_sun) / 70.0,
            float(np.sin(p.theta)),
            float(np.cos(p.theta)),
            1.0 if p.is_rotating else 0.0,
            1.0 if p.is_comet else 0.0,
            float(incoming_self.get(int(p.id), 0.0)) / 100.0,
            float(incoming_enemy.get(int(p.id), 0.0)) / 100.0,
            float(total_self_prod) / 50.0,
            float(total_enemy_prod) / 50.0,
            float(total_neutral_prod) / 50.0,
        ], dtype=np.float32)

    for i, src in enumerate(planets[:n]):
        src_own, src_enemy, _src_neutral = _owner_features(src.owner, player)
        for j, dst in enumerate(planets[:n]):
            if i == j:
                continue
            dst_own, dst_enemy, dst_neutral = _owner_features(dst.owner, player)
            dx = float(dst.x) - float(src.x)
            dy = float(dst.y) - float(src.y)
            dist = float(np.hypot(dx, dy))
            launch = compute_launch(src, dst, 1, planets, omega)
            reachable = 1.0 if launch is not None else 0.0
            eta = float(launch["eta"]) if launch is not None else 120.0
            edge_mask[i, j] = True
            edge[i, j] = np.asarray([
                reachable,
                min(eta, 120.0) / 120.0,
                min(dist, BOARD_SIZE * 2.0) / (BOARD_SIZE * 2.0),
                dx / BOARD_SIZE,
                dy / BOARD_SIZE,
                src_own,
                src_enemy,
                dst_own,
                dst_enemy,
                dst_neutral,
                float(src.ships) / 100.0,
                float(dst.ships) / 100.0,
                float(dst.production) / 5.0,
            ], dtype=np.float32)

    total_prod = max(1.0, total_self_prod + total_enemy_prod + total_neutral_prod)
    total_ships = max(1.0, total_self_ships + total_enemy_ships)
    global_features = np.asarray([
        total_self_prod / 50.0,
        total_enemy_prod / 50.0,
        total_neutral_prod / 50.0,
        (total_self_prod - total_enemy_prod) / total_prod,
        total_self_ships / 500.0,
        total_enemy_ships / 500.0,
        (total_self_ships - total_enemy_ships) / total_ships,
        float(len(planets)) / float(p_max),
    ], dtype=np.float32)

    return FinalObservation(
        node=node,
        edge=edge,
        global_features=global_features,
        valid_mask=valid,
        edge_mask=edge_mask,
        source_mask=source,
        target_mask=target,
        planet_ids=ids,
    )
