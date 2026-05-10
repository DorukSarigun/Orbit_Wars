"""Self-play rollout collector.

For each environment turn we:
  1. Build observations for player 0 and player 1 from the SAME env state.
  2. Run policy(.act()) for both observations.
  3. Decode each action to env format ([src_id, angle, ships]) and step.
  4. Compute rewards from the player's perspective.
  5. Push two transitions into the rollout buffer (one per side).

This is the simplest form of self-play: a single live policy plays both
sides. Both perspectives generate gradients in the next PPO update,
which is fine for a symmetric zero-sum game.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import torch
from kaggle_environments import make

from feature_extraction import parse_planets, parse_fleets, fleet_speed, Planet, Fleet
from navigation import predict_all_fleet_targets
from obs_builder import (
    build_observation,
    decode_action_with_launch,
    INCOMING_ETA_BUCKETS,
    PF_IDX_SHIPS,
    PF_IDX_INCOMING_SELF_B0,
    PF_IDX_INCOMING_ENEMY_B0,
    SHIP_REQUEST_CAP,
)
from model import OrbitWarsPolicy, pack_observations, _make_ship_dist

from .buffer import RolloutBuffer, Transition
from .rewards import RewardConfig, compute_step_reward, normalize_terminal_rewards
from .fleet_tracker import FleetTracker, compute_fleet_reward, compute_production_simple_fleet_combat_reward
from .defense_tracker import DefenseTracker, compute_defense_reward
from .planet_loss_tracker import PlanetLossTracker, compute_planet_loss_launch_penalty
from .planet_defense_tracker import PlanetDefenseTracker
from .defense_kills_tracker import DefenseKillsTracker
from .production_reward_tracker import (
    ProductionRewardTracker,
    compute_production_capture_reward,
    compute_production_neutral_gift_penalty,
    compute_production_reinforcement_reward,
    compute_production_capture_clawback,
)
from scripted_strategy import CaptureEfficiencyAgent, ScriptedConfig

MAX_LAUNCHES_PER_PLAYER_TURN = 8
MAX_PAIR_CANDIDATES = 64


def _incoming_bucket_idx(eta: float) -> int:
    for i, upper in enumerate(INCOMING_ETA_BUCKETS):
        if eta <= upper:
            return i
    return len(INCOMING_ETA_BUCKETS) - 1


def _compute_airborne_cost(*, ships_sent: int, est_distance: float, weight: float,
                           speed: Optional[float] = None,
                           risk_multiplier: float = 1.0) -> float:
    """Time-cost penalty for keeping ships airborne.

    `risk_multiplier` scales the cost up for launches from threatened
    planets (and especially for launches that leave a threatened planet
    near-defenseless). A safe launch uses 1.0; a risky one can grow up
    to ~5x.
    """
    speed = fleet_speed(ships_sent) if speed is None else speed
    eta = est_distance / max(speed, 1e-6)
    base = float(weight) * int(ships_sent) * eta
    return -base * float(max(risk_multiplier, 0.0))


def _planet_under_enemy_attack(obs, slot: int) -> bool:
    return bool(
        obs.planet_feats[
            slot,
            PF_IDX_INCOMING_ENEMY_B0:PF_IDX_INCOMING_ENEMY_B0 + len(INCOMING_ETA_BUCKETS),
        ].sum() > 0.0
    )


def _enemy_incoming_ships(obs, slot: int) -> float:
    return float(
        obs.planet_feats[
            slot,
            PF_IDX_INCOMING_ENEMY_B0:PF_IDX_INCOMING_ENEMY_B0 + len(INCOMING_ETA_BUCKETS),
        ].sum()
    ) * 100.0


def _source_threat_intensity(obs, slot: int, ships_remaining_after_launch: int) -> float:
    """Approximate threat-to-remaining ratio for a source planet.

    Enemy incoming ETA buckets store ships/100.0 (see rollout's
    self-bucket update), so multiplying the bucket sum by 100 gives an
    approximate raw enemy ship count en route to this planet. Dividing
    by `ships_remaining_after_launch` produces a threat intensity in
    units of "incoming enemy ships per defender". 0 means safe.
    """
    raw_enemy = _enemy_incoming_ships(obs, slot)
    if raw_enemy <= 0.0:
        return 0.0
    denom = float(max(int(ships_remaining_after_launch), 1))
    return raw_enemy / denom


def _compute_source_defense_deficit_penalty(*, enemy_incoming: float,
                                            ships_remaining_after_launch: int,
                                            weight: float = 1.0) -> float:
    deficit = max(0.0, float(enemy_incoming) - float(max(0, int(ships_remaining_after_launch))))
    return -float(weight) * deficit


def _compute_source_defense_shortfall(
    *,
    ships_remaining_after_launch: int,
    production: float,
    enemy_arrivals: list[tuple[float, float]],
    self_arrivals: Optional[list[tuple[float, float]]] = None,
) -> float:
    """Simulate garrison evolution at a source planet and return the
    total ship shortfall summed across all overwhelming enemy arrivals.

    Per-fleet simulation (no ETA bucketing):
        garrison0   = ships_remaining_after_launch
        At each fleet arrival (sorted by ETA):
            dt            = eta - prev_eta
            garrison     += production * dt      (new ships produced)
            garrison     += self_arrival_ships    (friendlies, if any)
            shortfall_i   = max(0, enemy_ships - garrison)
            total        += shortfall_i
            garrison      = max(0, garrison - enemy_ships)

    Returns the SUM of per-arrival shortfalls (in raw ship units).  Zero
    means the planet survives every incoming enemy fleet given its
    current garrison + production; positive means at least one arrival
    overwhelms the garrison.

    This replaces the older "sum all enemy buckets irrespective of ETA"
    heuristic: distant threats no longer trigger a penalty when the
    planet has enough time to produce a defense.

    Args:
        ships_remaining_after_launch: garrison after subtracting the
            just-launched fleet's ships.
        production: planet production per turn (>= 0).
        enemy_arrivals: list of (eta, ships) for enemy fleets predicted
            to hit this planet. ETAs may be float; order is irrelevant
            (we sort).
        self_arrivals: optional list of (eta, ships) for friendly fleets
            arriving at this planet (reinforcements).
    """
    if not enemy_arrivals:
        return 0.0
    events: list[tuple[float, float, bool]] = [
        (float(eta), float(ships), False) for eta, ships in enemy_arrivals
    ]
    if self_arrivals:
        events.extend(
            (float(eta), float(ships), True) for eta, ships in self_arrivals
        )
    events.sort(key=lambda ev: ev[0])

    garrison = float(max(0, int(ships_remaining_after_launch)))
    prev_eta = 0.0
    total_shortfall = 0.0
    for eta, ships, is_self in events:
        dt = max(0.0, eta - prev_eta)
        garrison += float(production) * dt
        if is_self:
            garrison += ships
        else:
            if ships > garrison:
                total_shortfall += (ships - garrison)
            garrison = max(0.0, garrison - ships)
        prev_eta = eta
    return total_shortfall


def _enemy_self_arrivals_at_planet(
    *,
    planet_id: int,
    player: int,
    fleets: list[Fleet],
    fleet_preds: dict[int, Optional[dict]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Partition predicted arrivals at `planet_id` into enemy and self
    lists of (eta, ships).

    `fleet_preds` is the output of `predict_all_fleet_targets`, keyed
    by fleet id. Fleets with no prediction (leaving board / hitting sun
    / beyond t_max) are skipped.
    """
    enemy: list[tuple[float, float]] = []
    mine: list[tuple[float, float]] = []
    for f in fleets:
        pred = fleet_preds.get(f.id)
        if pred is None:
            continue
        if int(pred["target_id"]) != int(planet_id):
            continue
        eta = float(pred["eta"])
        ships = float(f.ships)
        if f.owner == player:
            mine.append((eta, ships))
        elif f.owner != -1:
            enemy.append((eta, ships))
    return enemy, mine


def _airborne_risk_multiplier(threat_intensity: float, *,
                              base_threat_mult: float = 2.0,
                              cap: float = 3.0) -> float:
    """Map threat_intensity to a risk multiplier in [1, base+cap].

    threat_intensity = 0   -> 1.0   (safe)
    threat_intensity = 1   -> ~3.0  (incoming ~= remaining)
    threat_intensity >= 3  -> 5.0   (overrun risk)
    """
    if threat_intensity <= 0.0:
        return 1.0
    return 1.0 + base_threat_mult * min(float(threat_intensity), float(cap))


@dataclass
class EpisodeStats:
    length: int
    rewards: list[float]
    winner: int
    n_launches: list[int]

    @property
    def reward_p0(self) -> float:
        return self.rewards[0] if len(self.rewards) > 0 else 0.0

    @property
    def reward_p1(self) -> float:
        return self.rewards[1] if len(self.rewards) > 1 else 0.0

    @property
    def n_launches_p0(self) -> int:
        return self.n_launches[0] if len(self.n_launches) > 0 else 0

    @property
    def n_launches_p1(self) -> int:
        return self.n_launches[1] if len(self.n_launches) > 1 else 0


@dataclass
class ScriptedExpansionConfig:
    turns: int = 0
    nearest_k: int = 5


def _scripted_neutral_expansion_action(
    policy: OrbitWarsPolicy,
    out,
    obs,
    src_slot: int,
    planets: list,
    player: int,
    *,
    device: torch.device,
    nearest_k: int = 5,
) -> Optional[dict]:
    src_id = int(obs.planet_ids[src_slot])
    src = next((p for p in planets if p.id == src_id), None)
    if src is None or src.owner != player or src.ships <= 1:
        return None
    slot_by_planet_id = {
        int(pid): slot
        for slot, pid in enumerate(obs.planet_ids)
        if int(pid) >= 0
    }
    candidates = [
        p for p in planets
        if p.owner == -1 and not p.is_comet and p.id != src_id and p.id in slot_by_planet_id
    ]
    if not candidates:
        return None
    nearest = sorted(
        candidates,
        key=lambda p: (math.hypot(src.x - p.x, src.y - p.y), p.id),
    )[:max(1, int(nearest_k))]
    scored = []
    for tgt in nearest:
        # Scripted heuristic: must send strictly more than tgt.ships to
        # capture, must not exceed src.ships (otherwise no valid launch),
        # and must fit within SHIP_REQUEST_CAP (the continuous head's
        # absolute scale).
        needed_ships = int(tgt.ships) + 1
        if needed_ships > int(src.ships) or needed_ships > SHIP_REQUEST_CAP:
            continue
        distance = math.hypot(src.x - tgt.x, src.y - tgt.y)
        score = float(tgt.production) / max(1.0, float(tgt.ships))
        # Convert needed ship count to the [0, 1] fraction the Beta head emits.
        ship_value = float(needed_ships) / float(SHIP_REQUEST_CAP)
        scored.append((score, -distance, int(tgt.production), -int(tgt.ships), tgt.id, ship_value))
    if not scored:
        return None
    _, _, _, _, target_id, ship_value = max(scored)
    src_tensor = torch.tensor([src_slot], device=device, dtype=torch.long)
    tgt_tensor = torch.tensor([slot_by_planet_id[int(target_id)]], device=device, dtype=torch.long)
    ship_tensor = torch.tensor([ship_value], device=device, dtype=torch.float32)
    noop_tensor = torch.tensor([False], device=device, dtype=torch.bool)
    with torch.no_grad():
        ev = policy.evaluate_source_actions(out, src_tensor, tgt_tensor, ship_tensor, noop_tensor)
    return {
        "src": src_tensor,
        "tgt": tgt_tensor,
        "ship_value": ship_tensor,
        "is_noop": noop_tensor,
        "log_prob": ev["log_prob"],
    }


def _act_for_player(
    policy: OrbitWarsPolicy,
    env_obs: dict,
    player: int,
    *,
    device: torch.device,
    current_turn: int = 0,
    max_turns: int = 500,
    scripted_expansion: Optional[ScriptedExpansionConfig] = None,
) -> tuple:
    angular_velocity = env_obs.get("angular_velocity", 0.0) or 0.0
    obs = build_observation(env_obs, player=player, angular_velocity=angular_velocity,
                            p_max=policy.p_max, f_max=policy.f_max,
                            planet_feature_dim=policy.planet_dim,
                            max_turns=max_turns)
    parsed = parse_planets(env_obs)
    parsed_fleets = parse_fleets(env_obs)
    # Cache per-fleet ETA / target predictions once per decision step.
    # Used below to compute exact defense-shortfall per source planet
    # (replaces ETA-bucket aggregation that lumped distant threats in
    # with imminent ones).
    fleet_preds = (
        predict_all_fleet_targets(parsed_fleets, parsed, angular_velocity)
        if parsed_fleets else {}
    )
    planet_prod_by_id = {int(p.id): float(p.production) for p in parsed}
    my_total_ships = sum(int(p.ships) for p in parsed if p.owner == player and not p.is_comet)
    my_total_ships += sum(int(f.ships) for f in parsed_fleets if f.owner == player)
    enemy_total_ships = sum(
        int(p.ships)
        for p in parsed
        if p.owner not in (player, -1) and not p.is_comet
    )
    enemy_total_ships += sum(int(f.ships) for f in parsed_fleets if f.owner not in (player, -1))
    env_action = []
    decisions = []
    launch_count = 0
    invalid_count = 0
    obs_work = replace(
        obs,
        planet_feats=obs.planet_feats.copy(),
        source_mask=obs.source_mask.copy(),
    )
    source_slots = [int(i) for i, ok in enumerate(obs.source_mask) if ok]
    last_out = None

    if getattr(policy, "model_variant", "") == "grouped_planet":
        available_pairs = obs.planet_mask[None, :].repeat(policy.p_max, axis=0)
        available_pairs &= obs.source_mask[:, None]
        available_pairs &= ~torch.eye(policy.p_max, dtype=torch.bool).numpy()
        while launch_count < MAX_LAUNCHES_PER_PLAYER_TURN and any(bool(ok) for ok in obs_work.source_mask):
            obs_decision = replace(
                obs_work,
                planet_feats=obs_work.planet_feats.copy(),
                source_mask=obs_work.source_mask.copy(),
            )
            batch = pack_observations([obs_decision], device=device)
            selected = None
            with torch.no_grad():
                out = policy(**batch)
                if scripted_expansion is not None and current_turn < scripted_expansion.turns:
                    scripted_action = None
                    for src_slot in [int(i) for i, ok in enumerate(obs_decision.source_mask) if ok]:
                        scripted_action = _scripted_neutral_expansion_action(
                            policy,
                            out,
                            obs_decision,
                            src_slot,
                            parsed,
                            player,
                            device=device,
                            nearest_k=scripted_expansion.nearest_k,
                        )
                        if scripted_action is not None:
                            break
                    if scripted_action is not None:
                        action = scripted_action
                        src_slot = int(action["src"][0])
                        tgt_slot = int(action["tgt"][0])
                        ship_value = float(action["ship_value"][0])
                        is_noop = bool(action["is_noop"][0])
                        decoded = decode_action_with_launch(
                            obs_decision, src_slot, tgt_slot, ship_value,
                            planets=parsed, angular_velocity=angular_velocity, no_op=is_noop,
                        )
                        if decoded is not None:
                            selected = (action, decoded)
                        ranked = []
                    else:
                        pair_logits = out.tgt_logits[0].clone()
                        pair_logits[~torch.as_tensor(available_pairs, device=pair_logits.device)] = torch.finfo(pair_logits.dtype).min
                        ranked = torch.topk(
                            pair_logits.reshape(-1),
                            k=min(MAX_PAIR_CANDIDATES, pair_logits.numel()),
                        ).indices.tolist()
                else:
                    pair_logits = out.tgt_logits[0].clone()
                    pair_logits[~torch.as_tensor(available_pairs, device=pair_logits.device)] = torch.finfo(pair_logits.dtype).min
                    ranked = torch.topk(
                        pair_logits.reshape(-1),
                        k=min(MAX_PAIR_CANDIDATES, pair_logits.numel()),
                    ).indices.tolist()
            last_out = out
            if ranked:
                for pair_idx in ranked:
                    src_slot = int(pair_idx // policy.p_max)
                    tgt_slot = int(pair_idx % policy.p_max)
                    if not bool(obs_decision.source_mask[src_slot]):
                        continue
                    if not bool(obs_decision.planet_mask[tgt_slot]):
                        continue
                    if src_slot == tgt_slot:
                        continue
                    if not bool(available_pairs[src_slot, tgt_slot]):
                        continue
                    # Sample a single ship_value from the Beta head for
                    # this (src, tgt) pair. If decoding fails (invalid
                    # trajectory), fall through to the next pair.
                    ship_params_pair = out.ship_params[0, src_slot, tgt_slot]
                    ship_dist_pair = _make_ship_dist(ship_params_pair)
                    with torch.no_grad():
                        ship_value_sample = float(ship_dist_pair.sample().item())
                    decoded = decode_action_with_launch(
                        obs_decision, src_slot, tgt_slot, ship_value_sample,
                        planets=parsed, angular_velocity=angular_velocity, no_op=False,
                    )
                    if decoded is None:
                        continue
                    src_tensor = torch.tensor([src_slot], device=device, dtype=torch.long)
                    tgt_tensor = torch.tensor([tgt_slot], device=device, dtype=torch.long)
                    ship_tensor = torch.tensor([ship_value_sample], device=device, dtype=torch.float32)
                    noop_tensor = torch.tensor([False], device=device, dtype=torch.bool)
                    with torch.no_grad():
                        ev = policy.evaluate_source_actions(out, src_tensor, tgt_tensor, ship_tensor, noop_tensor)
                    action = {
                        "src": src_tensor,
                        "tgt": tgt_tensor,
                        "ship_value": ship_tensor,
                        "is_noop": noop_tensor,
                        "log_prob": ev["log_prob"],
                    }
                    selected = (action, decoded)
                    break
            if selected is None:
                with torch.no_grad():
                    for src_slot in [int(i) for i, ok in enumerate(obs_decision.source_mask) if ok]:
                        src_tensor = torch.tensor([src_slot], device=device, dtype=torch.long)
                        tgt_tensor = torch.tensor([0], device=device, dtype=torch.long)
                        # NO-OP rows: store a degenerate ship_value (0.5);
                        # the log_prob path masks ship contributions for noops.
                        ship_tensor = torch.tensor([0.5], device=device, dtype=torch.float32)
                        noop_tensor = torch.tensor([True], device=device, dtype=torch.bool)
                        ev = policy.evaluate_source_actions(out, src_tensor, tgt_tensor, ship_tensor, noop_tensor)
                        decisions.append((obs_decision, {
                            "src": src_tensor,
                            "tgt": tgt_tensor,
                            "ship_value": ship_tensor,
                            "is_noop": noop_tensor,
                            "log_prob": ev["log_prob"],
                        }, out, False, False, None))
                        obs_work.source_mask[src_slot] = False
                break

            action, decoded = selected
            src_slot = int(action["src"][0])
            tgt_slot = int(action["tgt"][0])
            move = decoded[0]
            launch_info = decoded[1]
            env_action.append(move)
            launch_count += 1
            sent_ships = float(move[2])
            obs_work.planet_feats[src_slot, PF_IDX_SHIPS] = max(
                0.0, obs_work.planet_feats[src_slot, PF_IDX_SHIPS] - sent_ships / 100.0
            )
            src_planet_id = int(obs_decision.planet_ids[src_slot])
            tgt_planet_id = int(obs_decision.planet_ids[tgt_slot])
            src_planet = next((p for p in parsed if p.id == src_planet_id), None)
            tgt_planet = next((p for p in parsed if p.id == tgt_planet_id), None)
            speed = fleet_speed(int(sent_ships))
            eta = float(launch_info["eta"]) if launch_info is not None else 0.0
            distance = eta * max(speed, 1e-6) if eta > 0.0 else 0.0
            if eta <= 0.0 and src_planet and tgt_planet:
                distance = math.hypot(src_planet.x - tgt_planet.x, src_planet.y - tgt_planet.y)
                eta = distance / max(speed, 1e-6)
            bucket = _incoming_bucket_idx(eta)
            obs_work.planet_feats[tgt_slot, PF_IDX_INCOMING_SELF_B0 + bucket] += sent_ships / 100.0
            src_ships_pre = int(round(float(obs_decision.planet_feats[src_slot, PF_IDX_SHIPS]) * 100.0))
            src_ships_remaining = max(0, src_ships_pre - int(sent_ships))
            if src_planet is not None:
                src_planet.ships = src_ships_remaining
            source_enemy_incoming = _enemy_incoming_ships(obs_decision, src_slot)
            source_threat = _source_threat_intensity(obs_decision, src_slot, src_ships_remaining)
            enemy_arrivals, self_arrivals = _enemy_self_arrivals_at_planet(
                planet_id=src_planet_id, player=player,
                fleets=parsed_fleets, fleet_preds=fleet_preds,
            )
            source_defense_shortfall = _compute_source_defense_shortfall(
                ships_remaining_after_launch=src_ships_remaining,
                production=planet_prod_by_id.get(src_planet_id, 0.0),
                enemy_arrivals=enemy_arrivals,
                self_arrivals=self_arrivals,
            )
            fleet_meta = {
                "launch_turn": current_turn,
                "player": player,
                "src_planet_id": src_planet_id,
                "tgt_planet_id": tgt_planet_id,
                "ships_sent": int(sent_ships),
                "angle": move[1],
                "distance": distance,
                "eta": eta,
                "launch_tgt_owner": int(tgt_planet.owner) if tgt_planet else -2,
                "launch_tgt_ships": int(tgt_planet.ships) if tgt_planet else 0,
                "source_had_enemy_incoming": _planet_under_enemy_attack(obs_decision, src_slot),
                "source_enemy_incoming": float(source_enemy_incoming),
                "source_defense_shortfall": float(source_defense_shortfall),
                "source_ships_remaining_after_launch": int(src_ships_remaining),
                "source_threat_intensity": float(source_threat),
                "my_total_ships_at_launch": int(my_total_ships),
                "enemy_total_ships_at_launch": int(enemy_total_ships),
                # Continuous Beta head: how many ships the policy *requested*
                # beyond the source's available garrison (>=0). Drives the
                # overflow penalty in the reward step.
                "overflow_ships": float(launch_info.get("overflow_ships", 0.0)) if launch_info is not None else 0.0,
                "raw_request": float(launch_info.get("raw_request", float(sent_ships))) if launch_info is not None else float(sent_ships),
            }
            obs_work.source_mask[src_slot] = obs_work.planet_feats[src_slot, PF_IDX_SHIPS] > 0.01
            available_pairs[src_slot, tgt_slot] = False
            if not bool(obs_work.source_mask[src_slot]):
                available_pairs[src_slot, :] = False
            decisions.append((obs_decision, action, out, True, False, fleet_meta))
        if last_out is None:
            batch = pack_observations([obs], device=device)
            with torch.no_grad():
                last_out = policy(**batch)
        return decisions, env_action, last_out, launch_count, invalid_count

    for src_slot in source_slots:
        obs_decision = replace(
            obs_work,
            planet_feats=obs_work.planet_feats.copy(),
            source_mask=obs_work.source_mask.copy(),
        )
        batch = pack_observations([obs_decision], device=device)
        with torch.no_grad():
            out = policy(**batch)
            action = policy.act_source(
                out,
                torch.tensor([src_slot], device=device),
                deterministic=False,
            )
            if scripted_expansion is not None and current_turn < scripted_expansion.turns:
                scripted_action = _scripted_neutral_expansion_action(
                    policy,
                    out,
                    obs_decision,
                    src_slot,
                    parsed,
                    player,
                    device=device,
                    nearest_k=scripted_expansion.nearest_k,
                )
                if scripted_action is not None:
                    action = scripted_action
        last_out = out

        tgt_slot = int(action["tgt"][0])
        ship_value = float(action["ship_value"][0])
        is_noop = bool(action["is_noop"][0])
        decoded = decode_action_with_launch(
            obs_decision, src_slot, tgt_slot, ship_value,
            planets=parsed, angular_velocity=angular_velocity, no_op=is_noop,
        )
        move = None if decoded is None else decoded[0]
        launch_info = None if decoded is None else decoded[1]
        if move is not None:
            env_action.append(move)
            launch_count += 1
            sent_ships = float(move[2])
            obs_work.planet_feats[src_slot, PF_IDX_SHIPS] = max(
                0.0, obs_work.planet_feats[src_slot, PF_IDX_SHIPS] - sent_ships / 100.0
            )
            src_planet_id = int(obs_decision.planet_ids[src_slot])
            tgt_planet_id = int(obs_decision.planet_ids[tgt_slot])
            src_planet = next((p for p in parsed if p.id == src_planet_id), None)
            tgt_planet = next((p for p in parsed if p.id == tgt_planet_id), None)
            speed = fleet_speed(int(sent_ships))
            distance = 0.0
            eta = float(launch_info["eta"]) if launch_info is not None else 0.0
            if eta > 0.0:
                distance = eta * max(speed, 1e-6)
            elif src_planet and tgt_planet:
                distance = math.hypot(src_planet.x - tgt_planet.x, src_planet.y - tgt_planet.y)
                eta = distance / max(speed, 1e-6)
            bucket = _incoming_bucket_idx(eta)
            obs_work.planet_feats[tgt_slot, PF_IDX_INCOMING_SELF_B0 + bucket] += sent_ships / 100.0
            # Source-planet threat assessment (used for risk-aware airborne).
            src_ships_pre = int(round(float(obs_decision.planet_feats[src_slot, PF_IDX_SHIPS]) * 100.0))
            src_ships_remaining = max(0, src_ships_pre - int(sent_ships))
            source_enemy_incoming = _enemy_incoming_ships(obs_decision, src_slot)
            source_threat = _source_threat_intensity(obs_decision, src_slot, src_ships_remaining)
            enemy_arrivals, self_arrivals = _enemy_self_arrivals_at_planet(
                planet_id=src_planet_id, player=player,
                fleets=parsed_fleets, fleet_preds=fleet_preds,
            )
            source_defense_shortfall = _compute_source_defense_shortfall(
                ships_remaining_after_launch=src_ships_remaining,
                production=planet_prod_by_id.get(src_planet_id, 0.0),
                enemy_arrivals=enemy_arrivals,
                self_arrivals=self_arrivals,
            )
            # Prepare fleet tracking metadata for later reward assignment.
            fleet_meta = {
                "launch_turn": current_turn,
                "player": player,
                "src_planet_id": src_planet_id,
                "tgt_planet_id": tgt_planet_id,
                "ships_sent": int(sent_ships),
                "angle": move[1],
                "distance": distance,
                "eta": eta,
                "launch_tgt_owner": int(tgt_planet.owner) if tgt_planet else -2,
                "launch_tgt_ships": int(tgt_planet.ships) if tgt_planet else 0,
                "source_had_enemy_incoming": _planet_under_enemy_attack(obs_decision, src_slot),
                "source_enemy_incoming": float(source_enemy_incoming),
                "source_defense_shortfall": float(source_defense_shortfall),
                "source_ships_remaining_after_launch": int(src_ships_remaining),
                "source_threat_intensity": float(source_threat),
                "my_total_ships_at_launch": int(my_total_ships),
                "enemy_total_ships_at_launch": int(enemy_total_ships),
                "overflow_ships": float(launch_info.get("overflow_ships", 0.0)) if launch_info is not None else 0.0,
                "raw_request": float(launch_info.get("raw_request", float(sent_ships))) if launch_info is not None else float(sent_ships),
            }
        else:
            fleet_meta = None
            if not is_noop:
                invalid_count += 1
        obs_work.source_mask[src_slot] = False
        decisions.append((obs_decision, action, out, move is not None, (move is None and not is_noop), fleet_meta))

    if last_out is None:
        batch = pack_observations([obs], device=device)
        with torch.no_grad():
            last_out = policy(**batch)

    return decisions, env_action, last_out, launch_count, invalid_count


def collect_episode(
    policy: OrbitWarsPolicy,
    buffer: RolloutBuffer,
    *,
    max_steps: int = 500,
    num_agents: int = 2,
    reward_cfg: Optional[RewardConfig] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    progress_cb=None,
    scripted_expansion: Optional[ScriptedExpansionConfig] = None,
    opponent_indices: Optional[set[int]] = None,
    scripted_opponent_cfg: Optional[ScriptedConfig] = None,
) -> EpisodeStats:
    """Play one game and append PPO-side transitions into ``buffer``.

    ``opponent_indices`` lists player slots controlled by the scripted
    ``CaptureEfficiencyAgent`` (one shared instance per call).  Those
    players act on the same env but no transitions, fleet/defense
    trackers, or reward bookkeeping are applied for them: the buffer
    only sees PPO-controlled players, keeping gradients clean.

    Pure self-play behavior is preserved when ``opponent_indices`` is
    None or empty.
    """
    if num_agents not in (2, 4):
        raise ValueError("Orbit Wars supports num_agents=2 or num_agents=4")
    device = device or torch.device("cpu")
    reward_cfg = reward_cfg or RewardConfig()
    opponent_indices = set(opponent_indices) if opponent_indices else set()
    scripted_agent: Optional[CaptureEfficiencyAgent] = (
        CaptureEfficiencyAgent(scripted_opponent_cfg) if opponent_indices else None
    )

    env = make("orbit_wars", debug=False)
    if seed is not None:
        env.configuration["randomSeed"] = int(seed)
    state = env.reset(num_agents=num_agents)
    # The env initializes planets only on the first step.
    state = env.step([[] for _ in range(num_agents)])

    n_launches = [0 for _ in range(num_agents)]
    cum_reward = [0.0 for _ in range(num_agents)]
    last_value = [0.0 for _ in range(num_agents)]      # for GAE bootstrapping at end-of-rollout
    fleet_trackers = [FleetTracker() for _ in range(num_agents)]  # One per player
    production_trackers = [
        ProductionRewardTracker(
            max_turns=reward_cfg.max_turns,
            num_agents=num_agents,
            reinforcement_decay_window=reward_cfg.production_reinforcement_decay_window,
            capture_clawback_window=reward_cfg.production_capture_clawback_window,
            capture_clawback_fraction=reward_cfg.production_capture_clawback_fraction,
        )
        for _ in range(num_agents)
    ]
    planet_loss_trackers = [
        PlanetLossTracker(horizon=reward_cfg.planet_loss_credit_horizon)
        for _ in range(num_agents)
    ]
    # Defense Window tracker (Tasarım 2). Active only in production_simple
    # since the legacy reward systems use the narrower PlanetLossTracker.
    planet_defense_trackers = [
        PlanetDefenseTracker(
            horizon=reward_cfg.planet_defense_horizon,
            depart_share=reward_cfg.planet_defense_depart_share,
            hold_share=reward_cfg.planet_defense_hold_share,
            max_turns=reward_cfg.max_turns,
            num_opponents=max(1, num_agents - 1),
            capture_time_curvature=reward_cfg.capture_time_curvature,
        )
        for _ in range(num_agents)
    ]
    defense_trackers = [
        DefenseTracker(
            horizon=reward_cfg.defense_hold_horizon,
            arrival_window=reward_cfg.defense_arrival_window,
        )
        for _ in range(num_agents)
    ]
    # Defense Kills tracker: rewards launches that defended my planets and
    # contributed to killing enemy ships (threat-aware replacement for the
    # legacy production_reinforcement bonus).
    initial_obs0 = state[0]["observation"]
    angular_velocity = float(initial_obs0.get("angular_velocity", 0.0) or 0.0)
    defense_kills_trackers = [
        DefenseKillsTracker(
            max_turns=reward_cfg.max_turns,
            angular_velocity=angular_velocity,
            production_reward_beta=reward_cfg.production_reward_beta,
            production_reward_reference=reward_cfg.production_reward_reference,
            capture_time_curvature=reward_cfg.capture_time_curvature,
        )
        for _ in range(num_agents)
    ]
    # Per-player rolling window of recent transition indices, used by the
    # `production_diff` reward system to retro-distribute the per-step
    # production-differential reward via eligibility traces. Each entry
    # is the list of transition buffer indices written for that turn.
    # `maxlen = horizon + 1` so we keep the current step plus H past steps.
    _trace_horizon = max(0, int(reward_cfg.eligibility_trace_horizon))
    trace_windows: list[deque] = [
        deque(maxlen=_trace_horizon + 1) for _ in range(num_agents)
    ]

    for t in range(max_steps):
        env_obs = [state[p]["observation"] for p in range(num_agents)]
        for p in range(num_agents):
            if p in opponent_indices:
                continue
            defense_trackers[p].prime(env_obs[p], p)

        # PPO players run the policy; opponent slots run the scripted bot
        # and do NOT contribute transitions or trackers.
        acted: list = []
        for p in range(num_agents):
            if p in opponent_indices:
                scripted_actions = scripted_agent.act(
                    env_obs[p], player=p,
                    current_turn=t, max_turns=reward_cfg.max_turns,
                )
                acted.append(("scripted", scripted_actions, len(scripted_actions)))
            else:
                acted.append(("ppo", _act_for_player(
                    policy, env_obs[p], p, device=device, current_turn=t,
                    max_turns=reward_cfg.max_turns,
                    scripted_expansion=scripted_expansion,
                )))
        env_actions = []
        invalid_actions = [0 for _ in range(num_agents)]
        for p, item in enumerate(acted):
            kind = item[0]
            if kind == "scripted":
                _, em, launches = item
                env_actions.append(em)
                n_launches[p] += launches
            else:
                _, payload = item
                _, em, _, launches, invalids = payload
                env_actions.append(em)
                n_launches[p] += launches
                invalid_actions[p] = invalids

        new_state = env.step(env_actions)
        new_obs = [new_state[p]["observation"] for p in range(num_agents)]

        # Observe fleet arrivals and assign delayed rewards.
        for p in range(num_agents):
            if p in opponent_indices:
                continue
            fleet_trackers[p].observe_turn(t, new_obs[p], p)
            for buffer_idx, outcome in fleet_trackers[p].pop_resolved(p):
                if reward_cfg.reward_system == "production_simple":
                    fleet_reward = compute_production_simple_fleet_combat_reward(
                        outcome,
                        kill_weight=reward_cfg.fleet_kill_weight,
                        reward_scale=reward_cfg.reward_scale,
                    )
                else:
                    fleet_reward = compute_fleet_reward(
                        outcome,
                        capture_weight=reward_cfg.fleet_capture_weight,
                        kill_weight=reward_cfg.fleet_kill_weight,
                        waste_weight=reward_cfg.fleet_waste_weight,
                        enemy_capture_radius_weight=reward_cfg.enemy_capture_radius_weight,
                        max_turns=reward_cfg.max_turns,
                        time_decay_floor=reward_cfg.time_decay_floor,
                    )
                buffer.add_delayed_reward(
                    buffer_idx,
                    fleet_reward,
                    event_turn=t,
                    discount=reward_cfg.delayed_reward_discount,
                )

        terminal = any(s.status == "DONE" for s in new_state) or env.done or (t == max_steps - 1)
        terminal_rewards = (
            normalize_terminal_rewards(
                [float(s.reward or 0.0) for s in new_state],
                draw_value=reward_cfg.draw_terminal_reward,
            )
            if terminal else [0.0 for _ in range(num_agents)]
        )
        if reward_cfg.reward_system == "production_diff":
            # Pure production-differential per-step reward. NO shaping, NO
            # tracker channels. Terminal step uses the +/-1 env signal.
            rewards = []
            opp_count = max(1, num_agents - 1)
            for p in range(num_agents):
                if terminal:
                    rewards.append(float(terminal_rewards[p]))
                    continue
                planets = parse_planets(new_obs[p])
                my_prod = sum(
                    float(planet.production) for planet in planets
                    if planet.owner == p and not getattr(planet, "is_comet", False)
                )
                opp_prod = sum(
                    float(planet.production) for planet in planets
                    if planet.owner != p and planet.owner != -1
                    and not getattr(planet, "is_comet", False)
                )
                rewards.append(
                    (my_prod - opp_prod / opp_count) * reward_cfg.reward_scale
                )
        else:
            rewards = [
                compute_step_reward(
                    env_obs[p], new_obs[p], p,
                    terminal=terminal,
                    env_terminal_reward=terminal_rewards[p],
                    cfg=reward_cfg,
                    invalid_actions=0,
                )
                for p in range(num_agents)
            ]
        for p, r in enumerate(rewards):
            if p in opponent_indices:
                continue
            cum_reward[p] += r - reward_cfg.invalid_action_penalty * invalid_actions[p]
            # acted[p] = ("ppo", (decisions, em, last_out, launches, invalids))
            last_value[p] = float(acted[p][1][2].value[0].item())

        for p, item in enumerate(acted):
            if p in opponent_indices:
                continue
            decisions, _, _, _, _ = item[1]
            r = rewards[p]
            new_indices = []
            for obs, act, out, launched, invalid, fleet_meta in decisions:
                reward = (
                    -reward_cfg.invalid_action_penalty
                    if invalid else 0.0
                )
                idx = buffer.add(Transition(
                    player=p,
                    planet_feats=obs.planet_feats,
                    fleet_feats=obs.fleet_feats,
                    global_feats=obs.global_feats,
                    edge_bias=obs.edge_bias,
                    planet_mask=obs.planet_mask,
                    fleet_mask=obs.fleet_mask,
                    source_mask=obs.source_mask,
                    src=int(act["src"][0]),
                    tgt=int(act["tgt"][0]),
                    ship_value=float(act["ship_value"][0]),
                    is_noop=bool(act["is_noop"][0]),
                    log_prob=float(act["log_prob"][0].item()),
                    value=float(out.value[0].item()),
                    reward=reward,
                    done=terminal,
                    turn=t,
                ))
                new_indices.append(idx)
                # Register fleet launch for delayed reward tracking.
                if launched and fleet_meta is not None:
                    defense_trackers[p].cancel_planet(fleet_meta["src_planet_id"])
                    risk_mult = _airborne_risk_multiplier(
                        fleet_meta["source_threat_intensity"],
                    )
                    airborne_cost = _compute_airborne_cost(
                        ships_sent=fleet_meta["ships_sent"],
                        est_distance=fleet_meta["distance"],
                        weight=reward_cfg.fleet_airborne_weight,
                        risk_multiplier=risk_mult,
                    ) * reward_cfg.reward_scale
                    buffer.add_reward(idx, airborne_cost)
                    # Continuous Beta ship head: penalize over-allocation
                    # whenever ship_value * SHIP_REQUEST_CAP exceeded the
                    # source garrison. Always wired (not just for
                    # production_simple), so the head learns to respect
                    # the available ship constraint regardless of the
                    # active reward system.
                    overflow_ships = float(fleet_meta.get("overflow_ships", 0.0))
                    if overflow_ships > 0.0:
                        overflow_penalty = (
                            -float(reward_cfg.ship_overflow_weight)
                            * overflow_ships
                            * reward_cfg.reward_scale
                        )
                        buffer.add_reward(idx, overflow_penalty)
                    if reward_cfg.reward_system == "production_simple":
                        # Use the ETA-exact shortfall: penalty reflects the
                        # net ship deficit across all actually-incoming enemy
                        # arrivals, after accounting for production + friendly
                        # reinforcements. A distant threat that production can
                        # cover contributes 0 shortfall.
                        defense_deficit_penalty = -float(reward_cfg.planet_loss_weight) * float(
                            fleet_meta["source_defense_shortfall"]
                        ) * reward_cfg.reward_scale
                        buffer.add_reward(idx, defense_deficit_penalty)
                    fleet_trackers[p].register_launch(
                        buffer_idx=idx,
                        launch_turn=fleet_meta["launch_turn"],
                        player=fleet_meta["player"],
                        src_planet_id=fleet_meta["src_planet_id"],
                        tgt_planet_id=fleet_meta["tgt_planet_id"],
                        ships_sent=fleet_meta["ships_sent"],
                        angle=fleet_meta["angle"],
                        est_distance=fleet_meta["distance"],
                        launch_tgt_owner=fleet_meta["launch_tgt_owner"],
                        launch_tgt_ships=fleet_meta["launch_tgt_ships"],
                        my_total_ships_at_launch=fleet_meta["my_total_ships_at_launch"],
                        enemy_total_ships_at_launch=fleet_meta["enemy_total_ships_at_launch"],
                    )
                    if reward_cfg.reward_system == "production_simple":
                        # Defense Window: this launch took ships AWAY from
                        # src_planet_id; if that planet is later lost, this
                        # transition shares part of the loss penalty.
                        planet_defense_trackers[p].register_depart(
                            buffer_idx=idx,
                            player=fleet_meta["player"],
                            src_planet_id=fleet_meta["src_planet_id"],
                            turn=fleet_meta["launch_turn"],
                            ships_sent=fleet_meta["ships_sent"],
                        )
                        production_trackers[p].register_capture_launch(
                            buffer_idx=idx,
                            launch_turn=fleet_meta["launch_turn"],
                            player=fleet_meta["player"],
                            target_planet_id=fleet_meta["tgt_planet_id"],
                            ships_sent=fleet_meta["ships_sent"],
                            est_arrival_turn=fleet_meta["launch_turn"] + fleet_meta["eta"],
                            launch_target_owner=fleet_meta["launch_tgt_owner"],
                            launch_target_ships=fleet_meta["launch_tgt_ships"],
                            my_total_ships_at_launch=fleet_meta["my_total_ships_at_launch"],
                        )
                        # Defense Kills: if I'm targeting one of my own
                        # planets that is currently under attack, register
                        # this launch as a defender. Reward is paid out
                        # later when the combat resolves.
                        if (
                            int(fleet_meta["launch_tgt_owner"]) == int(fleet_meta["player"])
                            and defense_kills_trackers[p].is_under_attack(
                                fleet_meta["player"], fleet_meta["tgt_planet_id"]
                            )
                        ):
                            defense_kills_trackers[p].register_defender_launch(
                                buffer_idx=idx,
                                player=fleet_meta["player"],
                                target_planet_id=fleet_meta["tgt_planet_id"],
                                launch_turn=fleet_meta["launch_turn"],
                                ships_sent=fleet_meta["ships_sent"],
                                est_arrival_turn=int(round(
                                    fleet_meta["launch_turn"] + fleet_meta["eta"]
                                )),
                            )
                    else:
                        planet_loss_trackers[p].register_launch(
                            buffer_idx=idx,
                            launch_turn=fleet_meta["launch_turn"],
                            player=fleet_meta["player"],
                            planet_id=fleet_meta["src_planet_id"],
                            ships_sent=fleet_meta["ships_sent"],
                            source_had_enemy_incoming=fleet_meta["source_had_enemy_incoming"],
                        )
                elif bool(act["is_noop"][0]) and not invalid:
                    src_slot = int(act["src"][0])
                    planet_id = int(obs.planet_ids[src_slot])
                    ships_held = int(round(float(obs.planet_feats[src_slot, PF_IDX_SHIPS]) * 100.0))
                    defense_trackers[p].register_hold(
                        buffer_idx=idx,
                        player=p,
                        planet_id=planet_id,
                        hold_turn=t,
                        ships_held=ships_held,
                    )
            if reward_cfg.reward_system == "production_diff":
                # Eligibility-trace distribution. Push current step's
                # transitions to the rolling window, then add the current
                # per-step reward `r` to ALL transitions in the window
                # with an exponential weight that depends on direction:
                #   forward (default off):  decay^k        -> newer gets max
                #   reverse (default ON):   decay^(L-1-k)  -> older gets max
                # where k=0 is the current turn, L is current window length.
                # Reverse is appropriate for delayed-effect domains:
                # the production-diff at turn t is caused by launches
                # made many turns ago whose fleets just arrived — those
                # OLD actions deserve the bulk of the credit.
                if new_indices:
                    trace_windows[p].append(list(new_indices))
                decay = float(reward_cfg.eligibility_trace_decay)
                noop_w = float(reward_cfg.noop_trace_weight)
                reverse = bool(reward_cfg.eligibility_trace_reverse)
                window_len = len(trace_windows[p])
                # Iterate window from newest (k=0) to oldest (k=L-1).
                for k, past_idxs in enumerate(reversed(trace_windows[p])):
                    if not past_idxs:
                        continue
                    if reverse:
                        # Older = larger weight: decay^(L-1-k)
                        exponent = (window_len - 1) - k
                    else:
                        # Newer = larger weight: decay^k
                        exponent = k
                    if decay > 0.0:
                        decay_k = decay ** exponent
                    else:
                        decay_k = 1.0 if exponent == 0 else 0.0
                    if decay_k <= 0.0:
                        # Forward: oldest is decay^L ~ 0; can break.
                        # Reverse: only fires for non-current with decay=0;
                        # in that case skip but don't break (current may
                        # have non-zero weight in forward only).
                        if not reverse:
                            break
                        continue
                    weights = [
                        (noop_w if buffer.data[idx].is_noop else 1.0)
                        for idx in past_idxs
                    ]
                    total_w = sum(weights)
                    if total_w <= 0.0:
                        continue
                    for idx, w in zip(past_idxs, weights):
                        share = r * decay_k * (w / total_w)
                        if share != 0.0:
                            buffer.add_reward(idx, share)
                # Skip all legacy tracker observe/pop blocks below; the
                # production_diff system uses NO trackers.
                continue
            if new_indices:
                reward = r / len(new_indices)
                for idx in new_indices:
                    buffer.add_reward(idx, reward)
            defense_trackers[p].observe_turn(t, new_obs[p], p)
            for buffer_idx, outcome in defense_trackers[p].pop_resolved():
                defense_reward = compute_defense_reward(
                    outcome,
                    hold_weight=reward_cfg.defense_hold_weight,
                )
                buffer.add_delayed_reward(
                    buffer_idx,
                    defense_reward,
                    event_turn=t,
                    discount=reward_cfg.delayed_reward_discount,
                )
            if reward_cfg.reward_system == "production_simple":
                # --- Defense Window (Tasarım 2) ---
                # Register passive HOLDs for every still-owned planet at
                # end of turn, attributed to this turn's last transition
                # (the agent's "decision pulse"). HOLDs for the lost
                # planets are skipped here; their previous-turn HOLDs are
                # already pending in the tracker and will be resolved by
                # observe_turn below.
                if new_indices:
                    pulse_idx = new_indices[-1]
                    for planet in parse_planets(new_obs[p]):
                        if planet.owner != p or getattr(planet, "is_comet", False):
                            continue
                        planet_defense_trackers[p].register_hold(
                            buffer_idx=pulse_idx,
                            player=p,
                            planet_id=int(planet.id),
                            turn=t,
                            ships_held=int(planet.ships),
                        )
                planet_defense_trackers[p].observe_turn(
                    t, env_obs[p], new_obs[p], p,
                    reward_scale=reward_cfg.reward_scale,
                    loss_weight=reward_cfg.planet_loss_weight,
                )
                for buffer_idx, signed_reward in planet_defense_trackers[p].pop_resolved(p):
                    buffer.add_delayed_reward(
                        buffer_idx,
                        signed_reward,
                        event_turn=t,
                        discount=reward_cfg.delayed_reward_discount,
                    )
                # Defense Kills: tag/untag under-attack episodes, detect
                # enemy fleet arrivals, and pay out kills credit to
                # eligible defenders. Must run AFTER register_defender_launch
                # for this turn (which already happened above) so that
                # defenders are registered before episode finalization can
                # complete in the same step (rare same-turn close).
                defense_kills_trackers[p].observe_turn(
                    t, env_obs[p], new_obs[p], p,
                    defense_kills_weight=reward_cfg.defense_kills_weight,
                    reward_scale=reward_cfg.reward_scale,
                )
                for buffer_idx, signed_reward, event_turn in defense_kills_trackers[p].pop_resolved(p):
                    buffer.add_delayed_reward(
                        buffer_idx,
                        signed_reward,
                        event_turn=event_turn,
                        discount=reward_cfg.delayed_reward_discount,
                    )
                production_trackers[p].observe_turn(t, env_obs[p], new_obs[p], p)
                for buffer_idx, outcome in production_trackers[p].pop_capture_resolved(p):
                    buffer.add_delayed_reward(
                        buffer_idx,
                        compute_production_capture_reward(
                            outcome,
                            reward_scale=reward_cfg.reward_scale,
                            production_reward_beta=reward_cfg.production_reward_beta,
                            production_reward_reference=reward_cfg.production_reward_reference,
                            capture_time_curvature=reward_cfg.capture_time_curvature,
                        ),
                        event_turn=t,
                        discount=reward_cfg.delayed_reward_discount,
                    )
                for buffer_idx, outcome in production_trackers[p].pop_neutral_gift_resolved(p):
                    buffer.add_delayed_reward(
                        buffer_idx,
                        compute_production_neutral_gift_penalty(
                            outcome,
                            reward_scale=reward_cfg.reward_scale,
                            production_reward_beta=reward_cfg.production_reward_beta,
                            production_reward_reference=reward_cfg.production_reward_reference,
                        ),
                        event_turn=t,
                        discount=reward_cfg.delayed_reward_discount,
                    )
                # Sorun 1: Post-capture reinforcement bonus.
                for buffer_idx, outcome in production_trackers[p].pop_reinforcement_resolved(p):
                    buffer.add_delayed_reward(
                        buffer_idx,
                        compute_production_reinforcement_reward(
                            outcome,
                            reinforcement_weight=reward_cfg.production_reinforcement_weight,
                            reward_scale=reward_cfg.reward_scale,
                            decay_window=reward_cfg.production_reinforcement_decay_window,
                            production_reward_beta=reward_cfg.production_reward_beta,
                            production_reward_reference=reward_cfg.production_reward_reference,
                        ),
                        event_turn=t,
                        discount=reward_cfg.delayed_reward_discount,
                    )
                # Sorun 2: Capture-then-lose clawback (negative reward).
                for buffer_idx, outcome in production_trackers[p].pop_capture_clawback_resolved(p):
                    buffer.add_delayed_reward(
                        buffer_idx,
                        compute_production_capture_clawback(
                            outcome,
                            reward_scale=reward_cfg.reward_scale,
                            production_reward_beta=reward_cfg.production_reward_beta,
                            production_reward_reference=reward_cfg.production_reward_reference,
                        ),
                        event_turn=t,
                        discount=reward_cfg.delayed_reward_discount,
                    )
            else:
                planet_loss_trackers[p].observe_turn(t, env_obs[p], new_obs[p], p)
                for buffer_idx, outcome in planet_loss_trackers[p].pop_resolved(p):
                    planet_loss_penalty = compute_planet_loss_launch_penalty(
                        outcome,
                        loss_weight=reward_cfg.planet_loss_weight,
                    )
                    buffer.add_delayed_reward(
                        buffer_idx,
                        planet_loss_penalty,
                        event_turn=t,
                        discount=reward_cfg.delayed_reward_discount,
                    )

        state = new_state
        if progress_cb is not None:
            progress_cb(t, state)
        if terminal:
            break

    # Determine winner from terminal env reward (env emits +1 to winner).
    final_r = [int(s.reward or 0) for s in state]
    best_reward = max(final_r) if final_r else 0
    winners = [p for p, r in enumerate(final_r) if r == best_reward]
    winner = winners[0] if len(winners) == 1 and best_reward > 0 else -1

    return EpisodeStats(
        length=t + 1,
        rewards=cum_reward,
        winner=winner,
        n_launches=n_launches,
    )
