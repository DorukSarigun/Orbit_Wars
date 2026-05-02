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
from dataclasses import dataclass, replace
from typing import Optional

import torch
from kaggle_environments import make

from feature_extraction import parse_planets, fleet_speed
from obs_builder import (
    build_observation,
    decode_action,
    INCOMING_ETA_BUCKETS,
    PF_IDX_SHIPS,
    PF_IDX_INCOMING_SELF_B0,
    PF_IDX_INCOMING_ENEMY_B0,
)
from model import OrbitWarsPolicy, pack_observations

from .buffer import RolloutBuffer, Transition
from .rewards import RewardConfig, compute_step_reward, normalize_terminal_rewards
from .fleet_tracker import FleetTracker, compute_fleet_reward, compute_production_simple_fleet_combat_reward
from .defense_tracker import DefenseTracker, compute_defense_reward
from .planet_loss_tracker import PlanetLossTracker, compute_planet_loss_launch_penalty
from .production_reward_tracker import (
    ProductionRewardTracker,
    compute_production_capture_reward,
    compute_production_planet_loss_penalty,
)


def _incoming_bucket_idx(eta: float) -> int:
    for i, upper in enumerate(INCOMING_ETA_BUCKETS):
        if eta <= upper:
            return i
    return len(INCOMING_ETA_BUCKETS) - 1


def _compute_airborne_cost(*, ships_sent: int, est_distance: float, weight: float,
                           speed: Optional[float] = None) -> float:
    speed = fleet_speed(ships_sent) if speed is None else speed
    eta = est_distance / max(speed, 1e-6)
    return -float(weight) * int(ships_sent) * eta


def _planet_under_enemy_attack(obs, slot: int) -> bool:
    return bool(
        obs.planet_feats[
            slot,
            PF_IDX_INCOMING_ENEMY_B0:PF_IDX_INCOMING_ENEMY_B0 + len(INCOMING_ETA_BUCKETS),
        ].sum() > 0.0
    )


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


def _act_for_player(
    policy: OrbitWarsPolicy,
    env_obs: dict,
    player: int,
    *,
    device: torch.device,
    current_turn: int = 0,
) -> tuple:
    angular_velocity = env_obs.get("angular_velocity", 0.0) or 0.0
    obs = build_observation(env_obs, player=player, angular_velocity=angular_velocity,
                            p_max=policy.p_max, f_max=policy.f_max,
                            planet_feature_dim=policy.planet_dim)
    parsed = parse_planets(env_obs)
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
        last_out = out

        tgt_slot = int(action["tgt"][0])
        ship_bin = int(action["ship_bin"][0])
        is_noop = bool(action["is_noop"][0])
        move = decode_action(
            obs_decision, src_slot, tgt_slot, ship_bin,
            planets=parsed, angular_velocity=angular_velocity, no_op=is_noop,
        )
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
            distance = 0.0
            if src_planet and tgt_planet:
                distance = math.hypot(src_planet.x - tgt_planet.x, src_planet.y - tgt_planet.y)
            eta = distance / max(fleet_speed(int(sent_ships)), 1e-6)
            bucket = _incoming_bucket_idx(eta)
            obs_work.planet_feats[tgt_slot, PF_IDX_INCOMING_SELF_B0 + bucket] += sent_ships / 500.0
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
                "defense_miss_candidates": [],
            }
            for threatened_slot, planet_id in enumerate(obs_decision.planet_ids):
                if planet_id < 0 or threatened_slot >= len(obs_decision.planet_mask):
                    continue
                if not obs_decision.planet_mask[threatened_slot]:
                    continue
                threatened_planet = next((p for p in parsed if p.id == int(planet_id)), None)
                if threatened_planet is None or threatened_planet.owner != player:
                    continue
                if not _planet_under_enemy_attack(obs_decision, threatened_slot):
                    continue
                if int(planet_id) != src_planet_id and tgt_planet_id == int(planet_id):
                    continue
                defense_distance = 0.0
                if src_planet:
                    defense_distance = math.hypot(src_planet.x - threatened_planet.x, src_planet.y - threatened_planet.y)
                defense_eta = defense_distance / max(fleet_speed(int(sent_ships)), 1e-6)
                fleet_meta["defense_miss_candidates"].append({
                    "threatened_planet_id": int(planet_id),
                    "est_defense_arrival_turn": current_turn + defense_eta,
                })
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
) -> EpisodeStats:
    """Play one self-play game and append all transitions (both players)
    into `buffer`. Returns episode-level statistics.
    """
    if num_agents not in (2, 4):
        raise ValueError("Orbit Wars supports num_agents=2 or num_agents=4")
    device = device or torch.device("cpu")
    reward_cfg = reward_cfg or RewardConfig()

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
        ProductionRewardTracker(max_turns=reward_cfg.max_turns, num_agents=num_agents)
        for _ in range(num_agents)
    ]
    planet_loss_trackers = [
        PlanetLossTracker(horizon=reward_cfg.planet_loss_credit_horizon)
        for _ in range(num_agents)
    ]
    defense_trackers = [
        DefenseTracker(
            horizon=reward_cfg.defense_hold_horizon,
            arrival_window=reward_cfg.defense_arrival_window,
        )
        for _ in range(num_agents)
    ]

    for t in range(max_steps):
        env_obs = [state[p]["observation"] for p in range(num_agents)]
        for p in range(num_agents):
            defense_trackers[p].prime(env_obs[p], p)

        # Both players act on the same underlying state but with their
        # own perspective (different feature normalization in obs_builder).
        acted = [
            _act_for_player(policy, env_obs[p], p, device=device, current_turn=t)
            for p in range(num_agents)
        ]
        env_actions = []
        invalid_actions = [0 for _ in range(num_agents)]
        for p, (_, em, _, launches, invalids) in enumerate(acted):
            env_actions.append(em)
            n_launches[p] += launches
            invalid_actions[p] = invalids

        new_state = env.step(env_actions)
        new_obs = [new_state[p]["observation"] for p in range(num_agents)]

        # Observe fleet arrivals and assign delayed rewards.
        for p in range(num_agents):
            fleet_trackers[p].observe_turn(t, new_obs[p], p)
            for buffer_idx, outcome in fleet_trackers[p].pop_resolved(p):
                if reward_cfg.reward_system == "production_simple":
                    fleet_reward = compute_production_simple_fleet_combat_reward(outcome)
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
        terminal_rewards = normalize_terminal_rewards([float(s.reward or 0.0) for s in new_state]) if terminal else [0.0 for _ in range(num_agents)]
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
            cum_reward[p] += r - reward_cfg.invalid_action_penalty * invalid_actions[p]
            last_value[p] = float(acted[p][2].value[0].item())

        for p, (decisions, _, _, _, _) in enumerate(acted):
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
                    ship_bin=int(act["ship_bin"][0]),
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
                    airborne_cost = _compute_airborne_cost(
                        ships_sent=fleet_meta["ships_sent"],
                        est_distance=fleet_meta["distance"],
                        weight=reward_cfg.fleet_airborne_weight,
                    )
                    buffer.add_reward(idx, airborne_cost)
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
                    )
                    if reward_cfg.reward_system == "production_simple":
                        production_trackers[p].register_capture_launch(
                            buffer_idx=idx,
                            launch_turn=fleet_meta["launch_turn"],
                            player=fleet_meta["player"],
                            target_planet_id=fleet_meta["tgt_planet_id"],
                            ships_sent=fleet_meta["ships_sent"],
                            est_arrival_turn=fleet_meta["launch_turn"] + fleet_meta["eta"],
                            launch_target_owner=fleet_meta["launch_tgt_owner"],
                        )
                        for candidate in fleet_meta["defense_miss_candidates"]:
                            production_trackers[p].register_defense_miss_candidate(
                                buffer_idx=idx,
                                launch_turn=fleet_meta["launch_turn"],
                                player=fleet_meta["player"],
                                threatened_planet_id=candidate["threatened_planet_id"],
                                ships_sent=fleet_meta["ships_sent"],
                                est_defense_arrival_turn=candidate["est_defense_arrival_turn"],
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
                production_trackers[p].observe_turn(t, env_obs[p], new_obs[p], p)
                for buffer_idx, outcome in production_trackers[p].pop_capture_resolved(p):
                    buffer.add_delayed_reward(
                        buffer_idx,
                        compute_production_capture_reward(outcome),
                        event_turn=t,
                        discount=reward_cfg.delayed_reward_discount,
                    )
                for buffer_idx, outcome in production_trackers[p].pop_loss_resolved(p):
                    buffer.add_delayed_reward(
                        buffer_idx,
                        compute_production_planet_loss_penalty(outcome),
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
