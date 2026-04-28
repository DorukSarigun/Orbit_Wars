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

from dataclasses import dataclass
from typing import Optional

import torch
from kaggle_environments import make

from feature_extraction import parse_planets
from obs_builder import build_observation, decode_action
from model import OrbitWarsPolicy, pack_observations

from .buffer import RolloutBuffer, Transition
from .rewards import RewardConfig, compute_step_reward


@dataclass
class EpisodeStats:
    length: int
    reward_p0: float
    reward_p1: float
    winner: int  # 0, 1, or -1 for draw / no-winner
    n_launches_p0: int
    n_launches_p1: int


def _act_for_player(
    policy: OrbitWarsPolicy,
    env_obs: dict,
    player: int,
    *,
    device: torch.device,
) -> tuple:
    """Run policy on a single env-observation; return (obs_dataclass,
    action_dict, env_action_list, output) so the caller can store the
    transition and step the env."""
    angular_velocity = env_obs.get("angular_velocity", 0.0) or 0.0
    obs = build_observation(env_obs, player=player, angular_velocity=angular_velocity,
                            p_max=policy.p_max, f_max=policy.f_max)
    batch = pack_observations([obs], device=device)
    with torch.no_grad():
        out = policy(**batch)
        action = policy.act(out, deterministic=False)

    src_slot = int(action["src"][0])
    tgt_slot = int(action["tgt"][0])
    ship_bin = int(action["ship_bin"][0])
    is_noop = bool(action["is_noop"][0])

    parsed = parse_planets(env_obs)
    move = decode_action(
        obs, src_slot, tgt_slot, ship_bin,
        planets=parsed, angular_velocity=angular_velocity, no_op=is_noop,
    )
    env_action = [move] if move is not None else []

    return obs, action, env_action, out, is_noop


def collect_episode(
    policy: OrbitWarsPolicy,
    buffer: RolloutBuffer,
    *,
    max_steps: int = 500,
    reward_cfg: Optional[RewardConfig] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    progress_cb=None,
) -> EpisodeStats:
    """Play one self-play game and append all transitions (both players)
    into `buffer`. Returns episode-level statistics.
    """
    device = device or torch.device("cpu")
    reward_cfg = reward_cfg or RewardConfig()

    env = make("orbit_wars", debug=False)
    if seed is not None:
        env.configuration["randomSeed"] = int(seed)
    state = env.reset(num_agents=2)
    # The env initializes planets only on the first step.
    state = env.step([[], []])

    n_launches = [0, 0]
    cum_reward = [0.0, 0.0]
    last_value = [0.0, 0.0]      # for GAE bootstrapping at end-of-rollout

    for t in range(max_steps):
        env_obs0 = state[0]["observation"]
        env_obs1 = state[1]["observation"]

        # Both players act on the same underlying state but with their
        # own perspective (different feature normalization in obs_builder).
        obs0, act0, em0, out0, noop0 = _act_for_player(policy, env_obs0, 0, device=device)
        obs1, act1, em1, out1, noop1 = _act_for_player(policy, env_obs1, 1, device=device)
        if not noop0:
            n_launches[0] += 1
        if not noop1:
            n_launches[1] += 1

        new_state = env.step([em0, em1])
        new_obs0 = new_state[0]["observation"]
        new_obs1 = new_state[1]["observation"]

        terminal = (new_state[0].status == "DONE") or env.done or (t == max_steps - 1)
        r0 = compute_step_reward(env_obs0, new_obs0, 0,
                                 terminal=terminal,
                                 env_terminal_reward=new_state[0].reward or 0,
                                 cfg=reward_cfg)
        r1 = compute_step_reward(env_obs1, new_obs1, 1,
                                 terminal=terminal,
                                 env_terminal_reward=new_state[1].reward or 0,
                                 cfg=reward_cfg)
        cum_reward[0] += r0
        cum_reward[1] += r1

        last_value[0] = float(out0.value[0].item())
        last_value[1] = float(out1.value[0].item())

        # Store both perspectives.
        for obs, act, out, r, p in [
            (obs0, act0, out0, r0, 0),
            (obs1, act1, out1, r1, 1),
        ]:
            buffer.add(Transition(
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
                reward=r,
                done=terminal,
            ))

        state = new_state
        if progress_cb is not None:
            progress_cb(t, state)
        if terminal:
            break

    # Determine winner from terminal env reward (env emits +1 to winner).
    final_r = [int(s.reward or 0) for s in state]
    if final_r[0] > final_r[1]:
        winner = 0
    elif final_r[1] > final_r[0]:
        winner = 1
    else:
        winner = -1

    return EpisodeStats(
        length=t + 1,
        reward_p0=cum_reward[0],
        reward_p1=cum_reward[1],
        winner=winner,
        n_launches_p0=n_launches[0],
        n_launches_p1=n_launches[1],
    )
