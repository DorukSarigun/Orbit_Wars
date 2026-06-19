"""Rollout collection and PPO update for the final neural policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from feature_extraction import parse_planets
from final_obs import P_MAX, FinalObservation, build_final_observation
from final_model import FinalNodePolicy
from final_micro_planner import MicroPlannerConfig, SourceOption, plan_target_attack
from navigation import compute_launch
from orbit_graph import build_orbit_graph
from orbit_match import production_share_margin


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 1.0


@dataclass
class RolloutBuffer:
    node: list[np.ndarray] = field(default_factory=list)
    edge: list[np.ndarray] = field(default_factory=list)
    global_features: list[np.ndarray] = field(default_factory=list)
    valid_mask: list[np.ndarray] = field(default_factory=list)
    edge_mask: list[np.ndarray] = field(default_factory=list)
    source_mask: list[np.ndarray] = field(default_factory=list)
    target_mask: list[np.ndarray] = field(default_factory=list)
    target_idx: list[int] = field(default_factory=list)
    source_idx: list[int] = field(default_factory=list)
    amount: list[float] = field(default_factory=list)
    logprob: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.reward)

    def add(self, obs: FinalObservation, action, reward: float, done: bool) -> None:
        self.node.append(obs.node)
        self.edge.append(obs.edge)
        self.global_features.append(obs.global_features)
        self.valid_mask.append(obs.valid_mask)
        self.edge_mask.append(obs.edge_mask)
        self.source_mask.append(obs.source_mask)
        self.target_mask.append(obs.target_mask)
        self.target_idx.append(int(action.target_idx.item()))
        self.source_idx.append(int(action.source_idx.item()))
        self.amount.append(float(action.amount.item()))
        self.logprob.append(float(action.logprob.item()))
        self.value.append(float(action.value.item()))
        self.reward.append(float(reward))
        self.done.append(bool(done))

    def compute_gae(self, *, gamma: float, lam: float) -> None:
        adv = 0.0
        advantages = [0.0 for _ in self.reward]
        returns = [0.0 for _ in self.reward]
        values = self.value + [0.0]
        for t in reversed(range(len(self.reward))):
            nonterminal = 0.0 if self.done[t] else 1.0
            delta = self.reward[t] + gamma * values[t + 1] * nonterminal - values[t]
            adv = delta + gamma * lam * nonterminal * adv
            advantages[t] = adv
            returns[t] = adv + values[t]
        self.advantages = advantages
        self.returns = returns

    def tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        adv = torch.tensor(self.advantages, dtype=torch.float32, device=device)
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
        return {
            "node": torch.tensor(np.asarray(self.node), dtype=torch.float32, device=device),
            "edge": torch.tensor(np.asarray(self.edge), dtype=torch.float32, device=device),
            "global": torch.tensor(np.asarray(self.global_features), dtype=torch.float32, device=device),
            "valid": torch.tensor(np.asarray(self.valid_mask), dtype=torch.bool, device=device),
            "edge_mask": torch.tensor(np.asarray(self.edge_mask), dtype=torch.bool, device=device),
            "source": torch.tensor(np.asarray(self.source_mask), dtype=torch.bool, device=device),
            "target": torch.tensor(np.asarray(self.target_mask), dtype=torch.bool, device=device),
            "target_idx": torch.tensor(self.target_idx, dtype=torch.long, device=device),
            "source_idx": torch.tensor(self.source_idx, dtype=torch.long, device=device),
            "amount": torch.tensor(self.amount, dtype=torch.float32, device=device),
            "old_logprob": torch.tensor(self.logprob, dtype=torch.float32, device=device),
            "returns": torch.tensor(self.returns, dtype=torch.float32, device=device),
            "advantages": adv,
        }


def _obs_to_tensors(obs: FinalObservation, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "node": torch.tensor(obs.node[None, ...], dtype=torch.float32, device=device),
        "edge": torch.tensor(obs.edge[None, ...], dtype=torch.float32, device=device),
        "global_features": torch.tensor(obs.global_features[None, ...], dtype=torch.float32, device=device),
        "valid_mask": torch.tensor(obs.valid_mask[None, ...], dtype=torch.bool, device=device),
        "edge_mask": torch.tensor(obs.edge_mask[None, ...], dtype=torch.bool, device=device),
        "source_mask": torch.tensor(obs.source_mask[None, ...], dtype=torch.bool, device=device),
        "target_mask": torch.tensor(obs.target_mask[None, ...], dtype=torch.bool, device=device),
    }


def _action_to_launch(env_obs: Any, obs: FinalObservation, action, player: int, *, current_turn: int, max_turns: int) -> tuple[list[list], str]:
    source_slot = int(action.source_idx.item())
    target_slot = int(action.target_idx.item())
    if source_slot < 0 or target_slot < 0:
        return [], "invalid_slot"
    source_id = int(obs.planet_ids[source_slot])
    target_id = int(obs.planet_ids[target_slot])
    if source_id < 0 or target_id < 0 or source_id == target_id:
        return [], "invalid_slot"
    planets = parse_planets(env_obs)
    by_id = {int(p.id): p for p in planets}
    source = by_id.get(source_id)
    target = by_id.get(target_id)
    if source is None or target is None or int(source.owner) != int(player):
        return [], "invalid_owner"
    ships = max(1, int(round(float(action.amount.item()) * float(source.ships))))
    ships = min(int(source.ships), ships)
    if ships <= 0:
        return [], "invalid_amount"
    omega = (
        float(env_obs.get("angular_velocity", 0.0) or 0.0)
        if isinstance(env_obs, dict)
        else float(getattr(env_obs, "angular_velocity", 0.0) or 0.0)
    )
    launch = compute_launch(source, target, ships, planets, omega)
    if launch is None:
        return [], "invalid_geometry"

    # Prefer the final micro-planner for non-owned targets.  The policy chooses
    # a target/source intent; the planner verifies capture-and-hold and may add
    # a larger amount from the chosen source.  If it cannot find a stable plan,
    # do not emit a weak raw attack; those undercommitted launches mostly add
    # reward noise during PPO.
    if int(target.owner) != int(player):
        try:
            og = build_orbit_graph(env_obs, player=int(player))
            plan = plan_target_attack(
                og=og,
                target=target,
                sources=[SourceOption(source=source, budget=int(source.ships))],
                player=int(player),
                current_turn=int(current_turn),
                max_turns=int(max_turns),
                omega=omega,
                cfg=MicroPlannerConfig(
                    hold_horizon=24.0,
                    beam_width=8,
                    max_launches_per_target=1,
                    top_m_sources=1,
                    min_send=1,
                    capture_margin=2,
                ),
            )
            if plan is not None:
                return plan.actions, "micro_capture"
            return [], "micro_reject_target"
        except Exception:
            return [], "micro_error_target"

    return [[int(source.id), float(launch["angle"]), int(launch["ships"])]], "raw_launch"


def collect_episode(
    policy: FinalNodePolicy,
    buffer: RolloutBuffer,
    *,
    device: torch.device,
    seed: int,
    max_steps: int = 300,
    num_agents: int = 2,
    shaping_weight: float = 0.05,
    terminal_weight: float = 1.0,
) -> dict[str, float]:
    from kaggle_environments import make
    import random as _random

    env = make("orbit_wars", debug=False)
    env.configuration["randomSeed"] = int(seed)
    episode_steps = int(getattr(env.configuration, "episodeSteps", 500) or 500)
    planner_max_turns = max(int(max_steps), episode_steps)
    _random.seed(int(seed))
    env.reset(num_agents=num_agents)
    state = env.step([[] for _ in range(num_agents)])
    prev_margin = [
        production_share_margin(state[p]["observation"], p, num_agents)
        for p in range(num_agents)
    ]
    total_rewards = [0.0 for _ in range(num_agents)]
    action_reasons: dict[str, int] = {}

    for t in range(int(max_steps)):
        pending = []
        env_actions: list[list] = []
        for p in range(num_agents):
            obs_raw = state[p]["observation"]
            obs = build_final_observation(obs_raw, p)
            tensors = _obs_to_tensors(obs, device)
            with torch.no_grad():
                action = policy.act(**tensors)
            launch_action, reason = _action_to_launch(
                obs_raw, obs, action, p, current_turn=t, max_turns=planner_max_turns)
            action_reasons[reason] = action_reasons.get(reason, 0) + 1
            env_actions.append(launch_action)
            pending.append((p, obs, action))

        state = env.step(env_actions)
        done = bool(env.done)
        for p, obs, action in pending:
            new_margin = production_share_margin(state[p]["observation"], p, num_agents)
            reward = float(shaping_weight) * (new_margin - prev_margin[p])
            prev_margin[p] = new_margin
            if done:
                env_reward = float(getattr(state[p], "reward", 0.0) or 0.0)
                reward += float(terminal_weight) * env_reward
            total_rewards[p] += reward
            buffer.add(obs, action, reward, done)
        if done:
            break

    return {
        "length": float(t + 1),
        "reward_p0": float(total_rewards[0]),
        "reward_mean": float(sum(total_rewards) / max(1, len(total_rewards))),
        "reward_abs_mean": float(sum(abs(r) for r in total_rewards) / max(1, len(total_rewards))),
        "noop_rate": float(sum(
            v for k, v in action_reasons.items()
            if k.startswith("invalid") or k.startswith("micro_reject") or k.startswith("micro_error")
        ) / max(1, sum(action_reasons.values()))),
    }


def ppo_update(policy: FinalNodePolicy, buffer: RolloutBuffer, optimizer, cfg: PPOConfig, device: torch.device) -> dict[str, float]:
    data = buffer.tensors(device)
    n = len(buffer)
    idx = torch.arange(n, device=device)
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    updates = 0

    for _epoch in range(int(cfg.epochs)):
        perm = idx[torch.randperm(n, device=device)]
        for start in range(0, n, int(cfg.minibatch_size)):
            mb = perm[start:start + int(cfg.minibatch_size)]
            logprob, entropy, value = policy.evaluate_actions(
                data["node"][mb],
                data["edge"][mb],
                data["global"][mb],
                data["valid"][mb],
                data["edge_mask"][mb],
                data["source"][mb],
                data["target"][mb],
                data["target_idx"][mb],
                data["source_idx"][mb],
                data["amount"][mb],
            )
            ratio = torch.exp(logprob - data["old_logprob"][mb])
            adv = data["advantages"][mb]
            clipped = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv
            policy_loss = -torch.min(ratio * adv, clipped).mean()
            value_loss = F.mse_loss(value, data["returns"][mb])
            entropy_loss = -entropy.mean()
            loss = policy_loss + float(cfg.value_coef) * value_loss + float(cfg.entropy_coef) * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.max_grad_norm))
            optimizer.step()

            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.mean().item())
            updates += 1

    return {
        "policy_loss": total_policy / max(1, updates),
        "value_loss": total_value / max(1, updates),
        "entropy": total_entropy / max(1, updates),
    }
