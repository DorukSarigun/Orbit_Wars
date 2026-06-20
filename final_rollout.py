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


def _planner_sources(
    *,
    planets,
    source_id: int,
    target,
    player: int,
    amount_frac: float,
    omega: float,
    top_m: int,
    budget_floor: float,
) -> tuple[list[SourceOption], str]:
    """Build source candidates for target-local planning.

    The policy-selected source is not discarded: it is forced into the candidate
    set first when owned. Other donors are added by ETA/budget order so the
    planner can solve multi-source captures.
    """

    scale = max(float(budget_floor), min(1.0, float(amount_frac)))
    rows: list[tuple[int, float, int, object]] = []
    for planet in planets:
        if int(planet.owner) != int(player) or int(planet.ships) <= 0:
            continue
        launch = compute_launch(planet, target, 1, planets, omega)
        if launch is None:
            continue
        selected_rank = 0 if int(planet.id) == int(source_id) else 1
        rows.append((selected_rank, float(launch["eta"]), -int(planet.ships), planet))
    if not rows:
        return [], "micro_reject_no_reachable_donor"

    rows.sort(key=lambda r: (r[0], r[1], r[2], int(r[3].id)))
    selected = [r for r in rows if int(r[3].id) == int(source_id)]
    others = [r for r in rows if int(r[3].id) != int(source_id)]
    picked = (selected + others)[: max(1, int(top_m))]
    out: list[SourceOption] = []
    for selected_rank, _eta, _neg_ships, planet in picked:
        raw_budget = int(round(float(planet.ships) * scale))
        budget = max(1, min(int(planet.ships), raw_budget))
        # A small preference keeps the policy-selected source meaningful without
        # forcing impossible one-source captures.
        strip_cost = -0.15 if int(selected_rank) == 0 else 0.0
        out.append(SourceOption(source=planet, budget=budget, strip_cost=strip_cost))
    return out, "ok"


def _action_to_launch(
    env_obs: Any,
    obs: FinalObservation,
    action,
    player: int,
    *,
    current_turn: int,
    max_turns: int,
    planner_top_m_sources: int = 5,
    planner_max_launches_per_target: int = 3,
    planner_budget_floor: float = 0.35,
    planner_hold_horizon: float = 24.0,
) -> tuple[list[list], str]:
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
    omega = (
        float(env_obs.get("angular_velocity", 0.0) or 0.0)
        if isinstance(env_obs, dict)
        else float(getattr(env_obs, "angular_velocity", 0.0) or 0.0)
    )

    # Prefer the final micro-planner for non-owned targets.  The policy chooses
    # a target/source intent; the planner verifies capture-and-hold and may add
    # a larger amount from the chosen source.  If it cannot find a stable plan,
    # do not emit a weak raw attack; those undercommitted launches mostly add
    # reward noise during PPO.
    if int(target.owner) != int(player):
        try:
            og = build_orbit_graph(env_obs, player=int(player))
            sources, source_reason = _planner_sources(
                planets=planets,
                source_id=int(source.id),
                target=target,
                player=int(player),
                amount_frac=float(action.amount.item()),
                omega=omega,
                top_m=int(planner_top_m_sources),
                budget_floor=float(planner_budget_floor),
            )
            if not sources:
                return [], source_reason
            plan = plan_target_attack(
                og=og,
                target=target,
                sources=sources,
                player=int(player),
                current_turn=int(current_turn),
                max_turns=int(max_turns),
                omega=omega,
                cfg=MicroPlannerConfig(
                    hold_horizon=float(planner_hold_horizon),
                    beam_width=8,
                    max_launches_per_target=int(planner_max_launches_per_target),
                    top_m_sources=int(planner_top_m_sources),
                    min_send=1,
                    capture_margin=2,
                ),
            )
            if plan is not None:
                return plan.actions, "micro_capture"
            return [], "micro_reject_target"
        except Exception:
            return [], "micro_error_target"

    ships = max(1, int(round(float(action.amount.item()) * float(source.ships))))
    ships = min(int(source.ships), ships)
    if ships <= 0:
        return [], "invalid_amount"
    launch = compute_launch(source, target, ships, planets, omega)
    if launch is None:
        return [], "invalid_geometry"
    return [[int(source.id), float(launch["angle"]), int(launch["ships"])]], "raw_launch"


def _action_validity_reward(
    reason: str,
    obs: FinalObservation,
    *,
    invalid_penalty: float,
    reject_penalty: float,
    capture_bonus: float,
    raw_launch_bonus: float,
) -> float:
    """Dense feedback for action validity, separate from game outcome reward."""

    has_source = bool(obs.source_mask.any())
    has_target = bool(obs.target_mask.any())
    if not has_source or not has_target:
        return 0.0
    if reason.startswith("invalid"):
        return -abs(float(invalid_penalty))
    if reason.startswith("micro_reject") or reason.startswith("micro_error"):
        return -abs(float(reject_penalty))
    if reason == "micro_capture":
        return abs(float(capture_bonus))
    if reason == "raw_launch":
        return abs(float(raw_launch_bonus))
    return 0.0


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
    invalid_action_penalty: float = 0.02,
    reject_action_penalty: float = 0.005,
    capture_action_bonus: float = 0.002,
    raw_launch_bonus: float = 0.0,
    planner_top_m_sources: int = 5,
    planner_max_launches_per_target: int = 3,
    planner_budget_floor: float = 0.35,
    planner_hold_horizon: float = 24.0,
    truncate_terminal_weight: float = 1.0,
    include_owned_targets: bool = False,
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
    validity_rewards = [0.0 for _ in range(num_agents)]
    action_reasons: dict[str, int] = {}

    for t in range(int(max_steps)):
        pending = []
        env_actions: list[list] = []
        for p in range(num_agents):
            obs_raw = state[p]["observation"]
            obs = build_final_observation(
                obs_raw, p, include_owned_targets=bool(include_owned_targets),
            )
            tensors = _obs_to_tensors(obs, device)
            with torch.no_grad():
                action = policy.act(**tensors)
            launch_action, reason = _action_to_launch(
                obs_raw,
                obs,
                action,
                p,
                current_turn=t,
                max_turns=planner_max_turns,
                planner_top_m_sources=int(planner_top_m_sources),
                planner_max_launches_per_target=int(planner_max_launches_per_target),
                planner_budget_floor=float(planner_budget_floor),
                planner_hold_horizon=float(planner_hold_horizon),
            )
            action_reasons[reason] = action_reasons.get(reason, 0) + 1
            validity_reward = _action_validity_reward(
                reason,
                obs,
                invalid_penalty=float(invalid_action_penalty),
                reject_penalty=float(reject_action_penalty),
                capture_bonus=float(capture_action_bonus),
                raw_launch_bonus=float(raw_launch_bonus),
            )
            validity_rewards[p] += float(validity_reward)
            env_actions.append(launch_action)
            pending.append((p, obs, action, validity_reward))

        state = env.step(env_actions)
        done = bool(env.done)
        truncated = bool(not done and t == int(max_steps) - 1)
        step_done = bool(done or truncated)
        for p, obs, action, validity_reward in pending:
            new_margin = production_share_margin(state[p]["observation"], p, num_agents)
            reward = float(validity_reward) + float(shaping_weight) * (new_margin - prev_margin[p])
            prev_margin[p] = new_margin
            if done:
                env_reward = float(getattr(state[p], "reward", 0.0) or 0.0)
                reward += float(terminal_weight) * env_reward
            elif truncated:
                reward += float(truncate_terminal_weight) * float(new_margin)
            total_rewards[p] += reward
            buffer.add(obs, action, reward, step_done)
        if done:
            break

    return {
        "length": float(t + 1),
        "reward_p0": float(total_rewards[0]),
        "reward_mean": float(sum(total_rewards) / max(1, len(total_rewards))),
        "reward_abs_mean": float(sum(abs(r) for r in total_rewards) / max(1, len(total_rewards))),
        "validity_reward_mean": float(sum(validity_rewards) / max(1, len(validity_rewards))),
        "validity_reward_abs_mean": float(sum(abs(r) for r in validity_rewards) / max(1, len(validity_rewards))),
        "truncated": float(1.0 if (not done and int(max_steps) > 0 and t == int(max_steps) - 1) else 0.0),
        "micro_capture_rate": float(action_reasons.get("micro_capture", 0) / max(1, sum(action_reasons.values()))),
        "raw_launch_rate": float(action_reasons.get("raw_launch", 0) / max(1, sum(action_reasons.values()))),
        "reject_rate": float(sum(
            v for k, v in action_reasons.items()
            if k.startswith("micro_reject") or k.startswith("micro_error")
        ) / max(1, sum(action_reasons.values()))),
        "invalid_rate": float(sum(
            v for k, v in action_reasons.items()
            if k.startswith("invalid")
        ) / max(1, sum(action_reasons.values()))),
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
