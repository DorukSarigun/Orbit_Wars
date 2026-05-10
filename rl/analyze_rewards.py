from __future__ import annotations

import argparse
import inspect
from collections import defaultdict
from pathlib import Path

import torch

from .buffer import RolloutBuffer
from .rewards import build_reward_config
from .rollout import collect_episode
from .train import load_checkpoint


_COMPONENT_RANGES = (
    ((798, 803), "fleet_combat"),
    ((873, 873), "airborne"),
    ((883, 883), "defense_deficit"),
    ((943, 943), "step_reward"),
    ((950, 955), "defense_hold"),
    ((982, 987), "planet_defense"),        # Tasarım 2 Defense Window
    ((990, 1000), "capture"),
    ((1002, 1012), "neutral_gift"),
    ((1015, 1027), "reinforcement"),       # Sorun 1: post-capture reinforcement
    ((1030, 1040), "capture_clawback"),    # Sorun 2: capture-then-lose
    ((1048, 1053), "planet_loss"),         # legacy PlanetLossTracker (current/aggressive)
)


def _classify_stack() -> str:
    for frame in inspect.stack(context=0):
        if not frame.filename.endswith("/rl/rollout.py"):
            continue
        for (lo, hi), name in _COMPONENT_RANGES:
            if lo <= frame.lineno <= hi:
                return name
    return "unknown"


class DiagnosticRolloutBuffer(RolloutBuffer):
    def __init__(self) -> None:
        super().__init__()
        self.components = defaultdict(list)

    def add_reward(self, idx: int, reward: float) -> None:
        self.components[_classify_stack()].append((int(idx), float(reward)))
        super().add_reward(idx, reward)

    def add_delayed_reward(self, idx: int, reward: float, *, event_turn: int, discount: float) -> None:
        delay = max(0, int(event_turn) - int(self.data[idx].turn))
        discounted = float(reward) * (float(discount) ** delay)
        self.components[_classify_stack()].append((int(idx), discounted))
        self.data[idx].reward += discounted


def _reward_cfg_from_checkpoint(ckpt: dict):
    reward_cfg = ckpt.get("reward_cfg", {})
    if isinstance(reward_cfg, dict):
        return build_reward_config(
            reward_cfg.get("reward_system", "production_simple"),
            **{k: v for k, v in reward_cfg.items() if k != "reward_system"},
        )
    return reward_cfg


def _print_summary(name: str, values: list[float]) -> None:
    if not values:
        print(f"{name:16s} n={0:4d}")
        return
    print(
        f"{name:16s} n={len(values):4d} "
        f"sum={sum(values):+.6f} mean={sum(values) / len(values):+.6f} "
        f"min={min(values):+.6f} max={max(values):+.6f} "
        f"pos={sum(v > 0 for v in values)} neg={sum(v < 0 for v in values)}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze rollout reward components for a PPO checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-agents", type=int, choices=(2, 4), default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    policy, ckpt = load_checkpoint(Path(args.ckpt), device=device)
    policy.eval()
    reward_cfg = _reward_cfg_from_checkpoint(ckpt)

    print("ckpt", args.ckpt)
    print("iteration", ckpt.get("iteration"))
    print("arch", ckpt.get("arch"))
    print(
        "reward_cfg_subset",
        {
            key: getattr(reward_cfg, key)
            for key in (
                "reward_system",
                "reward_scale",
                "fleet_airborne_weight",
                "fleet_kill_weight",
                "fleet_waste_weight",
                "planet_loss_weight",
            )
        },
    )

    buffer = DiagnosticRolloutBuffer()
    stats = collect_episode(
        policy,
        buffer,
        max_steps=args.max_steps,
        num_agents=args.num_agents,
        reward_cfg=reward_cfg,
        device=device,
        seed=args.seed,
    )
    print("episode", stats)
    print(
        "transitions",
        len(buffer),
        "launch_actions",
        sum(not tr.is_noop for tr in buffer.data),
        "noop_actions",
        sum(tr.is_noop for tr in buffer.data),
    )
    for component in sorted(buffer.components):
        _print_summary(component, [value for _, value in buffer.components[component]])
    _print_summary("launch_total", [tr.reward for tr in buffer.data if not tr.is_noop])
    _print_summary("noop_total", [tr.reward for tr in buffer.data if tr.is_noop])
    _print_summary("all_total", [tr.reward for tr in buffer.data])


if __name__ == "__main__":
    main()
