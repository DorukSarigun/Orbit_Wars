"""Smoke test: production_diff reward + eligibility trace.

Loads any checkpoint, overrides its reward config to use production_diff,
runs one episode, and prints the per-transition reward distribution to
verify trace decay actually propagates reward backward.
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
from collections import defaultdict

import torch

from rl.buffer import RolloutBuffer
from rl.rewards import build_reward_config
from rl.rollout import collect_episode
from rl.train import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--noop-trace", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=0.001)
    args = parser.parse_args()

    device = torch.device(args.device)
    policy, ckpt = load_checkpoint(args.ckpt, device=device)
    policy.eval()

    reward_cfg = build_reward_config(
        "production_diff",
        eligibility_trace_horizon=args.horizon,
        eligibility_trace_decay=args.decay,
        noop_trace_weight=args.noop_trace,
        reward_scale=args.reward_scale,
    )

    print(f"ckpt: {args.ckpt}")
    print(f"reward_system: {reward_cfg.reward_system}")
    print(f"horizon: {reward_cfg.eligibility_trace_horizon}")
    print(f"decay: {reward_cfg.eligibility_trace_decay}")
    print(f"noop_trace_weight: {reward_cfg.noop_trace_weight}")
    print(f"reward_scale: {reward_cfg.reward_scale}")
    print()

    buffer = RolloutBuffer()
    stats = collect_episode(
        policy, buffer,
        max_steps=args.max_steps,
        num_agents=2,
        reward_cfg=reward_cfg,
        device=device,
        seed=args.seed,
    )

    print(f"Episode: length={stats.length}, winner={stats.winner}, "
          f"launches={stats.n_launches}")
    print(f"Total transitions: {len(buffer)}")
    print()

    # Per-player reward stats
    by_player = defaultdict(list)
    by_turn = defaultdict(list)
    noop_rewards, launch_rewards = [], []
    for tr in buffer.data:
        by_player[tr.player].append(tr.reward)
        by_turn[tr.turn].append(tr.reward)
        (noop_rewards if tr.is_noop else launch_rewards).append(tr.reward)

    for p, rs in sorted(by_player.items()):
        print(f"Player {p}: n={len(rs):4d} "
              f"sum={sum(rs):+.4f} "
              f"mean={sum(rs)/len(rs):+.6f} "
              f"min={min(rs):+.4f} max={max(rs):+.4f}")
    print()
    print(f"NO-OP transitions:   n={len(noop_rewards):4d} "
          f"sum={sum(noop_rewards):+.4f} "
          f"mean={(sum(noop_rewards)/len(noop_rewards)) if noop_rewards else 0.0:+.6f}")
    print(f"Launch transitions:  n={len(launch_rewards):4d} "
          f"sum={sum(launch_rewards):+.4f} "
          f"mean={(sum(launch_rewards)/len(launch_rewards)) if launch_rewards else 0.0:+.6f}")
    print()

    # Distribution check: did any single transition accumulate from many turns?
    # (eligibility trace effect)
    nonzero = [tr.reward for tr in buffer.data if abs(tr.reward) > 1e-9]
    print(f"Non-zero reward transitions: {len(nonzero)}/{len(buffer)} "
          f"({100*len(nonzero)/max(1,len(buffer)):.1f}%)")

    # Print first 3 and last 3 turns' distribution
    sorted_turns = sorted(by_turn.keys())
    if sorted_turns:
        print()
        print("Sample turn rewards (first 3, last 3):")
        for t in sorted_turns[:3] + sorted_turns[-3:]:
            rs = by_turn[t]
            print(f"  turn={t:>3} n={len(rs):2d} "
                  f"min={min(rs):+.5f} max={max(rs):+.5f} "
                  f"mean={sum(rs)/len(rs):+.5f}")


if __name__ == "__main__":
    main()
