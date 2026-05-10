"""Debug DefenseKillsTracker: instrument live episode counters and report."""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import argparse
from collections import Counter

import torch

from rl.buffer import RolloutBuffer
from rl.rewards import build_reward_config
from rl.rollout import collect_episode
from rl.train import load_checkpoint
from rl.defense_kills_tracker import DefenseKillsTracker


_counters = Counter()


def install_instrumentation() -> None:
    orig_register = DefenseKillsTracker.register_defender_launch
    orig_observe = DefenseKillsTracker.observe_turn
    orig_pop = DefenseKillsTracker.pop_resolved

    def reg(self, *args, **kwargs):
        _counters["register_defender_calls"] += 1
        before = sum(len(ep.defenders) for ep in self._active.values())
        orig_register(self, *args, **kwargs)
        after = sum(len(ep.defenders) for ep in self._active.values())
        if after > before:
            _counters["defenders_appended"] += 1
        return None

    def obs(self, turn, prev_obs, curr_obs, player, *, defense_kills_weight, reward_scale):
        active_before = len(self._active)
        defenders_before = sum(len(ep.defenders) for ep in self._active.values())
        kills_before = sum(ep.enemy_killed_total for ep in self._active.values())
        orig_observe(self, turn, prev_obs, curr_obs, player,
                     defense_kills_weight=defense_kills_weight,
                     reward_scale=reward_scale)
        active_after = len(self._active)
        defenders_after = sum(len(ep.defenders) for ep in self._active.values())
        kills_after = sum(ep.enemy_killed_total for ep in self._active.values())
        _counters["observe_turn_calls"] += 1
        new_episodes = max(0, active_after - active_before)
        if new_episodes > 0:
            _counters["episodes_started"] += new_episodes
        if kills_after > kills_before:
            _counters["enemy_kills_recorded"] += 1
            _counters["enemy_kills_total_ships"] += int(kills_after - kills_before)
        # finalized = active dropped without going to _resolved? Hard to tell here.

    def pop(self, player):
        for item in orig_pop(self, player):
            _counters["resolved_yielded"] += 1
            buffer_idx, reward, event_turn = item
            _counters["resolved_reward_sum"] += reward
            yield item

    DefenseKillsTracker.register_defender_launch = reg
    DefenseKillsTracker.observe_turn = obs
    DefenseKillsTracker.pop_resolved = pop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    install_instrumentation()

    device = torch.device(args.device)
    policy, ckpt = load_checkpoint(args.ckpt, device=device)
    policy.eval()
    rcfg_dict = ckpt.get("reward_cfg", {})
    reward_cfg = build_reward_config(
        rcfg_dict.get("reward_system", "production_simple"),
        **{k: v for k, v in rcfg_dict.items() if k != "reward_system"},
    )

    print(f"ckpt: {args.ckpt}")
    print(f"reward_system: {reward_cfg.reward_system}")
    print(f"defense_kills_weight: {reward_cfg.defense_kills_weight}")
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

    print(f"Episode: length={stats.length}, winner={stats.winner}, launches={stats.n_launches}")
    print()
    print("DefenseKillsTracker instrumentation counters:")
    for k, v in sorted(_counters.items()):
        print(f"  {k:>30s}: {v}")


if __name__ == "__main__":
    main()
