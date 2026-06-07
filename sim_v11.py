"""Simulate v11 (P0) vs frozen v7 (P1) and dump the remastered reward breakdown.

Focus: does the "hoard + tiny-spray" behaviour earn reward? We look at per-campaign
reward vs ship size, and the launch-size distribution.
"""
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

from reworked_rollout import collect_episode, ReworkedBuffer, RewardConfig
from reworked_remastered_reward import RemasteredRewardConfig
from reworked_train import load_checkpoint

ckpt_p0 = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_remastered_v11/policy_iter00059.pt"
ckpt_p1 = sys.argv[2] if len(sys.argv) > 2 else "checkpoints_reworked_v7_campaign/latest.pt"
n_games = int(sys.argv[3]) if len(sys.argv) > 3 else 3

device = torch.device("cpu")
policy, ck0 = load_checkpoint(ckpt_p0, device=device)
frozen, ck1 = load_checkpoint(ckpt_p1, device=device)
policy_kind = ck0.get("arch", {}).get("policy_kind", "edge")
min_send = int(ck0.get("train_cfg", {}).get("min_send", 1) or 1)

reward_cfg = RewardConfig(enable_shaping=False, loss_weight=0.0, launch_weight=0.0,
                          enable_battle_credit=False, terminal_weight=0.0)
rcfg = RemasteredRewardConfig(w_casualty=0.35, w_delay=0.10, w_fail=0.25,
                              w_waste=0.5, waste_norm=50.0,
                              w_defense_loss=1.0, w_source_strip=0.5,
                              source_strip_window=10,
                              w_enemy_kill=1.0, terminal_weight=1.0)

print(f"P0={ckpt_p0}  kind={policy_kind} min_send={min_send}")
print(f"P1={ckpt_p1}\n")

agg_outcome_n = Counter()
agg_outcome_r = defaultdict(float)

for g in range(n_games):
    buf = ReworkedBuffer()
    stats = collect_episode(
        policy, buf, max_steps=500, num_agents=2,
        reward_cfg=reward_cfg, device=device, seed=g,
        opponent="frozen", frozen_policy=frozen, frozen_deterministic=True,
        policy_kind=policy_kind, min_send=min_send, remastered_cfg=rcfg,
    )
    detail = stats.episode_log or {}
    rs = detail.get("reward_stats", {})
    camps = detail.get("campaigns", [])
    winner = stats.winner
    print(f"--- game {g} (seed {g}): len={stats.length} winner={winner} "
          f"launches={stats.n_launches} total_R={rs.get('total',0):+.3f} "
          f"caps={detail.get('n_planet_captures')} losses={detail.get('n_planet_losses')}")
    by_oc_n = Counter()
    by_oc_r = defaultdict(float)
    for c in camps:
        oc = c["outcome"]
        by_oc_n[oc] += 1
        by_oc_r[oc] += c["reward"]
        agg_outcome_n[oc] += 1
        agg_outcome_r[oc] += c["reward"]
    for oc, n in by_oc_n.most_common():
        print(f"      {oc:22s} n={n:4d}  R={by_oc_r[oc]:+8.3f}  avg={by_oc_r[oc]/max(1,n):+.4f}")
    print(f"      source_strip: n={rs.get('n_source_strip',0)} R={rs.get('source_strip_reward',0.0):+.3f}")

print("\n=== AGGREGATE over", n_games, "games ===")
for oc, n in agg_outcome_n.most_common():
    r = agg_outcome_r[oc]
    print(f"  {oc:22s} n={n:5d}  R={r:+9.3f}  avg={r/max(1,n):+.4f}")
