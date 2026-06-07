"""Quick reward-composition analysis of the v11 remastered episodes log.

Answers: where does the current policy's reward actually come from, and does the
"hoard + tiny-spray" behaviour earn anything?
"""
import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_remastered_v11/episodes.jsonl"
n_tail = int(sys.argv[2]) if len(sys.argv) > 2 else 40

rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

rows = rows[-n_tail:]
print(f"Analyzing last {len(rows)} episodes from {path}\n")

outcome_n = Counter()
outcome_rew = defaultdict(float)
role_n = Counter()
tot_reward = 0.0
caps = 0
losses = 0
pos_campaigns = 0
zero_campaigns = 0
neg_campaigns = 0

for ep in rows:
    rs = ep.get("reward_stats", {})
    tot_reward += float(rs.get("total", 0.0))
    caps += int(ep.get("n_planet_captures", 0))
    losses += int(ep.get("n_planet_losses", 0))
    for c in ep.get("campaigns", []):
        oc = c.get("outcome", "?")
        r = float(c.get("reward", 0.0))
        outcome_n[oc] += 1
        outcome_rew[oc] += r
        role_n[c.get("role", "?")] += 1
        if r > 1e-6:
            pos_campaigns += 1
        elif r < -1e-6:
            neg_campaigns += 1
        else:
            zero_campaigns += 1

print(f"Total reward (sum over episodes): {tot_reward:+.3f}")
print(f"Avg reward / episode:            {tot_reward/max(1,len(rows)):+.3f}")
print(f"Planet captures: {caps}   losses: {losses}\n")

print("Campaign outcomes (count | summed reward | avg):")
for oc, n in outcome_n.most_common():
    r = outcome_rew[oc]
    print(f"  {oc:24s} n={n:5d}  R={r:+9.3f}  avg={r/max(1,n):+.4f}")

print(f"\nCampaign reward sign: +{pos_campaigns}  0:{zero_campaigns}  -{neg_campaigns}")
print(f"Roles: {dict(role_n)}")
