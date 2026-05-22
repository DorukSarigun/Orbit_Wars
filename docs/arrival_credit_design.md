# Arrival-Credit Eligibility Trace (Option B)

ETA-weighted **backward credit assignment** of the realized graph-metric impact
to the original launch transition. Default off; opt-in via
`--enable-arrival-credit`.

## Motivation

A launch made at turn `t_launch` doesn't pay off until the fleet lands at
`t_arrive = t_launch + τ`. Naïve per-step shaping spreads the credit thinly
across all τ intermediate steps, none of which the agent caused. The result
is a noisy gradient and slow learning when fleets have long ETAs.

We already partially address this with **flux-augmented potential shaping**
(Option A, already on `main`): in-flight fleets contribute ETA-discounted
flow into 4 node-level metric channels, which the potential function `φ(s)`
sees from the moment of launch. This is helpful but **doesn't see the 8
static node metrics** (influence, supportability, vulnerability, etc.) —
those only change when the fleet actually arrives.

Option B fills that gap by capturing the **realized 12-channel target-planet
delta** at arrival time and writing the corresponding ETA-discounted reward
**back to the launch transition** through `buffer.add_delayed_reward`.

## Pipeline

```
        launch (turn t_l)                     arrive (turn t_a)
        ----------------                      ------------------
1.  agent picks (src, tgt, ships).      1.  FleetTracker.pop_resolved
2.  rollout adds Transition to buffer   2.  yields (buffer_idx, outcome).
    (buffer_idx = idx).
3.  ArrivalCreditTracker.register:      3.  ArrivalCreditTracker.compute_credit:
    - tgt_slot = lookup(tgt_planet_id)    - curr_vec  = M_curr[tgt_slot]
    - launch_vec = M_launch[tgt_slot]     - delta     = curr_vec - launch_vec
    - stores (idx, tgt_id, eta, vec)      - weighted  = Σ_c w_c · delta_c
                                          - credit    = w(eta) · scale · weighted
                                          - removes the snapshot.
                                        4.  buffer.add_delayed_reward(
                                              idx, credit,
                                              event_turn=t_a,
                                              discount=delayed_reward_discount).
```

`M_launch` is the per-planet 12-channel metric matrix at launch time,
produced by `rl.rewards.compute_planet_metric_vectors(obs, cfg=...)`.
`M_curr` is the same matrix recomputed under the post-step environment
observation. Both reuse the existing graph-metric kernels from
`model_graph_pyg._graph_node_metrics`, so the credit signal is exactly the
same quantity the encoder and the potential shaping see.

## Credit formula

For each arrived fleet F (player p, launched at t_l with intended ETA τ,
target planet id g, buffer index i):

```
delta  = M_curr[slot(g)] - M_launch[slot(g)]              ∈ ℝ¹²
credit = w(τ) · arrival_credit_weight · reward_scale · ⟨channel_weights, delta⟩
buffer.rewards[i] += credit · delayed_reward_discount^(t_a − t_i)
```

with

- `channel_weights ∈ ℝ¹²` =
  `[graph_friendly_influence_weight, graph_enemy_influence_weight,
    graph_influence_balance_weight, graph_friendly_support_weight,
    graph_supportability_weight, graph_vulnerability_weight,
    graph_production_potential_weight, graph_strategic_potential_weight,
    graph_flux_friendly_in_weight, graph_flux_friendly_out_weight,
    graph_flux_enemy_in_weight, graph_flux_enemy_out_weight]`
  — i.e. the same 12-vector consumed by `compute_graph_native_reward`.
- `w(τ)` ∈ {1, max(0, 1-τ/η_norm), exp(-τ/τ_decay)} controlled by
  `arrival_credit_mode`.

When `arrival_credit_include_source=True`, the source-slot delta is
subtracted: `credit -= ⟨channel_weights, M_curr[slot(src)] - M_launch[slot(src)]⟩`,
which encodes the trade-off the launch makes (gain at target vs. cost at
source).

## "All 12 channels" guarantee

Every channel `c ∈ [0, 12)` contributes additively to the credit iff its
weight is non-zero. This is the user's hard requirement and is enforced by
a parametrized unit test:

```
tests/test_arrival_credit.py::test_all_twelve_channels_contribute[<c>]
```

which feeds a unit delta on each channel one at a time and asserts the
credit equals the channel's weight. Any future refactor that drops,
permutes, or zero-truncates the metric vector will surface here.

## Double-counting analysis

There are three sources of credit for the same arrival event:

| Source | What it credits | When it fires |
|---|---|---|
| Potential shaping (`r_t = γφ(s')−φ(s)`) | the **step-by-step** Δφ across all turns | every step from t_l to t_a |
| `FleetTracker` (capture / kill / waste) | the **categorical** outcome of the fleet | at t_a |
| **Arrival credit** (this module) | the **realized 12-channel target-planet delta** at t_a, projected back to t_l | at t_a |

These are intentionally additive — they emphasize different aspects of the
same event. To avoid the signals collapsing into one another:

1. The arrival credit is **opt-in**. Existing reward systems behave
   identically when the flag is off (verified by
   `test_rollout_smoke_disabled`).
2. The credit uses **per-planet** vectors at the target slot, not the global
   `node_sum` that potential shaping uses. So the same physical event is
   measured along two different axes (one global, one local).
3. The `arrival_credit_weight` defaults to `1.0` but should be tuned down
   (e.g. `0.25–0.5`) when both shaping and arrival credit are on together;
   otherwise `valL` variance can amplify.

## CLI

```
--enable-arrival-credit              # required to fire anything (store_true)
--arrival-credit-mode {raw,linear,exp_decay}
--arrival-credit-tau <float>         # decay timescale (exp_decay; default 10.0)
--arrival-credit-eta-norm <float>    # ETA cutoff (linear; default 50.0)
--arrival-credit-weight <float>      # global scale (default 1.0)
--arrival-credit-include-source      # also account for source-side delta
```

Example training run combining the existing graph-native reward, flux
metrics, and arrival credit:

```bash
python -m rl.train \
  --model-variant pyg_planet_graph \
  --reward-system graph_native --opponent scripted \
  --enable-flux-metrics --enable-arrival-credit \
  --arrival-credit-mode exp_decay --arrival-credit-tau 10.0 \
  --arrival-credit-weight 0.5 \
  --iterations 200 --episodes-per-iter 5 --max-episode-steps 500 \
  --d-model 96 --n-layers 2 --device cuda \
  --checkpoint-dir checkpoints/arrival_credit_run \
  --log-path checkpoints/arrival_credit_run/log.jsonl
```

## Performance

- O(P_max²) per launch for `compute_planet_metric_vectors(launch_obs)`.
- O(P_max²) per **affected turn per player** for the current matrix
  (lazy-built once per turn iff at least one arrival fires for that player).
- O(1) per arrival for the credit computation itself.

P_max = 44 → P_max² = 1936 ≈ negligible against the per-step PyG forward.

## Diagnostic counters

`ArrivalCreditTracker` exposes:

- `credits_emitted: int` — number of arrivals resolved this episode.
- `credits_total: float` — signed running sum of all credits issued.

These will be wired into the training-iteration log in a follow-up so the
user can see at a glance whether the trace is firing and which sign it
contributes.

## Files

| File | Role |
|---|---|
| `rl/arrival_credit.py` | `ArrivalCreditTracker`, `build_channel_weights`, `eta_weight` |
| `rl/rewards.py` | `RewardConfig` fields, `compute_planet_metric_vectors` helper |
| `rl/rollout.py` | snapshot at launch; credit at `pop_resolved` |
| `rl/train.py` | CLI flags |
| `tests/test_arrival_credit.py` | 34 tests (unit + integration smoke) |
