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
sees from the moment of launch. This is helpful but **doesn't see the 9
static node metrics** (influence, supportability, vulnerability, etc.) —
those only change when the fleet actually arrives.

Option B fills that gap by capturing the **realized 13-channel target-planet
delta** at arrival time and writing the corresponding ETA-discounted reward
**back to the launch transition** through `buffer.add_delayed_reward`.

The newer counterfactual mode changes the comparison from
`arrival state - launch state` to `actual arrival branch - no-launch branch`:
the arriving fleet is removed from the target, put back at the source, and
the graph metrics are recomputed.  If removing the fleet would leave the
target with negative ships, ownership is flipped to non-self in the
counterfactual branch, so friendly production/influence metrics are disabled
by the same graph metric code path.

## Pipeline

```
        launch (turn t_l)                     arrive (turn t_a)
        ----------------                      ------------------
1.  agent picks (src, tgt, ships).      1.  FleetTracker.pop_resolved
2.  rollout adds Transition to buffer   2.  yields (buffer_idx, outcome).
    (buffer_idx = idx).
3.  ArrivalCreditTracker.register:      3.  ArrivalCreditTracker.compute_credit:
    - tgt_slot = lookup(tgt_planet_id)    - legacy: delta = M_curr - M_launch
    - launch_vec = M_launch[tgt_slot]     - counterfactual:
    - stores idx, src/tgt, ships, eta       delta = M_actual - M_no_launch
                                          - weighted = Σ_c w_c · delta_c
                                          - credit = w(eta) · scale · weighted
                                          - removes the snapshot.
                                        4.  buffer.add_delayed_reward(
                                              idx, credit,
                                              event_turn=t_a,
                                              discount=delayed_reward_discount).
```

`M_launch` is the per-planet 13-channel metric matrix at launch time,
produced by `rl.rewards.compute_planet_metric_vectors(obs, cfg=...)`.
`M_curr` is the same matrix recomputed under the post-step environment
observation. Both reuse the existing graph-metric kernels from
`model_graph_pyg._graph_node_metrics`, so the credit signal is exactly the
same quantity the encoder and the potential shaping see.

## Credit formula

For each arrived fleet F (player p, launched at t_l with intended ETA τ,
target planet id g, buffer index i):

```
delta  = M_curr[slot(g)] - M_launch[slot(g)]              ∈ ℝ¹³
credit = w(τ) · arrival_credit_weight · reward_scale · ⟨channel_weights, delta⟩
buffer.rewards[i] += credit · delayed_reward_discount^(t_a − t_i)
```

With `arrival_credit_counterfactual=True`, `delta` is instead computed from
the actual and no-launch branches.  The scope is controlled by
`arrival_credit_counterfactual_scope`:

- `target`: only target planet vector.
- `target_source`: target plus source planet vectors.
- `full_graph`: summed metric delta across all planets, capturing diffusion
  effects from ownership and ship-count changes.

`arrival_credit_counterfactual_horizon > 0` optionally resolves the soonest
enemy arrival at affected self-owned planets if that ETA is inside the cap.
This captures cases where the fleet arrives now but saves the planet from an
enemy fleet a few turns later.

with

- `channel_weights ∈ ℝ¹³` =
  `[graph_friendly_influence_weight, graph_enemy_influence_weight,
    graph_influence_balance_weight, graph_friendly_support_weight,
    graph_supportability_weight, graph_vulnerability_weight,
    graph_production_potential_weight,
    graph_friendly_production_potential_weight,
    graph_strategic_potential_weight,
    graph_flux_friendly_in_weight, graph_flux_friendly_out_weight,
    graph_flux_enemy_in_weight, graph_flux_enemy_out_weight]`
  — i.e. the same 13-vector consumed by `compute_graph_native_reward`.
- `w(τ)` ∈ {1, max(0, 1-τ/η_norm), exp(-τ/τ_decay)} controlled by
  `arrival_credit_mode`.

When `arrival_credit_include_source=True`, the source-slot delta is
subtracted: `credit -= ⟨channel_weights, M_curr[slot(src)] - M_launch[slot(src)]⟩`,
which encodes the trade-off the launch makes (gain at target vs. cost at
source).

## No-launch reward policy

Do **not** give a generic direct reward to no-launch / NOOP actions.

The counterfactual arrival-credit signal already uses no-launch as the
baseline:

```
launch_credit = metrics(actual_arrival_branch)
              - metrics(no_launch_counterfactual_branch)
```

Therefore:

- Good launches receive positive delayed credit.
- Bad launches receive negative delayed credit.
- No-launch actions receive no separate direct bonus or penalty.

This avoids rewarding passivity twice.  If no-launch credit is ever needed,
it should be event-driven and defensive only: for example, a threatened source
planet gets a small hold credit only if keeping ships at home prevents a
future loss inside a short horizon.

## Counterfactual ownership rule

When building the no-launch branch, the arriving fleet's ships are removed
from the actual target state and added back to the source.

Critical rule:

```
if target_ships_after_removing_arrival < 0:
    target owner becomes non-self in the counterfactual branch
    remaining ships = abs(target_ships_after_removing_arrival)
```

This means the graph metric code naturally disables friendly-only sources:
`friendly_influence`, `friendly_support`, and
`friendly_production_potential` stop treating that planet as ours.

If `arrival_credit_counterfactual_horizon > 0`, the tracker also resolves the
soonest enemy arrival at affected self-owned planets when the ETA is within
that cap.  This captures cases where a fleet arrives now but the real value is
that it saves the planet from an enemy fleet a few turns later.

### `arrival_credit_counterfactual_return_to_source` (default `True`)

Controls how the *source* side of the no-launch branch is reconstructed.

- **`True` (default)** — full no-launch counterfactual: the arriving fleet's
  ships are removed from the target **and added back to the source**, as if
  the launch had never happened. The credit therefore covers the entire
  trade-off (target gain vs. source depletion) the agent committed to at
  launch time.

- **`False`** — partial counterfactual: the ships are still removed from the
  target, but the source is left at its actual (post-launch) ship count.
  Use this when you want to credit only the *arrival-side* impact and treat
  the source loss as already paid in earlier shaping. This isolates the
  capture/kill value of the fleet from the production foregone at the source.

CLI: pass `--no-arrival-credit-counterfactual-return-to-source` to set this
to `False`. The two source-side variants are exercised separately in
`tests/test_arrival_credit.py` (capture/source-rescue tests with
`return_to_source=True`, isolation tests with `return_to_source=False`).

## "All channels" guarantee

Every channel contributes additively to the credit iff its
weight is non-zero. This is the user's hard requirement and is enforced by
a parametrized unit test:

```
tests/test_arrival_credit.py::test_all_metric_channels_contribute[<c>]
```

which feeds a unit delta on each channel one at a time and asserts the
credit equals the channel's weight. Any future refactor that drops,
permutes, or zero-truncates the metric vector will surface here.

## Shaping vs. arrival credit — mutually exclusive

Step-level potential shaping and arrival credit are **mutually exclusive**.
When `enable_arrival_credit=True` the `graph_native` rollout path zeroes out
the per-step `r_t = γφ(s')−φ(s)` signal so that the same metric swing is
not credited twice — arrival credit fully replaces step shaping as the
"where the metric came from" signal.

Active reward sources by configuration:

| Configuration | Step reward | Launch delayed reward | Terminal |
|---|---|---|---|
| Legacy (`enable_arrival_credit=False`) | `γφ(s')−φ(s)` | — | ±1 |
| **Current default with credit on (`enable_arrival_credit=True`)** | **0** | counterfactual `M_actual − M_no_launch` | ±1 |

Other reward sources unaffected by this switch:

- `FleetTracker` capture / kill / waste categorical outcomes still fire at
  arrival; they encode a different (event-categorical) aspect, not the
  same metric delta arrival credit captures.
- `invalid_action_penalty` and ship-overflow penalties continue to apply.

Enforcement and regression tests:

- Implementation: a `shaping_active = not cfg.enable_arrival_credit`
  guard inside the `graph_native` branch of `collect_episode`
  (`rl/rollout.py`).
- Tests:
  - `tests/test_arrival_credit.py::test_rollout_arrival_credit_disables_step_shaping`
    monkey-patches `compute_graph_native_reward` and asserts call count
    is zero when arrival credit is on.
  - `tests/test_arrival_credit.py::test_rollout_arrival_credit_off_keeps_step_shaping`
    guards against the inverse regression (legacy path still exercised
    when arrival credit is off).
- The arrival credit flag is **opt-in**; legacy reward systems behave
  identically when it is off (`test_rollout_smoke_disabled`).

`arrival_credit_weight` defaults to `1.0`. With step shaping disabled
there is no longer a double-counting reason to tune it down, but
counterfactual `full_graph` scope can still produce large swings on
capture events; start with `0.5–1.0` and lower it only if PPO `valL`
variance is excessive.

## CLI

```
--enable-arrival-credit              # required to fire anything (store_true)
--arrival-credit-mode {raw,linear,exp_decay}
--arrival-credit-tau <float>         # decay timescale (exp_decay; default 10.0)
--arrival-credit-eta-norm <float>    # ETA cutoff (linear; default 50.0)
--arrival-credit-weight <float>      # global scale (default 1.0)
--arrival-credit-include-source      # also account for source-side delta
--arrival-credit-counterfactual
--arrival-credit-counterfactual-scope {target,target_source,full_graph}
--arrival-credit-counterfactual-horizon <float>
--no-arrival-credit-counterfactual-return-to-source
```

Example training run combining the existing graph-native reward, flux
metrics, and arrival credit:

```bash
python -m rl.train \
  --model-variant pyg_planet_graph \
  --reward-system graph_native --opponent scripted \
  --enable-flux-metrics --enable-arrival-credit \
  --arrival-credit-mode exp_decay --arrival-credit-tau 10.0 \
  --arrival-credit-counterfactual --arrival-credit-counterfactual-horizon 5 \
  --arrival-credit-weight 1.0 \
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

## AI handoff summary

If another AI continues this work, preserve these decisions:

- Use `friendly_production_potential`, not raw `production_potential`, as the
  ownership-aware production metric for capture/loss reward.
- Keep generic no-launch direct reward at `0`.
- Penalize bad launches through negative actual-vs-counterfactual arrival
  credit.
- Treat negative counterfactual target ships as ownership loss; never leave a
  negative ship count in metric inputs.
- Prefer `arrival_credit_counterfactual_scope="full_graph"` for diffusion
  effects, but tune `arrival_credit_weight` down when graph-native potential
  shaping is also active.
- Add any future no-launch reward only as defensive hold credit proven by a
  short-horizon counterfactual.

## Files

| File | Role |
|---|---|
| `rl/arrival_credit.py` | `ArrivalCreditTracker`, `build_channel_weights`, `eta_weight` |
| `rl/rewards.py` | `RewardConfig` fields, `compute_planet_metric_vectors` helper |
| `rl/rollout.py` | snapshot at launch; credit at `pop_resolved` |
| `rl/train.py` | CLI flags |
| `tests/test_arrival_credit.py` | 39 tests (unit + integration smoke) |
