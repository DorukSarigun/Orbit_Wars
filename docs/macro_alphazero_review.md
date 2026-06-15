# Macro AlphaZero Review

This note reviews the current macro AlphaZero-style path:

- `orbit_macro_mcts_agent.py`
- `macro_alphazero_net.py`
- `macro_selfplay.py`
- `train_macro_alphazero.py`

## Summary

The current implementation is not full AlphaZero in the strict sense. It is
better described as:

```text
abstract macro MCTS + neural prior/value imitation + heuristic value target
```

The search is useful, but the training target and eval loop are abstract-model
targets, not real Orbit Wars win/loss outcomes. This is fine if intentional, but
it should be treated as expert iteration over an abstract simulator rather than
as a direct AlphaZero implementation.

## Findings

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | `z` is a heuristic abstract score, not a game outcome | Critical | The value head learns the abstract score, not win probability. |
| 2 | `_rebase_state` drops accumulated reward | Critical | Past capture income can disappear when a live abstract state is shifted into a new search. |
| 3 | Reward perspective is ambiguous | Critical | A scalar `state.reward` is unsafe when `root_player` changes between macro self-play plies. |
| 4 | Eval-gate runs in the abstract model | Important | Promotion measures abstract strength, not real-env deployment strength. |
| 5 | Leaf value blends rollout and neural value | Important | If the value net is not calibrated to the rollout score scale, MCTS leaf values become noisy. |
| 6 | `gamma=1.0` can inflate value scale | Important | Long remaining horizons can produce large value targets and unstable value loss. |
| 7 | Abstract macro model creates a sim-to-real gap | Important | Macro actions are not the exact low-level launches that execute in the real env/scripted allocator. |

## Details

### `z` Is Not A Terminal Outcome

In `macro_selfplay.py`, the final target is produced by `_final_outcome(...)`,
which closes the game with:

```python
closer._score(closer._advance_to_horizon(closer._initial_state))
```

That is an abstract heuristic score. It is not a real `orbit_wars` terminal
reward and not a true win/loss label.

This means the value head learns:

```text
V(s) ~= abstract horizon metric
```

not:

```text
V(s) ~= probability of winning / final game outcome
```

That distinction matters for naming, value scaling, and how much trust we put
in the network during real-env play.

### `_rebase_state` Drops Reward

`MacroTimedMCTS._rebase_state(...)` shifts an evolved abstract state into a new
search-local relative turn zero. It currently rebuilds the state with:

```python
reward=0.0
```

If the incoming live state has already accumulated capture income or other
abstract rewards, that credit is discarded. The next value target can therefore
understate the value of earlier successful captures.

### Scalar Reward Is Not Enough

A quick patch would be:

```python
reward=state.reward
```

but this is only partially safe. In macro self-play, each ply creates a new
`MacroTimedMCTS` with:

```python
root_player = live.player_to_move
```

If `state.reward` is scalar and root-relative, its sign/meaning can become
ambiguous when the root player changes.

The robust fix is to store reward by player, for example:

```text
reward_by_player[player] += capture_income
```

Then `_score(state, root_player)` can compute:

```text
reward_by_player[root_player] - best_other_reward
```

or another explicit player-relative aggregation.

### Eval-Gate Is Abstract

`train_macro_alphazero.py::eval_gate` pits candidate and best nets against each
other inside the same abstract macro model. This is internally consistent for
abstract training, but it does not guarantee improvement in the real Orbit Wars
environment.

A safer promotion pipeline would use two gates:

```text
1. fast abstract eval-gate
2. slower real-env eval-gate against baseline agents
```

The real-env gate can be run less frequently to keep training affordable.

### Leaf Value Blend Requires Calibration

`MacroTimedMCTS._leaf_value(...)` blends:

```python
(1 - value_blend) * rollout_score + value_blend * neural_value
```

This is only sound if the neural value is trained on the same scale as
`_score(...)` and remains calibrated. If the value loss is large or the target
scale drifts, the blended leaf value injects noise into MCTS.

Practical mitigations:

- keep `value_blend` low early;
- log value target mean/std and prediction mean/std;
- normalize value targets or use a bounded value transform;
- increase blend only after calibration improves.

### `gamma=1.0` Scale Risk

The macro AlphaZero training config currently sets:

```python
gamma=1.0
```

This makes capture income scale linearly with remaining turns. With long
remaining horizons, value targets can become large and destabilize value loss.

Using `gamma < 1` or normalizing value targets would make the value head easier
to train.

## Recommended Fix Order

1. Replace scalar abstract reward with player-indexed reward.
2. Preserve reward when rebasing live abstract states.
3. Make `_score` explicitly root-player-relative.
4. Add value target scale logging.
5. Use `gamma < 1` by default for macro AlphaZero training, or normalize value
   targets.
6. Add an optional real-env eval-gate after abstract promotion.
7. Rename docs/log labels from "AlphaZero" to "AlphaZero-style" or "macro expert
   iteration" unless/until true terminal outcome training is added.

## Minimal Emergency Patch

If a fast tactical patch is needed before the player-indexed reward refactor,
change `_rebase_state` from:

```python
reward=0.0
```

to:

```python
reward=state.reward
```

This preserves accumulated abstract credit, but it does not fully solve the
root-perspective ambiguity. It should be treated as a temporary improvement, not
the final design.

