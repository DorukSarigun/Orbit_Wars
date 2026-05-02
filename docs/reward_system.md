# Orbit Wars Reward System

This document describes the current reward flow used by PPO/DQN training in the Orbit Wars project.

The implementation is split across:

- **`rl/rewards.py`**: step-level reward shaping and `RewardConfig`.
- **`rl/fleet_tracker.py`**: delayed fleet launch outcome rewards assigned to the original launch transition.
- **`rl/defense_tracker.py`**: delayed NO-OP defensive hold rewards assigned to the original NO-OP transition.
- **`rl/rollout.py`**: transition creation and reward assignment.
- **`rl/train.py`**: CLI flags that populate `RewardConfig`.

## High-level reward flow

Each environment step can add reward from three different paths:

```text
1. Immediate transition reward
   - Invalid action penalty on the invalid transition itself.

2. Launch-time fleet airborne cost
   - Applied immediately to the original launch transition.
   - Represents ships being unavailable while in flight.

3. Step-level reward
   - Computed from env_obs -> new_obs.
   - Split evenly across all transitions created by that player on that env step.

4. Delayed action-specific rewards
   - Fleet outcome reward goes back to the original launch transition.
   - Defense hold reward goes back to the original NO-OP transition.
```

The important design rule is:

```text
Delayed fleet/defense credit is not spread broadly.
It is added directly to the transition that caused it.
```

Eligibility trace distribution is disabled.

## RewardConfig defaults

Current defaults in `rl/rewards.py`:

| Parameter | Default | Meaning |
|---|---:|---|
| `terminal_scale` | `1.0` | Multiplier for sparse terminal env reward. |
| `capture_value_weight` | `0.01` | Small step-level reward for my production-value gain. Kept below fleet capture reward so delayed launch credit remains dominant. |
| `denial_value_weight` | `0.01` | Small step-level reward for opponent production-value loss. Equal to capture by default for symmetric production swing feedback. |
| `capture_value_kind` | `production` | Planet value basis: `production` or `radius`. |
| `max_turns` | `500` | Episode length used for time decay normalization. |
| `time_decay_floor` | `0.0` | Lower bound for capture-value time factor. |
| `ship_diff_weight` | `0.0005` | Step reward for ship advantage delta, normalized by 100. |
| `ship_loss_weight` | `0.001` | Step penalty for my ship count decreasing. |
| `fleet_airborne_weight` | `0.00002` | Launch-time cost per ship per estimated airborne step. |
| `fleet_capture_weight` | `0.05` | Delayed capture reward per captured planet production. |
| `fleet_kill_weight` | `0.001` | Small delayed reward per enemy garrison ship killed. Neutral ships are excluded. |
| `fleet_waste_weight` | `0.0015` | Delayed penalty per own ship wasted on failed fleet outcomes. |
| `enemy_capture_radius_weight` | `0.02` | Extra delayed reward for capturing enemy-owned planets, scaled by radius and remaining turns. |
| `planet_loss_weight` | `0.05` | Delayed penalty scale for losing a planet after launching from it under observed enemy incoming pressure. |
| `planet_loss_credit_horizon` | `30` | Max turns a risky launch can receive planet-loss credit. Prevents stale unrelated launch penalties. |
| `defense_hold_weight` | `0.001` | Reward per absorbed enemy ship for successful NO-OP defense. |
| `defense_hold_horizon` | `15` | Max turns a NO-OP hold can remain eligible. |
| `defense_arrival_window` | `1.5` | Enemy fleet ETA window used to identify arriving attacks. |
| `invalid_action_penalty` | `0.01` | Penalty for invalid action. |
| `step_cost` | `0.0` | Per-step alive cost. Not currently exposed in CLI. |
| `eligibility_trace_horizon` | `0` | Deprecated. Step rewards are no longer spread backward. |
| `eligibility_trace_decay` | `0.0` | Deprecated. |
| `noop_trace_weight` | `0.0` | Deprecated. NO-OP trace distribution disabled. |
| `planet_diff_weight` | `0.0` | Legacy unweighted planet count delta. Off by default. |
| `shape_on_terminal` | `False` | If false, dense shaping is suppressed on terminal step. Not currently exposed in CLI. |

## Reward system factory

Reward configs can be created through:

```text
rl/rewards.py::build_reward_config
```

Available reward systems:

```text
current
aggressive
production_simple
```

### `current`

`current` is the backward-compatible default. It returns the normal `RewardConfig()` values and preserves the existing reward behavior.

```bash
python -m rl.train --reward-system current
```

This is equivalent to not passing `--reward-system`.

### `aggressive`

`aggressive` starts from a separate preset that pushes harder toward opponent planet capture while keeping enemy kill reward small.

| Parameter | `current` | `aggressive` | Intent |
|---|---:|---:|---|
| `capture_value_weight` | `0.01` | `0.015` | Slightly stronger production-growth feedback. |
| `denial_value_weight` | `0.01` | `0.025` | Stronger reward for reducing opponent production. |
| `fleet_airborne_weight` | `0.00002` | `0.000025` | Slightly higher cost for long/oversized launches. |
| `fleet_capture_weight` | `0.05` | `0.06` | Stronger delayed capture credit. |
| `fleet_kill_weight` | `0.001` | `0.001` | Enemy ship damage stays small. |
| `fleet_waste_weight` | `0.0015` | `0.0015` | Failed attacks are not made cheaper. |
| `enemy_capture_radius_weight` | `0.02` | `0.035` | Enemy planet capture bonus is stronger. |
| `planet_loss_weight` | `0.05` | `0.075` | Risky launches that lose source planets are punished more. |
| `defense_hold_weight` | `0.001` | `0.0015` | Successful defensive holds get slightly clearer credit. |

Usage:

```bash
python -m rl.train --reward-system aggressive
```

Individual CLI flags override the selected preset:

```bash
python -m rl.train \
  --reward-system aggressive \
  --denial-value-weight 0.03 \
  --enemy-capture-radius-weight 0.04
```

### `production_simple`

`production_simple` is a separate PPO reward system focused on the core economic objective:

```text
maximize production controlled over the rest of the game
```

Standalone reference:

```text
docs/production_simple_reward.md
```

It does not use the default per-fleet capture reward. Instead, capture reward is assigned by `ProductionRewardTracker` to the launch actions that were sufficient to take the planet.

Implemented in:

```text
rl/production_reward_tracker.py
```

Usage:

```bash
python -m rl.train --reward-system production_simple
```

This system is PPO-rollout only. It is not exposed in `rl.dqn_train`, because it depends on delayed launch-credit trackers that are integrated in the PPO rollout path.

#### Capture reward

Neutral capture:

```text
capture_reward = planet_production * (max_turns - capture_turn)
```

Enemy capture:

```text
capture_reward = planet_production
               * (max_turns - capture_turn)
               * ((num_opponents + 1) / num_opponents)
```

Opponent multiplier:

| Opponents | Multiplier |
|---:|---:|
| 1 | `2.0` |
| 2 | `1.5` |
| 3 | `1.333...` |

Reward is split across contributing launch actions by shipped-fleet ratio:

```text
launch_reward = capture_reward * launch_ships / total_contributing_ships
```

Only launches that arrived by the capture turn are counted. Friendly fleets that arrive after the planet is already captured do not receive capture reward.

Example:

```text
production = 3
capture_turn = 100
max_turns = 500
remaining_steps = 400

neutral capture reward = 3 * 400 = +1200

launch A sent 20 ships
launch B sent 30 ships
total contributing ships = 50

launch A reward = 1200 * 20 / 50 = +480
launch B reward = 1200 * 30 / 50 = +720
```

Enemy capture in a 2-player game:

```text
opponents = 1
multiplier = 2

enemy capture reward = 3 * 400 * 2 = +2400
```

#### Planet loss penalty

When a friendly planet is under attack and is later lost, the lost production potential is assigned as negative reward to eligible non-defense launch actions.

Formula:

```text
loss_penalty = -planet_production
             * (max_turns - loss_turn)
             * ((num_opponents + 1) / num_opponents)
```

Eligible launch actions:

```text
1. Launches sent from the lost planet while it was under attack.
2. Launches from other planets that could have reached the threatened planet by the loss turn, but targeted somewhere else.
```

Launches from planets that were too far away to arrive in time are not penalized.

Penalty is split by shipped-fleet ratio:

```text
launch_penalty = loss_penalty * launch_ships / total_eligible_non_defense_ships
```

Example:

```text
lost planet production = 3
loss_turn = 100
max_turns = 500
remaining_steps = 400
opponents = 1
multiplier = 2

total loss penalty = -3 * 400 * 2 = -2400

eligible launch A sent 20 ships
eligible launch B sent 30 ships
late launch C could not arrive in time
total eligible ships = 50

launch A penalty = -2400 * 20 / 50 = -960
launch B penalty = -2400 * 30 / 50 = -1440
launch C penalty = 0
```

This lets the policy learn that abandoning a threatened planet can still be correct if the alternative launch captures a more valuable production opportunity.

#### Airborne and ship combat

Airborne fleet cost:

```text
airborne_penalty = -fleet_airborne_weight * ships_sent * eta
eta = distance / fleet_speed(ships_sent)
```

Because larger fleets are faster, increasing fleet size can reduce `eta`, but sending more ships still increases the cost term.

Ship combat:

```text
own ship lost = -1
enemy ship killed = +1
neutral ship killed = 0
```

`production_simple` preset values:

```text
capture_value_weight = 0.0
denial_value_weight = 0.0
ship_diff_weight = 0.0
ship_loss_weight = 0.0
fleet_airborne_weight = 0.01
fleet_capture_weight = 0.0
fleet_kill_weight = 1.0
fleet_waste_weight = 1.0
enemy_capture_radius_weight = 0.0
planet_loss_weight = 1.0
defense_hold_weight = 0.0
```

## Step-level reward

Implemented in:

```text
rl/rewards.py::compute_step_reward
```

The function receives:

```text
prev_env_obs
curr_env_obs
player
terminal
env_terminal_reward
RewardConfig
invalid_actions
```

### Time factor

```text
raw_factor = max(0, (max_turns - current_turn) / max_turns)
time_factor = max(time_decay_floor, raw_factor)
```

Early captures can be valued more than late captures if step-level capture shaping is enabled.

### Capture-value components

The helper computes:

```text
gain   = time_factor * delta(my total planet value)
denial = time_factor * -delta(opponents' total planet value)
```

Where planet value is:

```text
production, by default
```

or:

```text
radius, if capture_value_kind="radius"
```

Examples:

| Event | `gain` | `denial` |
|---|---:|---:|
| I capture neutral | positive | 0 |
| I steal enemy planet | positive | positive |
| I lose planet to enemy | negative | negative |
| Enemy captures neutral | 0 | negative |

Current default weights are small but non-zero:

```text
capture_value_weight = 0.01
denial_value_weight = 0.01
```

So this step-level capture shaping provides immediate state-change feedback, while `FleetTracker` remains the main launch-specific delayed capture credit.

### Ship advantage delta

The code computes:

```text
d_ship_diff = ((my ships now - opp ships now) - (my ships prev - opp ships prev)) / 100
```

Then adds:

```text
ship_diff_weight * d_ship_diff
```

Default:

```text
ship_diff_weight = 0.0005
```

This is a small dense signal for improving relative ship advantage.

### Ship loss penalty

The code computes:

```text
ship_loss = max(0, my_total_ships_prev - my_total_ships_now)
```

Then subtracts:

```text
ship_loss_weight * ship_loss
```

Default:

```text
ship_loss_weight = 0.001
```

Important credit-assignment detail:

```text
This is step-level reward.
It is assigned only to transitions created on that current env step.
It is not traced backward to old launch actions.
```

Old launch actions instead receive specific delayed fleet outcome reward through `FleetTracker`.

### Invalid action penalty

Invalid actions receive an immediate transition penalty:

```text
-invalid_action_penalty
```

Default:

```text
invalid_action_penalty = 0.01
```

`compute_step_reward` also supports `invalid_actions`, but rollout currently passes `invalid_actions=0` there and applies invalid penalties directly per transition.

### Terminal reward

At terminal step:

```text
terminal_term = terminal_scale * env_terminal_reward
```

Default:

```text
terminal_scale = 1.0
```

If:

```text
shape_on_terminal = False
```

then dense shaping is suppressed on the terminal step and only terminal reward remains.

## How step-level reward is assigned in rollout

Implemented in:

```text
rl/rollout.py::collect_episode
```

After `env.step(...)`, rollout computes:

```text
rewards[p] = compute_step_reward(env_obs[p], new_obs[p], p, ...)
```

Then each player may have multiple per-source decisions in the same env step. Rollout splits the step reward across those transitions:

```text
reward_per_transition = rewards[p] / len(new_indices)
```

Then:

```text
buffer.add_reward(idx, reward_per_transition)
```

This means:

```text
Step-level reward is assigned to the current step's actions only.
No eligibility trace is used.
```

## Delayed fleet outcome reward

Implemented in:

```text
rl/fleet_tracker.py
```

Registered from rollout when a fleet launch happens:

```text
FleetTracker.register_launch(buffer_idx=idx, ...)
```

The saved `buffer_idx` is the transition index of the launch action.

Later, when the fleet disappears or arrives, the tracker resolves an outcome and rollout applies:

```text
buffer.add_reward(buffer_idx, fleet_reward)
```

Therefore:

```text
Fleet capture/kill/waste reward goes directly to the original launch action.
```

## Launch-time fleet airborne cost

Implemented in:

```text
rl/rollout.py::collect_episode
```

When a valid launch transition is created, rollout estimates the fleet ETA:

```text
eta = distance / fleet_speed(ships_sent)
```

Then it immediately applies:

```text
airborne_cost = -fleet_airborne_weight * ships_sent * eta
```

Default:

```text
fleet_airborne_weight = 0.00002
```

This is mathematically equivalent to charging:

```text
-fleet_airborne_weight * ships_sent
```

for every step the fleet is expected to remain airborne, but it is assigned immediately to the launch transition for correct credit assignment.

Example:

```text
20 ships, ETA 40
= -0.00002 * 20 * 40
= -0.016
```

The launch transition can later receive additional delayed fleet outcome reward when the fleet arrives:

```text
launch transition total contribution
= immediate airborne cost
+ delayed capture/kill/waste outcome
+ current step reward share
```

### Fleet outcome types

`FleetOutcome` contains:

| Field | Meaning |
|---|---|
| `captured_planet` | Target ownership flipped from not-mine to mine. |
| `killed_enemy_ships` | Fleet damaged an enemy garrison without capture. |
| `destroyed_enroute` | Target despawned or fleet was lost before normal capture resolution. |
| `self_reinforce` | Fleet arrived at a planet already owned by the same player. |
| `failed_neutral_attack` | Fleet hit neutral, did not capture, and was wasted. |
| `captured_from_enemy` | Captured planet was enemy-owned before arrival, not neutral. |
| `planet_production` | Production of captured planet. |
| `planet_radius` | Radius of captured planet, used for enemy capture bonus. |
| `arrival_turn` | Turn when the fleet outcome resolved. |
| `enemy_ships_killed` | Enemy garrison ships killed, bounded by ships sent. |
| `own_waste` | Own ships committed/wasted by the launch. |

### Fleet reward formula

Implemented in:

```text
compute_fleet_reward(
    outcome,
    capture_weight=0.05,
    kill_weight=0.001,
    waste_weight=0.0015,
    enemy_capture_radius_weight=0.02,
    max_turns=500,
    time_decay_floor=0.0,
)
```

These values are exposed through `RewardConfig` and `rl.train` CLI flags.

#### Capture

If fleet captures target:

```text
reward = capture_weight * planet_production
```

Default:

```text
capture_weight = 0.05
```

Example:

```text
production 3 captured -> +0.15
```

If the captured planet was enemy-owned before arrival, an extra finishing/conquest bonus is added:

```text
enemy_bonus = enemy_capture_radius_weight * planet_radius * remaining_turn_factor
remaining_turn_factor = max(time_decay_floor, (max_turns - arrival_turn) / max_turns)
```

Default:

```text
enemy_capture_radius_weight = 0.02
```

Example:

```text
enemy planet, production 3, radius 5, captured at turn 250 of 500
base capture = 0.05 * 3 = +0.15
enemy radius bonus = 0.02 * 5 * 0.5 = +0.05
total delayed capture reward = +0.20
```

Neutral captures receive only the base production capture reward.

If the capture required 20 ships with ETA 40, the same launch also received:

```text
airborne_cost = -0.016
```

So the launch-specific fleet contribution before step-level reward is:

```text
+0.15 - 0.016 = +0.134
```

#### Enemy garrison damage without capture

If target was a real opponent planet and fleet killed enemy ships but did not capture:

```text
reward = kill_weight * enemy_ships_killed - waste_weight * own_waste
```

Defaults:

```text
kill_weight = 0.001
waste_weight = 0.0015
```

Example:

```text
20 enemy ships killed, 10 own ships wasted
= +0.020 - 0.015
= +0.005
```

#### Failed neutral attack

If target was neutral and fleet did not capture:

```text
reward = -waste_weight * own_waste
```

Neutral garrison reduction is not rewarded.

Example:

```text
16 ships sent to neutral, capture failed
= -0.0015 * 16
= -0.024
```

This prevents the agent from weakening neutral planets for opponents.

#### Destroyed en route

If target despawns or fleet is otherwise lost:

```text
reward = -waste_weight * own_waste
```

#### Self reinforce

If the fleet reinforces a planet already owned by the same player:

```text
reward = 0
```

This avoids rewarding self-pinging, but does not penalize potentially useful reinforcement.

## Delayed defense hold reward

Implemented in:

```text
rl/defense_tracker.py
```

A NO-OP source decision is treated as a defensive hold.

When rollout sees a valid NO-OP action:

```text
DefenseTracker.register_hold(buffer_idx=idx, player=p, planet_id=planet_id, hold_turn=t, ships_held=ships_held)
```

The saved `buffer_idx` is the transition index of that NO-OP action.

If that planet later successfully absorbs an enemy attack, rollout applies:

```text
buffer.add_reward(buffer_idx, defense_reward)
```

Therefore:

```text
Defense reward goes directly to the original NO-OP transition.
```

### Defense eligibility

A NO-OP hold can receive defense reward only if:

```text
1. The planet was owned by the player before the step.
2. An enemy fleet was predicted to arrive within defense_arrival_window.
3. After the step, the planet is still owned by the player.
4. The planet lost ships, indicating the attack was absorbed.
5. The hold is within defense_hold_horizon.
```

If the player launches from that planet, rollout cancels pending defensive holds for that planet:

```text
DefenseTracker.cancel_planet(src_planet_id)
```

This prevents a launch action from still receiving hold-defense credit.

### Defense reward formula

Implemented in:

```text
compute_defense_reward(outcome, hold_weight=0.001)
```

Formula:

```text
reward = defense_hold_weight * ships_absorbed
```

Default:

```text
defense_hold_weight = 0.001
```

Example:

```text
Enemy sends 10 ships.
Planet absorbs attack and stays mine.
Reward = 0.001 * 10 = +0.01
```

No enemy attack means no reward. Planet lost means no reward.

## Reward scale intuition with calculated examples

The coefficients look small because most rewards are attached to a single launch or NO-OP transition. The important comparison is the relative scale between outcomes.

These examples use the current defaults:

```text
fleet_airborne_weight = 0.00002
capture_value_weight = 0.01
denial_value_weight = 0.01
fleet_capture_weight = 0.05
fleet_kill_weight = 0.001
fleet_waste_weight = 0.0015
enemy_capture_radius_weight = 0.02
planet_loss_weight = 0.05
defense_hold_weight = 0.001
max_turns = 500
time_decay_floor = 0.0
```

### Scale table

| Event | Formula | Example reward |
|---|---|---:|
| Neutral capture | `0.05 * production` | production 3 -> `+0.150` |
| Enemy capture bonus | `0.02 * radius * remaining_turn_factor` | radius 5, turn 250 -> `+0.050` |
| Airborne cost | `-0.00002 * ships * eta` | 20 ships, ETA 10 -> `-0.004` |
| Failed neutral waste | `-0.0015 * ships` | 16 ships -> `-0.024` |
| Enemy damage only | `0.001 * kills - 0.0015 * own_waste` | 20 kills, 10 own -> `+0.005` |
| Risky planet loss | `-0.05 * production` split by launch ships | production 3 -> `-0.150` total |
| Successful defense hold | `0.001 * absorbed_ships` | 12 absorbed -> `+0.012` |

### Scenario 1: neutral capture with airborne cost

```text
Turn 10:
  launch 20 ships to neutral production-3 planet
  ETA = 10

Turn 20:
  fleet captures the planet
```

Launch-time airborne cost:

```text
-fleet_airborne_weight * ships * eta
= -0.00002 * 20 * 10
= -0.004
```

Delayed capture reward:

```text
fleet_capture_weight * production
= 0.05 * 3
= +0.150
```

Total reward on the Turn 10 launch transition:

```text
+0.150 - 0.004 = +0.146
```

### Scenario 2: enemy capture beats neutral capture

```text
Turn 10:
  launch 20 ships to enemy planet
  target production = 3
  target radius = 5
  ETA = 10

Turn 250 arrival context:
  max_turns = 500
  arrival_turn = 250
  capture succeeds
```

Remaining turn factor:

```text
(max_turns - arrival_turn) / max_turns
= (500 - 250) / 500
= 0.5
```

Base capture:

```text
0.05 * 3 = +0.150
```

Enemy capture bonus:

```text
0.02 * 5 * 0.5 = +0.050
```

Airborne cost:

```text
-0.00002 * 20 * 10 = -0.004
```

Total reward on the launch transition:

```text
+0.150 + 0.050 - 0.004 = +0.196
```

Comparison:

```text
same production neutral capture = +0.146
enemy capture at turn 250    = +0.196
difference                  = +0.050
```

The difference comes from direct enemy capture bonus, not from enemy ship kills.

### Scenario 3: enemy capture late in the game

```text
target production = 3
target radius = 5
arrival_turn = 450
max_turns = 500
ships_sent = 20
ETA = 10
```

Remaining turn factor:

```text
(500 - 450) / 500 = 0.1
```

Enemy capture bonus:

```text
0.02 * 5 * 0.1 = +0.010
```

Total:

```text
base capture = +0.150
enemy bonus = +0.010
airborne cost = -0.004
total = +0.156
```

Late enemy captures are still better than neutral captures of the same production, but the extra value is lower because fewer turns remain to benefit from the stolen planet.

### Scenario 4: failed neutral attack

```text
Turn 10:
  launch 16 ships to neutral planet
  ETA = 12

Turn 22:
  fleet arrives
  neutral remains neutral
```

Airborne cost:

```text
-0.00002 * 16 * 12 = -0.00384
```

Failed neutral waste penalty:

```text
-0.0015 * 16 = -0.024
```

Total reward on launch transition:

```text
-0.024 - 0.00384 = -0.02784
```

Neutral garrison damage gives no kill reward, so the agent is not paid for weakening neutral planets for opponents.

### Scenario 5: enemy damage without capture

```text
Turn 10:
  launch 10 ships to enemy planet
  ETA = 10

Turn 20:
  fleet kills 20 enemy ships
  capture fails
```

Enemy damage reward:

```text
0.001 * 20 = +0.020
```

Own waste penalty:

```text
-0.0015 * 10 = -0.015
```

Net fleet outcome:

```text
+0.020 - 0.015 = +0.005
```

Airborne cost:

```text
-0.00002 * 10 * 10 = -0.002
```

Total reward on launch transition:

```text
+0.005 - 0.002 = +0.003
```

This is intentionally tiny compared with a real capture:

```text
enemy damage only = +0.003
neutral capture example = +0.146
enemy capture example = +0.196
```

### Scenario 6: risky launch loses the source planet

```text
Turn 40:
  source planet is mine
  source planet production = 3
  source observation shows enemy incoming

Turn 40 launches:
  launch A sends 20 ships
  launch B sends 30 ships

Turn 45:
  source planet is lost
```

Total planet-loss penalty:

```text
-planet_loss_weight * production
= -0.05 * 3
= -0.150
```

Ship-ratio split:

```text
total sent = 20 + 30 = 50
launch A share = 20 / 50 = 0.4
launch B share = 30 / 50 = 0.6
```

Assigned to original launch transitions:

```text
launch A penalty = -0.150 * 0.4 = -0.060
launch B penalty = -0.150 * 0.6 = -0.090
```

This is not assigned to the current step. It goes back to the launch actions that depleted a threatened planet.

### Scenario 7: successful NO-OP defense

```text
Turn 50:
  source chooses NO-OP
  planet has incoming enemy attack

Turn 51:
  attack arrives
  planet loses 12 ships
  planet remains mine
```

Defense reward:

```text
defense_hold_weight * absorbed_ships
= 0.001 * 12
= +0.012
```

Assigned to:

```text
Turn 50 NO-OP transition
```

### Scenario 8: combined risky decision example

```text
Turn 100:
  my production-3 planet has enemy incoming
  agent launches 30 ships away from it
  ETA of launched fleet = 15

Turn 110:
  source planet is lost

Turn 115:
  launched fleet reaches neutral production-2 planet and captures it
```

Airborne cost:

```text
-0.00002 * 30 * 15 = -0.009
```

Neutral capture reward:

```text
0.05 * 2 = +0.100
```

Planet-loss penalty:

```text
-0.05 * 3 = -0.150
```

Total on the original launch transition:

```text
+0.100 - 0.009 - 0.150 = -0.059
```

Interpretation:

```text
Capturing a small neutral was not worth abandoning a threatened production-3 source planet.
```

## Deprecated eligibility trace behavior

The following config fields remain for compatibility:

```text
eligibility_trace_horizon = 0
eligibility_trace_decay = 0.0
noop_trace_weight = 0.0
```

Current behavior:

```text
Step rewards are not spread backward.
NO-OP trace leakage is disabled.
Delayed action-specific rewards are assigned by trackers to the original transition.
```

## Reward assignment examples

### Successful neutral capture

```text
Turn 10: launch 20 ships to neutral production-3 planet.
Turn 30: fleet arrives and captures.
```

Reward assignment:

```text
+0.15 goes to the Turn 10 launch transition.
```

Because:

```text
0.05 * production 3 = 0.15
```

### Failed neutral attack

```text
Turn 10: launch 16 ships to neutral planet with too many defenders.
Turn 30: fleet arrives, neutral remains neutral, garrison reduced.
```

Reward assignment:

```text
-0.024 goes to the Turn 10 launch transition.
```

Because:

```text
-0.0015 * 16 = -0.024
```

Plus any launch-time airborne cost:

```text
-0.00002 * 16 * ETA
```

No neutral kill reward is given.

### Enemy attack damage without capture

```text
Turn 10: launch 10 ships to enemy planet.
Turn 30: fleet kills 20 enemy ships but does not capture.
```

Reward assignment:

```text
+0.005 goes to the Turn 10 launch transition.
```

Because:

```text
0.001 * 20 - 0.0015 * 10 = +0.005
```

### Successful NO-OP defense

```text
Turn 50: planet chooses NO-OP.
Enemy fleet arrives.
Planet loses 12 ships but remains owned by player.
```

Reward assignment:

```text
+0.012 goes to the Turn 50 NO-OP transition.
```

Because:

```text
0.001 * 12 = +0.012
```

### Planet loss after risky launch

Planet loss is not added as a generic step-level penalty.

Instead, if a planet is lost after the agent launched ships from that planet while the source planet observation already showed enemy incoming fleets, the production-loss penalty is assigned back to those launch transitions.

Implemented in:

```text
rl/planet_loss_tracker.py
```

Formula:

```text
total_penalty = -planet_loss_weight * lost_planet_production
launch_penalty = total_penalty * ships_sent_by_launch / total_ships_sent_from_that_planet
```

Defaults:

```text
planet_loss_weight = 0.05
planet_loss_credit_horizon = 30
```

Example:

```text
lost planet production = 3
planet_loss_weight = 0.05
total penalty = -0.15

launch A sent 20 ships
launch B sent 30 ships
total sent = 50

launch A penalty = -0.15 * 20 / 50 = -0.06
launch B penalty = -0.15 * 30 / 50 = -0.09
```

If there was no enemy incoming in the source planet observation at launch time, the launch does not receive this planet-loss penalty.

## Training CLI reward flags

Defined in:

```text
rl/train.py
```

Available reward-related flags:

```bash
--reward-system current
--capture-value-weight 0.01
--denial-value-weight 0.01
--capture-value-kind production
--max-turns 500
--time-decay-floor 0.0
--planet-diff-weight 0.0
--ship-diff-weight 0.0005
--ship-loss-weight 0.001
--fleet-airborne-weight 0.00002
--fleet-capture-weight 0.05
--fleet-kill-weight 0.001
--fleet-waste-weight 0.0015
--enemy-capture-radius-weight 0.02
--planet-loss-weight 0.05
--planet-loss-credit-horizon 30
--defense-hold-weight 0.001
--defense-hold-horizon 15
--defense-arrival-window 1.5
--invalid-action-penalty 0.01
--eligibility-trace-horizon 0
--eligibility-trace-decay 0.0
--noop-trace-weight 0.0
```

Example current recommended training command:

```bash
orbit_wars_venv/bin/python -m rl.train \
  --device cuda \
  --checkpoint-dir checkpoints/planet_bucket_defense_waste_v1 \
  --log-path logs/planet_bucket_defense_waste_v1.jsonl \
  --model-variant planet_only \
  --episodes-per-iter 1 \
  --iterations 1000
```

Example with stronger defense hold reward:

```bash
orbit_wars_venv/bin/python -m rl.train \
  --device cuda \
  --checkpoint-dir checkpoints/planet_bucket_defense_waste_v2 \
  --log-path logs/planet_bucket_defense_waste_v2.jsonl \
  --model-variant planet_only \
  --episodes-per-iter 1 \
  --iterations 1000 \
  --defense-hold-weight 0.0015
```

Example with step-level planet loss/capture shaping re-enabled:

```bash
orbit_wars_venv/bin/python -m rl.train \
  --device cuda \
  --checkpoint-dir checkpoints/planet_bucket_shaping_v1 \
  --log-path logs/planet_bucket_shaping_v1.jsonl \
  --model-variant planet_only \
  --episodes-per-iter 1 \
  --iterations 1000 \
  --capture-value-weight 0.01 \
  --denial-value-weight 0.01
```

Use this carefully because fleet capture reward already gives direct delayed capture credit.

## Known caveats

### Step-level ship loss has broad current-step assignment

`ship_loss_weight` is step-level. It is split across current step transitions, not assigned to old launch transitions.

For launch-specific credit, rely on `FleetTracker` outcome rewards.

### Planet-only observation has implicit outcome observation

The agent observes capture/loss outcomes through changed planet owner/ships/features in subsequent observations, and through reward. It does not receive an explicit `my_previous_fleet_captured_target` flag.

## Recommended interpretation

The current reward design tries to enforce these behaviors:

- **Capture valuable planets:** positive delayed reward to the launch action.
- **Avoid failed neutral weakening:** negative delayed reward to the launch action.
- **Damage real enemies only when worthwhile:** enemy kill reward netted against own waste.
- **Do not reward passive waiting:** NO-OP defense reward only when an enemy attack is actually absorbed.
- **Keep credit assignment local:** no eligibility traces; delayed rewards go directly to causative transitions.
