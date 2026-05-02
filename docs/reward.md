# Orbit Wars Reward Reference

This document is the short reference for the active RL reward design. The full detailed document is `docs/reward_system.md`.

The standalone documentation for the new `production_simple` system is `docs/production_simple_reward.md`.

## Core principle

The reward should stay simple and should reward the desired strategic outcome directly:

```text
Neutral planet capture = base capture reward
Enemy planet capture = base capture reward + enemy capture bonus
```

The difference between neutral and enemy captures must not come mainly from enemy ship damage. Damaging an enemy garrison without capturing can still receive a small delayed signal, but it is not the primary incentive.

## Launch-time fleet cost

When a fleet is launched, it immediately receives an airborne cost on the launch transition:

```text
airborne_cost = -fleet_airborne_weight * ships_sent * eta
eta = distance / fleet_speed(ships_sent)
```

Default:

```text
fleet_airborne_weight = 0.00002
```

This discourages sending large fleets on long, low-impact trips.

## Delayed fleet reward

Fleet outcome rewards are assigned back to the original launch transition by `FleetTracker`.

### Neutral capture

If the fleet captures a neutral planet:

```text
reward = fleet_capture_weight * planet_production
```

Default:

```text
fleet_capture_weight = 0.05
```

Example:

```text
production = 3
reward = 0.05 * 3 = +0.15
```

### Enemy capture

If the fleet captures a planet that was enemy-owned before arrival:

```text
reward = fleet_capture_weight * planet_production
       + enemy_capture_radius_weight * planet_radius * remaining_turn_factor
```

Where:

```text
remaining_turn_factor = max(time_decay_floor, (max_turns - arrival_turn) / max_turns)
```

Defaults:

```text
enemy_capture_radius_weight = 0.02
max_turns = 500
time_decay_floor = 0.0
```

Example:

```text
production = 3
radius = 5
arrival_turn = 250
max_turns = 500

base = 0.05 * 3 = +0.15
bonus = 0.02 * 5 * 0.5 = +0.05
total = +0.20
```

This makes taking enemy planets more valuable than taking neutral planets without adding a complex “being ahead” reward.

### Enemy garrison damage without capture

If a fleet attacks an enemy planet, kills ships, but does not capture:

```text
reward = fleet_kill_weight * enemy_ships_killed - fleet_waste_weight * own_waste
```

Defaults:

```text
fleet_kill_weight = 0.001
fleet_waste_weight = 0.0015
```

This is intentionally small. It should not dominate the capture reward.

### Failed neutral attack

If a fleet attacks a neutral planet and fails to capture:

```text
reward = -fleet_waste_weight * own_waste
```

Neutral ship damage does not get kill reward.

### Self reinforce

Sending ships to a planet already owned by the same player gives no capture or kill reward:

```text
reward = 0
```

## Planet loss after risky launch

Planet loss is not added as a generic step-level penalty.

Instead, if a planet is lost after the agent launched ships from that planet while the source planet observation already showed enemy incoming fleets, the production-loss penalty is assigned back to those launch transitions.

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

## Step-level rewards

Step-level dense shaping is active at a small scale by default. It gives immediate state-change feedback, while delayed fleet trackers remain the main launch-specific credit signal.

Important defaults:

```text
capture_value_weight = 0.01
denial_value_weight = 0.01
planet_diff_weight = 0.0
noop_trace_weight = 0.0
```

Ship-level auxiliary signals remain:

```text
ship_diff_weight = 0.0005
ship_loss_weight = 0.001
invalid_action_penalty = 0.01
```

## Reward system factory

Reward configs are created with `build_reward_config`.

Available systems:

```text
current
aggressive
production_simple
```

`current` preserves the normal default behavior:

```bash
python -m rl.train --reward-system current
```

`aggressive` is a separate preset that increases opponent-capture incentives without increasing enemy ship kill reward:

```text
capture_value_weight = 0.015
denial_value_weight = 0.025
fleet_capture_weight = 0.06
fleet_kill_weight = 0.001
enemy_capture_radius_weight = 0.035
planet_loss_weight = 0.075
```

Usage:

```bash
python -m rl.train --reward-system aggressive
```

Individual reward flags override the selected preset.

`production_simple` is a PPO-only economic reward system:

```bash
python -m rl.train --reward-system production_simple
```

Core capture reward:

```text
reward = planet_production * (max_turns - capture_turn)
```

Enemy capture multiplier:

```text
multiplier = (num_opponents + 1) / num_opponents
```

Examples:

```text
production = 3
capture_turn = 100
max_turns = 500

neutral capture = 3 * 400 = +1200
enemy capture with 1 opponent = 3 * 400 * 2 = +2400
enemy capture with 2 opponents = 3 * 400 * 1.5 = +1800
```

Capture reward is split only across launch actions that arrived by the capture turn:

```text
launch_reward = reward * launch_ships / total_contributing_ships
```

Planet loss penalty mirrors capture value:

```text
penalty = -planet_production
        * (max_turns - loss_turn)
        * ((num_opponents + 1) / num_opponents)
```

It is split across non-defense launch actions that either came from the lost planet or could have reached the threatened planet in time but targeted elsewhere. Launches too far away to arrive in time are not penalized.

Ship combat in `production_simple`:

```text
own ship lost = -1
enemy ship killed = +1
neutral ship killed = 0
```

## Defense hold reward

NO-OP defensive holds can receive delayed reward if they successfully absorb incoming enemy pressure:

```text
reward = defense_hold_weight * absorbed_enemy_ships
```

Default:

```text
defense_hold_weight = 0.001
```

## Calculated reward examples

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
```

### Example 1: neutral capture, short route

```text
ships_sent = 20
eta = 10
target = neutral planet
target production = 3
capture succeeds
```

Launch-time airborne cost:

```text
-0.00002 * 20 * 10 = -0.004
```

Delayed capture reward:

```text
0.05 * 3 = +0.150
```

Total assigned to the launch transition:

```text
+0.150 - 0.004 = +0.146
```

### Example 2: enemy capture is better than neutral capture

```text
ships_sent = 20
eta = 10
target = enemy planet
target production = 3
target radius = 5
arrival_turn = 250
max_turns = 500
capture succeeds
```

Remaining turn factor:

```text
(500 - 250) / 500 = 0.5
```

Launch-time airborne cost:

```text
-0.00002 * 20 * 10 = -0.004
```

Base capture reward:

```text
0.05 * 3 = +0.150
```

Enemy capture bonus:

```text
0.02 * 5 * 0.5 = +0.050
```

Total assigned to the launch transition:

```text
+0.150 + 0.050 - 0.004 = +0.196
```

The same production-3 neutral capture was `+0.146`, while this enemy capture is `+0.196`.

### Example 3: enemy capture late game

```text
ships_sent = 20
eta = 10
target = enemy planet
target production = 3
target radius = 5
arrival_turn = 450
max_turns = 500
capture succeeds
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

Enemy capture is still better than neutral capture, but the extra bonus is smaller late in the game.

### Example 4: failed neutral attack

```text
ships_sent = 16
eta = 12
target = neutral planet
capture fails
```

Launch-time airborne cost:

```text
-0.00002 * 16 * 12 = -0.00384
```

Failed neutral waste penalty:

```text
-0.0015 * 16 = -0.024
```

Total assigned to the launch transition:

```text
-0.024 - 0.00384 = -0.02784
```

Neutral ship damage gives no kill reward.

### Example 5: enemy damage without capture is intentionally small

```text
ships_sent = 10
eta = 10
enemy_ships_killed = 20
capture fails
```

Launch-time airborne cost:

```text
-0.00002 * 10 * 10 = -0.002
```

Enemy damage reward net of waste:

```text
0.001 * 20 - 0.0015 * 10 = +0.020 - 0.015 = +0.005
```

Total assigned to the launch transition:

```text
+0.005 - 0.002 = +0.003
```

This is much smaller than capturing the planet. It should not teach the agent to only weaken enemies.

### Example 6: risky launch causes planet loss

```text
source planet production = 3
source planet had enemy incoming in observation
launch A sent 20 ships
launch B sent 30 ships
planet is lost within planet_loss_credit_horizon
```

Total planet-loss penalty:

```text
-0.05 * 3 = -0.150
```

Penalty split by ships sent:

```text
total sent = 20 + 30 = 50
launch A share = 20 / 50 = 0.4
launch B share = 30 / 50 = 0.6
```

Assigned penalties:

```text
launch A = -0.150 * 0.4 = -0.060
launch B = -0.150 * 0.6 = -0.090
```

This is not a step-level penalty. It is assigned directly to the launch transitions that depleted a threatened planet.

### Example 7: successful NO-OP defense

```text
NO-OP hold on owned planet
enemy attack arrives
planet absorbs 12 enemy ships
planet remains owned
```

Defense hold reward:

```text
0.001 * 12 = +0.012
```

This reward goes to the original NO-OP transition.

### Quick scale comparison

| Scenario | Approx reward |
|---|---:|
| Neutral production-3 capture, ETA 10, 20 ships | `+0.146` |
| Enemy production-3 radius-5 capture at turn 250, ETA 10, 20 ships | `+0.196` |
| Enemy production-3 radius-5 capture at turn 450, ETA 10, 20 ships | `+0.156` |
| Failed neutral attack, 16 ships, ETA 12 | `-0.02784` |
| Enemy damage only: 20 kills with 10 ships, ETA 10 | `+0.003` |
| Losing production-3 planet after risky 20/30 ship launches | `-0.060`, `-0.090` |
| Successful NO-OP defense absorbing 12 ships | `+0.012` |

## CLI flags

Current relevant training flags:

```bash
--reward-system current
--capture-value-weight 0.01
--denial-value-weight 0.01
--fleet-airborne-weight 0.00002
--fleet-capture-weight 0.05
--fleet-kill-weight 0.001
--fleet-waste-weight 0.0015
--enemy-capture-radius-weight 0.02
--planet-loss-weight 0.05
--planet-loss-credit-horizon 30
--max-turns 500
--time-decay-floor 0.0
--defense-hold-weight 0.001
--invalid-action-penalty 0.01
```

## Recommended behavior encouraged

The active reward setup encourages:

```text
1. Capture planets instead of only damaging them.
2. Prefer enemy captures over neutral captures when strategically possible.
3. Finish opponents by taking their planets.
4. Avoid wasting fleets on failed neutral attacks.
5. Avoid sending oversized fleets on long airborne routes.
```
