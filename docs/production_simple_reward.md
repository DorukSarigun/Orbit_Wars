# Production Simple Reward System

`production_simple` is a separate PPO reward system focused on the core economic objective of Orbit Wars:

```text
maximize production controlled over the rest of the game
```

It is selected with:

```bash
python -m rl.train --reward-system production_simple
```

This system is PPO-only. It is not exposed in `rl.dqn_train`, because it relies on delayed launch-credit trackers integrated in the PPO rollout path.

## Core idea

The value of a planet is the amount of production it can generate before the game ends.

```text
production_value = planet_production * remaining_steps
remaining_steps = max_turns - event_turn
```

With the default game length:

```text
max_turns = 500
```

Example:

```text
planet production = 3
capture_turn = 100
remaining_steps = 500 - 100 = 400

production value = 3 * 400 = +1200
```

## Reward system selection

Available PPO reward systems:

```text
current
aggressive
production_simple
```

Use this system with:

```bash
python -m rl.train --reward-system production_simple
```

`production_simple` disables step-level production shaping and uses delayed launch-specific credit instead.

```text
capture_value_weight = 0.0
denial_value_weight = 0.0
ship_diff_weight = 0.0
ship_loss_weight = 0.0
```

## Capture reward

### Neutral planet capture

Neutral capture reward:

```text
capture_reward = planet_production * (max_turns - capture_turn)
```

Example:

```text
planet production = 3
capture_turn = 100
max_turns = 500
remaining_steps = 400

capture_reward = 3 * 400 = +1200
```

### Enemy planet capture

Enemy capture is more valuable than neutral capture.

Formula:

```text
capture_reward = planet_production
               * (max_turns - capture_turn)
               * enemy_capture_multiplier
```

Where:

```text
enemy_capture_multiplier = (num_opponents + 1) / num_opponents
```

Multiplier table:

| Number of opponents | Multiplier |
|---:|---:|
| 1 | `2.0` |
| 2 | `1.5` |
| 3 | `1.333...` |

Example with 1 opponent:

```text
planet production = 3
capture_turn = 100
max_turns = 500
remaining_steps = 400
num_opponents = 1
multiplier = 2.0

capture_reward = 3 * 400 * 2.0 = +2400
```

Example with 2 opponents:

```text
planet production = 3
capture_turn = 100
max_turns = 500
remaining_steps = 400
num_opponents = 2
multiplier = 1.5

capture_reward = 3 * 400 * 1.5 = +1800
```

## Capture reward distribution

Capture reward is not assigned to the current step generically. It is assigned back to the launch actions that contributed to the capture.

Distribution formula:

```text
launch_reward = capture_reward * launch_ships / total_contributing_ships
```

Only launches that arrived by the capture turn are counted.

Friendly fleets that arrive after the planet is already captured do not receive capture reward.

Example:

```text
neutral capture reward = +1200

launch A sent 20 ships and arrived by capture
launch B sent 30 ships and arrived by capture
launch C sent 100 ships but arrives after capture

total_contributing_ships = 20 + 30 = 50

launch A reward = 1200 * 20 / 50 = +480
launch B reward = 1200 * 30 / 50 = +720
launch C reward = 0
```

## Planet loss penalty

If one of our planets is under attack and we lose it, the lost production potential is assigned as negative reward.

Formula:

```text
loss_penalty = -planet_production
             * (max_turns - loss_turn)
             * enemy_capture_multiplier
```

The same opponent-count multiplier is used:

```text
enemy_capture_multiplier = (num_opponents + 1) / num_opponents
```

Example with 1 opponent:

```text
lost planet production = 3
loss_turn = 100
max_turns = 500
remaining_steps = 400
num_opponents = 1
multiplier = 2.0

loss_penalty = -3 * 400 * 2.0 = -2400
```

## Planet loss penalty distribution

The penalty is assigned to launch actions that failed to defend the threatened planet.

Eligible launch actions:

```text
1. Launches sent from the lost planet while it was under attack.
2. Launches from other planets that could have reached the threatened planet by the loss turn but targeted somewhere else.
```

Launches from planets too far away to arrive in time are not penalized.

Distribution formula:

```text
launch_penalty = loss_penalty * launch_ships / total_eligible_non_defense_ships
```

Example:

```text
loss_penalty = -2400

eligible launch A sent 20 ships
eligible launch B sent 30 ships
late launch C sent 100 ships but could not arrive in time

total_eligible_non_defense_ships = 20 + 30 = 50

launch A penalty = -2400 * 20 / 50 = -960
launch B penalty = -2400 * 30 / 50 = -1440
launch C penalty = 0
```

This allows the agent to learn that not defending can still be correct if another launch captures a planet with greater production value.

## Airborne fleet penalty

Fleets are penalized while they are airborne because ships in transit are temporarily idle.

Formula:

```text
airborne_penalty = -fleet_airborne_weight * ships_sent * eta
eta = distance / fleet_speed(ships_sent)
```

Default in `production_simple`:

```text
fleet_airborne_weight = 0.01
```

This creates the intended tradeoff:

```text
ships_sent increases -> penalty increases
fleet_speed increases -> eta decreases -> penalty decreases
```

Because larger fleets move faster, sending a larger fleet can reduce travel time, but it is still not free.

Example:

```text
ships_sent = 30
eta = 15
fleet_airborne_weight = 0.01

airborne_penalty = -0.01 * 30 * 15 = -4.5
```

## Ship combat reward

Ship combat is deliberately simple.

```text
own ship lost = -1
enemy ship killed = +1
neutral ship killed = 0
```

Examples:

```text
10 own ships lost = -10
20 enemy ships killed = +20
15 neutral ships killed = 0
```

For enemy combat:

```text
net_combat_reward = enemy_ships_killed - own_ships_lost
```

For neutral failed attacks, only own ship loss is penalized. Neutral garrison damage does not produce positive reward.

## Preset values

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
max_turns = 500
```

## Implementation files

Main implementation:

```text
rl/production_reward_tracker.py
```

Rollout integration:

```text
rl/rollout.py
```

Factory preset:

```text
rl/rewards.py::build_reward_config
```

Fleet combat helper:

```text
rl/fleet_tracker.py::compute_production_simple_fleet_combat_reward
```

## Recommended training command

Use `cuda`, not `gpu`, if you want GPU training with PyTorch.

GPU command:

```bash
orbit_wars_venv/bin/python -m rl.train \
  --reward-system production_simple \
  --model-variant planet_only \
  --iterations 1000 \
  --episodes-per-iter 4 \
  --max-episode-steps 500 \
  --num-agents 2 \
  --batch-size 512 \
  --ppo-epochs 4 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --lr 3e-4 \
  --entropy-coef 0.01 \
  --checkpoint-every 25 \
  --checkpoint-dir checkpoints_production_simple \
  --log-path checkpoints_production_simple/train_log.jsonl \
  --device cuda
```

CPU command:

```bash
orbit_wars_venv/bin/python -m rl.train \
  --reward-system production_simple \
  --model-variant planet_only \
  --iterations 1000 \
  --episodes-per-iter 4 \
  --max-episode-steps 500 \
  --num-agents 2 \
  --batch-size 512 \
  --ppo-epochs 4 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --lr 3e-4 \
  --entropy-coef 0.01 \
  --checkpoint-every 25 \
  --checkpoint-dir checkpoints_production_simple \
  --log-path checkpoints_production_simple/train_log.jsonl \
  --device cpu
```

Short smoke test:

```bash
orbit_wars_venv/bin/python -m rl.train \
  --reward-system production_simple \
  --model-variant planet_only \
  --iterations 10 \
  --episodes-per-iter 1 \
  --max-episode-steps 500 \
  --num-agents 2 \
  --batch-size 256 \
  --ppo-epochs 4 \
  --gamma 0.995 \
  --gae-lambda 0.95 \
  --lr 3e-4 \
  --entropy-coef 0.01 \
  --checkpoint-every 5 \
  --checkpoint-dir checkpoints_production_simple_smoke \
  --log-path checkpoints_production_simple_smoke/train_log.jsonl \
  --device cpu
```
