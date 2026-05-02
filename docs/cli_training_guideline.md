---
title: CLI Training Guideline
---

# Orbit Wars CLI Training Guideline

## New GPU training run

Use a new checkpoint directory and log file so older runs are not overwritten.

```bash
orbit_wars_venv/bin/python -m rl.train \
  --device cuda \
  --checkpoint-dir checkpoints/planet_bucket_defense_v1 \
  --log-path logs/planet_bucket_defense_v1.jsonl \
  --model-variant planet_only \
  --episodes-per-iter 1 \
  --iterations 1000
```

## New CPU training run

```bash
orbit_wars_venv/bin/python -m rl.train \
  --device cpu \
  --checkpoint-dir checkpoints/planet_bucket_defense_v1 \
  --log-path logs/planet_bucket_defense_v1.jsonl \
  --model-variant planet_only \
  --episodes-per-iter 1 \
  --iterations 1000
```

## Resume an existing run

Resume from a checkpoint and keep writing into the same run folder.

```bash
orbit_wars_venv/bin/python -m rl.train \
  --device cuda \
  --resume-from checkpoints/planet_bucket_defense_v1/latest.pt \
  --checkpoint-dir checkpoints/planet_bucket_defense_v1 \
  --log-path logs/planet_bucket_defense_v1.jsonl \
  --model-variant planet_only \
  --episodes-per-iter 1 \
  --iterations 1000
```

## Watch the latest checkpoint as HTML replay

```bash
orbit_wars_venv/bin/python -m rl.watch \
  --device cuda \
  --ckpt checkpoints/planet_bucket_defense_v1/latest.pt \
  --html replays/planet_bucket_defense_v1_latest.html
```

Open the replay:

```bash
xdg-open replays/planet_bucket_defense_v1_latest.html
```

## Recommended run naming

Use a unique name per experiment:

```text
checkpoints/<experiment_name>/
logs/<experiment_name>.jsonl
replays/<experiment_name>_latest.html
```

Examples:

```text
planet_bucket_v1
planet_bucket_defense_v1
planet_bucket_loss_v1
planet_bucket_edge_v1
```

## Useful reward flags

```bash
--ship-loss-weight 0.001 \
--defense-hold-weight 0.001 \
--defense-hold-horizon 15 \
--defense-arrival-window 1.5
```

## Notes

- Use `--device cuda` for GPU training.
- Use a fresh `--checkpoint-dir` to avoid overwriting older experiments.
- Run as a module with `-m rl.train` or `-m rl.watch` to avoid relative import issues.
- The current recommended PPO model is `--model-variant planet_only`.
- `--episodes-per-iter 1` updates after every completed game.
