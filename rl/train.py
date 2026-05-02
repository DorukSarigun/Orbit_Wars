"""Main training loop for OrbitWarsPolicy via self-play PPO.

Usage:
    python -m rl.train --iterations 100 --episodes-per-iter 4

Each iteration:
    1. Roll out N self-play episodes -> rollout buffer
    2. Compute GAE advantages
    3. PPO update (K epochs)
    4. Log metrics, optionally checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch

from model import OrbitWarsPolicy
from model_planet_only import PlanetOnlyOrbitWarsPolicy

from .buffer import RolloutBuffer
from .metrics import (
    WeightSnapshot, grad_norms_by_module,
    round_metrics, weight_norms_by_module,
)
from .ppo import PPOConfig, ppo_update
from .rewards import REWARD_SYSTEMS, RewardConfig, build_reward_config
from .rollout import collect_episode


@dataclass
class TrainConfig:
    iterations: int = 100
    episodes_per_iter: int = 1
    max_episode_steps: int = 500
    num_agents: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    seed: int = 0
    checkpoint_every: int = 10
    checkpoint_dir: str = "checkpoints"
    log_path: str = "checkpoints/train_log.jsonl"
    device: str = "cpu"
    # Model arch (small by default; bump for serious training)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    model_variant: str = "planet_only"


def _make_policy(variant: str, **kwargs) -> OrbitWarsPolicy:
    if variant == "planet_only":
        return PlanetOnlyOrbitWarsPolicy(**kwargs)
    if variant == "full":
        return OrbitWarsPolicy(**kwargs)
    raise ValueError(f"Unknown model_variant={variant!r}")


def _save_ckpt(policy: OrbitWarsPolicy, optimizer, path: Path, *, iteration: int,
               train_cfg: TrainConfig, ppo_cfg: PPOConfig, reward_cfg: RewardConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": iteration,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_cfg": asdict(train_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "reward_cfg": asdict(reward_cfg),
        # Architecture is needed to rebuild the model on load.
        "arch": {
            "p_max": policy.p_max,
            "f_max": policy.f_max,
            "planet_dim": policy.planet_dim,
            "d_model": policy.d_model,
            "n_heads": policy.n_heads,
            "n_layers": len(policy.blocks),
            "n_ship_bins": policy.n_ship_bins,
            "model_variant": getattr(policy, "model_variant", "full"),
        },
    }, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple:
    """Load a checkpoint and rebuild policy. Returns (policy, ckpt_dict)."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    planet_dim = arch.get("planet_dim")
    if planet_dim is None:
        state = ckpt["policy_state_dict"]
        planet_dim = int(state["planet_proj.weight"].shape[1])
    policy = _make_policy(
        arch.get("model_variant", "full"),
        p_max=arch["p_max"], f_max=arch["f_max"], planet_dim=planet_dim,
        d_model=arch["d_model"], n_heads=arch["n_heads"],
        n_layers=arch["n_layers"], n_ship_bins=arch["n_ship_bins"],
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    return policy, ckpt


def train(
    train_cfg: Optional[TrainConfig] = None,
    ppo_cfg: Optional[PPOConfig] = None,
    reward_cfg: Optional[RewardConfig] = None,
    *,
    resume_from: Optional[str] = None,
) -> OrbitWarsPolicy:
    train_cfg = train_cfg or TrainConfig()
    ppo_cfg = ppo_cfg or PPOConfig()
    reward_cfg = reward_cfg or RewardConfig()

    torch.manual_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    if resume_from is not None:
        policy, ckpt = load_checkpoint(resume_from, device=device)
        start_iter = int(ckpt.get("iteration", 0)) + 1
        print(f"Resumed from {resume_from} at iter {start_iter}")
    else:
        policy = _make_policy(
            train_cfg.model_variant,
            d_model=train_cfg.d_model,
            n_heads=train_cfg.n_heads,
            n_layers=train_cfg.n_layers,
        ).to(device)
        start_iter = 0

    optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.learning_rate)
    n_params = sum(p.numel() for p in policy.parameters())
    variant = getattr(policy, "model_variant", "full")
    token_count = policy.p_max + 1 if variant == "planet_only" else policy.p_max + policy.f_max + 1
    print(f"Policy[{variant}]: {n_params:,} params, T={token_count}, device={device}")

    log_path = Path(train_cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for it in range(start_iter, train_cfg.iterations):
        t0 = time.time()
        buffer = RolloutBuffer()
        ep_stats_list = []
        for ep in range(train_cfg.episodes_per_iter):
            seed = train_cfg.seed * 100000 + it * 1000 + ep
            stats = collect_episode(
                policy, buffer,
                max_steps=train_cfg.max_episode_steps,
                num_agents=train_cfg.num_agents,
                reward_cfg=reward_cfg,
                device=device,
                seed=seed,
            )
            ep_stats_list.append(stats)

        # Bootstrap value for GAE: 0 since each episode ends terminally.
        # (collect_episode already sets done=True at the last stored step.)
        buffer.compute_gae(last_value=0.0,
                           gamma=train_cfg.gamma, lam=train_cfg.gae_lambda)
        # Snapshot weights before PPO so we can compute per-module update norms.
        snap = WeightSnapshot(policy)
        ppo_stats = ppo_update(policy, buffer, optimizer, ppo_cfg, device=device)
        # Per-module post-update stats.
        weight_norms = weight_norms_by_module(policy)
        update_norms = snap.compute_update_norms(policy)
        update_ratios = snap.compute_update_ratios(policy)
        current_lr = optimizer.param_groups[0]["lr"]

        wins_by_player = {
            f"p{p}": sum(1 for s in ep_stats_list if s.winner == p)
            for p in range(train_cfg.num_agents)
        }
        wins_p0 = wins_by_player.get("p0", 0)
        wins_p1 = wins_by_player.get("p1", 0)
        draws = sum(1 for s in ep_stats_list if s.winner == -1)
        avg_len = sum(s.length for s in ep_stats_list) / len(ep_stats_list)
        avg_launches = sum(sum(s.n_launches) / max(1, len(s.n_launches))
                           for s in ep_stats_list) / len(ep_stats_list)
        elapsed = time.time() - t0

        log = {
            "iter": it,
            "episodes": train_cfg.episodes_per_iter,
            "transitions": len(buffer),
            "avg_episode_length": round(avg_len, 1),
            "avg_launches_per_player": round(avg_launches, 1),
            "wins_p0": wins_p0, "wins_p1": wins_p1, "draws": draws,
            "wins_by_player": wins_by_player,
            "policy_loss": round(ppo_stats.policy_loss, 5),
            "value_loss": round(ppo_stats.value_loss, 5),
            "entropy": round(ppo_stats.entropy, 4),
            "approx_kl": round(ppo_stats.approx_kl, 5),
            "clip_fraction": round(ppo_stats.clip_fraction, 4),
            "elapsed_sec": round(elapsed, 1),
            # Hyperparameters in effect this iter.
            "learning_rate": current_lr,
            # Per-module training health metrics.
            "grad_norms": round_metrics(ppo_stats.last_grad_norms or {}, 4),
            "weight_norms": round_metrics(weight_norms, 3),
            "update_norms": round_metrics(update_norms, 5),
            "update_ratios": round_metrics(update_ratios, 6),
        }
        # Mean over update ratios is a one-number sanity check for lr scaling.
        ur_vals = list(update_ratios.values())
        ur_mean = sum(ur_vals) / max(1, len(ur_vals))
        wins_str = " ".join(f"P{p}={wins_by_player.get(f'p{p}', 0)}"
                            for p in range(train_cfg.num_agents))
        print(f"[epoch {it:>4}/{train_cfg.iterations}] "
              f"games={train_cfg.episodes_per_iter} "
              f"len={log['avg_episode_length']:>5.1f} "
              f"launch/p={log['avg_launches_per_player']:>4.1f} "
              f"{wins_str} D={draws} | "
              f"polL={log['policy_loss']:+.4f} valL={log['value_loss']:.3f} "
              f"H={log['entropy']:.3f} KL={log['approx_kl']:.4f} "
              f"clip={log['clip_fraction']:.2f} ur={ur_mean:.2e} "
              f"({elapsed:.1f}s)")

        with log_path.open("a") as fp:
            fp.write(json.dumps(log) + "\n")

        if (it + 1) % train_cfg.checkpoint_every == 0 or it == train_cfg.iterations - 1:
            ckpt_path = Path(train_cfg.checkpoint_dir) / f"policy_iter{it:05d}.pt"
            _save_ckpt(policy, optimizer, ckpt_path,
                       iteration=it, train_cfg=train_cfg,
                       ppo_cfg=ppo_cfg, reward_cfg=reward_cfg)
            latest = Path(train_cfg.checkpoint_dir) / "latest.pt"
            _save_ckpt(policy, optimizer, latest,
                       iteration=it, train_cfg=train_cfg,
                       ppo_cfg=ppo_cfg, reward_cfg=reward_cfg)
            print(f"  checkpoint saved -> {ckpt_path}")

    return policy


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OrbitWars self-play PPO")
    p.add_argument("--iterations", "--epochs", type=int, default=100,
                   dest="iterations",
                   help="Number of training epochs (rollout + update cycles)")
    p.add_argument("--episodes-per-iter", "--games-per-epoch", type=int, default=1,
                   dest="episodes_per_iter",
                   help="Self-play games per epoch (rolled into one update)")
    p.add_argument("--max-episode-steps", type=int, default=500,
                   help="Per-game turn cap (env's natural max is ~500)")
    p.add_argument("--num-agents", type=int, choices=(2, 4), default=2,
                   help="Number of self-play agents in each game")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--log-path", default="checkpoints/train_log.jsonl")
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--model-variant", choices=("full", "planet_only"), default="planet_only")
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4, dest="epochs",
                   help="Number of optimization epochs over each rollout buffer "
                        "(inner PPO update; not the outer training-epoch count)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--reward-system", choices=REWARD_SYSTEMS, default="current",
                   help="Reward preset to start from; individual reward flags override it.")
    p.add_argument("--capture-value-weight", type=float, default=None,
                   help="Reward for MY production growth (per planet captured). "
                        "Small default keeps state-change feedback active while "
                        "FleetTracker remains the main launch-specific capture credit.")
    p.add_argument("--denial-value-weight", type=float, default=None,
                   help="Reward for OPP's production loss (per planet they lose). "
                        "Equal to capture-value-weight gives symmetric (mine - opp) "
                        "swing; >capture-value-weight pushes aggressive play.")
    p.add_argument("--capture-value-kind", choices=("production", "radius"),
                   default=None)
    p.add_argument("--max-turns", type=int, default=None,
                   help="Game length used for time-decay normalization")
    p.add_argument("--time-decay-floor", type=float, default=None)
    p.add_argument("--planet-diff-weight", type=float, default=None,
                   help="Legacy un-weighted planet count delta (off by default)")
    p.add_argument("--ship-diff-weight", type=float, default=None)
    p.add_argument("--ship-loss-weight", type=float, default=None)
    p.add_argument("--fleet-airborne-weight", type=float, default=None)
    p.add_argument("--fleet-capture-weight", type=float, default=None)
    p.add_argument("--fleet-kill-weight", type=float, default=None)
    p.add_argument("--fleet-waste-weight", type=float, default=None)
    p.add_argument("--enemy-capture-radius-weight", type=float, default=None)
    p.add_argument("--planet-loss-weight", type=float, default=None)
    p.add_argument("--planet-loss-credit-horizon", type=int, default=None)
    p.add_argument("--defense-hold-weight", type=float, default=None)
    p.add_argument("--defense-hold-horizon", type=int, default=None)
    p.add_argument("--defense-arrival-window", type=float, default=None)
    p.add_argument("--delayed-reward-discount", type=float, default=None,
                   help="Multiplier per delayed-credit step: reward *= discount ** delay")
    p.add_argument("--invalid-action-penalty", type=float, default=None)
    p.add_argument("--eligibility-trace-horizon", type=int, default=None,
                   help="Deprecated; step rewards are no longer distributed backward.")
    p.add_argument("--eligibility-trace-decay", type=float, default=None,
                   help="Deprecated; step rewards are no longer distributed backward.")
    p.add_argument("--noop-trace-weight", type=float, default=None,
                   help="Deprecated; NO-OP trace reward distribution is disabled.")
    p.add_argument("--resume-from", default=None,
                   help="Checkpoint path to resume from")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    train_cfg = TrainConfig(
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        max_episode_steps=args.max_episode_steps,
        num_agents=args.num_agents,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        learning_rate=args.lr, seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        log_path=args.log_path,
        device=args.device,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        model_variant=args.model_variant,
    )
    ppo_cfg = PPOConfig(
        clip_range=args.clip_range,
        epochs=args.epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
    )
    reward_cfg = build_reward_config(
        args.reward_system,
        capture_value_weight=args.capture_value_weight,
        denial_value_weight=args.denial_value_weight,
        capture_value_kind=args.capture_value_kind,
        max_turns=args.max_turns,
        time_decay_floor=args.time_decay_floor,
        planet_diff_weight=args.planet_diff_weight,
        ship_diff_weight=args.ship_diff_weight,
        ship_loss_weight=args.ship_loss_weight,
        fleet_airborne_weight=args.fleet_airborne_weight,
        fleet_capture_weight=args.fleet_capture_weight,
        fleet_kill_weight=args.fleet_kill_weight,
        fleet_waste_weight=args.fleet_waste_weight,
        enemy_capture_radius_weight=args.enemy_capture_radius_weight,
        planet_loss_weight=args.planet_loss_weight,
        planet_loss_credit_horizon=args.planet_loss_credit_horizon,
        defense_hold_weight=args.defense_hold_weight,
        defense_hold_horizon=args.defense_hold_horizon,
        defense_arrival_window=args.defense_arrival_window,
        delayed_reward_discount=args.delayed_reward_discount,
        invalid_action_penalty=args.invalid_action_penalty,
        eligibility_trace_horizon=args.eligibility_trace_horizon,
        eligibility_trace_decay=args.eligibility_trace_decay,
        noop_trace_weight=args.noop_trace_weight,
    )
    train(train_cfg, ppo_cfg, reward_cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
