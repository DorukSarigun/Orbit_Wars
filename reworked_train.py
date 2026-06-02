"""Training driver for the reworked hierarchical Orbit Wars policy.

Clean, self-contained PPO self-play loop built on the reworked stack
(:mod:`reworked_model`, :mod:`reworked_rollout`).  Deliberately *not* a fork of
the legacy ``rl/train.py`` -- that script is welded to the old policy variants,
buffer, reward zoo and scripted-opponent plumbing.  This one only knows the
hierarchical policy + the potential-based economic reward.

Each iteration:
    1. roll out ``episodes_per_iter`` self-play games -> one shared buffer
    2. compute per-player GAE
    3. PPO update (with KL early-stop + optional entropy anneal)
    4. log a JSONL line + console summary, checkpoint periodically

Usage::

    python -m reworked_train --iterations 200 --episodes-per-iter 4
    # or, flat layout:
    python reworked_train.py --iterations 200 --episodes-per-iter 4
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import torch

try:  # package layout
    from .reworked_model import (
        ReworkedHierarchicalPolicy,
        REWORKED_NODE_FEATURE_DIM,
        REWORKED_EDGE_FEATURE_DIM,
        REWORKED_GLOBAL_FEATURE_DIM,
    )
    from .reworked_rollout import (
        ReworkedBuffer, collect_episode, ppo_update,
        PPOConfig, PPOStats, RewardConfig, economic_potential,
    )
except ImportError:  # pragma: no cover - flat layout / direct execution
    from reworked_model import (  # type: ignore
        ReworkedHierarchicalPolicy,
        REWORKED_NODE_FEATURE_DIM,
        REWORKED_EDGE_FEATURE_DIM,
        REWORKED_GLOBAL_FEATURE_DIM,
    )
    from reworked_rollout import (  # type: ignore
        ReworkedBuffer, collect_episode, ppo_update,
        PPOConfig, PPOStats, RewardConfig, economic_potential,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    iterations: int = 200
    episodes_per_iter: int = 4
    max_episode_steps: int = 300
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    seed: int = 0
    device: str = "cpu"
    routing_cost: bool = True
    # Model arch (rebuilt on load).
    d_model: int = 128
    n_layers: int = 3
    aux_owner: bool = True
    # Entropy anneal (linear from entropy_coef -> entropy_coef_final over the
    # first entropy_anneal_iters iters; constant if final is None / iters <= 0).
    entropy_coef: float = 0.02
    entropy_coef_final: Optional[float] = None
    entropy_anneal_iters: int = 0
    # Checkpoint / logging.
    checkpoint_every: int = 20
    checkpoint_dir: str = "checkpoints_reworked"
    log_path: str = "checkpoints_reworked/train_log.jsonl"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def _policy_arch(policy: ReworkedHierarchicalPolicy) -> dict:
    """Minimal arch needed to rebuild the policy on load."""
    return {
        "node_dim": REWORKED_NODE_FEATURE_DIM,
        "edge_dim": REWORKED_EDGE_FEATURE_DIM,
        "global_dim": REWORKED_GLOBAL_FEATURE_DIM,
        "d_model": int(policy.d_model),
        "n_layers": int(len(policy.encoder.layers)),
        "aux_owner": bool(policy.aux_owner_head is not None),
    }


def _save_ckpt(policy, optimizer, path: Path, *, iteration: int,
               train_cfg: TrainConfig, ppo_cfg: PPOConfig, reward_cfg: RewardConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": iteration,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_cfg": asdict(train_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "reward_cfg": asdict(reward_cfg),
        "arch": _policy_arch(policy),
    }, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple:
    """Load a checkpoint and rebuild the policy. Returns ``(policy, ckpt_dict)``."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    policy = ReworkedHierarchicalPolicy(
        node_dim=arch["node_dim"], edge_dim=arch["edge_dim"],
        global_dim=arch["global_dim"], d_model=arch["d_model"],
        n_layers=arch["n_layers"], aux_owner=arch["aux_owner"],
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy, ckpt


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _entropy_coef_at(it: int, train_cfg: TrainConfig, base: float) -> float:
    if train_cfg.entropy_coef_final is None or train_cfg.entropy_anneal_iters <= 0:
        return base
    progress = min(1.0, max(0.0, it / float(train_cfg.entropy_anneal_iters)))
    return (1.0 - progress) * base + progress * float(train_cfg.entropy_coef_final)


def train(
    train_cfg: Optional[TrainConfig] = None,
    ppo_cfg: Optional[PPOConfig] = None,
    reward_cfg: Optional[RewardConfig] = None,
    *,
    resume_from: Optional[str] = None,
) -> ReworkedHierarchicalPolicy:
    train_cfg = train_cfg or TrainConfig()
    ppo_cfg = ppo_cfg or PPOConfig(entropy_coef=train_cfg.entropy_coef)
    # Keep the shaping discount aligned with the RL discount unless overridden.
    reward_cfg = reward_cfg or RewardConfig(gamma=train_cfg.gamma)

    torch.manual_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    if resume_from is not None:
        policy, ckpt = load_checkpoint(resume_from, device=device)
        start_iter = int(ckpt.get("iteration", 0)) + 1
        print(f"Resumed from {resume_from} at iter {start_iter}")
    else:
        policy = ReworkedHierarchicalPolicy(
            d_model=train_cfg.d_model, n_layers=train_cfg.n_layers,
            aux_owner=train_cfg.aux_owner,
        ).to(device)
        start_iter = 0

    optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.learning_rate)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"ReworkedHierarchicalPolicy: {n_params:,} params, "
          f"d_model={train_cfg.d_model}, n_layers={train_cfg.n_layers}, device={device}")

    log_path = Path(train_cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for it in range(start_iter, train_cfg.iterations):
        t0 = time.time()
        buffer = ReworkedBuffer()
        ep_stats = []
        for ep in range(train_cfg.episodes_per_iter):
            seed = train_cfg.seed * 100_000 + it * 1_000 + ep
            stats = collect_episode(
                policy, buffer,
                max_steps=train_cfg.max_episode_steps,
                reward_cfg=reward_cfg,
                potential_fn=economic_potential,
                device=device,
                seed=seed,
                routing_cost=train_cfg.routing_cost,
            )
            ep_stats.append(stats)

        buffer.compute_gae(gamma=train_cfg.gamma, lam=train_cfg.gae_lambda)

        ent_now = _entropy_coef_at(it, train_cfg, ppo_cfg.entropy_coef)
        iter_ppo_cfg = replace(ppo_cfg, entropy_coef=ent_now)
        ppo_stats: PPOStats = ppo_update(policy, buffer, optimizer, iter_ppo_cfg, device=device)

        # Aggregate episode metrics.
        n_ep = len(ep_stats)
        avg_len = sum(s.length for s in ep_stats) / n_ep
        avg_launches = sum(s.n_launches for s in ep_stats) / n_ep
        avg_p0_reward = sum(s.rewards.get(0, 0.0) for s in ep_stats) / n_ep
        wins_p0 = sum(1 for s in ep_stats if s.winner == 0)
        wins_p1 = sum(1 for s in ep_stats if s.winner == 1)
        timeouts = sum(1 for s in ep_stats if s.winner is None)
        elapsed = time.time() - t0

        log = {
            "iter": it,
            "episodes": n_ep,
            "transitions": len(buffer),
            "avg_episode_length": round(avg_len, 1),
            "avg_launches": round(avg_launches, 1),
            "avg_p0_shaped_reward": round(avg_p0_reward, 4),
            "wins_p0": wins_p0, "wins_p1": wins_p1, "timeouts": timeouts,
            "policy_loss": round(ppo_stats.policy_loss, 5),
            "value_loss": round(ppo_stats.value_loss, 5),
            "entropy": round(ppo_stats.entropy, 4),
            "entropy_coef": round(float(ent_now), 5),
            "approx_kl": round(ppo_stats.approx_kl, 5),
            "ppo_epochs_run": ppo_stats.epochs_run,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_sec": round(elapsed, 1),
        }
        with log_path.open("a") as fp:
            fp.write(json.dumps(log) + "\n")

        print(f"[iter {it:>4}/{train_cfg.iterations}] "
              f"games={n_ep} len={avg_len:.0f} launch={avg_launches:.1f} "
              f"P0={wins_p0} P1={wins_p1} TO={timeouts} | "
              f"polL={log['policy_loss']:+.4f} valL={log['value_loss']:.4f} "
              f"H={log['entropy']:.3f} KL={log['approx_kl']:.4f} "
              f"e={ppo_stats.epochs_run}/{ppo_cfg.epochs} ({elapsed:.1f}s)")

        if (it + 1) % train_cfg.checkpoint_every == 0 or it == train_cfg.iterations - 1:
            ckpt_dir = Path(train_cfg.checkpoint_dir)
            _save_ckpt(policy, optimizer, ckpt_dir / f"policy_iter{it:05d}.pt",
                       iteration=it, train_cfg=train_cfg, ppo_cfg=ppo_cfg, reward_cfg=reward_cfg)
            _save_ckpt(policy, optimizer, ckpt_dir / "latest.pt",
                       iteration=it, train_cfg=train_cfg, ppo_cfg=ppo_cfg, reward_cfg=reward_cfg)
            print(f"  checkpoint -> {ckpt_dir / f'policy_iter{it:05d}.pt'}")

    return policy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reworked hierarchical Orbit Wars PPO self-play")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--episodes-per-iter", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-routing-cost", action="store_false", dest="routing_cost",
                   help="Disable the metric-based allocator routing cost matrix.")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--no-aux-owner", action="store_false", dest="aux_owner",
                   help="Disable the auxiliary per-node owner-prediction head.")
    # PPO.
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4, dest="epochs")
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--entropy-coef-final", type=float, default=None)
    p.add_argument("--entropy-anneal-iters", type=int, default=0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.05,
                   help="PPO KL early-stop threshold; <=0 disables.")
    # Reward.
    p.add_argument("--shaping-weight", type=float, default=1.0)
    p.add_argument("--terminal-weight", type=float, default=1.0)
    p.add_argument("--ship-weight", type=float, default=1.0)
    p.add_argument("--no-shaping", action="store_false", dest="enable_shaping",
                   help="Disable potential-based shaping (terminal reward only).")
    # I/O.
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--checkpoint-dir", default="checkpoints_reworked")
    p.add_argument("--log-path", default="checkpoints_reworked/train_log.jsonl")
    p.add_argument("--resume-from", default=None)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    train_cfg = TrainConfig(
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        max_episode_steps=args.max_episode_steps,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        learning_rate=args.lr, seed=args.seed, device=args.device,
        routing_cost=args.routing_cost,
        d_model=args.d_model, n_layers=args.n_layers, aux_owner=args.aux_owner,
        entropy_coef=args.entropy_coef,
        entropy_coef_final=args.entropy_coef_final,
        entropy_anneal_iters=args.entropy_anneal_iters,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        log_path=args.log_path,
    )
    ppo_cfg = PPOConfig(
        clip=args.clip, epochs=args.epochs, minibatch_size=args.minibatch_size,
        value_coef=args.value_coef, entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=(args.target_kl if args.target_kl and args.target_kl > 0 else None),
    )
    reward_cfg = RewardConfig(
        gamma=args.gamma,
        shaping_weight=args.shaping_weight,
        terminal_weight=args.terminal_weight,
        ship_weight=args.ship_weight,
        enable_shaping=args.enable_shaping,
    )
    train(train_cfg, ppo_cfg, reward_cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
