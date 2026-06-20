"""Train the final neural node policy with PPO self-play."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from final_model import FinalNodePolicy
from final_rollout import PPOConfig, RolloutBuffer, collect_episode, ppo_update


def save_checkpoint(path: Path, policy: FinalNodePolicy, optimizer, *, iteration: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": int(iteration),
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "arch": {
            "d_model": int(policy.d_model),
            "n_heads": int(policy.n_heads),
            "n_layers": int(policy.n_layers),
        },
    }, path)


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    policy = FinalNodePolicy(
        d_model=int(arch.get("d_model", 128)),
        n_heads=int(arch.get("n_heads", 4)),
        n_layers=int(arch.get("n_layers", 2)),
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy, ckpt


def train(args) -> None:
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    if args.resume_from:
        policy, ckpt = load_checkpoint(args.resume_from, device)
        start_iter = int(ckpt.get("iteration", -1)) + 1
    else:
        policy = FinalNodePolicy(
            d_model=int(args.d_model),
            n_heads=int(args.n_heads),
            n_layers=int(args.n_layers),
        ).to(device)
        start_iter = 0
        ckpt = None

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.lr))
    if ckpt is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as exc:
            print(f"warning: optimizer state not restored: {exc}")

    ppo_cfg = PPOConfig(
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip=float(args.clip),
        lr=float(args.lr),
        epochs=int(args.ppo_epochs),
        minibatch_size=int(args.minibatch_size),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        max_grad_norm=float(args.max_grad_norm),
    )
    ckpt_dir = Path(args.checkpoint_dir)
    log_path = Path(args.log_path or ckpt_dir / "train_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"FinalNodePolicy params={n_params:,} device={device} start_iter={start_iter}")

    for it in range(start_iter, int(args.iterations)):
        t0 = time.time()
        policy.train()
        buffer = RolloutBuffer()
        ep_stats = []
        for ep in range(int(args.episodes_per_iter)):
            seed = int(args.seed) * 100_000 + it * 1_000 + ep
            stats = collect_episode(
                policy,
                buffer,
                device=device,
                seed=seed,
                max_steps=int(args.max_steps),
                num_agents=int(args.num_agents),
                shaping_weight=float(args.shaping_weight),
                terminal_weight=float(args.terminal_weight),
                invalid_action_penalty=float(args.invalid_action_penalty),
                reject_action_penalty=float(args.reject_action_penalty),
                capture_action_bonus=float(args.capture_action_bonus),
                raw_launch_bonus=float(args.raw_launch_bonus),
                planner_top_m_sources=int(args.planner_top_m_sources),
                planner_max_launches_per_target=int(args.planner_max_launches_per_target),
                planner_budget_floor=float(args.planner_budget_floor),
                planner_hold_horizon=float(args.planner_hold_horizon),
                truncate_terminal_weight=float(args.truncate_terminal_weight),
                include_owned_targets=bool(args.include_owned_targets),
            )
            ep_stats.append(stats)

        buffer.compute_gae(gamma=ppo_cfg.gamma, lam=ppo_cfg.gae_lambda)
        update_stats = ppo_update(policy, buffer, optimizer, ppo_cfg, device)
        row = {
            "iter": it,
            "transitions": len(buffer),
            "episodes": int(args.episodes_per_iter),
            "avg_len": sum(s["length"] for s in ep_stats) / max(1, len(ep_stats)),
            "avg_reward": sum(s["reward_mean"] for s in ep_stats) / max(1, len(ep_stats)),
            "avg_reward_p0": sum(s["reward_p0"] for s in ep_stats) / max(1, len(ep_stats)),
            "avg_reward_abs": sum(s["reward_abs_mean"] for s in ep_stats) / max(1, len(ep_stats)),
            "avg_validity_reward": sum(s["validity_reward_mean"] for s in ep_stats) / max(1, len(ep_stats)),
            "avg_validity_reward_abs": sum(s["validity_reward_abs_mean"] for s in ep_stats) / max(1, len(ep_stats)),
            "noop_rate": sum(s.get("noop_rate", 0.0) for s in ep_stats) / max(1, len(ep_stats)),
            "micro_capture_rate": sum(s.get("micro_capture_rate", 0.0) for s in ep_stats) / max(1, len(ep_stats)),
            "raw_launch_rate": sum(s.get("raw_launch_rate", 0.0) for s in ep_stats) / max(1, len(ep_stats)),
            "reject_rate": sum(s.get("reject_rate", 0.0) for s in ep_stats) / max(1, len(ep_stats)),
            "invalid_rate": sum(s.get("invalid_rate", 0.0) for s in ep_stats) / max(1, len(ep_stats)),
            "truncated_rate": sum(s.get("truncated", 0.0) for s in ep_stats) / max(1, len(ep_stats)),
            "seconds": time.time() - t0,
            **update_stats,
        }
        with log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row) + "\n")
        print(
            f"[iter {it:04d}] steps={row['transitions']} "
            f"mean={row['avg_reward']:+.4f} "
            f"p0={row['avg_reward_p0']:+.4f} "
            f"abs={row['avg_reward_abs']:.4f} "
            f"valid={row['avg_validity_reward']:+.4f} "
            f"noop={row['noop_rate']:.3f} "
            f"cap={row['micro_capture_rate']:.3f} "
            f"raw={row['raw_launch_rate']:.3f} "
            f"rej={row['reject_rate']:.3f} "
            f"inv={row['invalid_rate']:.3f} "
            f"trunc={row['truncated_rate']:.3f} "
            f"pi={row['policy_loss']:+.4f} v={row['value_loss']:.4f} "
            f"ent={row['entropy']:.3f} sec={row['seconds']:.1f}",
            flush=True,
        )
        if (it + 1) % int(args.checkpoint_every) == 0 or it == int(args.iterations) - 1:
            save_checkpoint(ckpt_dir / f"policy_iter{it:05d}.pt", policy, optimizer, iteration=it, args=args)
            save_checkpoint(ckpt_dir / "latest.pt", policy, optimizer, iteration=it, args=args)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train final neural node policy")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--episodes-per-iter", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--num-agents", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--shaping-weight", type=float, default=0.05)
    p.add_argument("--terminal-weight", type=float, default=1.0)
    p.add_argument("--invalid-action-penalty", type=float, default=0.02)
    p.add_argument("--reject-action-penalty", type=float, default=0.005)
    p.add_argument("--capture-action-bonus", type=float, default=0.002)
    p.add_argument("--raw-launch-bonus", type=float, default=0.0)
    p.add_argument("--planner-top-m-sources", type=int, default=5)
    p.add_argument("--planner-max-launches-per-target", type=int, default=3)
    p.add_argument("--planner-budget-floor", type=float, default=0.35)
    p.add_argument("--planner-hold-horizon", type=float, default=24.0)
    p.add_argument("--truncate-terminal-weight", type=float, default=1.0)
    p.add_argument(
        "--include-owned-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow owned planets as launch targets (reinforcement/defense). "
             "Enable when resuming a BC checkpoint trained with owned targets.",
    )
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--checkpoint-dir", default="checkpoints_final_rl")
    p.add_argument("--log-path", default=None)
    p.add_argument("--resume-from", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    train(_build_argparser().parse_args(argv))


if __name__ == "__main__":
    main()
