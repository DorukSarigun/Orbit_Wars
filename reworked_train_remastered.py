"""Train the edge policy with the remastered attack-campaign reward (v10).

Clean entry point — no legacy ``--battle-credit`` / ``--loss-weight`` knobs.
The agent (``ReworkedEdgePolicy``) is unchanged; only the reward pass is new.

Usage::

    cd Orbit_Wars
    python -m reworked_train_remastered \\
        --opponent frozen --opponent-ckpt checkpoints_reworked_v7_campaign/latest.pt \\
        --checkpoint-dir checkpoints_remastered_v10 \\
        --iterations 400
"""

from __future__ import annotations

import argparse
from typing import Optional

try:
    from .reworked_train import TrainConfig, PPOConfig, RewardConfig, train
    from .reworked_remastered_reward import RemasteredRewardConfig
except ImportError:  # pragma: no cover
    from reworked_train import TrainConfig, PPOConfig, RewardConfig, train  # type: ignore
    from reworked_remastered_reward import RemasteredRewardConfig  # type: ignore


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Edge policy + remastered campaign reward")
    p.add_argument("--iterations", type=int, default=400)
    p.add_argument("--episodes-per-iter", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=300)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--entropy-coef-final", type=float, default=0.002)
    p.add_argument("--entropy-anneal-iters", type=int, default=80)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--checkpoint-dir", default="checkpoints_remastered")
    p.add_argument("--log-path", default=None)
    p.add_argument("--episode-log-path", default=None,
                   help="Per-game JSONL (default: {checkpoint-dir}/episodes.jsonl)")
    p.add_argument("--resume-from", default=None)
    p.add_argument("--opponent", choices=["self_play", "scripted", "frozen"], default="frozen")
    p.add_argument("--opponent-ckpt", default=None)
    p.add_argument("--scripted-strategy", default="production")
    p.add_argument("--min-send", type=int, default=1)
    p.add_argument("--eta-value-feature", action="store_true")
    # Remastered reward (3 knobs + terminal).
    p.add_argument("--w-casualty", type=float, default=0.35,
                   help="max fraction of capture value removed when K=S0")
    p.add_argument("--w-delay", type=float, default=0.10,
                   help="max fraction of capture value removed when campaign spans full horizon")
    p.add_argument("--w-fail", type=float, default=0.25,
                   help="failed/lost-race attack penalty scale (value-relative)")
    p.add_argument("--w-waste", type=float, default=0.0,
                   help="absolute wasted-ship penalty on failed attacks (prices tiny sprays)")
    p.add_argument("--waste-norm", type=float, default=50.0,
                   help="ships per 1.0 unit of waste for --w-waste")
    p.add_argument("--w-defense-loss", type=float, default=1.0,
                   help="scale on -dpp when an owned planet is lost")
    p.add_argument("--w-source-strip", type=float, default=0.0,
                   help="penalty for stripping a threatened source that then falls")
    p.add_argument("--source-strip-window", type=int, default=10,
                   help="turns before a loss in which outgoing strips are blamed")
    p.add_argument("--w-enemy-kill", type=float, default=1.0,
                   help="+scale * (enemy_ships/norm) when defense destroys an attack fleet")
    p.add_argument("--terminal-weight", type=float, default=1.0)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    if args.opponent == "frozen" and not args.opponent_ckpt:
        _build_argparser().error("--opponent frozen requires --opponent-ckpt")

    log_path = args.log_path or f"{args.checkpoint_dir}/train_log.jsonl"
    train_cfg = TrainConfig(
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        max_episode_steps=args.max_episode_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
        opponent=args.opponent,
        opponent_ckpt=args.opponent_ckpt,
        scripted_strategy=args.scripted_strategy,
        min_send=args.min_send,
        d_model=args.d_model,
        n_layers=args.n_layers,
        policy_kind="edge",
        eta_value_feature=args.eta_value_feature,
        entropy_coef=args.entropy_coef,
        entropy_coef_final=args.entropy_coef_final,
        entropy_anneal_iters=args.entropy_anneal_iters,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        log_path=log_path,
        episode_log_path=args.episode_log_path,
    )
    ppo_cfg = PPOConfig(
        entropy_coef=args.entropy_coef,
        target_kl=(args.target_kl if args.target_kl > 0 else None),
        aux_coef=0.1,
    )
    # Legacy reward path fully off for remastered runs.
    reward_cfg = RewardConfig(
        gamma=args.gamma,
        enable_shaping=False,
        loss_weight=0.0,
        launch_weight=0.0,
        enable_battle_credit=False,
        terminal_weight=0.0,
    )
    remastered_cfg = RemasteredRewardConfig(
        w_casualty=args.w_casualty,
        w_delay=args.w_delay,
        w_fail=args.w_fail,
        w_waste=args.w_waste,
        waste_norm=args.waste_norm,
        w_defense_loss=args.w_defense_loss,
        w_source_strip=args.w_source_strip,
        source_strip_window=args.source_strip_window,
        w_enemy_kill=args.w_enemy_kill,
        terminal_weight=args.terminal_weight,
    )
    train(train_cfg, ppo_cfg, reward_cfg,
          resume_from=args.resume_from, remastered_cfg=remastered_cfg)


if __name__ == "__main__":
    main()
