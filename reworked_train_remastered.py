"""Train the edge or hier policy with the remastered attack-campaign reward.

Clean entry point — no legacy ``--battle-credit`` / ``--loss-weight`` knobs.
The agents (``ReworkedEdgePolicy`` / ``ReworkedHierarchicalPolicy``) are
unchanged; only the reward pass is new.

* ``--policy edge`` (default): per-target credit slots (v10+).
* ``--policy hier``: hierarchical strategy heads + min-cost-flow allocator;
  campaign credits land on the launching turn's transition (turn-level).

Usage::

    cd Orbit_Wars
    python -m reworked_train_remastered \\
        --opponent frozen --opponent-ckpt checkpoints_reworked_v7_campaign/latest.pt \\
        --checkpoint-dir checkpoints_remastered_v10 \\
        --iterations 400

    # hier + LP allocator with the same reward:
    python -m reworked_train_remastered --policy hier \\
        --opponent scripted --scripted-strategy production \\
        --checkpoint-dir checkpoints_hier_remastered --iterations 400
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
    p = argparse.ArgumentParser(description="Edge/hier policy + remastered campaign reward")
    p.add_argument("--policy", choices=["edge", "hier"], default="edge",
                   dest="policy_kind",
                   help="edge = per-target credit (v10+); hier = hierarchical "
                        "strategy heads + min-cost-flow allocator, turn-level credit")
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
    p.add_argument("--minibatch-size", type=int, default=64,
                   help="PPO/BC minibatch (edge policy is memory-heavy; lower if CUDA OOM)")
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
    # Teacher forcing + behavior cloning.
    p.add_argument("--teacher-forcing", type=float, default=0.0,
                   help="per-decision probability the learner plays the scripted "
                        "bot's move (--scripted-strategy, e.g. 'orbit') instead "
                        "of its own; those turns are not PPO transitions")
    p.add_argument("--bc-coef", type=float, default=0.0,
                   help="behavior-cloning weight: supervised -log pi(a_teacher|s) "
                        "loss on teacher-forced turns (requires --teacher-forcing "
                        "> 0); 0 = teacher only steers, no imitation")
    # Remastered reward (3 knobs + terminal).
    p.add_argument("--w-casualty", type=float, default=0.35,
                   help="max fraction of capture value removed when K=S0")
    p.add_argument("--w-delay", type=float, default=0.10,
                   help="max fraction of capture value removed when campaign spans full horizon")
    p.add_argument("--w-fail", type=float, default=0.25,
                   help="failed/lost-race attack penalty scale (value-relative)")
    p.add_argument("--w-waste", type=float, default=0.25,
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
    p.add_argument("--terminal-weight", type=float, default=5.0,
                   help="episodic win/loss scale (campaign credit totals ~+10/ep, "
                        "so winning must outweigh farming)")
    p.add_argument("--enemy-capture-mult", type=float, default=1.5,
                   help="capture R multiplier when the target was enemy-owned (zero-sum swing)")
    p.add_argument("--recapture-decay", type=float, default=0.5,
                   help="n-th capture of the same planet earns R*decay**(n-1) (anti ping-pong)")
    p.add_argument("--timeout-terminal", type=float, default=1.0,
                   help="scale on the production-share soft terminal when the game times out; 0 disables")
    # v13 ship-equity units (default ON): every campaign reward in ships/norm.
    p.add_argument("--legacy-units", action="store_true",
                   help="use the legacy v10 mixed-unit formulas instead of the "
                        "v13 ship-equity currency (ships/norm)")
    p.add_argument("--eq-siege", type=float, default=1.0,
                   help="[equity] tempo cost scale: prod_tgt * (t_capture - first_arrival) / norm "
                        "-- prices dripped/split sieges by the defender's regrowth")
    p.add_argument("--eq-fail", type=float, default=1.25,
                   help="[equity] per-ship cost of failed/lost attack ships "
                        "(1.0 = a ship is a ship; the excess is anti-spray friction)")
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
        teacher_forcing=args.teacher_forcing,
        bc_coef=args.bc_coef,
        d_model=args.d_model,
        n_layers=args.n_layers,
        policy_kind=args.policy_kind,
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
        clip=0.2,
        epochs=4,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        target_kl=(args.target_kl if args.target_kl > 0 else None),
        aux_coef=0.1,
        bc_coef=args.bc_coef,
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
        enemy_capture_mult=args.enemy_capture_mult,
        recapture_decay=args.recapture_decay,
        timeout_terminal=args.timeout_terminal,
        ship_equity=not args.legacy_units,
        eq_siege=args.eq_siege,
        eq_fail=args.eq_fail,
    )
    if remastered_cfg.ship_equity:
        print(f"Reward units: v13 ship-equity (eq_siege={remastered_cfg.eq_siege}, "
              f"eq_fail={remastered_cfg.eq_fail}, norm={remastered_cfg.norm})")
    else:
        print("Reward units: legacy v10 mixed-unit formulas")
    train(train_cfg, ppo_cfg, reward_cfg,
          resume_from=args.resume_from, remastered_cfg=remastered_cfg)


if __name__ == "__main__":
    main()
