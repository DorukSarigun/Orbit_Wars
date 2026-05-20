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
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import torch

from model import OrbitWarsPolicy
from model_grouped_planet_edge import EdgeAwareGroupedPlanetOrbitWarsPolicy, PairwiseEdgeGroupedPlanetOrbitWarsPolicy
from model_grouped_planet import GroupedPlanetOrbitWarsPolicy
from model_planet_only import PlanetOnlyOrbitWarsPolicy

from .buffer import RolloutBuffer
from .metrics import (
    WeightSnapshot, grad_norms_by_module,
    round_metrics, weight_norms_by_module,
)
from .ppo import PPOConfig, ppo_update
from .rewards import REWARD_SYSTEMS, RewardConfig, build_reward_config
from .rollout import ScriptedExpansionConfig, collect_episode


@dataclass
class TrainConfig:
    iterations: int = 100
    episodes_per_iter: int = 1
    max_episode_steps: int = 500
    num_agents: int = 2
    random_num_agents: bool = False
    num_agent_choices: tuple[int, ...] = (2, 4)
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
    grouped_attention_type: str = "full"
    n_inducing_points: int = 32
    model_variant: str = "planet_only"
    save_game_batches: bool = False
    batch_save_dir: str = "checkpoints/game_batches"
    scripted_expansion_turns: int = 0
    scripted_expansion_nearest_k: int = 5
    # Opponent dispatch:
    #   "self_play"  -> all slots run the live policy (pure self-play).
    #   "scripted"   -> slot 0 runs the live policy, all other slots run
    #                   the CaptureEfficiencyAgent.
    opponent: str = "self_play"
    # Entropy schedule (linear anneal from PPOConfig.entropy_coef -> entropy_coef_final
    # over the first `entropy_anneal_iters` iterations; held constant afterwards).
    # If entropy_coef_final is None or entropy_anneal_iters <= 0, the schedule is
    # disabled and PPOConfig.entropy_coef is used as a constant.
    entropy_coef_final: Optional[float] = None
    entropy_anneal_iters: int = 0


def _make_policy(variant: str, **kwargs) -> OrbitWarsPolicy:
    if variant == "grouped_planet":
        return GroupedPlanetOrbitWarsPolicy(**kwargs)
    if variant == "grouped_planet_edge":
        return EdgeAwareGroupedPlanetOrbitWarsPolicy(**kwargs)
    if variant == "grouped_planet_edge_pair":
        return PairwiseEdgeGroupedPlanetOrbitWarsPolicy(**kwargs)
    kwargs.pop("attention_type", None)
    kwargs.pop("n_inducing_points", None)
    if variant == "planet_only":
        return PlanetOnlyOrbitWarsPolicy(**kwargs)
    if variant == "full":
        return OrbitWarsPolicy(**kwargs)
    raise ValueError(f"Unknown model_variant={variant!r}")


def _parse_num_agent_choices(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(part.strip()) for part in raw.split(",") if part.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--num-agent-choices must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("--num-agent-choices cannot be empty")
    invalid = [v for v in values if v not in (2, 4)]
    if invalid:
        raise argparse.ArgumentTypeError("Orbit Wars training currently supports num_agents choices 2 and 4")
    return values


def _episode_num_agents(train_cfg: TrainConfig, seed: int) -> int:
    if not train_cfg.random_num_agents:
        return int(train_cfg.num_agents)
    choices = tuple(train_cfg.num_agent_choices)
    return int(random.Random(int(seed)).choice(choices))


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
            "n_layers": len(policy.encoder.blocks) if getattr(policy, "model_variant", "full") in ("grouped_planet", "grouped_planet_edge", "grouped_planet_edge_pair") else len(policy.blocks),
            "model_variant": getattr(policy, "model_variant", "full"),
            "attention_type": getattr(policy, "attention_type", "full"),
            "n_inducing_points": getattr(policy, "n_inducing_points", 32),
        },
    }, path)


def _save_game_batch(buffer: RolloutBuffer, start_idx: int, end_idx: int, path: Path, *,
                     iteration: int, episode: int, seed: int, stats,
                     train_cfg: TrainConfig, ppo_cfg: PPOConfig,
                     reward_cfg: RewardConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = list(range(int(start_idx), int(end_idx)))
    batch = buffer.batch_from_indices(indices, normalize_advantages=False)
    torch.save({
        "iteration": int(iteration),
        "episode": int(episode),
        "seed": int(seed),
        "transition_start": int(start_idx),
        "transition_end": int(end_idx),
        "num_transitions": len(indices),
        "episode_stats": asdict(stats),
        "train_cfg": asdict(train_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "reward_cfg": asdict(reward_cfg),
        "batch": batch,
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
        n_layers=arch["n_layers"],
        attention_type=arch.get("attention_type", "full"),
        n_inducing_points=arch.get("n_inducing_points", 32),
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
            attention_type=train_cfg.grouped_attention_type,
            n_inducing_points=train_cfg.n_inducing_points,
        ).to(device)
        start_iter = 0

    optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.learning_rate)
    n_params = sum(p.numel() for p in policy.parameters())
    variant = getattr(policy, "model_variant", "full")
    if variant in ("grouped_planet", "grouped_planet_edge", "grouped_planet_edge_pair"):
        token_count = policy.p_max
    elif variant == "planet_only":
        token_count = policy.p_max + 1
    else:
        token_count = policy.p_max + policy.f_max + 1
    attention_info = ""
    if variant in ("grouped_planet", "grouped_planet_edge", "grouped_planet_edge_pair"):
        attention_info = f", attention={policy.attention_type}, M={policy.n_inducing_points}"
    print(f"Policy[{variant}]: {n_params:,} params, T={token_count}{attention_info}, device={device}")

    log_path = Path(train_cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for it in range(start_iter, train_cfg.iterations):
        t0 = time.time()
        buffer = RolloutBuffer()
        ep_stats_list = []
        for ep in range(train_cfg.episodes_per_iter):
            seed = train_cfg.seed * 100000 + it * 1000 + ep
            episode_num_agents = _episode_num_agents(train_cfg, seed)
            start_idx = len(buffer)
            opponent_indices: set[int] = (
                set(range(1, episode_num_agents))
                if train_cfg.opponent == "scripted" else set()
            )
            stats = collect_episode(
                policy, buffer,
                max_steps=train_cfg.max_episode_steps,
                num_agents=episode_num_agents,
                reward_cfg=reward_cfg,
                device=device,
                seed=seed,
                scripted_expansion=(
                    ScriptedExpansionConfig(
                        turns=train_cfg.scripted_expansion_turns,
                        nearest_k=train_cfg.scripted_expansion_nearest_k,
                    )
                    if train_cfg.scripted_expansion_turns > 0 else None
                ),
                opponent_indices=opponent_indices,
            )
            end_idx = len(buffer)
            ep_stats_list.append((ep, start_idx, end_idx, seed, stats))

        # Bootstrap value for GAE: 0 since each episode ends terminally.
        # (collect_episode already sets done=True at the last stored step.)
        buffer.compute_gae(last_value=0.0,
                           gamma=train_cfg.gamma, lam=train_cfg.gae_lambda)
        if train_cfg.save_game_batches:
            for ep, start_idx, end_idx, seed, stats in ep_stats_list:
                batch_path = (
                    Path(train_cfg.batch_save_dir)
                    / f"iter{it:05d}_game{ep:03d}_seed{seed}.pt"
                )
                _save_game_batch(
                    buffer,
                    start_idx,
                    end_idx,
                    batch_path,
                    iteration=it,
                    episode=ep,
                    seed=seed,
                    stats=stats,
                    train_cfg=train_cfg,
                    ppo_cfg=ppo_cfg,
                    reward_cfg=reward_cfg,
                )
        # Snapshot weights before PPO so we can compute per-module update norms.
        snap = WeightSnapshot(policy)
        # Linear entropy anneal: ppo_cfg.entropy_coef -> train_cfg.entropy_coef_final
        # over the first train_cfg.entropy_anneal_iters iterations.
        if (train_cfg.entropy_coef_final is not None
                and train_cfg.entropy_anneal_iters > 0):
            anneal_iters = int(train_cfg.entropy_anneal_iters)
            progress = min(1.0, max(0.0, float(it) / float(anneal_iters)))
            entropy_now = (
                (1.0 - progress) * float(ppo_cfg.entropy_coef)
                + progress * float(train_cfg.entropy_coef_final)
            )
            iter_ppo_cfg = replace(ppo_cfg, entropy_coef=entropy_now)
        else:
            iter_ppo_cfg = ppo_cfg
        ppo_stats = ppo_update(policy, buffer, optimizer, iter_ppo_cfg, device=device)
        # Per-module post-update stats.
        weight_norms = weight_norms_by_module(policy)
        update_norms = snap.compute_update_norms(policy)
        update_ratios = snap.compute_update_ratios(policy)
        current_lr = optimizer.param_groups[0]["lr"]

        stats_only = [item[4] for item in ep_stats_list]
        max_agents_seen = max(len(s.n_launches) for s in stats_only)
        episode_num_agents = [len(s.n_launches) for s in stats_only]
        agent_count_hist = {
            str(n): sum(1 for x in episode_num_agents if x == n)
            for n in sorted(set(episode_num_agents))
        }
        wins_by_player = {
            f"p{p}": sum(1 for s in stats_only if s.winner == p)
            for p in range(max_agents_seen)
        }
        wins_p0 = wins_by_player.get("p0", 0)
        wins_p1 = wins_by_player.get("p1", 0)
        draws = sum(1 for s in stats_only if s.winner == -1)
        avg_len = sum(s.length for s in stats_only) / len(stats_only)
        avg_launches = sum(sum(s.n_launches) / max(1, len(s.n_launches))
                           for s in stats_only) / len(stats_only)
        elapsed = time.time() - t0

        log = {
            "iter": it,
            "episodes": train_cfg.episodes_per_iter,
            "transitions": len(buffer),
            "avg_episode_length": round(avg_len, 1),
            "avg_launches_per_player": round(avg_launches, 1),
            "wins_p0": wins_p0, "wins_p1": wins_p1, "draws": draws,
            "wins_by_player": wins_by_player,
            "agent_count_hist": agent_count_hist,
            "episode_num_agents": episode_num_agents,
            "policy_loss": round(ppo_stats.policy_loss, 5),
            "value_loss": round(ppo_stats.value_loss, 5),
            "entropy": round(ppo_stats.entropy, 4),
            "entropy_coef": round(float(iter_ppo_cfg.entropy_coef), 5),
            "approx_kl": round(ppo_stats.approx_kl, 5),
            "clip_fraction": round(ppo_stats.clip_fraction, 4),
            "ratio_q05": round(ppo_stats.ratio_q05, 4),
            "ratio_q50": round(ppo_stats.ratio_q50, 4),
            "ratio_q95": round(ppo_stats.ratio_q95, 4),
            "ship_alpha": round(ppo_stats.ship_alpha_mean, 3),
            "ship_beta": round(ppo_stats.ship_beta_mean, 3),
            "ppo_epochs_run": ppo_stats.epochs_run,
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
                            for p in range(max_agents_seen))
        agents_str = ",".join(f"{n}p={c}" for n, c in agent_count_hist.items())
        print(f"[epoch {it:>4}/{train_cfg.iterations}] "
              f"games={train_cfg.episodes_per_iter} agents={agents_str} "
              f"len={avg_len:.1f} launch/p={avg_launches:.1f} "
              f"{wins_str} D={draws} | "
              f"polL={log['policy_loss']:+.4f} valL={log['value_loss']:.3f} "
              f"H={log['entropy']:.3f} KL={log['approx_kl']:.4f} "
              f"clip={log['clip_fraction']:.2f} "
              f"r=[{log['ratio_q05']:.2f},{log['ratio_q50']:.2f},{log['ratio_q95']:.2f}] "
              f"αβ=({log['ship_alpha']:.1f},{log['ship_beta']:.1f}) "
              f"e={ppo_stats.epochs_run}/{ppo_cfg.epochs} "
              f"ur={ur_mean:.2e} "
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
    p.add_argument("--random-num-agents", action="store_true",
                   help="Randomly choose the agent count separately for each episode")
    p.add_argument("--num-agent-choices", type=_parse_num_agent_choices, default=(2, 4),
                   help="Comma-separated episode agent-count choices for --random-num-agents")
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
    p.add_argument("--grouped-attention-type", choices=("full", "induced"), default="full")
    p.add_argument("--n-inducing-points", type=int, default=32)
    p.add_argument("--model-variant", choices=("full", "planet_only", "grouped_planet", "grouped_planet_edge", "grouped_planet_edge_pair"), default="planet_only")
    p.add_argument("--save-game-batches", action="store_true",
                   help="Save one .pt rollout/training batch per self-play game after GAE")
    p.add_argument("--batch-save-dir", default="checkpoints/game_batches")
    p.add_argument("--scripted-expansion-turns", type=int, default=0,
                   help="Force early neutral-expansion launches for this many turns")
    p.add_argument("--scripted-expansion-nearest-k", type=int, default=5,
                   help="Choose best production/ships neutral target among this many nearest planets")
    p.add_argument("--opponent", choices=["self_play", "scripted"], default="self_play",
                   help="self_play: all slots use the live policy. "
                        "scripted: slot 0 is the policy, all others are CaptureEfficiencyAgent.")
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4, dest="epochs",
                   help="Number of optimization epochs over each rollout buffer "
                        "(inner PPO update; not the outer training-epoch count)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--entropy-coef", type=float, default=0.05,
                   help="Initial entropy bonus coefficient. With --entropy-coef-final "
                        "and --entropy-anneal-iters, linearly anneals to the final value.")
    p.add_argument("--entropy-coef-final", type=float, default=None,
                   help="Target entropy bonus after annealing. Leave unset for constant "
                        "--entropy-coef.")
    p.add_argument("--entropy-anneal-iters", type=int, default=0,
                   help="Number of training iterations to linearly anneal entropy_coef "
                        "from --entropy-coef to --entropy-coef-final. 0 disables anneal.")
    p.add_argument("--target-kl", type=float, default=0.05,
                   help="Per-rollout PPO KL early-stopping threshold. If the "
                        "average approx_kl on an epoch's minibatches exceeds "
                        "this value, the remaining inner PPO epochs are "
                        "skipped. Set negative to disable. Recommended ~0.05 "
                        "for Beta+Categorical action spaces.")
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
    p.add_argument("--fleet-overcommit-weight", type=float, default=None)
    p.add_argument("--enemy-capture-radius-weight", type=float, default=None)
    p.add_argument("--planet-loss-weight", type=float, default=None)
    p.add_argument("--planet-loss-credit-horizon", type=int, default=None)
    p.add_argument("--defense-kills-weight", type=float, default=None,
                   help="Per enemy ship killed at MY planet, distributed across "
                        "my launches that defended it (registered while planet "
                        "was tagged under-attack). Threat-aware replacement for "
                        "the deprecated production_reinforcement_weight.")
    p.add_argument("--production-reinforcement-weight", type=float, default=None,
                   help="Legacy post-capture reinforcement bonus (DEPRECATED). "
                        "Default 0; setting > 0 re-enables the legacy behavior.")
    p.add_argument("--production-upstream-credit-weight", type=float, default=None,
                   help="1-hop upstream capture-credit weight for launches that reinforced "
                        "a source planet before it launched a successful capture attack.")
    p.add_argument("--production-upstream-credit-window", type=int, default=None,
                   help="Max turns between support arrival and direct capture-launch turn "
                        "for 1-hop upstream credit.")
    p.add_argument("--capture-time-curvature", type=float, default=None,
                   help="Power applied to ((max_turns - turn) / max_turns) "
                        "in capture/loss/defense-window time factor. "
                        "1.0 = legacy linear; >1 amplifies early/late ratio.")
    p.add_argument("--ship-overflow-weight", type=float, default=None,
                   help="Per-ship penalty when the continuous Beta head requests more ships "
                        "than the source garrison (penalty = -w * overflow_ships * reward_scale).")
    p.add_argument("--defense-hold-weight", type=float, default=None)
    p.add_argument("--defense-hold-horizon", type=int, default=None)
    p.add_argument("--defense-arrival-window", type=float, default=None)
    p.add_argument("--delayed-reward-discount", type=float, default=None,
                   help="Multiplier per delayed-credit step: reward *= discount ** delay")
    p.add_argument("--production-reward-beta", type=float, default=None,
                   help="Superlinear production reward exponent; 0 keeps production value linear")
    p.add_argument("--production-reward-reference", type=float, default=None,
                   help="Reference production value preserved by --production-reward-beta")
    p.add_argument("--invalid-action-penalty", type=float, default=None)
    p.add_argument("--reward-scale", type=float, default=None)
    p.add_argument("--draw-terminal-reward", type=float, default=None)
    p.add_argument("--eligibility-trace-horizon", type=int, default=None,
                   help="Number of past turns to spread the per-step reward "
                        "over (production_diff system only). 0 disables.")
    p.add_argument("--eligibility-trace-decay", type=float, default=None,
                   help="Exponential decay base for eligibility trace credit "
                        "assignment (production_diff system only). "
                        "0..1 typical.")
    p.add_argument("--noop-trace-weight", type=float, default=None,
                   help="Per-NO-OP credit weight inside the trace (relative "
                        "to launch transitions = 1.0). "
                        "production_diff system only.")
    p.add_argument("--eligibility-trace-reverse", action="store_true",
                   default=None,
                   help="Reverse eligibility trace: OLDER actions in the "
                        "window get more credit, NEWER ones get less. "
                        "Right default for delayed-effect domains where "
                        "current production-diff is caused by launches "
                        "made many turns ago. (production_diff default: True)")
    p.add_argument("--no-eligibility-trace-reverse", action="store_false",
                   dest="eligibility_trace_reverse",
                   help="Disable reverse trace; use standard forward trace "
                        "(newer actions get more credit).")
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
        random_num_agents=args.random_num_agents,
        num_agent_choices=args.num_agent_choices,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        learning_rate=args.lr, seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        log_path=args.log_path,
        device=args.device,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        grouped_attention_type=args.grouped_attention_type,
        n_inducing_points=args.n_inducing_points,
        model_variant=args.model_variant,
        save_game_batches=args.save_game_batches,
        batch_save_dir=args.batch_save_dir,
        scripted_expansion_turns=args.scripted_expansion_turns,
        scripted_expansion_nearest_k=args.scripted_expansion_nearest_k,
        opponent=args.opponent,
        entropy_coef_final=args.entropy_coef_final,
        entropy_anneal_iters=args.entropy_anneal_iters,
    )
    ppo_cfg = PPOConfig(
        clip_range=args.clip_range,
        epochs=args.epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        target_kl=args.target_kl if args.target_kl is not None and args.target_kl > 0 else None,
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
        fleet_overcommit_weight=args.fleet_overcommit_weight,
        enemy_capture_radius_weight=args.enemy_capture_radius_weight,
        planet_loss_weight=args.planet_loss_weight,
        planet_loss_credit_horizon=args.planet_loss_credit_horizon,
        capture_time_curvature=args.capture_time_curvature,
        defense_kills_weight=args.defense_kills_weight,
        production_reinforcement_weight=args.production_reinforcement_weight,
        production_upstream_credit_weight=args.production_upstream_credit_weight,
        production_upstream_credit_window=args.production_upstream_credit_window,
        ship_overflow_weight=args.ship_overflow_weight,
        defense_hold_weight=args.defense_hold_weight,
        defense_hold_horizon=args.defense_hold_horizon,
        defense_arrival_window=args.defense_arrival_window,
        delayed_reward_discount=args.delayed_reward_discount,
        production_reward_beta=args.production_reward_beta,
        production_reward_reference=args.production_reward_reference,
        invalid_action_penalty=args.invalid_action_penalty,
        reward_scale=args.reward_scale,
        draw_terminal_reward=args.draw_terminal_reward,
        eligibility_trace_horizon=args.eligibility_trace_horizon,
        eligibility_trace_decay=args.eligibility_trace_decay,
        noop_trace_weight=args.noop_trace_weight,
        eligibility_trace_reverse=args.eligibility_trace_reverse,
    )
    train(train_cfg, ppo_cfg, reward_cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
