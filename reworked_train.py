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
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import torch

try:  # package layout
    from .reworked_model import (
        ReworkedHierarchicalPolicy,
        ReworkedLaunchPolicy,
        ReworkedEdgePolicy,
        REWORKED_NODE_FEATURE_DIM,
        REWORKED_EDGE_FEATURE_DIM,
        REWORKED_GLOBAL_FEATURE_DIM,
    )
    from .reworked_rollout import (
        ReworkedBuffer, collect_episode, ppo_update,
        PPOConfig, PPOStats, RewardConfig, economic_potential,
    )
    from .reworked_remastered_reward import RemasteredRewardConfig
except ImportError:  # pragma: no cover - flat layout / direct execution
    from reworked_model import (  # type: ignore
        ReworkedHierarchicalPolicy,
        ReworkedLaunchPolicy,
        ReworkedEdgePolicy,
        REWORKED_NODE_FEATURE_DIM,
        REWORKED_EDGE_FEATURE_DIM,
        REWORKED_GLOBAL_FEATURE_DIM,
    )
    from reworked_rollout import (  # type: ignore
        ReworkedBuffer, collect_episode, ppo_update,
        PPOConfig, PPOStats, RewardConfig, economic_potential,
    )
    from reworked_remastered_reward import RemasteredRewardConfig  # type: ignore


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
    # Opponent: "self_play" | "scripted" | "frozen" (P0 always = live policy).
    num_agents: int = 2
    random_num_agents: bool = False
    num_agent_choices: tuple[int, ...] = (2, 4)
    opponent: str = "self_play"
    scripted_strategy: str = "production"
    opponent_ckpt: Optional[str] = None   # required when opponent == "frozen"
    # Teacher forcing: per-decision probability the LEARNING player executes the
    # scripted bot's move (``scripted_strategy``) instead of its own.  Those turns
    # are not stored as transitions; they only steer the trajectory into strong
    # states so the policy gets a win/advantage contrast to learn from.  0 = off.
    teacher_forcing: float = 0.0
    # Minimum ships per launch (v8 launch policy).  Raising it (e.g. 3-5) forces
    # the Dirichlet split to concentrate mass to get any launch through, directly
    # countering the "spray tiny fleets, capture nothing" failure mode.
    min_send: int = 1
    # Behavior cloning (v8 launch only): weight of the supervised loss that makes
    # the policy imitate the teacher's executed launches on teacher-forced turns.
    # Requires teacher_forcing > 0.  0 = off (teacher only steers, no imitation).
    bc_coef: float = 0.0
    # Model arch (rebuilt on load).
    d_model: int = 128
    n_layers: int = 3
    aux_owner: bool = True
    # Policy family: "hier" = hierarchical + min-cost-flow allocator (v6/v7);
    # "launch" = v8 source-centric launch policy (commit Beta + Dirichlet split,
    # allocator-free).  Different action spaces -> cannot resume across kinds.
    policy_kind: str = "hier"
    # ETA-aware capture-value node channel (node_dim 16 -> 17).  Off by default
    # so legacy checkpoints stay loadable; a new run is required to enable it
    # (feature dim changes -> cannot resume across the boundary).
    eta_value_feature: bool = False
    # Entropy anneal (linear from entropy_coef -> entropy_coef_final over the
    # first entropy_anneal_iters iters; constant if final is None / iters <= 0).
    entropy_coef: float = 0.02
    entropy_coef_final: Optional[float] = None
    entropy_anneal_iters: int = 0
    # Checkpoint / logging.
    checkpoint_every: int = 20
    checkpoint_dir: str = "checkpoints_reworked"
    log_path: str = "checkpoints_reworked/train_log.jsonl"
    # Per-game JSONL (planet captures, campaigns, winner).  None = auto beside log_path.
    episode_log_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def _policy_arch(policy) -> dict:
    """Minimal arch needed to rebuild the policy on load."""
    node_dim = int(getattr(policy, "node_dim", REWORKED_NODE_FEATURE_DIM))
    return {
        "node_dim": node_dim,
        "edge_dim": REWORKED_EDGE_FEATURE_DIM,
        "global_dim": REWORKED_GLOBAL_FEATURE_DIM,
        "d_model": int(policy.d_model),
        "n_layers": int(len(policy.encoder.layers)),
        "aux_owner": bool(policy.aux_owner_head is not None),
        # ETA-aware capture-value channel adds exactly one node feature.
        "eta_value_feature": node_dim > REWORKED_NODE_FEATURE_DIM,
        # Policy family ("hier" | "launch" | "edge") so load_checkpoint rebuilds
        # the right class with the right action space.
        "policy_kind": _policy_kind_of(policy),
    }


def _policy_kind_of(policy) -> str:
    if isinstance(policy, ReworkedEdgePolicy):
        return "edge"
    if isinstance(policy, ReworkedLaunchPolicy):
        return "launch"
    return "hier"


def _policy_class(kind: str):
    return {
        "edge": ReworkedEdgePolicy,
        "launch": ReworkedLaunchPolicy,
        "hier": ReworkedHierarchicalPolicy,
    }.get(kind, ReworkedHierarchicalPolicy)


def _save_ckpt(policy, optimizer, path: Path, *, iteration: int,
               train_cfg: TrainConfig, ppo_cfg: PPOConfig, reward_cfg: RewardConfig,
               remastered_cfg: Optional[RemasteredRewardConfig] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_cfg": asdict(train_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "reward_cfg": asdict(reward_cfg),
        "arch": _policy_arch(policy),
    }
    if remastered_cfg is not None:
        payload["remastered_cfg"] = asdict(remastered_cfg)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple:
    """Load a checkpoint and rebuild the policy. Returns ``(policy, ckpt_dict)``."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    cls = _policy_class(arch.get("policy_kind", "hier"))
    policy = cls(
        node_dim=arch["node_dim"], edge_dim=arch["edge_dim"],
        global_dim=arch["global_dim"], d_model=arch["d_model"],
        n_layers=arch["n_layers"], aux_owner=arch["aux_owner"],
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy, ckpt


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _parse_num_agent_choices(raw: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("--num-agent-choices cannot be empty")
    invalid = [v for v in values if v not in (2, 4)]
    if invalid:
        raise argparse.ArgumentTypeError(
            "Orbit Wars supports num_agents in {2, 4}; invalid: " + str(invalid))
    return values


def _episode_num_agents(train_cfg: TrainConfig, seed: int) -> int:
    if not train_cfg.random_num_agents:
        return int(train_cfg.num_agents)
    choices = tuple(train_cfg.num_agent_choices)
    return int(random.Random(int(seed)).choice(choices))


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
    remastered_cfg: Optional[RemasteredRewardConfig] = None,
) -> ReworkedHierarchicalPolicy:
    train_cfg = train_cfg or TrainConfig()
    ppo_cfg = ppo_cfg or PPOConfig(entropy_coef=train_cfg.entropy_coef)
    # Keep the shaping discount aligned with the RL discount unless overridden.
    reward_cfg = reward_cfg or RewardConfig(gamma=train_cfg.gamma)

    torch.manual_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)

    node_dim = REWORKED_NODE_FEATURE_DIM + (1 if train_cfg.eta_value_feature else 0)
    if resume_from is not None:
        policy, ckpt = load_checkpoint(resume_from, device=device)
        start_iter = int(ckpt.get("iteration", 0)) + 1
        ckpt_node_dim = int(getattr(policy, "node_dim", REWORKED_NODE_FEATURE_DIM))
        if ckpt_node_dim != node_dim:
            raise ValueError(
                f"--eta-value-feature mismatch: checkpoint has node_dim="
                f"{ckpt_node_dim} but this run wants {node_dim}. The ETA-value "
                f"channel changes the feature dim, so you cannot resume across "
                f"it -- start a fresh run instead.")
        ckpt_kind = ckpt.get("arch", {}).get("policy_kind", "hier")
        if ckpt_kind != train_cfg.policy_kind:
            raise ValueError(
                f"--policy mismatch: checkpoint is '{ckpt_kind}' but this run "
                f"wants '{train_cfg.policy_kind}'. The two policy families have "
                f"different action spaces, so you cannot resume across them -- "
                f"start a fresh run instead.")
        # Keep config in sync with the loaded arch.
        train_cfg = replace(train_cfg,
                            eta_value_feature=ckpt_node_dim > REWORKED_NODE_FEATURE_DIM,
                            policy_kind=ckpt_kind)
        print(f"Resumed from {resume_from} at iter {start_iter}")
    else:
        policy_cls = _policy_class(train_cfg.policy_kind)
        policy = policy_cls(
            node_dim=node_dim,
            d_model=train_cfg.d_model, n_layers=train_cfg.n_layers,
            aux_owner=train_cfg.aux_owner,
        ).to(device)
        start_iter = 0

    optimizer = torch.optim.Adam(policy.parameters(), lr=train_cfg.learning_rate)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"{type(policy).__name__} [{train_cfg.policy_kind}]: {n_params:,} params, "
          f"d_model={train_cfg.d_model}, n_layers={train_cfg.n_layers}, device={device}")

    frozen_policy = None
    if train_cfg.opponent == "frozen":
        if not train_cfg.opponent_ckpt:
            raise ValueError("TrainConfig.opponent='frozen' requires opponent_ckpt")
        frozen_policy, frozen_ckpt = load_checkpoint(train_cfg.opponent_ckpt, device=device)
        frozen_kind = frozen_ckpt.get("arch", {}).get("policy_kind", "hier")
        # The frozen opponent MAY be a different policy family than the learner
        # (e.g. a v8 launch policy training against a frozen v7 hier opponent).
        # They only have to agree on the FEATURE layout, since both consume the
        # same packed observation batch.
        frozen_node_dim = int(getattr(frozen_policy, "node_dim", REWORKED_NODE_FEATURE_DIM))
        if frozen_node_dim != node_dim:
            raise ValueError(
                f"Frozen opponent node_dim {frozen_node_dim} != training node_dim "
                f"{node_dim}.  Match --eta-value-feature between the frozen "
                f"checkpoint and this run (both share the same observation batch).")
        if frozen_kind != train_cfg.policy_kind:
            print(f"Note: frozen opponent is '{frozen_kind}', learner is "
                  f"'{train_cfg.policy_kind}' (mixed-family match).")
        frozen_policy.eval()
        for p in frozen_policy.parameters():
            p.requires_grad_(False)
        uses_4p = train_cfg.num_agents == 4 or (
            train_cfg.random_num_agents and 4 in train_cfg.num_agent_choices)
        extra = f"; P2–P3 scripted ({train_cfg.scripted_strategy})" if uses_4p else ""
        print(f"Frozen opponent (P1): {train_cfg.opponent_ckpt}{extra}")
    if train_cfg.random_num_agents:
        print(f"Random num_agents per episode from {tuple(train_cfg.num_agent_choices)}")
    else:
        print(f"num_agents={train_cfg.num_agents}")
    if train_cfg.teacher_forcing > 0.0:
        bc_note = (f" + behavior cloning (bc_coef={ppo_cfg.bc_coef})"
                   if ppo_cfg.bc_coef > 0.0 else " (steer only, not trained on)")
        print(f"Teacher forcing: {train_cfg.teacher_forcing:.0%} of learner turns "
              f"play scripted '{train_cfg.scripted_strategy}'{bc_note}")
    if train_cfg.policy_kind in ("launch", "edge") and train_cfg.min_send != 1:
        print(f"min_send={train_cfg.min_send} (launches below this are dropped)")
    if remastered_cfg is not None:
        if train_cfg.policy_kind != "edge":
            raise ValueError("remastered reward requires --policy edge")
        print("Remastered attack-campaign reward (edge v10): "
              f"w_casualty={remastered_cfg.w_casualty}, "
              f"w_delay={remastered_cfg.w_delay}, w_fail={remastered_cfg.w_fail}")
    elif train_cfg.policy_kind == "edge":
        print("Edge policy (v9): per-target credit; over-extension taught via "
              f"--credit-source-loss-weight (={reward_cfg.credit_source_loss_weight}).")

    log_path = Path(train_cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    episode_log_path = Path(
        train_cfg.episode_log_path or str(log_path.parent / "episodes.jsonl"))
    episode_log_path.parent.mkdir(parents=True, exist_ok=True)
    if remastered_cfg is not None:
        print(f"Episode log -> {episode_log_path}")

    for it in range(start_iter, train_cfg.iterations):
        t0 = time.time()
        buffer = ReworkedBuffer()
        ep_stats = []
        episode_num_agents_list: list[int] = []
        for ep in range(train_cfg.episodes_per_iter):
            seed = train_cfg.seed * 100_000 + it * 1_000 + ep
            n_agents = _episode_num_agents(train_cfg, seed)
            episode_num_agents_list.append(n_agents)
            stats = collect_episode(
                policy, buffer,
                max_steps=train_cfg.max_episode_steps,
                num_agents=n_agents,
                reward_cfg=reward_cfg,
                potential_fn=economic_potential,
                device=device,
                seed=seed,
                routing_cost=train_cfg.routing_cost,
                opponent=train_cfg.opponent,
                scripted_strategy=train_cfg.scripted_strategy,
                frozen_policy=frozen_policy,
                include_value_rate=train_cfg.eta_value_feature,
                policy_kind=train_cfg.policy_kind,
                teacher_forcing=train_cfg.teacher_forcing,
                min_send=train_cfg.min_send,
                record_bc=(ppo_cfg.bc_coef > 0.0 and train_cfg.teacher_forcing > 0.0),
                remastered_cfg=remastered_cfg,
            )
            ep_stats.append(stats)
            ep_record = {
                "iter": it,
                "ep": ep,
                "seed": seed,
                "num_agents": n_agents,
                "length": stats.length,
                "winner": stats.winner,
                "rewards": {str(k): float(v) for k, v in stats.rewards.items()},
                "n_launches": stats.n_launches,
                "credit_total": round(float(stats.credit_total), 4),
                "n_capture": stats.n_capture,
                "n_defense": stats.n_defense,
            }
            if stats.episode_log:
                ep_record.update(stats.episode_log)
            with episode_log_path.open("a") as ep_fp:
                ep_fp.write(json.dumps(ep_record) + "\n")

        buffer.compute_gae(gamma=train_cfg.gamma, lam=train_cfg.gae_lambda)

        ent_now = _entropy_coef_at(it, train_cfg, ppo_cfg.entropy_coef)
        iter_ppo_cfg = replace(ppo_cfg, entropy_coef=ent_now)
        ppo_stats: PPOStats = ppo_update(policy, buffer, optimizer, iter_ppo_cfg, device=device)

        # Aggregate episode metrics.
        n_ep = len(ep_stats)
        avg_len = sum(s.length for s in ep_stats) / n_ep
        avg_launches = sum(s.n_launches for s in ep_stats) / n_ep
        avg_p0_reward = sum(s.rewards.get(0, 0.0) for s in ep_stats) / n_ep
        wins_by_player = {
            p: sum(1 for s in ep_stats if s.winner == p)
            for p in range(max(episode_num_agents_list) if episode_num_agents_list else 2)
        }
        wins_p0 = wins_by_player.get(0, 0)
        wins_p1 = wins_by_player.get(1, 0)
        timeouts = sum(1 for s in ep_stats if s.winner is None)
        n_agents_hist = {
            str(n): sum(1 for x in episode_num_agents_list if x == n)
            for n in sorted(set(episode_num_agents_list))
        }
        avg_credit = sum(s.credit_total for s in ep_stats) / n_ep
        avg_capture = sum(s.n_capture for s in ep_stats) / n_ep
        avg_defense = sum(s.n_defense for s in ep_stats) / n_ep
        avg_planet_cap = sum(
            (s.episode_log or {}).get("n_planet_captures", 0) for s in ep_stats) / n_ep
        avg_planet_loss = sum(
            (s.episode_log or {}).get("n_planet_losses", 0) for s in ep_stats) / n_ep
        elapsed = time.time() - t0

        log = {
            "iter": it,
            "episodes": n_ep,
            "transitions": len(buffer),
            "avg_episode_length": round(avg_len, 1),
            "avg_launches": round(avg_launches, 1),
            "avg_p0_shaped_reward": round(avg_p0_reward, 4),
            "wins_p0": wins_p0, "wins_p1": wins_p1, "timeouts": timeouts,
            "wins_by_player": wins_by_player,
            "episode_num_agents": episode_num_agents_list,
            "num_agents_hist": n_agents_hist,
            "policy_loss": round(ppo_stats.policy_loss, 5),
            "value_loss": round(ppo_stats.value_loss, 5),
            "entropy": round(ppo_stats.entropy, 4),
            "entropy_coef": round(float(ent_now), 5),
            "approx_kl": round(ppo_stats.approx_kl, 5),
            "ppo_epochs_run": ppo_stats.epochs_run,
            "bc_loss": round(ppo_stats.bc_loss, 4),
            "bc_samples": ppo_stats.bc_samples,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "credit_total": round(avg_credit, 4),
            "n_capture": round(avg_capture, 1),
            "n_defense": round(avg_defense, 1),
            "n_planet_captures": round(avg_planet_cap, 1),
            "n_planet_losses": round(avg_planet_loss, 1),
            "elapsed_sec": round(elapsed, 1),
        }
        with log_path.open("a") as fp:
            fp.write(json.dumps(log) + "\n")

        win_str = " ".join(f"P{p}={wins_by_player.get(p, 0)}" for p in sorted(wins_by_player))
        na_str = ",".join(f"{k}:{v}" for k, v in n_agents_hist.items()) if n_agents_hist else "2:0"
        print(f"[iter {it:>4}/{train_cfg.iterations}] "
              f"games={n_ep} na={na_str} len={avg_len:.0f} launch={avg_launches:.1f} "
              f"{win_str} TO={timeouts} | "
              f"polL={log['policy_loss']:+.4f} valL={log['value_loss']:.4f} "
              f"H={log['entropy']:.3f} KL={log['approx_kl']:.4f} "
              + (f"cr={log['credit_total']:+.3f} cap={log['n_capture']:.1f} "
                 f"def={log['n_defense']:.1f} "
                 if (reward_cfg.enable_battle_credit or remastered_cfg is not None) else "")
              + (f"bc={ppo_stats.bc_loss:.3f}({ppo_stats.bc_samples}) "
                 if ppo_stats.bc_samples else "")
              + f"e={ppo_stats.epochs_run}/{ppo_cfg.epochs} ({elapsed:.1f}s)")

        if (it + 1) % train_cfg.checkpoint_every == 0 or it == train_cfg.iterations - 1:
            ckpt_dir = Path(train_cfg.checkpoint_dir)
            _save_ckpt(policy, optimizer, ckpt_dir / f"policy_iter{it:05d}.pt",
                       iteration=it, train_cfg=train_cfg, ppo_cfg=ppo_cfg,
                       reward_cfg=reward_cfg, remastered_cfg=remastered_cfg)
            _save_ckpt(policy, optimizer, ckpt_dir / "latest.pt",
                       iteration=it, train_cfg=train_cfg, ppo_cfg=ppo_cfg,
                       reward_cfg=reward_cfg, remastered_cfg=remastered_cfg)
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
    p.add_argument("--num-agents", type=int, default=2, choices=[2, 4],
                   help="Players per episode when --random-num-agents is off.")
    p.add_argument("--random-num-agents", action="store_true",
                   help="Sample 2 or 4 players per episode from --num-agent-choices.")
    p.add_argument("--num-agent-choices", type=_parse_num_agent_choices, default="2,4",
                   help="Comma-separated player counts for --random-num-agents (2 and/or 4).")
    p.add_argument("--opponent", choices=["self_play", "scripted", "frozen"],
                   default="self_play",
                   help="self_play: all seats = live policy. "
                        "scripted: P0 = train, others = scripted bot. "
                        "frozen: P0 = train, P1 = --opponent-ckpt; in 4p P2–P3 scripted.")
    p.add_argument("--opponent-ckpt", default=None,
                   help="Checkpoint path for P1 when --opponent=frozen.")
    p.add_argument("--teacher-forcing", type=float, default=0.0,
                   help="Per-decision probability the learning player executes the "
                        "scripted bot's move (teacher forcing) instead of its own. "
                        "These turns are not trained on; they steer the trajectory "
                        "into strong states. e.g. 0.1 = 10%%. Default 0 (off).")
    p.add_argument("--min-send", type=int, default=1,
                   help="Minimum ships per launch (v8 launch policy). Raise to "
                        "3-5 to force the Dirichlet split to concentrate (anti-spray). "
                        "Default 1.")
    p.add_argument("--bc-coef", type=float, default=0.0,
                   help="Behavior-cloning weight (v8 launch). With --teacher-forcing, "
                        "adds a supervised loss so the policy imitates the teacher's "
                        "executed launches (learns to concentrate/capture). e.g. 0.5. "
                        "Default 0 (off).")
    p.add_argument("--scripted-strategy", default="production",
                   help="Scripted bot strategy when --opponent=scripted "
                        "(see reworked_scripted.SCRIPTED_STRATEGIES).")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--policy", choices=["hier", "launch", "edge"], default="hier", dest="policy_kind",
                   help="'hier' = hierarchical + MCF allocator (v6/v7); 'launch' = "
                        "v8 source-centric launch policy (commit Beta + Dirichlet "
                        "split, allocator-free); 'edge' = v9 directed-edge policy "
                        "(independent per-edge gate+magnitude, per-target credit). "
                        "Different action spaces -> cannot resume across kinds "
                        "(start a fresh run).")
    p.add_argument("--no-aux-owner", action="store_false", dest="aux_owner",
                   help="Disable the auxiliary per-node owner-prediction head.")
    p.add_argument("--eta-value-feature", action="store_true", dest="eta_value_feature",
                   help="Add the ETA-aware capture-value node channel (node_dim "
                        "16->17): an arrival- and cost-normalised target value so "
                        "the priority head ranks nearby high-production captures "
                        "first.  Changes the feature dim -> requires a fresh run "
                        "(cannot resume from a 16-dim checkpoint).")
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
    p.add_argument("--loss-weight", type=float, default=1.0,
                   help="penalty per OWN ship destroyed (combat + sun)")
    p.add_argument("--launch-weight", type=float, default=0.1,
                   help="small penalty per ship committed to a launch")
    p.add_argument("--loss-norm", type=float, default=50.0,
                   help="ship-count scale for the loss/launch cost terms")
    p.add_argument("--battle-credit", action="store_true", dest="enable_battle_credit",
                   help="decision-indexed delayed credit via FIFO inventory accounting "
                        "(captures/defences credited to the deciding turn; drops the "
                        "production term from the potential to avoid double counting)")
    p.add_argument("--credit-capture-weight", type=float, default=1.0,
                   help="scale on +dpp(target) for a successful capture")
    p.add_argument("--credit-defense-weight", type=float, default=1.0,
                   help="scale on +/-dpp for a resolved defence (hold/loss)")
    p.add_argument("--credit-lost-race-weight", type=float, default=0.0,
                   help="penalty scale on -dpp(target) when a rival captures a "
                        "planet my fleets were committed to (campaign ledger)")
    p.add_argument("--credit-fail-weight", type=float, default=0.0,
                   help="penalty scale on -dpp(target) for attack waves that "
                        "never capture their target by episode end")
    p.add_argument("--credit-source-loss-weight", type=float, default=0.0,
                   help="over-extension penalty: scale on -dpp(source) charged to "
                        "decisions that recently stripped a planet that then fell "
                        "(the departed ships escaped the defence-loss penalty)")
    p.add_argument("--credit-source-loss-window", type=int, default=10,
                   help="only departures within this many turns before a planet "
                        "falls are blamed by --credit-source-loss-weight")
    p.add_argument("--airborne-weight", type=float, default=0.0,
                   help="attack-only penalty: scale on ship-turns in flight "
                        "(ships * (resolve_turn - launch_turn) / --airborne-norm); "
                        "requires --battle-credit (uses the same fleet log)")
    p.add_argument("--airborne-norm", type=float, default=50.0,
                   help="normaliser for --airborne-weight (defaults to loss scale)")
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
    if args.opponent == "frozen" and not args.opponent_ckpt:
        _build_argparser().error("--opponent frozen requires --opponent-ckpt")
    train_cfg = TrainConfig(
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        max_episode_steps=args.max_episode_steps,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        learning_rate=args.lr, seed=args.seed, device=args.device,
        routing_cost=args.routing_cost,
        num_agents=args.num_agents,
        random_num_agents=args.random_num_agents,
        num_agent_choices=args.num_agent_choices,
        opponent=args.opponent,
        opponent_ckpt=args.opponent_ckpt,
        scripted_strategy=args.scripted_strategy,
        teacher_forcing=args.teacher_forcing,
        min_send=args.min_send,
        d_model=args.d_model, n_layers=args.n_layers, aux_owner=args.aux_owner,
        policy_kind=args.policy_kind,
        eta_value_feature=args.eta_value_feature,
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
        bc_coef=args.bc_coef,
    )
    reward_cfg = RewardConfig(
        gamma=args.gamma,
        shaping_weight=args.shaping_weight,
        terminal_weight=args.terminal_weight,
        ship_weight=args.ship_weight,
        enable_shaping=args.enable_shaping,
        loss_weight=args.loss_weight,
        launch_weight=args.launch_weight,
        loss_norm=args.loss_norm,
        enable_battle_credit=args.enable_battle_credit,
        credit_capture_weight=args.credit_capture_weight,
        credit_defense_weight=args.credit_defense_weight,
        credit_lost_race_weight=args.credit_lost_race_weight,
        credit_fail_weight=args.credit_fail_weight,
        credit_source_loss_weight=args.credit_source_loss_weight,
        credit_source_loss_window=args.credit_source_loss_window,
        airborne_weight=args.airborne_weight,
        airborne_norm=args.airborne_norm,
    )
    train(train_cfg, ppo_cfg, reward_cfg, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
