"""Behavior-cloning (BC) warm start for :class:`final_model.FinalNodePolicy`.

A random policy samples (target, source) almost uniformly, so the micro-planner
rejects ~75% of picks and PPO never sees a useful gradient.  This module teaches
the policy to imitate a scripted teacher's *capture* launches before RL, moving
it into a region where it picks feasible enemy/neutral targets from good donors.

Pipeline
--------
1. Play teacher-vs-teacher games on the real ``orbit_wars`` env.
2. For every teacher launch ``[from_id, angle, ships]`` recover the target id by
   matching ``compute_launch`` angles, and turn it into a
   ``(target_slot, source_slot, amount_frac)`` label.
3. Supervised loss: target CE + source CE + amount Beta-NLL (+ optional value
   regression to the production-share margin).

By default (``include_owned_targets=True``) launches toward owned planets
(reinforcement / defense / support) are also cloned, matching the observation's
expanded ``target_mask``.  With ``include_owned_targets=False`` only non-owned
(attack / expansion) targets are kept, matching the legacy action space.

The saved checkpoint is compatible with ``final_train_rl.load_checkpoint`` so
RL can continue via ``--resume-from``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from feature_extraction import parse_planets
from final_model import FinalNodePolicy
from final_obs import FinalObservation, build_final_observation
from navigation import compute_launch
from orbit_match import production_share_margin
from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig


# ---------------------------------------------------------------------------
# Teacher action -> policy label helpers
# ---------------------------------------------------------------------------
def _omega(env_obs: Any) -> float:
    if isinstance(env_obs, dict):
        return float(env_obs.get("angular_velocity", 0.0) or 0.0)
    return float(getattr(env_obs, "angular_velocity", 0.0) or 0.0)


def _angle_diff(a: float, b: float) -> float:
    d = (float(a) - float(b)) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d


def _recover_target_id(
    source,
    angle: float,
    ships: int,
    planets,
    omega: float,
    *,
    tol: float = 0.2,
) -> int:
    """Invert a teacher launch back to its target planet.

    The teacher built the launch with ``compute_launch`` so the true target
    reproduces the same angle.  We pick the reachable planet whose recomputed
    launch angle is closest to the teacher angle.
    """

    best_id = -1
    best_err = float("inf")
    for tgt in planets:
        if int(tgt.id) == int(source.id):
            continue
        launch = compute_launch(source, tgt, int(ships), planets, omega)
        if launch is None:
            continue
        err = abs(_angle_diff(float(launch["angle"]), float(angle)))
        if err < best_err:
            best_err = err
            best_id = int(tgt.id)
    return best_id if best_err <= float(tol) else -1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class BCDataset:
    obs: list[FinalObservation] = field(default_factory=list)
    obs_index: list[int] = field(default_factory=list)
    source_idx: list[int] = field(default_factory=list)
    target_idx: list[int] = field(default_factory=list)
    amount: list[float] = field(default_factory=list)
    value_target: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.source_idx)


def _stack_obs(obs_list: list[FinalObservation], device: torch.device) -> dict[str, torch.Tensor]:
    node = np.stack([o.node for o in obs_list])
    edge = np.stack([o.edge for o in obs_list])
    glob = np.stack([o.global_features for o in obs_list])
    valid = np.stack([o.valid_mask for o in obs_list])
    edge_mask = np.stack([o.edge_mask for o in obs_list])
    source = np.stack([o.source_mask for o in obs_list])
    target = np.stack([o.target_mask for o in obs_list])
    return {
        "node": torch.tensor(node, dtype=torch.float32, device=device),
        "edge": torch.tensor(edge, dtype=torch.float32, device=device),
        "global_features": torch.tensor(glob, dtype=torch.float32, device=device),
        "valid_mask": torch.tensor(valid, dtype=torch.bool, device=device),
        "edge_mask": torch.tensor(edge_mask, dtype=torch.bool, device=device),
        "source_mask": torch.tensor(source, dtype=torch.bool, device=device),
        "target_mask": torch.tensor(target, dtype=torch.bool, device=device),
    }


def _labels_from_launches(
    obs: FinalObservation,
    obs_raw: Any,
    launches: list,
    player: int,
    *,
    target_match_tol: float,
    include_owned_targets: bool = False,
) -> list[tuple[int, int, float]]:
    """Turn expert launch triples into policy slot labels for one turn.

    When ``include_owned_targets`` is False (legacy), launches toward owned
    planets are dropped because the legacy ``target_mask`` cannot represent
    them.  When True, reinforcement/defense launches are kept.
    """

    planets = parse_planets(obs_raw)
    by_id = {int(pl.id): pl for pl in planets}
    omega = _omega(obs_raw)
    id_to_slot = {
        int(pid): slot
        for slot, pid in enumerate(obs.planet_ids)
        if int(pid) >= 0
    }
    pending: list[tuple[int, int, float]] = []
    for launch in launches or []:
        try:
            sid = int(launch[0])
            angle = float(launch[1])
            ships = int(launch[2])
        except (TypeError, ValueError, IndexError):
            continue
        src = by_id.get(sid)
        if src is None or int(src.owner) != int(player) or int(src.ships) <= 0:
            continue
        tgt_id = _recover_target_id(
            src, angle, ships, planets, omega, tol=float(target_match_tol),
        )
        if tgt_id < 0:
            continue
        tgt = by_id.get(tgt_id)
        if tgt is None:
            continue
        if int(tgt.owner) == int(player) and not include_owned_targets:
            continue
        if sid not in id_to_slot or tgt_id not in id_to_slot:
            continue
        frac = float(ships) / max(1.0, float(src.ships))
        frac = min(1.0 - 1e-4, max(1e-4, frac))
        pending.append((id_to_slot[sid], id_to_slot[tgt_id], frac))
    return pending


def collect_bc_from_replay(
    replay: dict[str, Any],
    *,
    expert_player: int,
    target_match_tol: float = 0.2,
    max_steps: int | None = None,
    include_owned_targets: bool = True,
) -> BCDataset:
    """Extract BC samples from one Kaggle ``competitions replay`` JSON object."""

    ds = BCDataset()
    sample_player: list[int] = []
    steps = list(replay.get("steps") or [])
    if not steps:
        return ds

    num_agents = len(steps[0])
    expert = int(expert_player)
    if expert < 0 or expert >= num_agents:
        raise ValueError(f"expert_player={expert} out of range for {num_agents} agents")

    limit = len(steps) if max_steps is None else min(len(steps), int(max_steps) + 1)
    game_start = 0

    for turn_idx in range(limit):
        step = steps[turn_idx]
        if expert >= len(step):
            continue
        agent_state = step[expert]
        if isinstance(agent_state, dict):
            status = str(agent_state.get("status", "ACTIVE"))
            obs_raw = agent_state.get("observation")
            action = agent_state.get("action") or []
        else:
            status = str(getattr(agent_state, "status", "ACTIVE"))
            obs_raw = getattr(agent_state, "observation", None)
            action = getattr(agent_state, "action", None) or []

        if status != "ACTIVE" or obs_raw is None:
            continue

        obs = build_final_observation(
            obs_raw, expert, include_owned_targets=bool(include_owned_targets),
        )
        pending = _labels_from_launches(
            obs, obs_raw, list(action), expert,
            target_match_tol=float(target_match_tol),
            include_owned_targets=bool(include_owned_targets),
        )
        if pending:
            obs_idx = len(ds.obs)
            ds.obs.append(obs)
            for src_slot, tgt_slot, frac in pending:
                ds.obs_index.append(obs_idx)
                ds.source_idx.append(src_slot)
                ds.target_idx.append(tgt_slot)
                ds.amount.append(frac)
                ds.value_target.append(0.0)
                sample_player.append(expert)

    final_step = steps[-1]
    if expert < len(final_step):
        final_agent = final_step[expert]
        final_obs = (
            final_agent.get("observation")
            if isinstance(final_agent, dict)
            else getattr(final_agent, "observation", None)
        )
        if final_obs is not None:
            z = float(np.clip(
                production_share_margin(final_obs, expert, num_agents), -1.0, 1.0,
            ))
            for i in range(game_start, len(ds.source_idx)):
                ds.value_target[i] = z

    return ds


def collect_bc_from_leader_replays(
    manifest_path: Path | str,
    *,
    target_match_tol: float = 0.2,
    max_steps: int | None = None,
    include_owned_targets: bool = True,
    verbose: bool = True,
) -> BCDataset:
    """Load leader replay files indexed by ``kaggle_replays.download_leader_replays``."""

    from kaggle_replays import LeaderReplayManifest, load_replay_json

    manifest = LeaderReplayManifest.load(manifest_path)
    merged = BCDataset()
    total_launches = 0

    for ep, path in manifest.iter_episode_paths():
        if not path.is_file():
            if verbose:
                print(f"[bc-replay] skip missing {path}", flush=True)
            continue
        replay = load_replay_json(path)
        part = collect_bc_from_replay(
            replay,
            expert_player=int(ep.expert_agent_index),
            target_match_tol=float(target_match_tol),
            max_steps=max_steps,
            include_owned_targets=bool(include_owned_targets),
        )
        total_launches += len(part)
        base_obs = len(merged.obs)
        merged.obs.extend(part.obs)
        for obs_i in part.obs_index:
            merged.obs_index.append(base_obs + obs_i)
        merged.source_idx.extend(part.source_idx)
        merged.target_idx.extend(part.target_idx)
        merged.amount.extend(part.amount)
        merged.value_target.extend(part.value_target)
        if verbose:
            print(
                f"[bc-replay] episode={ep.episode_id} team={ep.expert_team_name!r} "
                f"expert={ep.expert_agent_index} samples={len(part)}",
                flush=True,
            )

    if verbose:
        print(
            f"[bc-replay] done: {len(merged)} samples from "
            f"{len(list(manifest.iter_episode_paths()))} episodes",
            flush=True,
        )
    return merged


def collect_bc_dataset(
    *,
    teacher_factory: Callable[[], Any],
    games: int,
    seed: int,
    max_steps: int,
    num_agents: int,
    target_match_tol: float = 0.2,
    include_owned_targets: bool = True,
    verbose: bool = True,
) -> BCDataset:
    from kaggle_environments import make

    ds = BCDataset()
    sample_player: list[int] = []  # parallel to source_idx, for value targets
    total_launches = 0
    used_launches = 0

    for g in range(int(games)):
        env = make("orbit_wars", debug=False)
        env.configuration["randomSeed"] = int(seed) * 100_000 + g
        episode_steps = int(getattr(env.configuration, "episodeSteps", 500) or 500)
        planner_max_turns = max(int(max_steps), episode_steps)
        env.reset(num_agents=num_agents)
        state = env.step([[] for _ in range(num_agents)])
        agents = [teacher_factory() for _ in range(num_agents)]
        game_start = len(ds.source_idx)

        for t in range(int(max_steps)):
            env_actions: list[list] = []
            for p in range(num_agents):
                obs_raw = state[p]["observation"]
                obs = build_final_observation(
                    obs_raw, p, include_owned_targets=bool(include_owned_targets),
                )
                teacher_actions = agents[p].act(
                    obs_raw, p, current_turn=int(t), max_turns=int(planner_max_turns),
                )
                env_actions.append(teacher_actions)

                for launch in teacher_actions:
                    total_launches += 1
                pending = _labels_from_launches(
                    obs, obs_raw, teacher_actions, p,
                    target_match_tol=float(target_match_tol),
                    include_owned_targets=bool(include_owned_targets),
                )

                if pending:
                    obs_idx = len(ds.obs)
                    ds.obs.append(obs)
                    for src_slot, tgt_slot, frac in pending:
                        ds.obs_index.append(obs_idx)
                        ds.source_idx.append(src_slot)
                        ds.target_idx.append(tgt_slot)
                        ds.amount.append(frac)
                        ds.value_target.append(0.0)
                        sample_player.append(int(p))
                        used_launches += 1

            state = env.step(env_actions)
            if bool(env.done):
                break

        # Assign value targets (production-share margin z) for this game.
        z = [
            float(np.clip(production_share_margin(state[p]["observation"], p, num_agents), -1.0, 1.0))
            for p in range(num_agents)
        ]
        for i in range(game_start, len(ds.source_idx)):
            ds.value_target[i] = z[sample_player[i]]

        if verbose:
            print(
                f"[bc-collect] game {g + 1}/{games} samples={len(ds)} "
                f"used/total_launches={used_launches}/{total_launches}",
                flush=True,
            )

    if verbose:
        kept = (used_launches / total_launches * 100.0) if total_launches else 0.0
        print(
            f"[bc-collect] done: {len(ds)} samples, "
            f"{used_launches}/{total_launches} launches kept ({kept:.1f}% attack launches)",
            flush=True,
        )
    return ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def bc_train(
    policy: FinalNodePolicy,
    ds: BCDataset,
    *,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 3e-4,
    amount_coef: float = 0.5,
    value_coef: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, float]:
    if len(ds) == 0:
        raise ValueError("BC dataset is empty; no usable teacher launches were collected.")

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(lr))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n = len(ds)
    src = torch.tensor(ds.source_idx, dtype=torch.long, device=device)
    tgt = torch.tensor(ds.target_idx, dtype=torch.long, device=device)
    amt = torch.tensor(ds.amount, dtype=torch.float32, device=device)
    val = torch.tensor(ds.value_target, dtype=torch.float32, device=device)
    obs_index = ds.obs_index

    last: dict[str, float] = {}
    policy.train()
    for epoch in range(int(epochs)):
        perm = torch.randperm(n, generator=g).tolist()
        tot = {"loss": 0.0, "tgt": 0.0, "src": 0.0, "amt": 0.0, "val": 0.0,
               "tgt_acc": 0.0, "src_acc": 0.0}
        steps = 0
        for start in range(0, n, int(batch_size)):
            mb = perm[start:start + int(batch_size)]
            obs_batch = [ds.obs[obs_index[i]] for i in mb]
            tensors = _stack_obs(obs_batch, device)
            mb_idx = torch.tensor(mb, dtype=torch.long, device=device)

            h, target_logits, source_logits, value = policy.logits(**tensors)
            tgt_b = tgt[mb_idx]
            src_b = src[mb_idx]
            amt_b = amt[mb_idx].clamp(1e-4, 1.0 - 1e-4)
            val_b = val[mb_idx]

            target_loss = F.cross_entropy(target_logits, tgt_b)
            source_loss = F.cross_entropy(source_logits, src_b)
            amount_dist = policy._amount_dist(h, src_b)
            amount_loss = -amount_dist.log_prob(amt_b).mean()
            value_loss = F.mse_loss(value, val_b)

            loss = (
                target_loss
                + source_loss
                + float(amount_coef) * amount_loss
                + float(value_coef) * value_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                tgt_acc = (target_logits.argmax(dim=-1) == tgt_b).float().mean().item()
                src_acc = (source_logits.argmax(dim=-1) == src_b).float().mean().item()
            tot["loss"] += float(loss.item())
            tot["tgt"] += float(target_loss.item())
            tot["src"] += float(source_loss.item())
            tot["amt"] += float(amount_loss.item())
            tot["val"] += float(value_loss.item())
            tot["tgt_acc"] += tgt_acc
            tot["src_acc"] += src_acc
            steps += 1

        last = {k: v / max(1, steps) for k, v in tot.items()}
        if verbose:
            print(
                f"[bc-train] epoch {epoch + 1}/{epochs} "
                f"loss={last['loss']:.4f} tgt={last['tgt']:.4f}(acc {last['tgt_acc']:.3f}) "
                f"src={last['src']:.4f}(acc {last['src_acc']:.3f}) "
                f"amt={last['amt']:.4f} val={last['val']:.4f}",
                flush=True,
            )
    return last


def save_bc_checkpoint(path: Path, policy: FinalNodePolicy, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": -1,  # so final_train_rl resumes at iter 0
        "policy_state_dict": policy.state_dict(),
        "args": vars(args),
        "arch": {
            "d_model": int(policy.d_model),
            "n_heads": int(policy.n_heads),
            "n_layers": int(policy.n_layers),
        },
    }, path)


def _teacher_factory(kind: str) -> Callable[[], Any]:
    kind = str(kind).lower()
    if kind == "scripted":
        return lambda: OrbitScriptedAgent(OrbitScriptedConfig())
    if kind == "mcts":
        from orbit_mcts import MCTSConfig
        from orbit_mcts_agent import OrbitMCTSAgent
        return lambda: OrbitMCTSAgent(OrbitScriptedConfig(), MCTSConfig())
    raise ValueError(f"unknown teacher kind: {kind!r}")


def train(args) -> None:
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))

    policy = FinalNodePolicy(
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"FinalNodePolicy params={n_params:,} device={device} teacher={args.teacher}")

    t0 = time.time()
    if args.from_replays:
        ds = collect_bc_from_leader_replays(
            args.from_replays,
            target_match_tol=float(args.target_match_tol),
            max_steps=int(args.max_steps) if int(args.max_steps) > 0 else None,
            include_owned_targets=bool(args.include_owned_targets),
            verbose=True,
        )
        if int(args.games) > 0:
            teacher_ds = collect_bc_dataset(
                teacher_factory=_teacher_factory(args.teacher),
                games=int(args.games),
                seed=int(args.seed),
                max_steps=int(args.max_steps),
                num_agents=int(args.num_agents),
                target_match_tol=float(args.target_match_tol),
                include_owned_targets=bool(args.include_owned_targets),
                verbose=True,
            )
            base_obs = len(ds.obs)
            ds.obs.extend(teacher_ds.obs)
            for obs_i in teacher_ds.obs_index:
                ds.obs_index.append(base_obs + obs_i)
            ds.source_idx.extend(teacher_ds.source_idx)
            ds.target_idx.extend(teacher_ds.target_idx)
            ds.amount.extend(teacher_ds.amount)
            ds.value_target.extend(teacher_ds.value_target)
            print(f"[bc] mixed {len(teacher_ds)} teacher samples", flush=True)
    else:
        ds = collect_bc_dataset(
            teacher_factory=_teacher_factory(args.teacher),
            games=int(args.games),
            seed=int(args.seed),
            max_steps=int(args.max_steps),
            num_agents=int(args.num_agents),
            target_match_tol=float(args.target_match_tol),
            include_owned_targets=bool(args.include_owned_targets),
            verbose=True,
        )
    print(f"[bc] collected {len(ds)} samples in {time.time() - t0:.1f}s", flush=True)

    stats = bc_train(
        policy,
        ds,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        amount_coef=float(args.amount_coef),
        value_coef=float(args.value_coef),
        seed=int(args.seed),
        verbose=True,
    )

    out = Path(args.out)
    save_bc_checkpoint(out, policy, args)
    log_path = out.parent / "bc_log.json"
    with log_path.open("w", encoding="utf-8") as fp:
        json.dump({"samples": len(ds), "final": stats, "args": vars(args)}, fp, indent=2)
    print(f"[bc] saved warm-start checkpoint -> {out}", flush=True)
    print(f"[bc] resume RL with: --resume-from {out}", flush=True)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Behavior-cloning warm start for FinalNodePolicy")
    p.add_argument(
        "--from-replays",
        default=None,
        help="Path to data/leader_replays/manifest.json (top-player Kaggle replays)",
    )
    p.add_argument("--games", type=int, default=30,
                   help="Teacher self-play games (0 when using --from-replays only)")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--num-agents", type=int, default=2)
    p.add_argument("--teacher", default="scripted", choices=["scripted", "mcts"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--amount-coef", type=float, default=0.5)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--target-match-tol", type=float, default=0.2)
    p.add_argument(
        "--include-owned-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clone reinforcement/defense launches toward owned planets too.",
    )
    p.add_argument("--out", default="checkpoints_final_bc/bc.pt")
    return p


def main(argv: list[str] | None = None) -> None:
    train(_build_argparser().parse_args(argv))


if __name__ == "__main__":
    main()
