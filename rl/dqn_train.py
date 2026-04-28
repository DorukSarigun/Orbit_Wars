"""Self-play DQN / Dueling DQN training loop.

Run as:
    python -m rl.dqn_train --iterations 100 --episodes-per-iter 4
    python -m rl.dqn_train --dueling --iterations 100

Each environment step both players act with the same online Q-net under
ε-greedy. Two transitions per env step (one per perspective) are pushed
into the replay buffer, exactly mirroring the PPO self-play structure.

After each episode we run a number of gradient updates proportional to
the new transitions pushed (`updates_per_step`).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from kaggle_environments import make

from feature_extraction import parse_planets
from obs_builder import build_observation, decode_action
from model import pack_observations
from model_q import OrbitWarsQNet

from .replay import DQNTransition, ReplayBuffer
from .dqn import DQNConfig, dqn_update, hard_update_target, soft_update_target
from .rewards import RewardConfig, compute_step_reward


@dataclass
class DQNTrainConfig:
    iterations: int = 100
    episodes_per_iter: int = 4
    max_episode_steps: int = 500
    updates_per_step: float = 1.0       # ratio: gradient steps per env step
    learn_starts: int = 1_000           # min transitions before any update
    batch_size: int = 256
    replay_capacity: int = 100_000
    target_sync_interval: int = 1_000   # env steps between hard syncs
    target_tau: float = 0.0             # soft updates if > 0; else hard sync
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    seed: int = 0
    checkpoint_every: int = 10
    checkpoint_dir: str = "checkpoints_dqn"
    log_path: str = "checkpoints_dqn/train_log.jsonl"
    device: str = "cpu"
    # Architecture.
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dueling: bool = False


def _epsilon_at(step: int, cfg: DQNTrainConfig) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = step / cfg.eps_decay_steps
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac


def _act_q(
    online: OrbitWarsQNet,
    env_obs,
    player: int,
    *,
    epsilon: float,
    device: torch.device,
    rng_torch: Optional[torch.Generator] = None,
) -> tuple:
    """Build observation, run Q-net + ε-greedy action selection, decode env move."""
    omega = env_obs.get("angular_velocity", 0.0) or 0.0
    obs = build_observation(env_obs, player=player, angular_velocity=omega,
                            p_max=online.p_max, f_max=online.f_max)
    batch = pack_observations([obs], device=device)
    with torch.no_grad():
        out = online(**batch)
        actions = OrbitWarsQNet.select_actions(
            out,
            source_mask=batch["source_mask"],
            planet_mask=batch["planet_mask"],
            epsilon=epsilon,
            rng=rng_torch,
        )

    src_slot = int(actions["src"][0])
    tgt_slot = int(actions["tgt"][0])
    ship_bin = int(actions["ship_bin"][0])
    is_noop = bool(actions["is_noop"][0])

    parsed = parse_planets(env_obs)
    move = decode_action(obs, src_slot, tgt_slot, ship_bin,
                         planets=parsed, angular_velocity=omega, no_op=is_noop)
    env_action = [move] if move is not None else []
    return obs, src_slot, tgt_slot, ship_bin, is_noop, env_action


def _save_ckpt(online, target_net, optimizer, path: Path, *, iteration: int,
               train_cfg: DQNTrainConfig, dqn_cfg: DQNConfig,
               reward_cfg: RewardConfig, total_env_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": iteration,
        "total_env_steps": total_env_steps,
        "online_state_dict": online.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_cfg": asdict(train_cfg),
        "dqn_cfg": asdict(dqn_cfg),
        "reward_cfg": asdict(reward_cfg),
        "arch": {
            "p_max": online.p_max,
            "f_max": online.f_max,
            "d_model": online.d_model,
            "n_heads": online.n_heads,
            "n_layers": len(online.blocks),
            "n_ship_bins": online.n_ship_bins,
            "dueling": online.dueling,
        },
        "kind": "dqn",
    }, path)


def load_dqn_checkpoint(path: str | Path, device: torch.device | None = None):
    """Rebuild Q-net (and target copy) from a DQN checkpoint."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    online = OrbitWarsQNet(
        p_max=arch["p_max"], f_max=arch["f_max"],
        d_model=arch["d_model"], n_heads=arch["n_heads"],
        n_layers=arch["n_layers"], n_ship_bins=arch["n_ship_bins"],
        dueling=arch.get("dueling", False),
    ).to(device)
    online.load_state_dict(ckpt["online_state_dict"])
    return online, ckpt


def train_dqn(
    train_cfg: Optional[DQNTrainConfig] = None,
    dqn_cfg: Optional[DQNConfig] = None,
    reward_cfg: Optional[RewardConfig] = None,
) -> OrbitWarsQNet:
    train_cfg = train_cfg or DQNTrainConfig()
    dqn_cfg = dqn_cfg or DQNConfig(gamma=train_cfg.gamma)
    reward_cfg = reward_cfg or RewardConfig()

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    rng_torch = torch.Generator(device=device).manual_seed(train_cfg.seed)
    rng_np = np.random.default_rng(train_cfg.seed)

    online = OrbitWarsQNet(
        d_model=train_cfg.d_model, n_heads=train_cfg.n_heads,
        n_layers=train_cfg.n_layers, dueling=train_cfg.dueling,
    ).to(device)
    target_net = copy.deepcopy(online).to(device)
    target_net.eval()
    for p in target_net.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(online.parameters(), lr=train_cfg.learning_rate)
    replay = ReplayBuffer(capacity=train_cfg.replay_capacity)

    n_params = sum(p.numel() for p in online.parameters())
    print(f"Q-net{' (Dueling)' if train_cfg.dueling else ''}: "
          f"{n_params:,} params, T={online.p_max+online.f_max+1}, device={device}")

    log_path = Path(train_cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_env_steps = 0
    update_count = 0
    pending_updates = 0.0

    for it in range(train_cfg.iterations):
        t0 = time.time()
        ep_lengths, ep_rewards_p0, ep_rewards_p1, ep_winners, ep_launches = [], [], [], [], []
        update_stats = {"dqn_loss": [], "td_error": [], "q_mean": [], "grad_norm": []}

        for ep in range(train_cfg.episodes_per_iter):
            env = make("orbit_wars", debug=False)
            env.configuration["randomSeed"] = (
                train_cfg.seed * 100000 + it * 1000 + ep
            )
            state = env.reset(num_agents=2)
            state = env.step([[], []])
            n_launches = [0, 0]
            cum_reward = [0.0, 0.0]

            for t in range(train_cfg.max_episode_steps):
                eps = _epsilon_at(total_env_steps, train_cfg)
                env_obs0 = state[0]["observation"]
                env_obs1 = state[1]["observation"]

                obs0, src0, tgt0, sh0, no0, em0 = _act_q(
                    online, env_obs0, 0, epsilon=eps, device=device, rng_torch=rng_torch)
                obs1, src1, tgt1, sh1, no1, em1 = _act_q(
                    online, env_obs1, 1, epsilon=eps, device=device, rng_torch=rng_torch)
                if not no0: n_launches[0] += 1
                if not no1: n_launches[1] += 1

                new_state = env.step([em0, em1])
                terminal = (new_state[0].status == "DONE") or env.done \
                    or (t == train_cfg.max_episode_steps - 1)

                new_obs0 = new_state[0]["observation"]
                new_obs1 = new_state[1]["observation"]
                r0 = compute_step_reward(env_obs0, new_obs0, 0,
                                         terminal=terminal,
                                         env_terminal_reward=new_state[0].reward or 0,
                                         cfg=reward_cfg)
                r1 = compute_step_reward(env_obs1, new_obs1, 1,
                                         terminal=terminal,
                                         env_terminal_reward=new_state[1].reward or 0,
                                         cfg=reward_cfg)
                cum_reward[0] += r0; cum_reward[1] += r1

                # We need next-state observations from each player's view.
                from obs_builder import build_observation as _bo
                omega_next = new_obs0.get("angular_velocity", 0.0) or 0.0
                next_obs0 = _bo(new_obs0, player=0, angular_velocity=omega_next,
                                p_max=online.p_max, f_max=online.f_max)
                next_obs1 = _bo(new_obs1, player=1, angular_velocity=omega_next,
                                p_max=online.p_max, f_max=online.f_max)

                for cur, nxt, sl, tg, sh, no, r in [
                    (obs0, next_obs0, src0, tgt0, sh0, no0, r0),
                    (obs1, next_obs1, src1, tgt1, sh1, no1, r1),
                ]:
                    replay.add(DQNTransition(
                        planet_feats=cur.planet_feats, fleet_feats=cur.fleet_feats,
                        global_feats=cur.global_feats, edge_bias=cur.edge_bias,
                        planet_mask=cur.planet_mask, fleet_mask=cur.fleet_mask,
                        source_mask=cur.source_mask,
                        src=sl, tgt=tg, ship_bin=sh, is_noop=no,
                        reward=r, done=terminal,
                        next_planet_feats=nxt.planet_feats, next_fleet_feats=nxt.fleet_feats,
                        next_global_feats=nxt.global_feats, next_edge_bias=nxt.edge_bias,
                        next_planet_mask=nxt.planet_mask, next_fleet_mask=nxt.fleet_mask,
                        next_source_mask=nxt.source_mask,
                    ))

                total_env_steps += 1
                pending_updates += train_cfg.updates_per_step

                # Run gradient steps.
                while (pending_updates >= 1.0
                       and replay.can_sample(train_cfg.batch_size)
                       and len(replay) >= train_cfg.learn_starts):
                    pending_updates -= 1.0
                    sample = replay.sample(train_cfg.batch_size,
                                           device=device, rng=rng_np)
                    stats = dqn_update(online, target_net, sample, optimizer, cfg=dqn_cfg)
                    for k, v in stats.items():
                        update_stats[k].append(v)
                    update_count += 1

                    # Target network update.
                    if train_cfg.target_tau > 0.0:
                        soft_update_target(target_net, online, tau=train_cfg.target_tau)
                    elif update_count % max(1, train_cfg.target_sync_interval // max(1, int(train_cfg.updates_per_step))) == 0:
                        hard_update_target(target_net, online)

                state = new_state
                if terminal:
                    break

            final_r = [int(s.reward or 0) for s in state]
            if final_r[0] > final_r[1]: winner = 0
            elif final_r[1] > final_r[0]: winner = 1
            else: winner = -1
            ep_lengths.append(t + 1)
            ep_rewards_p0.append(cum_reward[0])
            ep_rewards_p1.append(cum_reward[1])
            ep_winners.append(winner)
            ep_launches.append((n_launches[0] + n_launches[1]) / 2)

        elapsed = time.time() - t0
        wins_p0 = sum(1 for w in ep_winners if w == 0)
        wins_p1 = sum(1 for w in ep_winners if w == 1)
        draws = sum(1 for w in ep_winners if w == -1)
        log = {
            "iter": it,
            "total_env_steps": total_env_steps,
            "replay_size": len(replay),
            "epsilon": round(_epsilon_at(total_env_steps, train_cfg), 4),
            "avg_episode_length": round(sum(ep_lengths)/len(ep_lengths), 1),
            "avg_launches_per_player": round(sum(ep_launches)/len(ep_launches), 1),
            "wins_p0": wins_p0, "wins_p1": wins_p1, "draws": draws,
            "n_updates": len(update_stats["dqn_loss"]),
            "dqn_loss": round(np.mean(update_stats["dqn_loss"]) if update_stats["dqn_loss"] else 0.0, 5),
            "td_error": round(np.mean(update_stats["td_error"]) if update_stats["td_error"] else 0.0, 4),
            "q_mean": round(np.mean(update_stats["q_mean"]) if update_stats["q_mean"] else 0.0, 4),
            "elapsed_sec": round(elapsed, 1),
        }
        print(f"[iter {it:>4}] env_steps={total_env_steps:>6} replay={len(replay):>6} "
              f"eps={log['epsilon']:.3f} len={log['avg_episode_length']:>5.1f} "
              f"P0={wins_p0} P1={wins_p1} D={draws} | "
              f"loss={log['dqn_loss']:.4f} td={log['td_error']:.3f} q={log['q_mean']:+.3f} "
              f"upd={log['n_updates']} ({elapsed:.1f}s)")
        with log_path.open("a") as fp:
            fp.write(json.dumps(log) + "\n")

        if (it + 1) % train_cfg.checkpoint_every == 0 or it == train_cfg.iterations - 1:
            ckpt_path = Path(train_cfg.checkpoint_dir) / f"qnet_iter{it:05d}.pt"
            _save_ckpt(online, target_net, optimizer, ckpt_path,
                       iteration=it, train_cfg=train_cfg, dqn_cfg=dqn_cfg,
                       reward_cfg=reward_cfg, total_env_steps=total_env_steps)
            latest = Path(train_cfg.checkpoint_dir) / "latest.pt"
            _save_ckpt(online, target_net, optimizer, latest,
                       iteration=it, train_cfg=train_cfg, dqn_cfg=dqn_cfg,
                       reward_cfg=reward_cfg, total_env_steps=total_env_steps)
            print(f"  checkpoint saved -> {ckpt_path}")

    return online


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OrbitWars self-play DQN / Dueling DQN")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--episodes-per-iter", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--updates-per-step", type=float, default=1.0)
    p.add_argument("--learn-starts", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--replay-capacity", type=int, default=100_000)
    p.add_argument("--target-sync-interval", type=int, default=1000)
    p.add_argument("--target-tau", type=float, default=0.0)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=50_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--checkpoint-dir", default="checkpoints_dqn")
    p.add_argument("--log-path", default="checkpoints_dqn/train_log.jsonl")
    p.add_argument("--device", default="cpu")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dueling", action="store_true",
                   help="Use Dueling DQN architecture (V + A decomposition)")
    p.add_argument("--no-double-dqn", action="store_true",
                   help="Disable Double DQN (use vanilla DQN target)")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    train_cfg = DQNTrainConfig(
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        max_episode_steps=args.max_episode_steps,
        updates_per_step=args.updates_per_step,
        learn_starts=args.learn_starts,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_sync_interval=args.target_sync_interval,
        target_tau=args.target_tau,
        eps_start=args.eps_start, eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        gamma=args.gamma, learning_rate=args.lr,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir, log_path=args.log_path,
        device=args.device,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dueling=args.dueling,
    )
    dqn_cfg = DQNConfig(
        gamma=args.gamma,
        double_dqn=not args.no_double_dqn,
    )
    train_dqn(train_cfg, dqn_cfg)


if __name__ == "__main__":
    main()
