"""Load a checkpoint and play a self-play game with rendering.

Three ways to use (most to least recommended):

  **Interactive HTML replay** (kaggle_environments native player):
    python -m rl.watch --ckpt checkpoints/latest.pt --html replay.html
    python -m rl.watch --ckpt ... --html replays/g.html --games 5
    # -> open replay.html in any browser: full scrub-bar animation.

  **Notebook inline**: import `record_game_html` and embed the result
    via IPython.display.IFrame, or call `play_self_play_game(render=True)`
    for matplotlib snapshots every N steps.

  **CLI matplotlib popup** (GUI desktop only):
    python -m rl.watch --ckpt checkpoints/latest.pt --render-every 20
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch
from kaggle_environments import make

from feature_extraction import parse_planets, fleet_speed
from obs_builder import (
    build_observation,
    decode_action,
    INCOMING_ETA_BUCKETS,
    PF_IDX_SHIPS,
    PF_IDX_INCOMING_SELF_B0,
)
from model import OrbitWarsPolicy, pack_observations

from .train import load_checkpoint as load_ppo_checkpoint
from .dqn_train import load_dqn_checkpoint
from model_q import OrbitWarsQNet
from playground import render_board


def _incoming_bucket_idx(eta: float) -> int:
    for i, upper in enumerate(INCOMING_ETA_BUCKETS):
        if eta <= upper:
            return i
    return len(INCOMING_ETA_BUCKETS) - 1


def load_any_checkpoint(path, device=None):
    """Auto-detect PPO vs DQN checkpoint and return (net, kind, ckpt)."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("kind") == "dqn" or "online_state_dict" in ckpt:
        net, ckpt = load_dqn_checkpoint(path, device=device)
        return net, "dqn", ckpt
    net, ckpt = load_ppo_checkpoint(path, device=device)
    return net, "ppo", ckpt


def _act(net, env_obs, player: int, *,
         deterministic: bool, device: torch.device) -> tuple:
    """Build obs, sample/argmax action, decode to env format. Works for
    both `OrbitWarsPolicy` (PPO) and `OrbitWarsQNet` (DQN)."""
    omega = env_obs.get("angular_velocity", 0.0) or 0.0
    obs = build_observation(env_obs, player=player, angular_velocity=omega,
                            p_max=net.p_max, f_max=net.f_max,
                            planet_feature_dim=net.planet_dim)
    batch = pack_observations([obs], device=device)

    info_extra = {}
    if isinstance(net, OrbitWarsQNet):
        with torch.no_grad():
            out = net(**batch)
            action = OrbitWarsQNet.select_actions(
                out, source_mask=batch["source_mask"], planet_mask=batch["planet_mask"],
                epsilon=0.0,  # always greedy at watch time
            )
        info_extra["v_state"] = (
            float(out.v_state[0].item()) if out.v_state is not None else None
        )
    else:
        with torch.no_grad():
            out = net(**batch)
            action = net.act(out, deterministic=deterministic)
        info_extra["value"] = float(out.value[0].item())

    src_slot = int(action["src"][0])
    tgt_slot = int(action["tgt"][0])
    ship_bin = int(action["ship_bin"][0])
    is_noop = bool(action["is_noop"][0])

    parsed = parse_planets(env_obs)
    move = decode_action(obs, src_slot, tgt_slot, ship_bin,
                         planets=parsed, angular_velocity=omega, no_op=is_noop)
    return [move] if move is not None else [], {
        "src_slot": src_slot, "tgt_slot": tgt_slot,
        "ship_bin": ship_bin, "is_noop": is_noop,
        **info_extra,
    }


def _act_all_sources(net, env_obs, player: int, *,
                     deterministic: bool, device: torch.device) -> tuple:
    if isinstance(net, OrbitWarsQNet):
        return _act(net, env_obs, player, deterministic=deterministic, device=device)

    omega = env_obs.get("angular_velocity", 0.0) or 0.0
    obs = build_observation(env_obs, player=player, angular_velocity=omega,
                            p_max=net.p_max, f_max=net.f_max,
                            planet_feature_dim=net.planet_dim)
    parsed = parse_planets(env_obs)
    env_actions = []
    infos = []
    obs_work = replace(
        obs,
        planet_feats=obs.planet_feats.copy(),
        source_mask=obs.source_mask.copy(),
    )

    for src_slot in [int(i) for i, ok in enumerate(obs.source_mask) if ok]:
        obs_decision = replace(
            obs_work,
            planet_feats=obs_work.planet_feats.copy(),
            source_mask=obs_work.source_mask.copy(),
        )
        batch = pack_observations([obs_decision], device=device)
        with torch.no_grad():
            out = net(**batch)
            action = net.act_source(
                out,
                torch.tensor([src_slot], device=device),
                deterministic=deterministic,
            )

        tgt_slot = int(action["tgt"][0])
        ship_bin = int(action["ship_bin"][0])
        is_noop = bool(action["is_noop"][0])
        move = decode_action(
            obs_decision, src_slot, tgt_slot, ship_bin,
            planets=parsed, angular_velocity=omega, no_op=is_noop,
        )
        if move is not None:
            env_actions.append(move)
            sent_ships = float(move[2])
            obs_work.planet_feats[src_slot, PF_IDX_SHIPS] = max(
                0.0, obs_work.planet_feats[src_slot, PF_IDX_SHIPS] - sent_ships / 100.0
            )
            src_planet_id = int(obs_decision.planet_ids[src_slot])
            tgt_planet_id = int(obs_decision.planet_ids[tgt_slot])
            src_planet = next((p for p in parsed if p.id == src_planet_id), None)
            tgt_planet = next((p for p in parsed if p.id == tgt_planet_id), None)
            distance = 0.0
            if src_planet and tgt_planet:
                distance = ((src_planet.x - tgt_planet.x) ** 2 + (src_planet.y - tgt_planet.y) ** 2) ** 0.5
            eta = distance / max(fleet_speed(int(sent_ships)), 1e-6)
            bucket = _incoming_bucket_idx(eta)
            obs_work.planet_feats[tgt_slot, PF_IDX_INCOMING_SELF_B0 + bucket] += sent_ships / 500.0
        obs_work.source_mask[src_slot] = False
        infos.append({
            "src_slot": src_slot,
            "tgt_slot": tgt_slot,
            "ship_bin": ship_bin,
            "is_noop": is_noop,
            "value": float(out.value[0].item()),
        })

    return env_actions, infos


def _winner_from_rewards(final_r: list[int]) -> int:
    best_reward = max(final_r) if final_r else 0
    winners = [p for p, r in enumerate(final_r) if r == best_reward]
    return winners[0] if len(winners) == 1 and best_reward > 0 else -1


def _planet_counts_by_owner(env_obs: dict, num_agents: int) -> dict:
    final_planets = parse_planets(env_obs)
    counts = {
        f"planets_p{p}": sum(1 for pl in final_planets if pl.owner == p and not pl.is_comet)
        for p in range(num_agents)
    }
    counts["planets_neutral"] = sum(1 for pl in final_planets if pl.owner == -1 and not pl.is_comet)
    return counts


def play_self_play_game(
    policy: OrbitWarsPolicy,
    *,
    max_steps: int = 500,
    deterministic: bool = False,
    render: bool = False,
    render_every: int = 20,
    num_agents: int = 2,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """Play one self-play game. Returns summary dict.

    If `render=True`, draws the board with `playground.render_board` every
    `render_every` turns and at terminal.
    """
    device = device or torch.device("cpu")
    policy.eval().to(device)

    env = make("orbit_wars", debug=False)
    if seed is not None:
        env.configuration["randomSeed"] = int(seed)
    state = env.reset(num_agents=num_agents)
    state = env.step([[] for _ in range(num_agents)])

    frames = []
    history = []
    if render:
        render_board(state[0]["observation"], title="Step 0 — start")

    for t in range(max_steps):
        acted = [
            _act_all_sources(
                policy, state[p]["observation"], p,
                deterministic=deterministic, device=device,
            )
            for p in range(num_agents)
        ]
        state = env.step([actions for actions, _ in acted])
        history.append({
            "step": t,
            **{f"p{p}": info for p, (_, info) in enumerate(acted)},
        })

        if render and ((t + 1) % render_every == 0):
            render_board(state[0]["observation"], title=f"Step {t+1}")

        if any(s.status == "DONE" for s in state) or env.done:
            break

    if render:
        render_board(state[0]["observation"], title=f"Final — step {t+1}")

    final_r = [int(s.reward or 0) for s in state]
    winner = _winner_from_rewards(final_r)
    planet_counts = _planet_counts_by_owner(state[0]["observation"], num_agents)

    summary = {
        "length": t + 1,
        "winner": winner,
        "final_reward": final_r,
        **planet_counts,
        "history": history,
    }
    if verbose:
        winner_str = f"P{winner}" if winner >= 0 else "DRAW"
        counts_str = " / ".join(
            f"P{p}={planet_counts.get(f'planets_p{p}', 0)}"
            for p in range(num_agents)
        )
        print(f"Game over @ step {t+1}: winner={winner_str}, "
              f"planets {counts_str} / neutral={planet_counts['planets_neutral']}")
    return summary


def record_game_html(
    net,
    out_path: str | Path,
    *,
    max_steps: int = 500,
    deterministic: bool = False,
    num_agents: int = 2,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """Play one game, write kaggle_environments' native interactive HTML
    replay to `out_path`, and return a summary dict.

    The resulting .html is a full replay player (scrub bar, play/pause,
    planet/fleet visualization) -- open it in any browser.
    """
    device = device or torch.device("cpu")
    net.eval().to(device)

    env = make("orbit_wars", debug=False)
    if seed is not None:
        env.configuration["randomSeed"] = int(seed)
    state = env.reset(num_agents=num_agents)
    state = env.step([[] for _ in range(num_agents)])

    for t in range(max_steps):
        acted = [
            _act_all_sources(
                net, state[p]["observation"], p,
                deterministic=deterministic, device=device,
            )
            for p in range(num_agents)
        ]
        state = env.step([actions for actions, _ in acted])
        if any(s.status == "DONE" for s in state) or env.done:
            break

    html = env.render(mode="html")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    final_r = [int(s.reward or 0) for s in state]
    winner = _winner_from_rewards(final_r)
    planet_counts = _planet_counts_by_owner(state[0]["observation"], num_agents)
    summary = {
        "length": t + 1,
        "winner": winner,
        "final_reward": final_r,
        **planet_counts,
        "html_path": str(out_path),
        "html_size_kb": round(len(html) / 1024, 1),
    }
    if verbose:
        winner_str = f"P{winner}" if winner >= 0 else "DRAW"
        print(f"[replay] {winner_str} wins @ step {t+1} -> {out_path} "
              f"({summary['html_size_kb']} KB)")
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Watch self-play with a trained checkpoint")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true",
                   help="Argmax actions (no sampling)")
    p.add_argument("--no-render", action="store_true",
                   help="Don't open matplotlib windows")
    p.add_argument("--render-every", type=int, default=20)
    p.add_argument("--num-agents", type=int, default=2,
                   help="Number of self-play agents in each game")
    p.add_argument("--html", default=None, metavar="PATH",
                   help="Export interactive HTML replay to PATH and skip "
                        "matplotlib rendering. Open PATH in a browser.")
    p.add_argument("--games", type=int, default=1,
                   help="Number of games to play/record (with --html, each "
                        "becomes a separate file PATH_NNN.html).")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    net, kind, _ = load_any_checkpoint(args.ckpt, device=torch.device(args.device))
    print(f"Loaded {kind.upper()} checkpoint: {sum(p.numel() for p in net.parameters()):,} params")

    if args.html:
        base = Path(args.html)
        for i in range(args.games):
            if args.games == 1:
                out = base
            else:
                out = base.with_stem(f"{base.stem}_{i:03d}")
            seed = (args.seed or 0) + i if args.seed is not None else None
            record_game_html(
                net, out, max_steps=args.max_steps,
                deterministic=args.deterministic, num_agents=args.num_agents, seed=seed,
                device=torch.device(args.device),
            )
        return

    play_self_play_game(
        net,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        render=not args.no_render,
        render_every=args.render_every,
        num_agents=args.num_agents,
        seed=args.seed,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
