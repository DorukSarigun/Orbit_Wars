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
    decode_action_with_launch,
    INCOMING_ETA_BUCKETS,
    PF_IDX_SHIPS,
    PF_IDX_INCOMING_SELF_B0,
)
from model import OrbitWarsPolicy, pack_observations, _beta_concentrations

from .train import load_checkpoint as load_ppo_checkpoint
from .dqn_train import load_dqn_checkpoint
from model_q import OrbitWarsQNet
from playground import render_board
from scripted_strategy import CaptureEfficiencyAgent, ScriptedConfig

MAX_LAUNCHES_PER_PLAYER_TURN = 8
MAX_PAIR_CANDIDATES = 64


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
        # DQN keeps a discrete ship_bin head; map bin -> ship_value [0,1]
        # so it can call the (now continuous) decode_action.
        n_bins = max(1, int(net.n_ship_bins))
        ship_bin = int(action["ship_bin"][0])
        ship_value = float(min(n_bins, ship_bin + 1)) / float(n_bins)
    else:
        with torch.no_grad():
            out = net(**batch)
            action = net.act(out, deterministic=deterministic)
        info_extra["value"] = float(out.value[0].item())
        ship_value = float(action["ship_value"][0])

    src_slot = int(action["src"][0])
    tgt_slot = int(action["tgt"][0])
    is_noop = bool(action["is_noop"][0])

    parsed = parse_planets(env_obs)
    move = decode_action(obs, src_slot, tgt_slot, ship_value,
                         planets=parsed, angular_velocity=omega, no_op=is_noop)
    return [move] if move is not None else [], {
        "src_slot": src_slot, "tgt_slot": tgt_slot,
        "ship_value": ship_value, "is_noop": is_noop,
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

    if getattr(net, "model_variant", "") in ("grouped_planet", "grouped_planet_edge"):
        available_pairs = obs.planet_mask[None, :].repeat(net.p_max, axis=0)
        available_pairs &= obs.source_mask[:, None]
        available_pairs &= ~torch.eye(net.p_max, dtype=torch.bool).numpy()
        while len(env_actions) < MAX_LAUNCHES_PER_PLAYER_TURN and any(bool(ok) for ok in obs_work.source_mask):
            obs_decision = replace(
                obs_work,
                planet_feats=obs_work.planet_feats.copy(),
                source_mask=obs_work.source_mask.copy(),
            )
            batch = pack_observations([obs_decision], device=device)
            with torch.no_grad():
                out = net(**batch)
            selected = None
            pair_logits = out.tgt_logits[0].clone()
            pair_logits[~torch.as_tensor(available_pairs, device=pair_logits.device)] = torch.finfo(pair_logits.dtype).min
            ranked = torch.topk(
                pair_logits.reshape(-1),
                k=min(MAX_PAIR_CANDIDATES, pair_logits.numel()),
            ).indices.tolist()
            for pair_idx in ranked:
                src_slot = int(pair_idx // net.p_max)
                tgt_slot = int(pair_idx % net.p_max)
                if not bool(obs_decision.source_mask[src_slot]):
                    continue
                if not bool(obs_decision.planet_mask[tgt_slot]):
                    continue
                if src_slot == tgt_slot:
                    continue
                if not bool(available_pairs[src_slot, tgt_slot]):
                    continue
                # Continuous Beta head: sample ship_value once per pair.
                # Use the deterministic mean alpha/(alpha+beta) for replay
                # so episodes are reproducible. ship_params is either
                # (B, p_max, p_max, 2) for grouped variants or (B, p_max, 2)
                # for the source-only full variant.
                if out.ship_params.dim() == 4:
                    ship_params_sel = out.ship_params[0, src_slot, tgt_slot]
                else:
                    ship_params_sel = out.ship_params[0, src_slot]
                alpha, beta = _beta_concentrations(ship_params_sel)
                ship_value_sample = float((alpha / (alpha + beta)).item())
                decoded = decode_action_with_launch(
                    obs_decision, src_slot, tgt_slot, ship_value_sample,
                    planets=parsed, angular_velocity=omega, no_op=False,
                )
                if decoded is not None:
                    selected = (src_slot, tgt_slot, ship_value_sample, decoded)
                    break
            if selected is None:
                break
            src_slot, tgt_slot, ship_value, decoded = selected
            move, launch_info = decoded
            env_actions.append(move)
            sent_ships = float(move[2])
            obs_work.planet_feats[src_slot, PF_IDX_SHIPS] = max(
                0.0, obs_work.planet_feats[src_slot, PF_IDX_SHIPS] - sent_ships / 100.0
            )
            src_planet_id = int(obs_decision.planet_ids[src_slot])
            src_planet = next((p for p in parsed if p.id == src_planet_id), None)
            if src_planet is not None:
                src_planet.ships = max(0, int(round(float(obs_work.planet_feats[src_slot, PF_IDX_SHIPS]) * 100.0)))
            eta = float(launch_info["eta"]) if launch_info is not None else 0.0
            if eta <= 0.0:
                tgt_planet_id = int(obs_decision.planet_ids[tgt_slot])
                tgt_planet = next((p for p in parsed if p.id == tgt_planet_id), None)
                if src_planet and tgt_planet:
                    distance = ((src_planet.x - tgt_planet.x) ** 2 + (src_planet.y - tgt_planet.y) ** 2) ** 0.5
                    eta = distance / max(fleet_speed(int(sent_ships)), 1e-6)
            bucket = _incoming_bucket_idx(eta)
            obs_work.planet_feats[tgt_slot, PF_IDX_INCOMING_SELF_B0 + bucket] += sent_ships / 100.0
            obs_work.source_mask[src_slot] = obs_work.planet_feats[src_slot, PF_IDX_SHIPS] > 0.01
            available_pairs[src_slot, tgt_slot] = False
            if not bool(obs_work.source_mask[src_slot]):
                available_pairs[src_slot, :] = False
            infos.append({
                "src_slot": src_slot,
                "tgt_slot": tgt_slot,
                "ship_value": ship_value,
                "is_noop": False,
                "value": float(out.value[0].item()),
            })
        return env_actions, infos

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
        ship_value = float(action["ship_value"][0])
        is_noop = bool(action["is_noop"][0])
        move = decode_action(
            obs_decision, src_slot, tgt_slot, ship_value,
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
            obs_work.planet_feats[tgt_slot, PF_IDX_INCOMING_SELF_B0 + bucket] += sent_ships / 100.0
        obs_work.source_mask[src_slot] = False
        infos.append({
            "src_slot": src_slot,
            "tgt_slot": tgt_slot,
            "ship_value": ship_value,
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


def _act_dispatch(net, env_obs, player: int, *, deterministic: bool,
                  device: torch.device, opponent_indices: set,
                  scripted_agent: Optional[CaptureEfficiencyAgent],
                  current_turn: int, max_turns: int) -> tuple:
    """Pick an action for `player`. Scripted slots get scripted actions;
    everyone else uses the loaded policy/Q-net.

    Returns the same tuple shape as `_act_all_sources` so callers can
    treat both paths uniformly: `(actions, info_extra)`.
    """
    if player in opponent_indices and scripted_agent is not None:
        actions = scripted_agent.act(
            env_obs, player=player,
            current_turn=int(current_turn), max_turns=int(max_turns),
        )
        return actions, {"scripted": True, "n_launches": len(actions)}
    return _act_all_sources(
        net, env_obs, player,
        deterministic=deterministic, device=device,
    )


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
    opponent_indices: Optional[set[int]] = None,
    scripted_cfg: Optional[ScriptedConfig] = None,
) -> dict:
    """Play one self-play game. Returns summary dict.

    If `render=True`, draws the board with `playground.render_board` every
    `render_every` turns and at terminal.
    """
    device = device or torch.device("cpu")
    policy.eval().to(device)
    opponent_indices = set(opponent_indices) if opponent_indices else set()
    scripted_agent: Optional[CaptureEfficiencyAgent] = (
        CaptureEfficiencyAgent(scripted_cfg) if opponent_indices else None
    )

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
            _act_dispatch(
                policy, state[p]["observation"], p,
                deterministic=deterministic, device=device,
                opponent_indices=opponent_indices,
                scripted_agent=scripted_agent,
                current_turn=t, max_turns=max_steps,
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
    opponent_indices: Optional[set[int]] = None,
    scripted_cfg: Optional[ScriptedConfig] = None,
) -> dict:
    """Play one game, write kaggle_environments' native interactive HTML
    replay to `out_path`, and return a summary dict.

    The resulting .html is a full replay player (scrub bar, play/pause,
    planet/fleet visualization) -- open it in any browser.
    """
    device = device or torch.device("cpu")
    net.eval().to(device)
    opponent_indices = set(opponent_indices) if opponent_indices else set()
    scripted_agent: Optional[CaptureEfficiencyAgent] = (
        CaptureEfficiencyAgent(scripted_cfg) if opponent_indices else None
    )

    env = make("orbit_wars", debug=False)
    if seed is not None:
        env.configuration["randomSeed"] = int(seed)
    state = env.reset(num_agents=num_agents)
    state = env.step([[] for _ in range(num_agents)])

    for t in range(max_steps):
        acted = [
            _act_dispatch(
                net, state[p]["observation"], p,
                deterministic=deterministic, device=device,
                opponent_indices=opponent_indices,
                scripted_agent=scripted_agent,
                current_turn=t, max_turns=max_steps,
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
    p.add_argument("--opponent", choices=["self_play", "scripted"],
                   default="self_play",
                   help="self_play (default): all slots use the loaded "
                        "policy. scripted: slot 0 = policy, all other "
                        "slots = CaptureEfficiencyAgent (matches "
                        "training-vs-scripted setup).")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    net, kind, _ = load_any_checkpoint(args.ckpt, device=torch.device(args.device))
    print(f"Loaded {kind.upper()} checkpoint: {sum(p.numel() for p in net.parameters()):,} params")

    opponent_indices: set[int] = set()
    if args.opponent == "scripted":
        opponent_indices = set(range(1, args.num_agents))
        print(f"Opponent slots (scripted): {sorted(opponent_indices)} | "
              f"Policy slots: [0]")

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
                opponent_indices=opponent_indices,
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
        opponent_indices=opponent_indices,
    )


if __name__ == "__main__":
    main()
