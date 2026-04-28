"""Load a checkpoint and play a self-play game with rendering.

Two ways to use:

  CLI / stdout play (no rendering):
    python -m rl.watch --ckpt checkpoints/latest.pt --no-render

  CLI with matplotlib popups every N steps:
    python -m rl.watch --ckpt checkpoints/latest.pt --render-every 20

  Notebook: import `play_self_play_game` and call it; pass `render=True`
  to draw frames inline as the game progresses.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from kaggle_environments import make

from feature_extraction import parse_planets
from obs_builder import build_observation, decode_action
from model import OrbitWarsPolicy, pack_observations

from .train import load_checkpoint as load_ppo_checkpoint
from .dqn_train import load_dqn_checkpoint
from model_q import OrbitWarsQNet
from playground import render_board


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
                            p_max=net.p_max, f_max=net.f_max)
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


def play_self_play_game(
    policy: OrbitWarsPolicy,
    *,
    max_steps: int = 500,
    deterministic: bool = False,
    render: bool = False,
    render_every: int = 20,
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
    state = env.reset(num_agents=2)
    state = env.step([[], []])

    frames = []
    history = []
    if render:
        render_board(state[0]["observation"], title="Step 0 — start")

    for t in range(max_steps):
        env_obs0 = state[0]["observation"]
        env_obs1 = state[1]["observation"]
        em0, info0 = _act(policy, env_obs0, 0, deterministic=deterministic, device=device)
        em1, info1 = _act(policy, env_obs1, 1, deterministic=deterministic, device=device)
        state = env.step([em0, em1])
        history.append({"step": t, "p0": info0, "p1": info1})

        if render and ((t + 1) % render_every == 0):
            render_board(state[0]["observation"], title=f"Step {t+1}")

        if state[0].status == "DONE" or env.done:
            break

    if render:
        render_board(state[0]["observation"], title=f"Final — step {t+1}")

    final_r = [int(s.reward or 0) for s in state]
    if final_r[0] > final_r[1]:
        winner = 0
    elif final_r[1] > final_r[0]:
        winner = 1
    else:
        winner = -1

    # Final ownership tally for a quick "who won what" summary.
    final_planets = parse_planets(state[0]["observation"])
    own_p0 = sum(1 for p in final_planets if p.owner == 0 and not p.is_comet)
    own_p1 = sum(1 for p in final_planets if p.owner == 1 and not p.is_comet)
    own_neutral = sum(1 for p in final_planets if p.owner == -1 and not p.is_comet)

    summary = {
        "length": t + 1,
        "winner": winner,
        "final_reward": final_r,
        "planets_p0": own_p0,
        "planets_p1": own_p1,
        "planets_neutral": own_neutral,
        "history": history,
    }
    if verbose:
        winner_str = {0: "P0", 1: "P1", -1: "DRAW"}[winner]
        print(f"Game over @ step {t+1}: winner={winner_str}, "
              f"planets P0={own_p0} / P1={own_p1} / neutral={own_neutral}")
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
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    net, kind, _ = load_any_checkpoint(args.ckpt, device=torch.device(args.device))
    print(f"Loaded {kind.upper()} checkpoint: {sum(p.numel() for p in net.parameters()):,} params")
    play_self_play_game(
        net,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        render=not args.no_render,
        render_every=args.render_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
