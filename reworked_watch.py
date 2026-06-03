"""Watch a trained reworked policy play — writes an interactive HTML replay.

Loads a :mod:`reworked_train` checkpoint and plays one game (policy vs the
scripted production bot, or self-play), then writes kaggle_environments' native
interactive HTML replay (scrub bar, play/pause, planet/fleet view) — open it in
any browser.

Usage::

    python -m reworked_watch --ckpt checkpoints_reworked_scripted/latest.pt \
        --opponent scripted --html replay.html
    python -m reworked_watch --ckpt .../latest.pt --opponent self_play --html r.html
    # P0 vs a second checkpoint (e.g. v5 vs v6):
    python -m reworked_watch --ckpt checkpoints_reworked_v5/policy_iter00139.pt \
        --ckpt-p1 checkpoints_reworked_v6_credit/policy_iter00139.pt \
        --opponent policy --html v5_vs_v6.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

try:  # package layout
    from .obs_builder import build_observation
    from .feature_extraction import parse_planets
    from .reworked_pipeline import pack_reworked_batch
    from .reworked_rollout import (
        action_to_env_launches, _target_masks, P_MAX, F_MAX,
    )
    from .reworked_scripted import scripted_action_row
    from .reworked_train import load_checkpoint
except ImportError:  # pragma: no cover - flat layout / direct execution
    from obs_builder import build_observation  # type: ignore
    from feature_extraction import parse_planets  # type: ignore
    from reworked_pipeline import pack_reworked_batch  # type: ignore
    from reworked_rollout import (  # type: ignore
        action_to_env_launches, _target_masks, P_MAX, F_MAX,
    )
    from reworked_scripted import scripted_action_row  # type: ignore
    from reworked_train import load_checkpoint  # type: ignore


def _policy_env_action(policy, env_obs, player, t, max_steps, device, deterministic):
    """One turn's env launches from the trained policy (greedy by default)."""
    av = env_obs.get("angular_velocity", 0.0) or 0.0
    obs = build_observation(env_obs, player=player, angular_velocity=av,
                            p_max=P_MAX, f_max=F_MAX, turn=t, max_turns=max_steps)
    batch = pack_reworked_batch([(obs, player)], device=device, p_max=P_MAX)
    attack_np, defend_np = _target_masks(batch["states"][0], P_MAX, player)
    action, _, _ = policy.act(
        node_feats=batch["node_feats"], edge_feats=batch["edge_feats"],
        global_feats=batch["global_feats"], planet_mask=batch["planet_mask"],
        attack_mask=torch.from_numpy(attack_np)[None].to(device),
        defend_mask=torch.from_numpy(defend_np)[None].to(device),
        deterministic=deterministic,
    )
    row = {
        "attack_sel": action["attack_sel"][0].cpu().numpy(),
        "defend_sel": action["defend_sel"][0].cpu().numpy(),
        "order": action["order"][0].cpu().numpy(),
        "commit_frac": action["commit_frac"][0].cpu().numpy(),
        "potential_weight": float(action["potential_weight"][0]),
        "horizon": float(action["horizon"][0]),
    }
    ea, _ = action_to_env_launches(obs, player, row, parse_planets(env_obs), av)
    return ea


def _scripted_env_action(env_obs, player, t, max_steps, device, strategy):
    av = env_obs.get("angular_velocity", 0.0) or 0.0
    obs = build_observation(env_obs, player=player, angular_velocity=av,
                            p_max=P_MAX, f_max=F_MAX, turn=t, max_turns=max_steps)
    batch = pack_reworked_batch([(obs, player)], device=device, p_max=P_MAX)
    row = scripted_action_row(strategy, batch["states"][0], player, p_max=P_MAX)
    ea, _ = action_to_env_launches(obs, player, row, parse_planets(env_obs), av)
    return ea


def record_game_html(
    policy,
    out_path: str | Path,
    *,
    policy_p1: Optional[torch.nn.Module] = None,
    num_agents: int = 2,
    max_steps: int = 300,
    opponent: str = "scripted",
    scripted_strategy: str = "production",
    deterministic: bool = True,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """Play one game and write the native interactive HTML replay."""
    from kaggle_environments import make
    import random as _random

    device = device or torch.device("cpu")
    num_agents = int(num_agents)
    if num_agents not in (2, 4):
        raise ValueError("num_agents must be 2 or 4")
    policy.eval().to(device)
    if policy_p1 is not None:
        policy_p1.eval().to(device)
    players = list(range(num_agents))
    if opponent == "scripted":
        scripted_players = set(range(1, num_agents))
    elif opponent == "policy":
        if policy_p1 is None:
            raise ValueError("opponent='policy' requires policy_p1 (use --ckpt-p1).")
        if num_agents != 2:
            raise ValueError("opponent='policy' (head-to-head) supports num_agents=2 only")
        scripted_players = set()
    else:
        scripted_players = set()

    env = make("orbit_wars", debug=False)
    if seed is not None:
        env.configuration["randomSeed"] = int(seed)
        _random.seed(int(seed))
    env.reset(num_agents=num_agents)
    state = env.step([[] for _ in players])

    t = 0
    with torch.no_grad():
        for t in range(max_steps):
            env_actions = []
            for p in players:
                env_obs = state[p]["observation"]
                if p in scripted_players:
                    ea = _scripted_env_action(env_obs, p, t, max_steps, device, scripted_strategy)
                elif opponent == "policy" and p == 1:
                    ea = _policy_env_action(policy_p1, env_obs, p, t, max_steps, device, deterministic)
                elif opponent == "self_play":
                    ea = _policy_env_action(policy, env_obs, p, t, max_steps, device, deterministic)
                else:
                    ea = _policy_env_action(policy, env_obs, p, t, max_steps, device, deterministic)
                env_actions.append(ea)
            state = env.step(env_actions)
            if env.done or any(getattr(state[p], "status", "ACTIVE") == "DONE" for p in players):
                break

    html = env.render(mode="html")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    rewards = [float(getattr(state[p], "reward", 0.0) or 0.0) for p in players]
    winner = int(max(range(len(rewards)), key=lambda i: rewards[i])) if rewards else -1
    if rewards.count(max(rewards)) > 1:
        winner = -1
    summary = {
        "length": t + 1,
        "winner": winner,
        "rewards": rewards,
        "num_agents": num_agents,
        "opponent": opponent,
        "html_path": str(out_path),
        "html_size_kb": round(len(html) / 1024, 1),
    }
    if verbose:
        who = "P0" if winner == 0 else (f"P{winner}" if winner >= 0 else "DRAW")
        if opponent == "scripted":
            opp = f"vs {scripted_strategy} bot"
        elif opponent == "policy":
            opp = "vs policy(P1)"
        else:
            opp = "self-play"
        print(f"[replay] {who} wins @ step {t+1} ({opp}) -> {out_path} "
              f"({summary['html_size_kb']} KB)")
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Watch a trained reworked policy (HTML replay)")
    p.add_argument("--ckpt", required=True, help="Path to a reworked checkpoint .pt")
    p.add_argument("--html", default="replay.html", help="Output HTML replay path")
    p.add_argument("--opponent", choices=["self_play", "scripted", "policy"], default="scripted",
                   help="'policy' = P0 uses --ckpt, P1 uses --ckpt-p1 (two checkpoints).")
    p.add_argument("--ckpt-p1", default=None,
                   help="Second checkpoint for player 1 (--opponent policy).")
    p.add_argument("--scripted-strategy", default="production")
    p.add_argument("--num-agents", type=int, default=2, choices=[2, 4],
                   help="Number of players in the replay.")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions instead of greedy/argmax (default deterministic).")
    p.add_argument("--games", type=int, default=1, help="Number of replays to record.")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    device = torch.device(args.device)
    policy, _ = load_checkpoint(args.ckpt, device=device)
    policy_p1 = None
    if args.opponent == "policy":
        if not args.ckpt_p1:
            p.error("--opponent policy requires --ckpt-p1")
        policy_p1, _ = load_checkpoint(args.ckpt_p1, device=device)

    base = Path(args.html)
    for g in range(args.games):
        out = base if args.games == 1 else base.with_name(f"{base.stem}_{g:03d}{base.suffix}")
        record_game_html(
            policy, out,
            policy_p1=policy_p1,
            num_agents=args.num_agents,
            max_steps=args.max_steps,
            opponent=args.opponent,
            scripted_strategy=args.scripted_strategy,
            deterministic=not args.stochastic,
            seed=(args.seed if args.seed is None else args.seed + g),
            device=device,
        )


if __name__ == "__main__":
    main()
