"""Render an HTML replay of a trained macro-AlphaZero agent in the REAL env.

Loads a ``MacroAlphaZeroNet`` checkpoint (the GNN or MLP you trained), wraps it
in the shipped :class:`MacroAlphaZeroAgent`, plays a real ``orbit_wars`` game
against a chosen opponent, and writes ``env.render(mode="html")`` so you can
open it in a browser and actually watch the agent play.

Examples
--------
Watch the trained GNN vs the scripted LP baseline::

    python -m scripts.watch_macro_alphazero \
        --checkpoint checkpoints_macro_alphazero_gnn/best.pt \
        --opponent lp --out replays/macro_az_vs_lp.html

Self-play (trained net on both sides)::

    python -m scripts.watch_macro_alphazero --opponent self
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.disable(logging.INFO)

import torch  # noqa: E402
from kaggle_environments import make  # noqa: E402

from orbit_macro_mcts_agent import MacroMCTSConfig  # noqa: E402
from orbit_timed_mcts import TimedMCTSConfig  # noqa: E402
from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig  # noqa: E402
from macro_alphazero_net import MacroAlphaZeroAgent, load_net  # noqa: E402


def _macro_cfg(simulations: int, horizon: int) -> MacroMCTSConfig:
    return MacroMCTSConfig(
        timed=TimedMCTSConfig(
            simulations=int(simulations),
            horizon_turns=int(horizon),
            source_reserve_frac=0.0,
            gamma=1.0,
        ),
        value_blend=0.5,
    )


def _lp_cfg() -> OrbitScriptedConfig:
    return OrbitScriptedConfig(attack_allocator="lp", reinforcement_allocator="lp")


def _load_genome(path: str) -> dict[str, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw["genome"] if isinstance(raw, dict) and "genome" in raw else raw


def _make_opponent(
    kind: str,
    net,
    macro_cfg,
    device,
    *,
    genome_path: str,
    mcts_simulations: int,
):
    if kind == "self":
        return MacroAlphaZeroAgent(net, macro_cfg=macro_cfg, device=device)
    if kind == "lp":
        return OrbitScriptedAgent(cfg=_lp_cfg())
    if kind == "scripted":
        return OrbitScriptedAgent()
    if kind in ("timed_mcts", "tmcts"):
        from orbit_timed_mcts_agent import OrbitTimedMCTSAgent
        return OrbitTimedMCTSAgent(
            cfg=_lp_cfg(),
            timed_mcts_cfg=TimedMCTSConfig(simulations=64, horizon_turns=8, top_k_actions=3),
        )
    if kind in ("ga_mcts", "mcts_ga"):
        from orbit_ga import config_from_genome
        from orbit_mcts import MCTSConfig
        from orbit_mcts_agent import OrbitMCTSAgent
        cfg, _look = config_from_genome(_load_genome(genome_path))
        return OrbitMCTSAgent(
            cfg=cfg,
            mcts_cfg=MCTSConfig(simulations=int(mcts_simulations)),
        )
    raise ValueError(f"unknown opponent {kind!r}")


def record_replay(
    out_path: Path,
    *,
    checkpoint: str,
    opponent: str,
    seed: int | None,
    steps: int,
    simulations: int,
    horizon: int,
    swap: bool,
    device: str,
    genome_path: str,
    mcts_simulations: int,
) -> dict:
    if seed is None:
        seed = random.randrange(2**31)
    dev = torch.device(device)
    net = load_net(checkpoint, device=dev)
    macro_cfg = _macro_cfg(simulations, horizon)

    az = MacroAlphaZeroAgent(net, macro_cfg=macro_cfg, device=dev)
    opp = _make_opponent(
        opponent, net, macro_cfg, dev,
        genome_path=genome_path, mcts_simulations=mcts_simulations,
    )
    controllers = [opp, az] if swap else [az, opp]
    az_seat = 1 if swap else 0

    env = make("orbit_wars", debug=False)
    env.configuration["randomSeed"] = int(seed)
    random.seed(int(seed))
    env.reset(num_agents=2)
    state = env.step([[], []])

    total_actions = [0, 0]
    t = 0
    for t in range(int(steps)):
        actions = []
        for player, controller in enumerate(controllers):
            env_obs = state[player]["observation"]
            action = controller.act(
                env_obs, player, current_turn=t, max_turns=int(steps)) or []
            total_actions[player] += len(action)
            actions.append(action)
        state = env.step(actions)
        if env.done or any(getattr(state[p], "status", "ACTIVE") == "DONE" for p in range(2)):
            break

    html = env.render(mode="html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    rewards = [float(getattr(state[p], "reward", 0.0) or 0.0) for p in range(2)]
    winner = -1
    if rewards and rewards.count(max(rewards)) == 1:
        winner = int(max(range(2), key=lambda i: rewards[i]))
    return {
        "html_path": str(out_path.resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "arch": getattr(net, "arch", "gnn"),
        "steps": t + 1,
        "seed": int(seed),
        "opponent": opponent,
        "genome": str(Path(genome_path).resolve()) if opponent in ("ga_mcts", "mcts_ga") else None,
        "mcts_simulations": int(mcts_simulations) if opponent in ("ga_mcts", "mcts_ga") else None,
        "az_seat": az_seat,
        "az_won": bool(winner == az_seat),
        "winner": winner,
        "simulations": int(simulations),
        "rewards": rewards,
        "actions": total_actions,
        "html_size_kb": round(len(html) / 1024, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="HTML replay of a trained macro-AlphaZero agent")
    p.add_argument("--checkpoint", default="checkpoints_macro_alphazero_gnn/best.pt")
    p.add_argument("--opponent", default="lp",
                   choices=["lp", "scripted", "self", "timed_mcts", "tmcts", "ga_mcts", "mcts_ga"])
    p.add_argument("--genome", default="ga_runs/mcts/best_genome.json",
                   help="GA genome for --opponent ga_mcts (Kaggle MCTS+GA submission)")
    p.add_argument("--mcts-simulations", type=int, default=48,
                   help="MCTS sims for ga_mcts opponent (GA was tuned at 48)")
    p.add_argument("--out", default="replays/macro_az.html")
    p.add_argument("--seed", type=int, default=None,
                   help="Board seed (default: random each run)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--simulations", type=int, default=32)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--swap", action="store_true", help="Put the AZ agent in seat P1")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    summary = record_replay(
        Path(args.out),
        checkpoint=args.checkpoint,
        opponent=args.opponent,
        seed=args.seed,
        steps=args.steps,
        simulations=args.simulations,
        horizon=args.horizon,
        swap=bool(args.swap),
        device=args.device,
        genome_path=args.genome,
        mcts_simulations=args.mcts_simulations,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
