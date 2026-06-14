"""Write an HTML replay for timed-MCTS+LP against MCTS variants."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.disable(logging.INFO)

from kaggle_environments import make  # noqa: E402

from orbit_mcts import MCTSConfig  # noqa: E402
from orbit_mcts_agent import OrbitMCTSAgent  # noqa: E402
from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig  # noqa: E402
from orbit_timed_mcts import TimedMCTSConfig  # noqa: E402
from orbit_timed_mcts_agent import OrbitTimedMCTSAgent  # noqa: E402


def _scripted_cfg(*, lp: bool) -> OrbitScriptedConfig:
    common = dict(
        max_launches=4,
        eta_horizon=35.0,
        lp_top_k_targets=2,
        lp_support_top_k_targets=2,
    )
    if not lp:
        return OrbitScriptedConfig(**common)
    return OrbitScriptedConfig(
        **common,
        attack_allocator="lp",
        reinforcement_allocator="lp",
    )


def _timed_mcts_lp_agent(simulations: int) -> OrbitTimedMCTSAgent:
    return OrbitTimedMCTSAgent(
        cfg=_scripted_cfg(lp=True),
        timed_mcts_cfg=TimedMCTSConfig(
            simulations=int(simulations),
            horizon_turns=5,
            top_k_actions=3,
        ),
    )


def _mcts_agent(kind: str, simulations: int):
    if kind == "legacy_mcts":
        return OrbitMCTSAgent(
            cfg=_scripted_cfg(lp=False),
            mcts_cfg=MCTSConfig(
                simulations=int(simulations),
                rollout_depth=2,
                top_k_actions=3,
            ),
        )
    if kind == "timed_mcts":
        return OrbitTimedMCTSAgent(
            cfg=_scripted_cfg(lp=False),
            timed_mcts_cfg=TimedMCTSConfig(
                simulations=int(simulations),
                horizon_turns=5,
                top_k_actions=3,
            ),
        )
    raise ValueError(f"unknown opponent {kind!r}")


def record_replay(
    out_path: Path,
    *,
    opponent: str,
    seed: int,
    steps: int,
    plan_max_turns: int,
    simulations: int,
    swap: bool,
) -> dict:
    p_lp = _timed_mcts_lp_agent(simulations)
    p_mcts = _mcts_agent(opponent, simulations)
    controllers = [p_mcts, p_lp] if swap else [p_lp, p_mcts]

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
                env_obs,
                player,
                current_turn=t,
                max_turns=int(plan_max_turns),
            ) or []
            total_actions[player] += len(action)
            actions.append(action)
        state = env.step(actions)
        if env.done or any(getattr(state[p], "status", "ACTIVE") == "DONE" for p in range(2)):
            break

    html = env.render(mode="html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    rewards = [float(getattr(state[p], "reward", 0.0) or 0.0) for p in range(2)]
    summary = {
        "html_path": str(out_path.resolve()),
        "steps": t + 1,
        "seed": int(seed),
        "opponent": opponent,
        "simulations": int(simulations),
        "swap": bool(swap),
        "p0": opponent if swap else "timed_mcts_lp",
        "p1": "timed_mcts_lp" if swap else opponent,
        "rewards": rewards,
        "actions": total_actions,
        "html_size_kb": round(len(html) / 1024, 1),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="replays/timed_mcts_lp_vs_timed_mcts.html")
    parser.add_argument("--opponent", choices=["timed_mcts", "legacy_mcts"], default="timed_mcts")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--plan-max-turns", type=int, default=300)
    parser.add_argument("--simulations", type=int, default=2)
    parser.add_argument("--swap", action="store_true")
    args = parser.parse_args()

    summary = record_replay(
        Path(args.out),
        opponent=args.opponent,
        seed=args.seed,
        steps=args.steps,
        plan_max_turns=args.plan_max_turns,
        simulations=args.simulations,
        swap=bool(args.swap),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
