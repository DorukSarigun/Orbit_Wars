"""Fast in-process self-play fuzzer for the GA-configured agent.

Plays many self-play games at different seeds, calling the agent's act() in
process and capturing any exception with a full traceback (the kind of crash
that surfaces on Kaggle as 'Validation Episode failed'). Also validates every
emitted action against basic env legality (owned source, ships available,
finite angle).
"""

from __future__ import annotations

import logging
import random
import sys
import traceback
from pathlib import Path

logging.disable(logging.INFO)

STAGING = Path("dist/kaggle_submission_staging").resolve()
sys.path.insert(0, str(STAGING))

from kaggle_environments import make  # noqa: E402
import mcts_inference  # noqa: E402


def _planets(obs):
    return obs["planets"]


def validate_actions(actions, obs, player):
    """Return a problem string if any action is illegal, else None."""
    if not actions:
        return None
    owned = {}
    for row in obs["planets"]:
        pid, owner, x, y, r, ships, prod = row
        owned[int(pid)] = (int(owner), int(ships))
    for a in actions:
        if not (isinstance(a, (list, tuple)) and len(a) == 3):
            return f"malformed action {a!r}"
        sid, angle, ships = a
        import math as _m
        if not _m.isfinite(float(angle)):
            return f"non-finite angle in {a!r}"
        if int(sid) not in owned:
            return f"source {sid} not a planet"
        o, sh = owned[int(sid)]
        if o != player:
            return f"source {sid} owned by {o}, not player {player}"
        if int(ships) <= 0:
            return f"non-positive ships in {a!r}"
    return None


def play(num_agents: int, seed: int, max_steps: int = 500):
    random.seed(seed)
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=num_agents)
    state = env.step([[] for _ in range(num_agents)])

    # Fresh agent per game to mimic a new Kaggle process.
    mcts_inference._AGENT = None
    agent = mcts_inference.get_agent()

    for t in range(max_steps):
        actions = []
        for p in range(num_agents):
            obs = state[p]["observation"]
            if state[p]["status"] != "ACTIVE":
                actions.append([])
                continue
            try:
                moves = agent.act(
                    obs, int(p),
                    current_turn=int(obs.get("step", t)),
                    max_turns=500,
                ) or []
            except Exception:
                return ("CRASH", num_agents, seed, t, p, traceback.format_exc())
            prob = validate_actions(moves, obs, p)
            if prob is not None:
                return ("INVALID", num_agents, seed, t, p, prob)
            actions.append(moves)
        state = env.step(actions)
        if env.done:
            break
    return ("OK", num_agents, seed, t + 1, None, None)


def main():
    import os
    seeds = [int(x) for x in os.environ.get("FUZZ_SEEDS", "1,2,3,4,5,6").split(",")]
    max_steps = int(os.environ.get("FUZZ_STEPS", "500"))
    modes = [int(x) for x in os.environ.get("FUZZ_MODES", "2,4").split(",")]
    for na in modes:
        for s in seeds:
            res = play(na, s, max_steps=max_steps)
            tag = res[0]
            if tag == "OK":
                print(f"{na}p seed={s}: OK ({res[3]} turns)", flush=True)
            else:
                print(f"\n!!! {tag} {na}p seed={s} turn={res[3]} player={res[4]}", flush=True)
                print(res[5], flush=True)
                return 1
    print("\nAll games clean.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
