"""Targeted fuzzer: exercise the GA attack/partial-wave paths fast.

Builds one real board, then repeatedly randomizes planet ownership/ship counts
to synthesize mid-game states with enemy planets, and calls the GA agent's
act() on each. Catches any exception (the crash that fails Kaggle validation)
with a full traceback. Much faster than full self-play games.
"""

from __future__ import annotations

import copy
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


def base_obs():
    random.seed(0)
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=2)
    env.step([[], []])
    return copy.deepcopy(env.state[0]["observation"])


def randomize(obs, rng, num_players=2):
    o = copy.deepcopy(obs)
    planets = o["planets"]
    for row in planets:
        # row = [id, owner, x, y, r, ships, prod]
        roll = rng.random()
        if roll < 0.35:
            row[1] = 0
        elif roll < 0.7:
            row[1] = rng.randrange(1, num_players)
        else:
            row[1] = -1
        row[5] = rng.randrange(0, 60)
    o["fleets"] = []
    o["step"] = rng.randrange(0, 500)
    return o


def main():
    obs0 = base_obs()
    rng = random.Random(12345)
    N = 4000
    for player in (0, 1):
        mcts_inference._AGENT = None
        agent = mcts_inference.get_agent()
        for i in range(N):
            obs = randomize(obs0, rng)
            obs["player"] = player
            try:
                agent.act(
                    obs, player,
                    current_turn=int(obs["step"]),
                    max_turns=500,
                )
            except Exception:
                print(f"\n!!! CRASH player={player} iter={i}", flush=True)
                print("planets:", obs["planets"], flush=True)
                print(traceback.format_exc(), flush=True)
                return 1
            if i % 500 == 0:
                print(f"player={player} iter={i} ok", flush=True)
    print("\nAll synthetic boards clean.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
