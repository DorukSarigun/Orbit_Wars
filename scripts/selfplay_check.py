"""Self-play validation harness mirroring Kaggle.

Runs the bundled main.py agent against ITSELF via env.run (separate agent
processes, like Kaggle), for both 2p and 4p, and reports any agent error
status and per-turn timing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.disable(logging.INFO)

STAGING = Path("dist/kaggle_submission_staging").resolve()
MAIN = str(STAGING / "main.py")

from kaggle_environments import make  # noqa: E402


def run(num_agents: int, seed: int | None = None):
    import random as _r
    if seed is not None:
        _r.seed(seed)
    env = make("orbit_wars", debug=True)
    agents = [MAIN] * num_agents
    env.run(agents)

    statuses = [s.status for s in env.state]
    print(f"\n=== {num_agents}p self-play ===")
    print("final statuses:", statuses)

    # Per-agent max action duration from logs.
    for i in range(num_agents):
        durs = [
            step[i]["duration"]
            for step in env.logs
            if step and i < len(step) and step[i] and "duration" in step[i]
        ]
        if durs:
            durs_sorted = sorted(durs)
            p99 = durs_sorted[int(0.99 * (len(durs_sorted) - 1))]
            print(
                f"agent {i}: turns={len(durs)} "
                f"mean={sum(durs)/len(durs)*1000:.0f}ms "
                f"p99={p99*1000:.0f}ms max={max(durs)*1000:.0f}ms"
            )

    # Surface any ERROR/TIMEOUT with the captured stderr.
    for i, s in enumerate(env.state):
        if s.status not in ("DONE", "ACTIVE", "INACTIVE"):
            print(f"!!! agent {i} status={s.status}")
            for step in env.logs:
                if step and i < len(step) and step[i]:
                    err = step[i].get("stderr", "")
                    if err and not err.isspace():
                        print(f"--- agent {i} stderr ---\n{err[:3000]}")
                        break
    return statuses


def main():
    modes = [int(a) for a in sys.argv[1:]] or [2, 4]
    for i, m in enumerate(modes):
        run(m, seed=100 + i)


if __name__ == "__main__":
    sys.exit(main())
