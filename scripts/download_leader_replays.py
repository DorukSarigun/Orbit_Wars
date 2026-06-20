#!/usr/bin/env python3
"""Download Orbit Wars replays from the top-N Kaggle leaderboard teams.

Requires ``~/.kaggle/kaggle.json`` and competition rules accepted for ``orbit-wars``.

Examples::

    cd Orbit_Wars
    ../orbit_wars_env/bin/python scripts/download_leader_replays.py \\
        --top-n 3 --episodes-per-team 12

    # BC warm-start from downloaded expert games
    ../orbit_wars_env/bin/python -m final_bc \\
        --from-replays data/leader_replays/manifest.json \\
        --games 0 --epochs 12 --out checkpoints_final_bc/bc_leader.pt

    ../orbit_wars_env/bin/python -m final_train_rl \\
        --resume-from checkpoints_final_bc/bc_leader.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_replays import download_leader_replays  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Download top leaderboard Orbit Wars replays")
    p.add_argument("--out-dir", default="data/leader_replays", help="Output directory")
    p.add_argument("--top-n", type=int, default=3, help="Number of leaderboard teams")
    p.add_argument("--episodes-per-team", type=int, default=12, help="Max episodes per team")
    p.add_argument(
        "--max-submissions-per-team",
        type=int,
        default=2,
        help="How many best submissions to pull episodes from",
    )
    p.add_argument("--competition", default="orbit-wars")
    args = p.parse_args()

    try:
        manifest = download_leader_replays(
            out_dir=Path(args.out_dir),
            top_n=int(args.top_n),
            episodes_per_team=int(args.episodes_per_team),
            max_submissions_per_team=int(args.max_submissions_per_team),
            competition=str(args.competition),
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Install the CLI with `pip install -U kaggle` and place credentials at "
            "~/.kaggle/kaggle.json (Kaggle → Account → Create New Token).",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:  # type: ignore[name-defined]
        print(f"Kaggle API failed: {exc}", file=sys.stderr)
        sys.exit(exc.returncode)

    n_eps = sum(len(t.episodes) for t in manifest.teams)
    print(f"Done: {len(manifest.teams)} teams, {n_eps} episodes -> {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
