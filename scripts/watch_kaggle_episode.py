#!/usr/bin/env python3
"""Download a Kaggle Orbit Wars episode replay and render interactive HTML.

Use this to watch **any** public ladder game (e.g. the #1 player's episode),
not just your own submissions.

Setup (once)::

    # Kaggle → Account → Create New Token → save as ~/.kaggle/kaggle.json
    pip install kaggle

Examples::

    # Download episode 80044090 and open replay in browser
    python scripts/watch_kaggle_episode.py 80044090

    # Custom output path
    python scripts/watch_kaggle_episode.py 80044090 \\
        --out replays/leader_ep80044090.html

    # Re-render from an already-downloaded JSON (no API call)
    python scripts/watch_kaggle_episode.py --json replays/80044090.json \\
        --out replays/leader.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.disable(logging.INFO)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_replays import download_episode_replay  # noqa: E402


def _download_replay(episode_id: int, dest_dir: Path) -> Path:
    """Fetch replay JSON via ``kaggle competitions replay``."""
    print(f"Downloading episode {episode_id} …")
    path = download_episode_replay(episode_id, dest_dir)
    print(f"Replay JSON: {path}")
    return path


def render_replay_json(json_path: Path, out_html: Path) -> None:
    """Build interactive HTML from a Kaggle episode replay file."""
    from kaggle_environments import make

    replay = json.loads(json_path.read_text(encoding="utf-8"))
    env_name = replay.get("name", "orbit_wars")
    if env_name != "orbit_wars":
        print(f"Warning: replay name is {env_name!r}, rendering with orbit_wars anyway.")

    env = make(
        "orbit_wars",
        configuration=replay.get("configuration"),
        steps=replay.get("steps", []),
        info=replay.get("info", {}),
    )
    html = env.render(mode="html")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")
    print(f"HTML replay: {out_html.resolve()} ({len(html) // 1024} KB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Watch a Kaggle Orbit Wars episode locally")
    p.add_argument("episode_id", type=int, nargs="?", help="Kaggle episode ID (e.g. 80044090)")
    p.add_argument("--json", type=str, default=None, help="Skip download; use existing replay JSON")
    p.add_argument("--download-dir", type=str, default="replays/kaggle",
                   help="Where to save downloaded JSON")
    p.add_argument("--out", type=str, default=None,
                   help="Output HTML path (default: replays/episode_<id>.html)")
    p.add_argument("--open", action="store_true", help="Open HTML in default browser (macOS)")
    args = p.parse_args()

    if args.json:
        json_path = Path(args.json)
        eid = args.episode_id or int(json_path.stem.split("_")[0]) if json_path.stem.isdigit() else 0
    elif args.episode_id is not None:
        eid = int(args.episode_id)
        json_path = _download_replay(eid, Path(args.download_dir))
    else:
        p.error("Provide episode_id or --json")

    out = Path(args.out or f"replays/episode_{eid or json_path.stem}.html")
    render_replay_json(json_path, out)

    if args.open:
        import webbrowser
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
