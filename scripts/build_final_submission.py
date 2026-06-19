#!/usr/bin/env python3
"""Build and smoke-test the final deterministic micro-planner submission.

The PPO policy is useful for training experiments, but shipping a torch
checkpoint inside Kaggle's Orbit Wars agent path is a separate packaging
decision.  This builder packages the dependency-free final micro agent.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGING = ROOT / "dist" / "final_submission_staging"
DEFAULT_OUT = ROOT / "dist" / "submission_final_micro.tar.gz"

SEEDS = ["final_submission_agent.py"]
FORCE_INCLUDE = {
    "orbit_production_pressure.py",  # imported lazily by orbit_graph at act time
    "minigraph.py",
}
NETWORKX_FILES = {"orbit_graph.py", "orbit_production_pressure.py"}

MAIN_PY = '''"""Kaggle Orbit Wars final micro-planner submission."""

from __future__ import annotations

import os
import sys

for _p in list(sys.path):
    if _p and os.path.isfile(os.path.join(_p, "final_submission_agent.py")):
        if _p not in sys.path[:1]:
            sys.path.insert(0, _p)
        break

from final_submission_agent import agent
'''

NX_PATCH = """from minigraph import MiniDiGraph


class _NxShim:
    DiGraph = MiniDiGraph


nx = _NxShim()
"""


def _local_module_files(path: Path) -> set[str]:
    out: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module.split(".")[0]]
        for mod in mods:
            rel = f"{mod}.py"
            if (ROOT / rel).is_file():
                out.add(rel)
    return out


def dependency_closure(seeds: list[str]) -> set[str]:
    queue = list(seeds)
    seen: set[str] = set()
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(rel)
        for dep in sorted(_local_module_files(path)):
            if dep not in seen:
                queue.append(dep)
    return seen | set(FORCE_INCLUDE)


def _copy_minigraph(dst: Path) -> None:
    candidates = [
        ROOT / "minigraph.py",
        ROOT / "kaggle_submission" / "minigraph.py",
    ]
    for src in candidates:
        if src.is_file():
            shutil.copy2(src, dst)
            return
    raise FileNotFoundError("minigraph.py")


def _patch_networkx(text: str, name: str) -> str:
    if "import networkx as nx" not in text:
        raise RuntimeError(f"{name} missing expected networkx import")
    return text.replace("import networkx as nx", NX_PATCH.strip(), 1)


def build_staging(staging: Path) -> set[str]:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    deps = dependency_closure(SEEDS)
    for rel in sorted(deps):
        dst = staging / Path(rel).name
        if rel == "minigraph.py":
            _copy_minigraph(dst)
            continue
        src = ROOT / rel
        text = src.read_text(encoding="utf-8")
        if rel in NETWORKX_FILES:
            text = _patch_networkx(text, rel)
        dst.write_text(text, encoding="utf-8")

    (staging / "main.py").write_text(MAIN_PY, encoding="utf-8")
    (staging / "SUBMISSION.txt").write_text(
        "agent=final_micro\n"
        "policy=deterministic timeline micro-planner\n"
        "checkpoint=none\n",
        encoding="utf-8",
    )

    for path in staging.glob("*.py"):
        if "import networkx" in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"networkx import still present in {path.name}")
    return deps


def smoke_test(staging: Path, games: int, max_steps: int) -> dict:
    logging.disable(logging.INFO)
    sys.path.insert(0, str(staging))
    try:
        sys.modules.pop("networkx", None)
        for key in list(sys.modules):
            if key.startswith("networkx."):
                sys.modules.pop(key, None)
        for key in list(sys.modules):
            if key in {"main", "final_submission_agent"} or key.startswith("final_"):
                sys.modules.pop(key, None)

        from kaggle_environments import make  # noqa: WPS433

        import main as submission_main  # noqa: WPS433

        env = make("orbit_wars", debug=False)
        timings: list[float] = []
        results: list[tuple[int, float]] = []

        for _ in range(max(1, int(games))):
            env.reset(num_agents=2)
            state = env.step([[], []])
            for turn in range(max(1, int(max_steps))):
                actions = []
                for player in range(2):
                    obs = state[player]["observation"]
                    if player == 0:
                        t0 = time.perf_counter()
                        moves = submission_main.agent(obs)
                        timings.append(time.perf_counter() - t0)
                    else:
                        moves = []
                    actions.append(moves or [])
                state = env.step(actions)
                if env.done:
                    break
            reward = float(getattr(state[0], "reward", 0.0) or 0.0)
            results.append((turn + 1, reward))

        p99_idx = int(0.99 * max(len(timings) - 1, 0))
        p99 = sorted(timings)[p99_idx] if timings else 0.0
        return {
            "mean_turn_sec": sum(timings) / max(1, len(timings)),
            "p99_turn_sec": p99,
            "max_turn_sec": max(timings) if timings else 0.0,
            "results": results,
        }
    finally:
        if str(staging) in sys.path:
            sys.path.remove(str(staging))
        logging.disable(logging.NOTSET)


def make_tarball(staging: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(staging / "__pycache__", ignore_errors=True)
    with tarfile.open(out, "w:gz") as tar:
        for path in sorted(staging.iterdir()):
            if path.name == "__pycache__":
                continue
            tar.add(path, arcname=path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=80)
    args = parser.parse_args()

    deps = build_staging(args.staging)
    print(f"staged {len(list(args.staging.iterdir()))} files ({len(deps)} deps)")
    if not args.skip_test:
        stats = smoke_test(args.staging, args.games, args.max_steps)
        print(
            f"smoke: mean {stats['mean_turn_sec']*1000:.0f}ms, "
            f"p99 {stats['p99_turn_sec']*1000:.0f}ms, "
            f"max {stats['max_turn_sec']*1000:.0f}ms, "
            f"results {stats['results']}"
        )
        if stats["p99_turn_sec"] > 0.95:
            print("WARNING: p99 > 950ms; Kaggle actTimeout risk")
    make_tarball(args.staging, args.out)
    print(f"-> {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
