#!/usr/bin/env python3
"""Build and smoke-test Kaggle Orbit Wars MCTS submission tarballs.

Examples::

    python scripts/build_kaggle_submission.py --variant ga
    python scripts/build_kaggle_submission.py --variant default
    kaggle competitions submit orbit-wars -f dist/submission_mcts_ga.tar.gz -m "mcts ga gen9"
    kaggle competitions submit orbit-wars -f dist/submission_mcts_default.tar.gz -m "mcts default"
"""

from __future__ import annotations

import argparse
import ast
import logging
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "kaggle_submission"

MCTS_DEPS = [
    "feature_extraction.py",
    "navigation.py",
    "obs_builder.py",
    "orbit_attack_allocate.py",
    "orbit_attack_targets.py",
    "orbit_defense.py",
    "orbit_enemy_weakness.py",
    "orbit_expansion.py",
    "orbit_graph.py",
    "orbit_lookahead_agent.py",
    "orbit_mcts.py",
    "orbit_mcts_agent.py",
    "orbit_production_pressure.py",
    "orbit_scripted_agent.py",
    "reworked_features.py",
    "scripted_strategy.py",
]

BUNDLE_FILES = [
    "main.py",
    "mcts_inference.py",
    "genome_config.py",
    "minigraph.py",
]

_NX_PATCH = """from minigraph import MiniDiGraph


class _NxShim:
    DiGraph = MiniDiGraph


nx = _NxShim()
"""

_NX_FILES = ("orbit_graph.py", "orbit_production_pressure.py")


def _patch_networkx_imports(text: str) -> str:
    if "import networkx as nx" not in text:
        return text
    return text.replace("import networkx as nx", _NX_PATCH.strip(), 1)


def _patch_file_networkx(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    if "import networkx as nx" not in text:
        raise RuntimeError(f"{src.name} missing expected networkx import")
    dst.write_text(_patch_networkx_imports(text), encoding="utf-8")


def _closure_from(seeds: list[str]) -> set[str]:
    local: set[str] = set()
    queue = list(seeds)
    while queue:
        rel = queue.pop()
        if rel in local:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        local.add(rel)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module.split(".")[0]]
            elif isinstance(node, ast.Import):
                mods = [n.name.split(".")[0] for n in node.names]
            for mod in mods:
                cand = f"{mod}.py"
                if (ROOT / cand).is_file() and cand not in local:
                    queue.append(cand)
    return local


def verify_closure(deps: list[str]) -> None:
    actual = _closure_from(["orbit_mcts_agent.py", "mcts_inference.py"])
    missing = set(deps) - actual
    if missing:
        raise RuntimeError(f"MCTS bundle missing imported modules: {sorted(missing)}")


def build_staging(
    staging: Path,
    *,
    variant: str,
    genome: Path | None,
) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    verify_closure(MCTS_DEPS)
    for rel in MCTS_DEPS:
        src = ROOT / rel
        dst = staging / Path(rel).name
        if rel in _NX_FILES:
            _patch_file_networkx(src, dst)
        else:
            shutil.copy2(src, dst)

    for name in BUNDLE_FILES:
        shutil.copy2(BUNDLE / name, staging / name)

    meta_lines = [f"agent=mcts", f"variant={variant}", "simulations=12"]
    if variant == "ga":
        if genome is None or not genome.is_file():
            raise FileNotFoundError(f"genome required for variant=ga: {genome}")
        shutil.copy2(genome, staging / "genome.json")
        meta_lines.append(f"genome={genome}")
    else:
        meta_lines.append("genome=default OrbitScriptedConfig")

    (staging / "SUBMISSION.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    for path in staging.glob("*.py"):
        if "import networkx" in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"networkx import still present in bundle: {path.name}")


def smoke_test(staging: Path, games: int = 1) -> dict:
    logging.disable(logging.INFO)
    sys.path.insert(0, str(staging))
    try:
        # Ensure we are not accidentally using repo networkx.
        sys.modules.pop("networkx", None)
        for key in list(sys.modules):
            if key.startswith("networkx."):
                sys.modules.pop(key, None)

        from kaggle_environments import make  # noqa: WPS433

        import main as submission_main  # noqa: WPS433

        env = make("orbit_wars", debug=False)
        timings: list[float] = []
        results: list[tuple[int, float]] = []

        for _ in range(games):
            env.reset(num_agents=2)
            state = env.step([[], []])
            for t in range(500):
                actions = []
                for p in range(2):
                    obs = state[p]["observation"]
                    if p == 0:
                        t0 = time.perf_counter()
                        moves = submission_main.agent(obs)
                        timings.append(time.perf_counter() - t0)
                    else:
                        moves = []
                    actions.append(moves or [])
                state = env.step(actions)
                if env.done:
                    break
            rewards = [float(getattr(state[p], "reward", 0.0) or 0.0) for p in range(2)]
            results.append((t + 1, rewards[0]))

        p99 = sorted(timings)[int(0.99 * max(len(timings) - 1, 0))] if timings else 0.0
        return {
            "mean_turn_sec": sum(timings) / max(len(timings), 1),
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


def _build_one(
    variant: str,
    genome: Path | None,
    staging: Path,
    out: Path,
    *,
    skip_test: bool,
    games: int,
) -> None:
    build_staging(staging, variant=variant, genome=genome)
    print(f"[{variant}] staged {len(list(staging.iterdir()))} files")

    if not skip_test:
        stats = smoke_test(staging, games=games)
        print(
            f"  smoke: mean {stats['mean_turn_sec']*1000:.0f}ms, "
            f"p99 {stats['p99_turn_sec']*1000:.0f}ms, "
            f"max {stats['max_turn_sec']*1000:.0f}ms, "
            f"results {stats['results']}"
        )
        if stats["p99_turn_sec"] > 0.95:
            print("  WARNING: p99 > 950ms — may hit actTimeout on Kaggle")

    make_tarball(staging, out)
    print(f"  -> {out} ({out.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    p = argparse.ArgumentParser(description="Build Kaggle MCTS submissions")
    p.add_argument(
        "--variant",
        choices=("ga", "default", "both"),
        default="both",
        help="ga=GA genome, default=no genome, both=build two tarballs",
    )
    p.add_argument("--genome", type=Path, default=ROOT / "ga_runs" / "mcts" / "best_genome.json")
    p.add_argument("--staging", type=Path, default=ROOT / "dist" / "kaggle_submission_staging")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--games", type=int, default=1)
    args = p.parse_args()

    variants = ["ga", "default"] if args.variant == "both" else [args.variant]
    for variant in variants:
        out = ROOT / "dist" / f"submission_mcts_{variant}.tar.gz"
        genome = args.genome if variant == "ga" else None
        _build_one(
            variant, genome, args.staging, out,
            skip_test=args.skip_test, games=args.games,
        )


if __name__ == "__main__":
    main()
