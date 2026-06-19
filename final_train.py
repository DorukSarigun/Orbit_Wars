"""GA tuner for the final V1 micro-planner agent.

This is the practical "full training" entrypoint for the contest-facing final
agent.  It optimizes planner/agent hyperparameters by paired-seed matches,
instead of trying to train a large neural policy under the remaining contest
time.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from final_agent import FinalAgentConfig, FinalMicroAgent
from final_micro_planner import MicroPlannerConfig
from orbit_match import make_controller, play_match


GENE_SPACE: dict[str, tuple[float, float, str]] = {
    "hold_horizon": (12.0, 50.0, "float"),
    "beam_width": (4.0, 32.0, "int"),
    "max_launches_per_target": (1.0, 5.0, "int"),
    "top_m_sources": (2.0, 8.0, "int"),
    "capture_margin": (1.0, 8.0, "int"),
    "ship_cost_weight": (0.0, 0.08, "float"),
    "delay_weight": (0.0, 0.05, "float"),
    "source_strip_weight": (0.0, 4.0, "float"),
    "true_overkill_weight": (0.0, 0.05, "float"),
    "enemy_gift_penalty": (1.0, 20.0, "float"),
    "value_weight": (0.25, 2.0, "float"),
    "enemy_denial_weight": (0.0, 2.0, "float"),
    "influence_weight": (0.0, 3.0, "float"),
    "production_pressure_weight": (0.0, 3.0, "float"),
    "enemy_weakness_weight": (0.0, 3.0, "float"),
    "enemy_empire_support_weight": (0.0, 3.0, "float"),
}


def _clip_gene(name: str, value: float) -> float | int:
    lo, hi, kind = GENE_SPACE[name]
    value = min(float(hi), max(float(lo), float(value)))
    if kind == "int":
        return int(round(value))
    return float(value)


def random_genome(rng: random.Random) -> dict[str, float | int]:
    return {
        name: _clip_gene(name, rng.uniform(lo, hi))
        for name, (lo, hi, _kind) in GENE_SPACE.items()
    }


def mutate_genome(genome: dict[str, float | int], rng: random.Random, sigma: float) -> dict[str, float | int]:
    child: dict[str, float | int] = {}
    for name, value in genome.items():
        lo, hi, kind = GENE_SPACE[name]
        span = float(hi) - float(lo)
        if rng.random() < 0.75:
            value = float(value) + rng.gauss(0.0, float(sigma) * span)
        if rng.random() < 0.05:
            value = rng.uniform(float(lo), float(hi))
        child[name] = _clip_gene(name, float(value))
    return child


def crossover(a: dict[str, float | int], b: dict[str, float | int], rng: random.Random) -> dict[str, float | int]:
    return {name: (a[name] if rng.random() < 0.5 else b[name]) for name in GENE_SPACE}


def config_from_genome(genome: dict[str, Any]) -> FinalAgentConfig:
    micro = MicroPlannerConfig(
        hold_horizon=float(genome["hold_horizon"]),
        beam_width=int(genome["beam_width"]),
        max_launches_per_target=int(genome["max_launches_per_target"]),
        top_m_sources=int(genome["top_m_sources"]),
        capture_margin=int(genome["capture_margin"]),
        ship_cost_weight=float(genome["ship_cost_weight"]),
        delay_weight=float(genome["delay_weight"]),
        source_strip_weight=float(genome["source_strip_weight"]),
        true_overkill_weight=float(genome["true_overkill_weight"]),
        enemy_gift_penalty=float(genome["enemy_gift_penalty"]),
        value_weight=float(genome["value_weight"]),
        enemy_denial_weight=float(genome["enemy_denial_weight"]),
    )
    return FinalAgentConfig(
        influence_weight=float(genome["influence_weight"]),
        production_pressure_weight=float(genome["production_pressure_weight"]),
        enemy_weakness_weight=float(genome["enemy_weakness_weight"]),
        enemy_empire_support_weight=float(genome["enemy_empire_support_weight"]),
        micro=micro,
    )


def _score_result(res, *, final_player: int) -> float:
    if final_player == 0:
        win = 1.0 if res.winner == 0 else (-1.0 if res.winner == 1 else 0.0)
        prod = float(res.prod_margin_p0)
        ship = float(res.ship_margin_p0)
    else:
        win = 1.0 if res.winner == 1 else (-1.0 if res.winner == 0 else 0.0)
        prod = -float(res.prod_margin_p0)
        ship = -float(res.ship_margin_p0)
    finish_penalty = 0.02 * max(0.0, float(res.length) / 500.0)
    return 5.0 * win + 1.0 * prod + 0.25 * ship - finish_penalty


def evaluate_genome(
    genome: dict[str, Any],
    *,
    opponent: str,
    seeds: list[int],
    max_steps: int,
) -> dict[str, float]:
    cfg = config_from_genome(genome)
    total = 0.0
    wins = 0
    games = 0
    for seed in seeds:
        final = FinalMicroAgent(cfg)
        opp = make_controller(opponent)
        res0 = play_match([final, opp], num_agents=2, max_steps=max_steps, seed=int(seed))
        total += _score_result(res0, final_player=0)
        wins += int(res0.winner == 0)
        games += 1

        final = FinalMicroAgent(cfg)
        opp = make_controller(opponent)
        res1 = play_match([opp, final], num_agents=2, max_steps=max_steps, seed=int(seed))
        total += _score_result(res1, final_player=1)
        wins += int(res1.winner == 1)
        games += 1

    return {
        "score": float(total / max(1, games)),
        "winrate": float(wins / max(1, games)),
        "games": float(games),
    }


def save_best(path: Path, genome: dict[str, Any], metrics: dict[str, float], *, generation: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generation": int(generation),
        "genome": genome,
        "metrics": metrics,
        "config": asdict(config_from_genome(genome)),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_ga(args) -> None:
    rng = random.Random(int(args.seed))
    seeds = [int(args.seed) + i for i in range(int(args.games_per_eval))]
    population = [random_genome(rng) for _ in range(int(args.population))]
    best_genome: dict[str, Any] | None = None
    best_metrics = {"score": -math.inf, "winrate": 0.0, "games": 0.0}
    out = Path(args.out)

    for gen in range(int(args.generations)):
        scored: list[tuple[float, dict[str, Any], dict[str, float]]] = []
        for genome in population:
            metrics = evaluate_genome(
                genome,
                opponent=str(args.opponent),
                seeds=seeds,
                max_steps=int(args.max_steps),
            )
            scored.append((float(metrics["score"]), genome, metrics))
            if float(metrics["score"]) > float(best_metrics["score"]):
                best_genome = dict(genome)
                best_metrics = dict(metrics)
                save_best(out, best_genome, best_metrics, generation=gen)
        scored.sort(key=lambda row: row[0], reverse=True)
        leader = scored[0]
        print(
            f"[gen {gen:03d}] best={leader[0]:+.3f} "
            f"winrate={leader[2]['winrate']:.3f} global_best={best_metrics['score']:+.3f}",
            flush=True,
        )

        elites = [dict(g) for _score, g, _m in scored[: max(1, int(args.elites))]]
        next_pop = list(elites)
        while len(next_pop) < int(args.population):
            a = rng.choice(elites)
            b = rng.choice(scored[: max(len(elites), int(args.parent_pool))])[1]
            child = crossover(a, b, rng)
            child = mutate_genome(child, rng, sigma=float(args.mutation_sigma))
            next_pop.append(child)
        population = next_pop

    if best_genome is not None:
        save_best(out, best_genome, best_metrics, generation=int(args.generations))
        print(f"best saved to {out}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GA tune the final micro-planner agent")
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--elites", type=int, default=3)
    p.add_argument("--parent-pool", type=int, default=6)
    p.add_argument("--mutation-sigma", type=float, default=0.12)
    p.add_argument("--games-per-eval", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--opponent", default="orbit",
                   help="Any orbit_match controller kind, e.g. orbit, timed_mcts, mcts")
    p.add_argument("--out", default="ga_runs/final_micro/best_genome.json")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    run_ga(args)


if __name__ == "__main__":
    main()

