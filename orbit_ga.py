"""Genetic-algorithm tuner for the orbit scripted / lookahead agent.

Evolves the numeric fields of :class:`OrbitScriptedConfig` (and, optionally, the
:class:`LookaheadConfig` planner knobs) by playing headless Orbit Wars matches
against a fixed benchmark opponent.  The agent code itself is **never** modified
— each genome is just a config instance fed to ``OrbitScriptedAgent`` /
``OrbitLookaheadAgent``.

Genome
------
A genome is a ``dict[str, float]`` over the parameters in :data:`GENE_SPEC`.
:func:`config_from_genome` splits it into an ``OrbitScriptedConfig`` and a
``LookaheadConfig``; ints/bools are rounded/thresholded from the stored float so
crossover and mutation can stay continuous.

Fitness
-------
Each genome plays ``games`` seeded matches as **P0** vs the benchmark opponent;
fitness averages a win bonus, the final production-share margin, a ship-margin
term, and a small bonus for finishing (eliminating the opponent) quickly.

CLI
---
::

    python -m orbit_ga --opponent production --pop 24 --generations 15 \
        --games 4 --max-steps 150 --workers 12 --out ga_runs/run1

Resumable via ``--resume ga_runs/run1/state.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from orbit_scripted_agent import OrbitScriptedConfig


# ---------------------------------------------------------------------------
# Gene specification: name -> (low, high, kind)
# kind: "f" float, "i" int, "b" bool (>=0.5 True), "ll" lookahead float/int
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Gene:
    low: float
    high: float
    kind: str = "f"          # f | i | b
    group: str = "config"    # config | lookahead


GENE_SPEC: dict[str, Gene] = {
    # --- defense / horizon ---
    "horizon_cap": Gene(10.0, 40.0, "f"),
    "potential_weight": Gene(0.0, 1.0, "f"),
    "critical_threshold": Gene(-25.0, -2.0, "f"),
    "comet_evac_turns": Gene(0.0, 5.0, "i"),
    "max_rescue_launches": Gene(1.0, 5.0, "i"),
    # --- attack target weights ---
    "eta_horizon": Gene(25.0, 90.0, "f"),
    "denial_weight": Gene(0.0, 3.0, "f"),
    "influence_weight": Gene(0.0, 3.0, "f"),
    "production_pressure_weight": Gene(0.0, 3.0, "f"),
    "enemy_weakness_weight": Gene(0.0, 3.0, "f"),
    "enemy_empire_support_weight": Gene(0.0, 3.0, "f"),
    "prioritize_enemy_weakspots": Gene(0.0, 1.0, "b"),
    "capture_margin": Gene(0.0, 0.5, "f"),
    "max_wave_launches": Gene(1.0, 6.0, "i"),
    "allow_partial_enemy_waves": Gene(0.0, 1.0, "b"),
    # --- support ---
    "support_reserve_slots": Gene(0.0, 4.0, "i"),
    "support_threat_threshold": Gene(2.0, 20.0, "f"),
    "support_min_source_surplus": Gene(1.0, 15.0, "f"),
    "support_max_source_threat": Gene(0.0, 12.0, "f"),
    "support_cushion": Gene(0.0, 10.0, "f"),
    "max_support_launches_per_target": Gene(1.0, 4.0, "i"),
    # --- consolidation ---
    "consolidation_min_surplus": Gene(3.0, 25.0, "f"),
    "consolidation_fraction": Gene(0.1, 0.9, "f"),
    "consolidation_min_send": Gene(1.0, 15.0, "i"),
    # --- lookahead planner ---
    "lookahead_weight": Gene(0.0, 4.0, "f", group="lookahead"),
    "max_depth": Gene(1.0, 3.0, "i", group="lookahead"),
    "beam_width": Gene(2.0, 6.0, "i", group="lookahead"),
    "unlock_decay": Gene(0.2, 0.9, "f", group="lookahead"),
}

_LOOKAHEAD_FIELDS = {"lookahead_weight", "max_depth", "beam_width", "unlock_decay"}


def _coerce(name: str, raw: float) -> Any:
    g = GENE_SPEC[name]
    v = min(g.high, max(g.low, float(raw)))
    if g.kind == "i":
        return int(round(v))
    if g.kind == "b":
        return bool(v >= 0.5)
    return float(v)


def default_genome() -> dict[str, float]:
    """Genome matching the current shipped defaults (clamped into range)."""
    cfg = OrbitScriptedConfig()
    out: dict[str, float] = {}
    for name, g in GENE_SPEC.items():
        if name in _LOOKAHEAD_FIELDS:
            # Lookahead defaults: off-by-default weight, standard planner knobs.
            defaults = {"lookahead_weight": 1.0, "max_depth": 2.0,
                        "beam_width": 4.0, "unlock_decay": 0.6}
            out[name] = float(defaults[name])
            continue
        val = getattr(cfg, name)
        out[name] = float(1.0 if val is True else (0.0 if val is False else val))
    return out


def random_genome(rng: random.Random) -> dict[str, float]:
    return {name: rng.uniform(g.low, g.high) for name, g in GENE_SPEC.items()}


def config_from_genome(genome: dict[str, float]):
    """Split a genome into ``(OrbitScriptedConfig, LookaheadConfig)``."""
    from orbit_lookahead_agent import LookaheadConfig

    cfg_kwargs: dict[str, Any] = {}
    look_kwargs: dict[str, Any] = {}
    for name in GENE_SPEC:
        val = _coerce(name, genome.get(name, GENE_SPEC[name].low))
        if name in _LOOKAHEAD_FIELDS:
            look_kwargs[name] = val
        else:
            cfg_kwargs[name] = val
    # Keep lookahead's eta_horizon tied to the attack horizon for consistency.
    look_kwargs["eta_horizon"] = float(cfg_kwargs.get("eta_horizon", 60.0))
    return OrbitScriptedConfig(**cfg_kwargs), LookaheadConfig(**look_kwargs)


# ---------------------------------------------------------------------------
# Fitness (runs in worker process)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FitnessConfig:
    opponent: str = "production"     # production | orbit | lookahead | mcts | frozen:<ckpt>
    games: int = 4
    max_steps: int = 150
    num_agents: int = 2
    base_seed: int = 1000
    candidate: str = "lookahead"     # orbit | lookahead | mcts (P0 agent family)
    mcts_simulations: int = 48       # only used when candidate == "mcts"
    win_bonus: float = 1.0
    prod_weight: float = 1.0
    ship_weight: float = 0.25
    speed_weight: float = 0.5        # bonus for finishing early (when decisive)


def evaluate_genome(genome: dict[str, float], fc: FitnessConfig) -> dict[str, float]:
    """Average match score for ``genome`` as P0 vs the benchmark opponent."""
    from orbit_match import play_match, make_controller

    cfg, look = config_from_genome(genome)
    if fc.candidate == "mcts":
        from orbit_mcts import MCTSConfig
        p0 = make_controller(
            "mcts", cfg=cfg,
            mcts_cfg=MCTSConfig(simulations=int(fc.mcts_simulations)),
        )
    elif fc.candidate == "lookahead":
        p0 = make_controller("lookahead", cfg=cfg, lookahead_cfg=look)
    else:
        p0 = make_controller("orbit", cfg=cfg)

    def opp():
        return make_controller(fc.opponent)

    total = 0.0
    wins = 0
    margins = 0.0
    for i in range(int(fc.games)):
        controllers = [p0] + [opp() for _ in range(fc.num_agents - 1)]
        res = play_match(
            controllers,
            num_agents=fc.num_agents,
            max_steps=fc.max_steps,
            seed=fc.base_seed + i,
        )
        win = 1.0 if res.winner == 0 else (0.0 if res.winner >= 0 else 0.5)
        wins += 1 if res.winner == 0 else 0
        margins += res.prod_margin_p0
        score = (
            fc.win_bonus * (win - 0.5) * 2.0
            + fc.prod_weight * res.prod_margin_p0
            + fc.ship_weight * res.ship_margin_p0
        )
        if res.finished and res.winner == 0:
            score += fc.speed_weight * (1.0 - res.length / float(fc.max_steps))
        total += score

    n = max(1, int(fc.games))
    return {
        "fitness": total / n,
        "winrate": wins / n,
        "prod_margin": margins / n,
    }


# Module-level worker shim (picklable for ProcessPoolExecutor).
_WORKER_FC: Optional[FitnessConfig] = None


def _worker_init(fc_dict: dict) -> None:
    global _WORKER_FC
    _WORKER_FC = FitnessConfig(**fc_dict)


def _worker_eval(genome: dict[str, float]) -> dict[str, float]:
    assert _WORKER_FC is not None
    return evaluate_genome(genome, _WORKER_FC)


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------
def crossover(a: dict[str, float], b: dict[str, float], rng: random.Random) -> dict[str, float]:
    """Per-gene blend (BLX-0.5) crossover."""
    child: dict[str, float] = {}
    for name, g in GENE_SPEC.items():
        x, y = a[name], b[name]
        lo, hi = min(x, y), max(x, y)
        d = hi - lo
        child[name] = rng.uniform(lo - 0.5 * d, hi + 0.5 * d)
        child[name] = min(g.high, max(g.low, child[name]))
    return child


def mutate(genome: dict[str, float], rng: random.Random, *, rate: float, scale: float) -> dict[str, float]:
    """Gaussian mutation (per-gene, sigma = ``scale`` * gene range)."""
    out = dict(genome)
    for name, g in GENE_SPEC.items():
        if rng.random() < rate:
            sigma = scale * (g.high - g.low)
            out[name] = min(g.high, max(g.low, out[name] + rng.gauss(0.0, sigma)))
    return out


def tournament(pop: list[dict], fits: list[float], rng: random.Random, k: int) -> dict:
    best_i = None
    for _ in range(max(2, k)):
        i = rng.randrange(len(pop))
        if best_i is None or fits[i] > fits[best_i]:
            best_i = i
    return pop[best_i]


# ---------------------------------------------------------------------------
# GA driver
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    pop_size: int = 24
    generations: int = 15
    elite: int = 4
    tournament_k: int = 3
    mutation_rate: float = 0.25
    mutation_scale: float = 0.15
    seed_with_default: bool = True
    workers: int = max(1, (os.cpu_count() or 2) - 2)
    rng_seed: int = 0


def _evaluate_population(
    pop: list[dict],
    fc: FitnessConfig,
    workers: int,
) -> list[dict[str, float]]:
    if workers <= 1:
        return [evaluate_genome(g, fc) for g in pop]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(asdict(fc),),
    ) as ex:
        return list(ex.map(_worker_eval, pop))


def run_ga(
    ga: GAConfig,
    fc: FitnessConfig,
    out_dir: str | Path,
    *,
    resume: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    log_path = out_dir / "ga_log.jsonl"

    rng = random.Random(ga.rng_seed)

    start_gen = 0
    best_overall: Optional[dict] = None
    if resume and Path(resume).exists():
        saved = json.loads(Path(resume).read_text())
        pop = saved["population"]
        start_gen = int(saved["generation"]) + 1
        best_overall = saved.get("best")
        rng.seed(ga.rng_seed + start_gen)
        if verbose:
            print(f"[ga] resumed from gen {start_gen} ({len(pop)} genomes)")
    else:
        pop = [random_genome(rng) for _ in range(ga.pop_size)]
        if ga.seed_with_default:
            pop[0] = default_genome()

    for gen in range(start_gen, ga.generations):
        t0 = time.time()
        stats = _evaluate_population(pop, fc, ga.workers)
        fits = [s["fitness"] for s in stats]

        order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        pop = [pop[i] for i in order]
        stats = [stats[i] for i in order]
        fits = [fits[i] for i in order]

        gen_best = {"genome": pop[0], **stats[0], "generation": gen}
        if best_overall is None or gen_best["fitness"] > best_overall["fitness"]:
            best_overall = gen_best

        dt = time.time() - t0
        rec = {
            "generation": gen,
            "best_fitness": fits[0],
            "mean_fitness": sum(fits) / len(fits),
            "best_winrate": stats[0]["winrate"],
            "best_prod_margin": stats[0]["prod_margin"],
            "elapsed_s": round(dt, 1),
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if verbose:
            print(
                f"[gen {gen:>3}/{ga.generations}] "
                f"best={fits[0]:+.3f} mean={rec['mean_fitness']:+.3f} "
                f"WR={stats[0]['winrate']:.2f} prodM={stats[0]['prod_margin']:+.3f} "
                f"({dt:.0f}s)"
            )

        # Persist (resumable) — keep the *evaluated, sorted* population.
        state_path.write_text(json.dumps({
            "generation": gen,
            "population": pop,
            "best": best_overall,
            "ga_config": asdict(ga),
            "fitness_config": asdict(fc),
        }, indent=2))
        (out_dir / "best_genome.json").write_text(
            json.dumps(best_overall, indent=2))

        if gen == ga.generations - 1:
            break

        # Next generation: elitism + tournament/crossover/mutation.
        elite = max(0, min(ga.elite, len(pop)))
        next_pop = [dict(pop[i]) for i in range(elite)]
        while len(next_pop) < ga.pop_size:
            pa = tournament(pop, fits, rng, ga.tournament_k)
            pb = tournament(pop, fits, rng, ga.tournament_k)
            child = crossover(pa, pb, rng)
            child = mutate(child, rng, rate=ga.mutation_rate, scale=ga.mutation_scale)
            next_pop.append(child)
        pop = next_pop

    if verbose and best_overall is not None:
        print(f"[ga] done. best fitness={best_overall['fitness']:+.3f} "
              f"WR={best_overall['winrate']:.2f} -> {out_dir/'best_genome.json'}")
    return best_overall or {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Genetic-algorithm tuner for the orbit agent")
    p.add_argument("--opponent", default="production",
                   help="production | orbit | lookahead | frozen:<ckpt.pt>")
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--elite", type=int, default=4)
    p.add_argument("--games", type=int, default=4, help="seeded matches per genome")
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--num-agents", type=int, default=2, choices=[2, 4])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--mutation-rate", type=float, default=0.25)
    p.add_argument("--mutation-scale", type=float, default=0.15)
    p.add_argument("--tournament-k", type=int, default=3)
    p.add_argument("--candidate", default=None,
                   choices=["orbit", "lookahead", "mcts"],
                   help="P0 agent family to evolve (default: lookahead). "
                        "'mcts' tunes OrbitScriptedConfig under Monte-Carlo target search.")
    p.add_argument("--mcts-simulations", type=int, default=48,
                   help="MCTS rollouts per turn when --candidate mcts")
    p.add_argument("--no-lookahead", action="store_true",
                   help="shorthand for --candidate orbit (plain greedy agent)")
    p.add_argument("--no-seed-default", action="store_true",
                   help="do not inject the shipped-default genome into gen 0")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--rng-seed", type=int, default=0)
    p.add_argument("--out", default="ga_runs/run")
    p.add_argument("--resume", default=None, help="path to a state.json to resume")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    # Ensure output dir exists before shell ``tee`` or other wrappers open log files.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ga = GAConfig(
        pop_size=args.pop,
        generations=args.generations,
        elite=args.elite,
        tournament_k=args.tournament_k,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        seed_with_default=not args.no_seed_default,
        workers=args.workers,
        rng_seed=args.rng_seed,
    )
    candidate = args.candidate or ("orbit" if args.no_lookahead else "lookahead")
    fc = FitnessConfig(
        opponent=args.opponent,
        games=args.games,
        max_steps=args.max_steps,
        num_agents=args.num_agents,
        base_seed=args.base_seed,
        candidate=candidate,
        mcts_simulations=args.mcts_simulations,
    )
    resume = args.resume or (str(Path(args.out) / "state.json")
                             if (Path(args.out) / "state.json").exists() else None)
    run_ga(ga, fc, args.out, resume=resume)


if __name__ == "__main__":
    main()
