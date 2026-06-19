"""Evaluate a trained/tuned final micro-planner genome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from final_agent import FinalMicroAgent
from final_train import config_from_genome
from orbit_match import make_controller, play_match


def _load_genome(path: str):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw["genome"] if isinstance(raw, dict) and "genome" in raw else raw


def evaluate(args) -> None:
    genome = _load_genome(args.genome)
    cfg = config_from_genome(genome)
    wins = 0
    games = 0
    score_sum = 0.0
    for i in range(int(args.games)):
        seed = int(args.seed) + i
        final = FinalMicroAgent(cfg)
        opp = make_controller(str(args.opponent))
        res0 = play_match([final, opp], num_agents=2, max_steps=int(args.max_steps), seed=seed)
        wins += int(res0.winner == 0)
        games += 1
        score_sum += float(res0.prod_margin_p0)
        print(f"[p0 game {i:03d}] seed={seed} winner={res0.winner} prod={res0.prod_margin_p0:+.3f} len={res0.length}")

        if bool(args.seat_swap):
            final = FinalMicroAgent(cfg)
            opp = make_controller(str(args.opponent))
            res1 = play_match([opp, final], num_agents=2, max_steps=int(args.max_steps), seed=seed)
            wins += int(res1.winner == 1)
            games += 1
            score_sum += -float(res1.prod_margin_p0)
            print(f"[p1 game {i:03d}] seed={seed} winner={res1.winner} prod_p1={-res1.prod_margin_p0:+.3f} len={res1.length}")

    print(
        f"[summary] games={games} winrate={wins / max(1, games):.3f} "
        f"avg_prod_margin={score_sum / max(1, games):+.3f}"
    )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate final micro-planner genome")
    p.add_argument("--genome", required=True)
    p.add_argument("--opponent", default="orbit")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--no-seat-swap", action="store_false", dest="seat_swap")
    p.set_defaults(seat_swap=True)
    return p


def main(argv: list[str] | None = None) -> None:
    evaluate(_build_argparser().parse_args(argv))


if __name__ == "__main__":
    main()

