"""Evaluate final neural PPO checkpoints."""

from __future__ import annotations

import argparse

from final_rl_controller import FinalRLController
from orbit_match import make_controller, play_match


def evaluate(args) -> None:
    wins = 0
    games = 0
    prod_sum = 0.0
    for i in range(int(args.games)):
        seed = int(args.seed) + i
        final = FinalRLController(args.ckpt, device=args.device, greedy=not args.sample)
        opp = make_controller(args.opponent)
        res0 = play_match([final, opp], num_agents=2, max_steps=int(args.max_steps), seed=seed)
        wins += int(res0.winner == 0)
        games += 1
        prod_sum += float(res0.prod_margin_p0)
        print(f"[p0 {i:03d}] seed={seed} winner={res0.winner} prod={res0.prod_margin_p0:+.3f} len={res0.length}")

        if args.seat_swap:
            final = FinalRLController(args.ckpt, device=args.device, greedy=not args.sample)
            opp = make_controller(args.opponent)
            res1 = play_match([opp, final], num_agents=2, max_steps=int(args.max_steps), seed=seed)
            wins += int(res1.winner == 1)
            games += 1
            prod_sum += -float(res1.prod_margin_p0)
            print(f"[p1 {i:03d}] seed={seed} winner={res1.winner} prod_p1={-res1.prod_margin_p0:+.3f} len={res1.length}")

    print(f"[summary] games={games} winrate={wins / max(1, games):.3f} prod={prod_sum / max(1, games):+.3f}")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate final neural checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--opponent", default="orbit")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--no-seat-swap", action="store_false", dest="seat_swap")
    p.set_defaults(seat_swap=True)
    return p


def main(argv: list[str] | None = None) -> None:
    evaluate(_build_argparser().parse_args(argv))


if __name__ == "__main__":
    main()

