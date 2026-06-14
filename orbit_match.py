"""Headless Orbit Wars match runner (no HTML render) for evaluation loops.

Used by the genetic-algorithm tuner (:mod:`orbit_ga`) and the lookahead agent
benchmarks.  A *controller* is anything with

    ``act(env_obs, player, *, current_turn, max_turns) -> list[[from, angle, ships], ...]``

so :class:`orbit_scripted_agent.OrbitScriptedAgent` (and the lookahead agent)
plug in directly.  This module imports **no** torch; the heavy RL / production
opponents are wired lazily only when requested via :func:`make_controller`.
"""

from __future__ import annotations

import argparse
import json
import logging as _logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

# kaggle_environments / OpenSpiel emit a flood of INFO import logs in every
# worker process (they set their own logger level at import, so a per-logger
# setLevel is overridden).  Globally disabling <=INFO logging keeps GA / eval
# progress readable; WARNING and above still surface.  ``print`` is unaffected.
_logging.disable(_logging.INFO)

from feature_extraction import parse_planets, parse_fleets


class Controller(Protocol):
    def act(
        self, env_obs: Any, player: int, *, current_turn: int, max_turns: int,
    ) -> list[list]:
        ...


# ---------------------------------------------------------------------------
# Scoring helpers (production-share margin; mirrors reworked_rollout)
# ---------------------------------------------------------------------------
def _own_production(env_obs: Any, player: int) -> float:
    return float(sum(p.production for p in parse_planets(env_obs) if p.owner == player))


def _own_ships(env_obs: Any, player: int) -> float:
    g = sum(float(p.ships) for p in parse_planets(env_obs) if p.owner == player)
    f = sum(float(fl.ships) for fl in parse_fleets(env_obs) if fl.owner == player)
    return float(g + f)


def _alive(env_obs: Any, player: int) -> bool:
    return any(p.owner == player for p in parse_planets(env_obs))


def production_share_margin(env_obs: Any, player: int, num_agents: int) -> float:
    """``(my_prod - best_other_prod) / total_prod`` in ``[-1, 1]`` (0 if empty)."""
    prods = [_own_production(env_obs, p) for p in range(num_agents)]
    total = sum(prods)
    if total <= 0.0:
        return 0.0
    mine = prods[player]
    others = [prods[p] for p in range(num_agents) if p != player]
    best_other = max(others) if others else 0.0
    return float((mine - best_other) / total)


@dataclass
class MatchResult:
    winner: int                     # player index, or -1 for draw/timeout-tie
    length: int
    rewards: list[float]
    prod_margin_p0: float           # production-share margin from P0's view
    ship_margin_p0: float
    finished: bool                  # True if a player was eliminated / env done
    survivors: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Match loop
# ---------------------------------------------------------------------------
def play_match(
    controllers: list[Controller],
    *,
    num_agents: int = 2,
    max_steps: int = 300,
    seed: Optional[int] = None,
) -> MatchResult:
    """Play one headless game; ``controllers[p]`` drives player ``p``."""
    from kaggle_environments import make
    import random as _random

    if len(controllers) != num_agents:
        raise ValueError(
            f"need {num_agents} controllers, got {len(controllers)}")

    env = make("orbit_wars", debug=False)
    if seed is not None:
        env.configuration["randomSeed"] = int(seed)
        _random.seed(int(seed))
    env.reset(num_agents=num_agents)
    state = env.step([[] for _ in range(num_agents)])

    t = 0
    for t in range(max_steps):
        env_actions = []
        for p in range(num_agents):
            env_obs = state[p]["observation"]
            if _alive(env_obs, p):
                ea = controllers[p].act(
                    env_obs, p, current_turn=t, max_turns=max_steps)
            else:
                ea = []
            env_actions.append(ea or [])
        state = env.step(env_actions)
        done = env.done or any(
            getattr(state[p], "status", "ACTIVE") == "DONE"
            for p in range(num_agents)
        )
        if done:
            break

    final_obs = state[0]["observation"]
    rewards = [float(getattr(state[p], "reward", 0.0) or 0.0) for p in range(num_agents)]
    survivors = [p for p in range(num_agents) if _alive(final_obs, p)]

    best = max(rewards) if rewards else 0.0
    winner = -1
    if rewards.count(best) == 1:
        winner = int(max(range(len(rewards)), key=lambda i: rewards[i]))

    return MatchResult(
        winner=winner,
        length=t + 1,
        rewards=rewards,
        prod_margin_p0=production_share_margin(final_obs, 0, num_agents),
        ship_margin_p0=(
            (_own_ships(final_obs, 0) - max(
                (_own_ships(final_obs, p) for p in range(1, num_agents)), default=0.0))
            / max(1.0, sum(_own_ships(final_obs, p) for p in range(num_agents)))
        ),
        finished=bool(env.done or len(survivors) < num_agents),
        survivors=survivors,
    )


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------
def make_controller(kind: str, **kwargs: Any) -> Controller:
    """Build a controller by name.

    ``kind``:
        * ``"orbit"``      — :class:`OrbitScriptedAgent` (optional ``cfg=``).
        * ``"lookahead"``  — :class:`OrbitLookaheadAgent` (optional ``cfg=``,
          ``lookahead_cfg=``).
        * ``"timed_mcts"`` / ``"tmcts"`` — time-aware capture MCTS attack
          bonus (optional ``cfg=``, ``timed_mcts_cfg=``).
        * ``"production"`` / ``"scripted:<name>"`` — the action_row scripted bot
          (lazily pulls the torch stack via :mod:`reworked_watch`).
        * ``"frozen:<ckpt>"`` — a trained RL checkpoint as P-controller.
    """
    if kind == "orbit":
        from orbit_scripted_agent import OrbitScriptedAgent
        return OrbitScriptedAgent(cfg=kwargs.get("cfg"))
    if kind == "mcts":
        from orbit_mcts_agent import OrbitMCTSAgent
        return OrbitMCTSAgent(cfg=kwargs.get("cfg"), mcts_cfg=kwargs.get("mcts_cfg"))
    if kind in ("timed_mcts", "tmcts"):
        from orbit_timed_mcts_agent import OrbitTimedMCTSAgent
        return OrbitTimedMCTSAgent(
            cfg=kwargs.get("cfg"), timed_mcts_cfg=kwargs.get("timed_mcts_cfg"))
    if kind == "lookahead":
        from orbit_lookahead_agent import OrbitLookaheadAgent
        return OrbitLookaheadAgent(
            cfg=kwargs.get("cfg"), lookahead_cfg=kwargs.get("lookahead_cfg"))
    if kind == "production" or kind.startswith("scripted:"):
        strategy = "production" if kind == "production" else kind.split(":", 1)[1]
        return _ScriptedRowController(strategy)
    if kind.startswith("frozen:"):
        return _FrozenPolicyController(kind.split(":", 1)[1])
    raise ValueError(f"unknown controller kind {kind!r}")


class _ScriptedRowController:
    """Adapts the action_row scripted bots (e.g. 'production') to Controller."""

    def __init__(self, strategy: str):
        self.strategy = strategy
        import torch  # noqa: F401  (ensure available)
        from reworked_watch import _scripted_env_action
        self._fn = _scripted_env_action
        import torch as _t
        self._device = _t.device("cpu")

    def act(self, env_obs, player, *, current_turn, max_turns):
        return self._fn(
            env_obs, player, current_turn, max_turns, self._device, self.strategy)


class _FrozenPolicyController:
    """Adapts a trained RL checkpoint to the Controller protocol (greedy)."""

    def __init__(self, ckpt_path: str):
        import torch
        from reworked_train import load_checkpoint
        from reworked_watch import _policy_env_action
        self._torch = torch
        self._device = torch.device("cpu")
        self._policy, ckpt = load_checkpoint(ckpt_path, device=self._device)
        self._policy.eval()
        arch = ckpt.get("arch", {})
        self._kind = arch.get("policy_kind", "hier")
        self._ivr = bool(arch.get("eta_value_feature", False))
        self._min_send = int(ckpt.get("train_cfg", {}).get("min_send", 1) or 1)
        self._fn = _policy_env_action
        self._cap: dict[int, int] = {}

    def act(self, env_obs, player, *, current_turn, max_turns):
        with self._torch.no_grad():
            return self._fn(
                self._policy, env_obs, player, current_turn, max_turns,
                self._device, True, self._ivr, self._kind, self._min_send,
                cap_counts=self._cap)


__all__ = [
    "Controller",
    "MatchResult",
    "play_match",
    "make_controller",
    "production_share_margin",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_genome_config(path: str | None):
    if not path:
        return None, None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    genome = raw["genome"] if isinstance(raw, dict) and "genome" in raw else raw
    from orbit_ga import config_from_genome

    return config_from_genome(genome)


def _make_cli_controller(kind: str, *, genome_path: str | None, args) -> Controller:
    cfg, look = _load_genome_config(genome_path)
    if kind == "mcts":
        from orbit_mcts import MCTSConfig

        return make_controller(
            "mcts",
            cfg=cfg,
            mcts_cfg=MCTSConfig(simulations=int(args.mcts_simulations)),
        )
    if kind in ("timed_mcts", "tmcts"):
        from orbit_timed_mcts import TimedMCTSConfig

        return make_controller(
            "timed_mcts",
            cfg=cfg,
            timed_mcts_cfg=TimedMCTSConfig(
                simulations=int(args.timed_mcts_simulations),
                horizon_turns=int(args.timed_horizon_turns),
                eta_horizon=float(args.timed_eta_horizon),
                bonus_weight=float(args.timed_bonus_weight),
                source_reserve_frac=float(args.timed_source_reserve_frac),
                gamma=float(args.timed_gamma),
                capture_income_weight=float(args.timed_capture_income_weight),
                ship_cost_weight=float(args.timed_ship_cost_weight),
                eta_cost_weight=float(args.timed_eta_cost_weight),
            ),
        )
    if kind == "lookahead":
        return make_controller("lookahead", cfg=cfg, lookahead_cfg=look)
    if kind == "orbit":
        return make_controller("orbit", cfg=cfg)
    return make_controller(kind)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Headless Orbit Wars match runner")
    p.add_argument("--p0", default="timed_mcts",
                   help="P0 controller: timed_mcts | mcts | orbit | lookahead | production | scripted:<name> | frozen:<ckpt>")
    p.add_argument("--p1", default="mcts",
                   help="P1 controller: timed_mcts | mcts | orbit | lookahead | production | scripted:<name> | frozen:<ckpt>")
    p.add_argument("--p0-genome", default=None,
                   help="Optional GA best_genome.json for P0 scripted config")
    p.add_argument("--p1-genome", default=None,
                   help="Optional GA best_genome.json for P1 scripted config")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-steps", type=int, default=300)

    p.add_argument("--mcts-simulations", type=int, default=12,
                   help="Legacy OrbitMCTSAgent simulations per turn")
    p.add_argument("--timed-mcts-simulations", type=int, default=96,
                   help="TimedCaptureMCTS simulations per turn")
    p.add_argument("--timed-horizon-turns", type=int, default=10,
                   help="TimedCaptureMCTS turn horizon")
    p.add_argument("--timed-eta-horizon", type=float, default=60.0,
                   help="Max ETA considered reachable by timed MCTS")
    p.add_argument("--timed-bonus-weight", type=float, default=0.03,
                   help="Scale timed MCTS metric improvement before attack ranking")
    p.add_argument("--timed-source-reserve-frac", type=float, default=0.25)
    p.add_argument("--timed-gamma", type=float, default=0.97)
    p.add_argument("--timed-capture-income-weight", type=float, default=1.0)
    p.add_argument("--timed-ship-cost-weight", type=float, default=0.05)
    p.add_argument("--timed-eta-cost-weight", type=float, default=0.05)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    wins = 0
    draws = 0
    prod_sum = 0.0
    ship_sum = 0.0
    length_sum = 0
    games = max(1, int(args.games))

    for i in range(games):
        controllers = [
            _make_cli_controller(args.p0, genome_path=args.p0_genome, args=args),
            _make_cli_controller(args.p1, genome_path=args.p1_genome, args=args),
        ]
        res = play_match(
            controllers,
            num_agents=2,
            max_steps=int(args.max_steps),
            seed=int(args.seed) + i,
        )
        wins += int(res.winner == 0)
        draws += int(res.winner < 0)
        prod_sum += float(res.prod_margin_p0)
        ship_sum += float(res.ship_margin_p0)
        length_sum += int(res.length)
        print(
            f"[game {i:03d}] seed={int(args.seed) + i} winner={res.winner} "
            f"prod={res.prod_margin_p0:+.3f} ship={res.ship_margin_p0:+.3f} "
            f"len={res.length} finished={int(res.finished)}"
        )

    print(
        f"[summary] P0={args.p0} P1={args.p1} games={games} "
        f"winrate={wins / games:.3f} drawrate={draws / games:.3f} "
        f"prod={prod_sum / games:+.3f} ship={ship_sum / games:+.3f} "
        f"avg_len={length_sum / games:.1f}"
    )


if __name__ == "__main__":
    main()
