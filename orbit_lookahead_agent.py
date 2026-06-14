"""Multi-step (stepping-stone) planning layer over :class:`OrbitScriptedAgent`.

The base agent is purely greedy: each turn it picks the locally best capture and
never reasons "take the 18-garrison world first *because* it unlocks the 5 behind
it."  :class:`OrbitLookaheadAgent` keeps **all** of the base agent's mechanics
(defense triage, all-or-nothing waves, support, consolidation) untouched and only
adds a *priority bonus* to the attack ranking via the base
``_attack_priority_bonus`` hook.

How the bonus is computed
-------------------------
Planet routes are pure geometry (precomputed on every directed edge), so a
planet's reachable set depends only on *which planets we own*.  We therefore run
a small depth-limited search:

1. ``reach0`` = targets reachable from our current planets (within the ETA
   horizon).
2. For each directly-reachable first step ``T``, pretend we capture it (add ``T``
   to the owned set) and compute which **new** targets that unlocks.
3. The bonus for ``T`` is the (discounted, time-shifted) production those newly
   unlocked worlds would yield — recursively to ``max_depth``, with branching
   capped at ``beam_width``.

A gateway like the 18-world that opens the path to the 5-world (and beyond) thus
gets a large bonus and is captured first; a dead-end world gets none.  When
nothing is unlocked the bonus is empty and ranking is identical to the base
agent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from orbit_expansion import production_value_until_end
from orbit_graph import OrbitGraph
from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig
from orbit_defense import PlanetDefenseState


@dataclass(frozen=True)
class LookaheadConfig:
    """Hyperparameters for the stepping-stone planner."""

    lookahead_weight: float = 1.0
    """Scale applied to unlocked-production value before it joins the ranking."""
    max_depth: int = 2
    """How many sequential captures deep to plan (1 = unlock-only)."""
    beam_width: int = 4
    """Top-K newly-unlocked worlds expanded per depth (branching cap)."""
    eta_horizon: float = 60.0
    """Max ETA for a route to count as reachable during planning."""
    gamma: float = 0.97
    """Discount for production value over the remaining horizon."""
    unlock_decay: float = 0.6
    """Per-depth decay on deeper unlocked value (uncertainty about getting there)."""
    min_bonus: float = 1e-6
    """Drop bonuses below this (keeps ranking identical when nothing unlocks)."""


def _owned_ids(og: OrbitGraph) -> set[int]:
    return {
        int(p.id) for p in og.planets
        if int(p.owner) == int(og.player) and not getattr(p, "is_comet", False)
    }


def _candidate_target_ids(og: OrbitGraph) -> list[int]:
    """Non-owned, non-comet planets eligible as capture targets."""
    out = []
    for p in og.planets:
        if int(p.owner) == int(og.player) or getattr(p, "is_comet", False):
            continue
        out.append(int(p.id))
    return out


def _reachable_etas(
    og: OrbitGraph,
    owned: set[int],
    candidates: list[int],
    *,
    eta_horizon: float,
) -> dict[int, float]:
    """``tgt -> min reachable ETA`` from any planet in ``owned`` (geometry only)."""
    reach: dict[int, float] = {}
    G = og.G
    for tgt in candidates:
        if tgt in owned:
            continue
        best = math.inf
        for src in owned:
            if src == tgt or not G.has_edge(src, tgt):
                continue
            data = G.edges[src, tgt]
            if not data.get("reachable", False):
                continue
            eta = float(data.get("eta", math.inf))
            if eta < best:
                best = eta
        if math.isfinite(best) and best <= float(eta_horizon):
            reach[tgt] = best
    return reach


class _StepStonePlanner:
    """Depth-limited unlocked-production search (see module docstring)."""

    def __init__(
        self,
        og: OrbitGraph,
        cfg: LookaheadConfig,
        *,
        current_turn: int,
        max_turns: int,
    ):
        self.og = og
        self.cfg = cfg
        self.current_turn = int(current_turn)
        self.max_turns = int(max_turns)
        self.candidates = _candidate_target_ids(og)
        self._prod = {int(p.id): int(p.production) for p in og.planets}

    def _pv(self, pid: int, eta: float) -> float:
        pv, _hold = production_value_until_end(
            self._prod.get(int(pid), 0),
            eta=float(eta),
            current_turn=self.current_turn,
            max_turns=self.max_turns,
            gamma=float(self.cfg.gamma),
        )
        return float(pv)

    def _unlocked_value(
        self,
        owned: set[int],
        reach_prev: dict[int, float],
        captured_id: int,
        eta_to_captured: float,
        depth: int,
    ) -> float:
        """Discounted production unlocked by capturing ``captured_id``."""
        if depth <= 0:
            return 0.0
        owned2 = owned | {int(captured_id)}
        reach2 = _reachable_etas(
            self.og, owned2, self.candidates, eta_horizon=self.cfg.eta_horizon)
        newly = {
            x: e for x, e in reach2.items()
            if x not in reach_prev and x != int(captured_id)
        }
        if not newly:
            return 0.0

        ranked = sorted(
            newly.items(),
            key=lambda kv: self._prod.get(kv[0], 0),
            reverse=True,
        )[: max(1, int(self.cfg.beam_width))]

        decay = float(self.cfg.unlock_decay)
        total = 0.0
        for x, eta_from_captured in ranked:
            total_eta = float(eta_to_captured) + float(eta_from_captured)
            total += self._pv(x, total_eta)
            total += decay * self._unlocked_value(
                owned2, reach2, x, total_eta, depth - 1)
        return total

    def bonus(self) -> dict[int, float]:
        owned = _owned_ids(self.og)
        if not owned:
            return {}
        reach0 = _reachable_etas(
            self.og, owned, self.candidates, eta_horizon=self.cfg.eta_horizon)
        if not reach0:
            return {}

        out: dict[int, float] = {}
        for tgt, eta0 in reach0.items():
            unlocked = self._unlocked_value(
                owned, reach0, tgt, eta0, int(self.cfg.max_depth))
            val = float(self.cfg.lookahead_weight) * unlocked
            if val > float(self.cfg.min_bonus):
                out[int(tgt)] = val
        return out


class OrbitLookaheadAgent(OrbitScriptedAgent):
    """:class:`OrbitScriptedAgent` + stepping-stone attack prioritization.

    Identical to the base agent except the attack phase promotes capture targets
    that unlock further expansion.  Set ``lookahead_cfg.lookahead_weight = 0`` (or
    pass an empty plan) to recover exact base behavior.
    """

    def __init__(
        self,
        cfg: OrbitScriptedConfig | None = None,
        lookahead_cfg: LookaheadConfig | None = None,
    ):
        super().__init__(cfg=cfg)
        self.lookahead_cfg = lookahead_cfg or LookaheadConfig()

    def _attack_priority_bonus(
        self,
        og: OrbitGraph,
        plan: Any,
        defense_states: list[PlanetDefenseState],
        *,
        omega: float,
        current_turn: int,
        max_turns: int,
    ) -> Optional[dict[int, float]]:
        if float(self.lookahead_cfg.lookahead_weight) == 0.0:
            return None
        planner = _StepStonePlanner(
            og, self.lookahead_cfg,
            current_turn=int(current_turn), max_turns=int(max_turns))
        bonus = planner.bonus()
        return bonus or None


__all__ = [
    "OrbitLookaheadAgent",
    "LookaheadConfig",
]
