"""Macro action MCTS for Orbit Wars (AlphaGo-flavoured search core).

Where :mod:`orbit_timed_mcts` searches a *capture-only* macro tree, this module
generalises the same abstract turn-horizon model to **typed macro actions**:

* ``Capture(target)``     -- fund a capture wave at a neutral/enemy planet.
* ``Reinforce(target)``   -- schedule a friendly wave onto a threatened owned
                              planet so it survives an incoming enemy arrival.
* ``Pass``                -- bank production / ships for a turn.

(``Evacuate`` / ``Consolidate`` are reserved for a later stage; see the plan in
``docs/macro_mcts_agent_plan.md``.)

The forward model is intentionally the same abstract one as the timed MCTS:
owned planets produce ships, capture/reinforce waves consume source garrisons
*now* and schedule an arrival event, and arrivals resolve after their ETA.  A
key property we lean on: the abstract model already credits **defense** for
free.  Enemy fleets in flight are seeded as pending events, so an undefended
owned planet flips to the enemy (a reward *loss*); a successful ``Reinforce``
adds friendly ships before the enemy lands and prevents that loss.  No extra
defense bookkeeping is needed for the value to be roughly correct.

AlphaGo / AlphaZero hooks
-------------------------
The search is structured so a neural network can drop in without rewrites:

* ``prior_provider`` -- maps ``(state, candidate macro actions)`` to prior
  logits keyed by :meth:`MacroAction.key`, used inside PUCT selection.  Default
  is uniform (classic UCT-style behaviour under the PUCT formula).
* ``leaf_value_fn`` -- evaluates a leaf abstract state with a learned value in
  the same scale as :meth:`MacroTimedMCTS._score`.  Blended with the heuristic
  rollout via ``value_blend`` (0 = pure rollout, 1 = pure network value).

:meth:`MacroTimedMCTS.run` returns a :class:`MacroSearchResult` carrying the
root visit counts / Q-values per macro action -- exactly the policy target
(visit distribution) and value bootstrap needed for self-play training.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Protocol

from orbit_defense import PlanetDefenseState
from orbit_graph import OrbitGraph
from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig
from orbit_timed_mcts import (
    TimedCaptureMCTS,
    TimedMCTSConfig,
    _CaptureEvent,
    _TimedState,
)


class MacroActionType(str, Enum):
    """Macro decision categories searched by :class:`MacroTimedMCTS`."""

    CAPTURE = "capture"
    REINFORCE = "reinforce"
    EVACUATE = "evacuate"
    CONSOLIDATE = "consolidate"
    PASS = "pass"


@dataclass(frozen=True)
class MacroAction:
    """One abstract macro decision plus its funded source plan.

    ``target`` is the planet the wave lands on (capture target / reinforced
    owned planet / consolidation destination).  For ``PASS`` it is ``-1``.
    ``sources`` is the ``(source_id, ships)`` funding plan chosen by the abstract
    planner; the real LP allocator re-derives launches at execution time.
    """

    type: MacroActionType
    target: int = -1
    arrival_turn: int = 0
    ships: int = 0
    sources: tuple[tuple[int, int], ...] = ()
    heuristic: float = 0.0

    def key(self) -> tuple[str, int]:
        """Hashable identity for the search tree (one plan per type+target)."""
        return (self.type.value, int(self.target))


PASS_ACTION = MacroAction(type=MacroActionType.PASS, target=-1)


# --------------------------------------------------------------------------- #
# Learned-prior hook (Phase 3): same spirit as orbit_learned_mcts_agent.
# --------------------------------------------------------------------------- #
MACRO_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "is_capture",
    "is_reinforce",
    "is_evacuate",
    "is_consolidate",
    "is_pass",
    "target_prod_norm",
    "target_ships_norm",
    "target_is_enemy",
    "target_is_neutral",
    "arrival_norm",
    "ships_sent_norm",
    "value_norm",
    "heuristic_norm",
    "turn_frac",
    "root_owned_frac",
)


@dataclass(frozen=True)
class MacroPriorRequest:
    """Input passed to a learned macro-prior provider."""

    root_player: int
    player_to_move: int
    relative_turn: int
    current_turn: int
    max_turns: int
    actions: tuple[MacroAction, ...]
    features_by_key: Mapping[tuple[str, int], Mapping[str, float]]


class MacroPriorProvider(Protocol):
    """Callable branch prior used by :class:`MacroTimedMCTS`."""

    def __call__(self, request: MacroPriorRequest) -> Mapping[tuple[str, int], float]:
        """Return unnormalised prior logits keyed by :meth:`MacroAction.key`."""


# State-level value hook (Phase 2): abstract leaf -> scalar value.
LeafValueFn = Callable[[_TimedState, int], float]


@dataclass(frozen=True)
class MacroMCTSConfig:
    """Hyperparameters for :class:`MacroTimedMCTS`.

    Reuses the abstract-model knobs from :class:`TimedMCTSConfig` (horizon,
    discount, capture margins, score weights, simulations) and adds macro
    branching caps plus the PUCT / value-blend controls for the AlphaGo hooks.
    """

    timed: TimedMCTSConfig = field(default_factory=TimedMCTSConfig)

    top_k_capture: int = 4
    top_k_reinforce: int = 3
    top_k_evac: int = 2
    top_k_consolidate: int = 1
    include_pass: bool = True

    # Staged action types (plan stages 3-4): off by default so the base agent
    # keeps the Capture + Reinforce + Pass behaviour until explicitly enabled.
    enable_evacuate: bool = False
    enable_consolidate: bool = False

    c_puct: float = 1.4
    prior_temperature: float = 1.0
    min_prior: float = 1e-4

    # Hybrid leaf evaluation: value = (1 - blend) * rollout + blend * leaf_value.
    value_blend: float = 0.0

    # Reinforce planning.
    reinforce_margin: float = 1.0
    """Extra friendly ships above the projected shortfall to commit a save."""

    # Root exploration noise (AlphaZero self-play): mix Dir(alpha) into the
    # root priors.  Disabled by default (alpha = 0) for deterministic play.
    root_dirichlet_alpha: float = 0.0
    root_noise_frac: float = 0.25

    # Root output scaling (capture priority bonus for the scripted allocator).
    bonus_weight: float = 1.0
    min_bonus: float = 1e-6


@dataclass
class _MacroNode:
    state: _TimedState
    untried: list[MacroAction]
    priors: dict[tuple[str, int], float]
    actions: dict[tuple[str, int], MacroAction] = field(default_factory=dict)
    children: dict[tuple[str, int], "_MacroNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    parent: Optional["_MacroNode"] = None
    action_key: Optional[tuple[str, int]] = None

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


@dataclass(frozen=True)
class MacroSearchResult:
    """Root statistics from one macro search (policy + value targets)."""

    baseline: float
    root_value: float
    visits: dict[tuple[str, int], int]
    q_values: dict[tuple[str, int], float]
    actions: dict[tuple[str, int], MacroAction]
    priors: dict[tuple[str, int], float]

    def visit_distribution(self, *, temperature: float = 1.0) -> dict[tuple[str, int], float]:
        """Normalised visit-count policy target ``pi(a)`` (AlphaZero policy)."""
        if not self.visits:
            return {}
        temp = max(1e-6, float(temperature))
        if abs(temp - 1.0) < 1e-9:
            counts = {k: float(v) for k, v in self.visits.items()}
        else:
            counts = {k: float(v) ** (1.0 / temp) for k, v in self.visits.items()}
        total = sum(counts.values())
        if total <= 0.0:
            n = len(counts)
            return {k: 1.0 / n for k in counts}
        return {k: v / total for k, v in counts.items()}

    def best_action(self) -> Optional[MacroAction]:
        """Most-visited macro action (greedy play)."""
        if not self.visits:
            return None
        best_key = max(self.visits, key=lambda k: (self.visits[k], self.q_values.get(k, 0.0)))
        return self.actions.get(best_key)

    def capture_bonus(self, *, weight: float, min_bonus: float) -> dict[int, float]:
        """Per-target capture priority bonus for the scripted attack allocator."""
        out: dict[int, float] = {}
        for key, action in self.actions.items():
            if action.type is not MacroActionType.CAPTURE:
                continue
            if self.visits.get(key, 0) <= 0:
                continue
            improvement = self.q_values.get(key, 0.0) - self.baseline
            val = float(weight) * float(improvement)
            if val > float(min_bonus):
                out[int(action.target)] = val
        return out


class MacroTimedMCTS(TimedCaptureMCTS):
    """Typed-macro MCTS over the abstract turn-horizon model (PUCT selection)."""

    def __init__(
        self,
        og: OrbitGraph,
        cfg: MacroMCTSConfig,
        *,
        root_player: int,
        current_turn: int = 0,
        max_turns: int = 500,
        prior_provider: Optional[MacroPriorProvider] = None,
        leaf_value_fn: Optional[LeafValueFn] = None,
        init_state: Optional[_TimedState] = None,
        rng: Optional[random.Random] = None,
    ):
        self.macro_cfg = cfg
        self.prior_provider = prior_provider
        self.leaf_value_fn = leaf_value_fn
        super().__init__(
            og,
            cfg.timed,
            root_player=root_player,
            current_turn=current_turn,
            max_turns=max_turns,
            rng=rng,
        )
        if init_state is not None:
            # Self-play hands an *evolved* abstract state (owners / ships /
            # pending) that supersedes the board built from ``og``.  The board
            # geometry (productions, edge ETAs, candidates) stays fixed; only
            # the dynamic state is replaced and rebased to a relative turn 0.
            self._initial_state = self._rebase_state(init_state)
            self._baseline = self._score(self._advance_to_horizon(self._initial_state))

    def _rebase_state(self, state: _TimedState) -> _TimedState:
        """Shift an external live state to this search's relative turn 0."""
        shift = int(state.turn)
        owners = state.owner_map()
        pending = []
        for e in state.pending:
            arr = int(e.arrival_turn) - shift
            if arr <= 0:
                continue  # already resolved before this decision
            if int(e.target) not in owners:
                continue
            pending.append(_CaptureEvent(
                arrival_turn=arr,
                owner=int(e.owner),
                target=int(e.target),
                ships=int(e.ships),
            ))
        return self._state(
            turn=0,
            player_to_move=int(state.player_to_move),
            owners=owners,
            ships=state.ship_map(),
            pending=pending,
            reward=0.0,
        )

    # ------------------------------------------------------------------ #
    # Threat / reinforce planning on the abstract state.
    # ------------------------------------------------------------------ #
    def _threatened_owned(
        self, state: _TimedState, player: int
    ) -> dict[int, tuple[float, int]]:
        """Owned planets with pending enemy arrivals.

        Returns ``target -> (summed_enemy_ships, earliest_relative_eta)`` over
        future enemy events (approximate: per-event flips are not summed, but
        the aggregate is a fine candidate-generation signal).
        """
        owners = state.owner_map()
        out: dict[int, tuple[float, int]] = {}
        for e in state.pending:
            if int(e.owner) == int(player) or int(e.owner) < 0:
                continue
            tgt = int(e.target)
            if owners.get(tgt, -1) != int(player):
                continue
            rel = int(e.arrival_turn) - int(state.turn)
            if rel <= 0:
                continue
            ships, eta = out.get(tgt, (0.0, rel))
            out[tgt] = (ships + float(e.ships), min(eta, rel))
        return out

    def _plan_reinforce(
        self,
        state: _TimedState,
        player: int,
        target: int,
        enemy_ships: float,
        enemy_rel_eta: int,
    ) -> Optional[MacroAction]:
        """All-or-nothing friendly wave that lands *before* the enemy arrival."""
        if int(state.turn) >= int(self.horizon_turns):
            return None
        owners = state.owner_map()
        ships = state.ship_map()
        target = int(target)
        if owners.get(target, -1) != int(player):
            return None

        prod = float(self._prod.get(target, 0))
        # Projected defenders when the enemy lands (garrison + production).
        defense = float(ships.get(target, 0)) + prod * float(enemy_rel_eta)
        need = float(enemy_ships) - defense + float(self.macro_cfg.reinforce_margin)
        if need <= 0.0:
            return None  # already safe; not a real reinforce candidate
        need_i = max(1, int(math.ceil(need)))

        threatened = self._threatened_owned(state, player)
        enemy_arrival_turn = int(state.turn) + int(enemy_rel_eta)

        cands: list[tuple[float, int, int, int]] = []  # (eta, src, surplus, arrival)
        for src, owner in owners.items():
            src = int(src)
            if int(owner) != int(player) or src == target:
                continue
            if src in threatened:  # never strip another threatened planet
                continue
            if not self.og.G.has_edge(src, target):
                continue
            data = self.og.G.edges[src, target]
            if not data.get("reachable", False):
                continue
            eta = float(data.get("eta", math.inf))
            if not math.isfinite(eta) or eta > float(self.cfg.eta_horizon):
                continue
            arrival = int(state.turn) + max(1, int(math.ceil(eta)))
            if arrival >= enemy_arrival_turn:
                continue  # must land strictly before the enemy wave
            src_ships = int(ships.get(src, 0))
            surplus = int(math.floor(src_ships * (1.0 - float(self.cfg.source_reserve_frac))))
            if surplus < int(self.cfg.min_send):
                continue
            cands.append((eta, src, surplus, arrival))
        cands.sort(key=lambda x: x[0])

        chosen: list[tuple[int, int]] = []
        total = 0
        arrival_turn = int(state.turn) + 1
        for eta, src, surplus, arrival in cands:
            if total >= need_i:
                break
            send = min(int(surplus), need_i - total)
            if send < int(self.cfg.min_send):
                continue
            chosen.append((int(src), int(send)))
            total += int(send)
            arrival_turn = max(arrival_turn, int(arrival))
        if total < need_i or not chosen:
            return None  # all-or-nothing: cannot save -> not a candidate

        # Value of the save = production income we would otherwise lose, minus
        # the ships committed.  (The board score also keeps this planet on our
        # side, so the search Q reflects the avoided enemy capture directly.)
        saved = self._capture_income(target, enemy_arrival_turn)
        ship_penalty = float(self.cfg.ship_cost_weight) * (
            float(total) / max(1e-6, float(self.cfg.ship_cost_norm))
        )
        heuristic = float(saved) - float(ship_penalty)
        return MacroAction(
            type=MacroActionType.REINFORCE,
            target=target,
            arrival_turn=int(arrival_turn),
            ships=int(total),
            sources=tuple(chosen),
            heuristic=float(heuristic),
        )

    def _reachable_from(self, src: int, tgt: int) -> Optional[float]:
        """ETA of a reachable ``src -> tgt`` leg, else ``None``."""
        src, tgt = int(src), int(tgt)
        if src == tgt or not self.og.G.has_edge(src, tgt):
            return None
        data = self.og.G.edges[src, tgt]
        if not data.get("reachable", False):
            return None
        eta = float(data.get("eta", math.inf))
        if not math.isfinite(eta) or eta > float(self.cfg.eta_horizon):
            return None
        return eta

    def _doomed_owned(
        self, state: _TimedState, player: int
    ) -> dict[int, tuple[float, int]]:
        """Threatened owned planets whose current trajectory loses them.

        ``target -> (enemy_ships, earliest_relative_eta)`` where projected
        defenders (garrison + production to the enemy arrival) fall short.
        """
        owners = state.owner_map()
        ships = state.ship_map()
        out: dict[int, tuple[float, int]] = {}
        for tgt, (enemy_ships, eta) in self._threatened_owned(state, player).items():
            prod = float(self._prod.get(int(tgt), 0))
            defense = float(ships.get(int(tgt), 0)) + prod * float(eta)
            if defense < float(enemy_ships):
                out[int(tgt)] = (float(enemy_ships), int(eta))
        return out

    def _safe_friendly(self, state: _TimedState, player: int) -> set[int]:
        """Owned planets not currently under any pending enemy threat."""
        owners = state.owner_map()
        threatened = set(self._threatened_owned(state, player))
        return {
            int(pid) for pid, o in owners.items()
            if int(o) == int(player) and int(pid) not in threatened
        }

    def _plan_evacuate(
        self, state: _TimedState, player: int, source: int, enemy_rel_eta: int
    ) -> Optional[MacroAction]:
        """Rescue a doomed planet's garrison to the nearest safe friendly planet."""
        if int(state.turn) >= int(self.horizon_turns):
            return None
        ships = state.ship_map()
        source = int(source)
        garrison = int(ships.get(source, 0))
        if garrison < int(self.cfg.min_send):
            return None
        enemy_arrival = int(state.turn) + int(enemy_rel_eta)

        best_dest = None
        best_eta = math.inf
        for dest in self._safe_friendly(state, player):
            if dest == source:
                continue
            eta = self._reachable_from(source, dest)
            if eta is None:
                continue
            arrival = int(state.turn) + max(1, int(math.ceil(eta)))
            if arrival >= enemy_arrival:
                continue  # ships must leave before the planet falls
            if eta < best_eta:
                best_eta = eta
                best_dest = (int(dest), arrival)
        if best_dest is None:
            return None
        dest, arrival = best_dest

        prod = float(self._prod.get(source, 0))
        # Rescued ships are kept on our side; the planet (and its production) is
        # written off.  Net heuristic favours saving a large garrison.
        rescued = float(garrison) / max(1e-6, float(self.cfg.ship_cost_norm))
        abandoned = prod * 0.1
        return MacroAction(
            type=MacroActionType.EVACUATE,
            target=int(dest),
            arrival_turn=int(arrival),
            ships=int(garrison),
            sources=((source, garrison),),
            heuristic=float(rescued - abandoned),
        )

    def _plan_consolidate(
        self, state: _TimedState, player: int, target: int
    ) -> Optional[MacroAction]:
        """Shift backline surplus onto a frontier owned planet."""
        if int(state.turn) >= int(self.horizon_turns):
            return None
        owners = state.owner_map()
        ships = state.ship_map()
        target = int(target)
        if owners.get(target, -1) != int(player):
            return None
        threatened = set(self._threatened_owned(state, player))

        best = None
        best_eta = math.inf
        for src, owner in owners.items():
            src = int(src)
            if int(owner) != int(player) or src == target or src in threatened:
                continue
            # Backline = not itself adjacent to a non-owned planet.
            adj_frontier = any(
                owners.get(int(nb), -1) != int(player)
                and self._reachable_from(src, int(nb)) is not None
                for nb in owners
            )
            if adj_frontier:
                continue
            eta = self._reachable_from(src, target)
            if eta is None:
                continue
            surplus = int(math.floor(int(ships.get(src, 0)) * (1.0 - float(self.cfg.source_reserve_frac))))
            if surplus < int(self.cfg.min_send):
                continue
            if eta < best_eta:
                best_eta = eta
                best = (src, surplus, int(state.turn) + max(1, int(math.ceil(eta))))
        if best is None:
            return None
        src, surplus, arrival = best
        return MacroAction(
            type=MacroActionType.CONSOLIDATE,
            target=int(target),
            arrival_turn=int(arrival),
            ships=int(surplus),
            sources=((int(src), int(surplus)),),
            heuristic=float(surplus) / max(1e-6, float(self.cfg.ship_cost_norm)) * 0.25,
        )

    def _frontier_owned(self, state: _TimedState, player: int) -> list[int]:
        """Owned planets adjacent (reachable) to a non-owned planet."""
        owners = state.owner_map()
        out: list[int] = []
        for pid, owner in owners.items():
            if int(owner) != int(player):
                continue
            if any(
                owners.get(int(nb), -1) != int(player)
                and self._reachable_from(int(pid), int(nb)) is not None
                for nb in owners
            ):
                out.append(int(pid))
        return out

    # ------------------------------------------------------------------ #
    # Macro action enumeration / application.
    # ------------------------------------------------------------------ #
    def _macro_actions(self, state: _TimedState) -> list[MacroAction]:
        player = int(state.player_to_move)
        owners = state.owner_map()

        captures: list[MacroAction] = []
        for target in self.candidates:
            if owners.get(int(target), -1) == player:
                continue
            plan = self._plan_capture(state, player, int(target))
            if plan is None:
                continue
            captures.append(MacroAction(
                type=MacroActionType.CAPTURE,
                target=int(plan.target),
                arrival_turn=int(plan.arrival_turn),
                ships=int(plan.ships),
                sources=tuple(plan.sources),
                heuristic=float(plan.heuristic),
            ))
        captures.sort(key=lambda a: a.heuristic, reverse=True)
        captures = captures[: max(0, int(self.macro_cfg.top_k_capture))]

        reinforces: list[MacroAction] = []
        for target, (enemy_ships, enemy_eta) in self._threatened_owned(state, player).items():
            action = self._plan_reinforce(state, player, int(target), enemy_ships, int(enemy_eta))
            if action is not None:
                reinforces.append(action)
        reinforces.sort(key=lambda a: a.heuristic, reverse=True)
        reinforces = reinforces[: max(0, int(self.macro_cfg.top_k_reinforce))]

        evacuations: list[MacroAction] = []
        if bool(self.macro_cfg.enable_evacuate):
            for source, (_enemy_ships, enemy_eta) in self._doomed_owned(state, player).items():
                action = self._plan_evacuate(state, player, int(source), int(enemy_eta))
                if action is not None:
                    evacuations.append(action)
            evacuations.sort(key=lambda a: a.heuristic, reverse=True)
            evacuations = evacuations[: max(0, int(self.macro_cfg.top_k_evac))]

        consolidations: list[MacroAction] = []
        if bool(self.macro_cfg.enable_consolidate):
            for target in self._frontier_owned(state, player):
                action = self._plan_consolidate(state, player, int(target))
                if action is not None:
                    consolidations.append(action)
            consolidations.sort(key=lambda a: a.heuristic, reverse=True)
            consolidations = consolidations[: max(0, int(self.macro_cfg.top_k_consolidate))]

        actions = captures + reinforces + evacuations + consolidations
        if bool(self.macro_cfg.include_pass) or not actions:
            actions.append(PASS_ACTION)
        return actions

    def _apply_launch(
        self,
        state: _TimedState,
        *,
        player: int,
        target: int,
        arrival_turn: int,
        ships: int,
        sources: tuple[tuple[int, int], ...],
    ) -> _TimedState:
        owners = state.owner_map()
        ship_map = state.ship_map()
        for src, send in sources:
            ship_map[int(src)] = max(0, int(ship_map.get(int(src), 0)) - int(send))
        pending = list(state.pending)
        pending.append(_CaptureEvent(
            arrival_turn=int(arrival_turn),
            owner=int(player),
            target=int(target),
            ships=int(ships),
        ))
        nxt, wrapped = self._next_player(int(player))
        out = self._state(
            turn=state.turn,
            player_to_move=nxt,
            owners=owners,
            ships=ship_map,
            pending=pending,
            reward=state.reward,
        )
        return self._advance_one_turn(out) if wrapped else out

    def _apply_macro(self, state: _TimedState, action: MacroAction) -> _TimedState:
        if action.type is MacroActionType.PASS:
            return self._pass(state)
        if action.type in (
            MacroActionType.CAPTURE,
            MacroActionType.REINFORCE,
            MacroActionType.EVACUATE,
            MacroActionType.CONSOLIDATE,
        ):
            if not action.sources:
                return self._pass(state)
            return self._apply_launch(
                state,
                player=int(state.player_to_move),
                target=int(action.target),
                arrival_turn=int(action.arrival_turn),
                ships=int(action.ships),
                sources=action.sources,
            )
        return self._pass(state)

    # ------------------------------------------------------------------ #
    # Prior handling (uniform default; learned provider for AlphaGo).
    # ------------------------------------------------------------------ #
    def _macro_features(self, state: _TimedState, action: MacroAction) -> dict[str, float]:
        owners = state.owner_map()
        ships = state.ship_map()
        owner = int(owners.get(int(action.target), -1))
        total_owned = sum(1 for o in owners.values() if int(o) == self.root_player)
        total_planets = max(1, len(owners))
        horizon = max(1.0, float(self.horizon_turns))
        return {
            "bias": 1.0,
            "is_capture": 1.0 if action.type is MacroActionType.CAPTURE else 0.0,
            "is_reinforce": 1.0 if action.type is MacroActionType.REINFORCE else 0.0,
            "is_evacuate": 1.0 if action.type is MacroActionType.EVACUATE else 0.0,
            "is_consolidate": 1.0 if action.type is MacroActionType.CONSOLIDATE else 0.0,
            "is_pass": 1.0 if action.type is MacroActionType.PASS else 0.0,
            "target_prod_norm": float(self._prod.get(int(action.target), 0)) / 10.0,
            "target_ships_norm": float(ships.get(int(action.target), 0)) / 100.0,
            "target_is_enemy": 1.0 if owner >= 0 and owner != int(state.player_to_move) else 0.0,
            "target_is_neutral": 1.0 if owner < 0 and action.type is MacroActionType.CAPTURE else 0.0,
            "arrival_norm": float(int(action.arrival_turn) - int(state.turn)) / horizon,
            "ships_sent_norm": float(action.ships) / 100.0,
            "value_norm": 0.0,
            "heuristic_norm": float(action.heuristic) / 1000.0,
            "turn_frac": float(self.current_turn + int(state.turn)) / max(1.0, float(self.max_turns)),
            "root_owned_frac": float(total_owned) / float(total_planets),
        }

    def _prior_logits(
        self, state: _TimedState, actions: list[MacroAction]
    ) -> dict[tuple[str, int], float]:
        if self.prior_provider is None or not actions:
            return {a.key(): 0.0 for a in actions}
        features = {a.key(): self._macro_features(state, a) for a in actions}
        request = MacroPriorRequest(
            root_player=int(self.root_player),
            player_to_move=int(state.player_to_move),
            relative_turn=int(state.turn),
            current_turn=int(self.current_turn),
            max_turns=int(self.max_turns),
            actions=tuple(actions),
            features_by_key=features,
        )
        logits = dict(self.prior_provider(request))
        return {a.key(): float(logits.get(a.key(), 0.0)) for a in actions}

    def _softmax_priors(
        self, logits: Mapping[tuple[str, int], float]
    ) -> dict[tuple[str, int], float]:
        if not logits:
            return {}
        temp = max(1e-6, float(self.macro_cfg.prior_temperature))
        scaled = {k: float(v) / temp for k, v in logits.items()}
        max_logit = max(scaled.values())
        exp_vals = {
            k: math.exp(max(-60.0, min(60.0, float(v) - max_logit)))
            for k, v in scaled.items()
        }
        total = sum(exp_vals.values())
        if total <= 0.0:
            return {k: 1.0 / len(exp_vals) for k in exp_vals}
        floor = max(0.0, float(self.macro_cfg.min_prior))
        priors = {k: max(floor, v / total) for k, v in exp_vals.items()}
        norm = sum(priors.values())
        return {k: v / norm for k, v in priors.items()}

    def _node_for_state(
        self,
        state: _TimedState,
        *,
        parent: Optional[_MacroNode] = None,
        action_key: Optional[tuple[str, int]] = None,
    ) -> _MacroNode:
        actions = self._macro_actions(state)
        logits = self._prior_logits(state, actions)
        priors = self._softmax_priors(logits)
        return _MacroNode(
            state=state,
            untried=list(actions),
            priors=priors,
            actions={a.key(): a for a in actions},
            parent=parent,
            action_key=action_key,
        )

    def _expand(self, node: _MacroNode) -> _MacroNode:
        action = node.untried.pop(0)
        child_state = self._apply_macro(node.state, action)
        child = self._node_for_state(child_state, parent=node, action_key=action.key())
        node.children[action.key()] = child
        return child

    def _apply_root_noise(self, root: _MacroNode) -> None:
        """Mix Dirichlet noise into the root priors (AlphaZero exploration)."""
        alpha = float(self.macro_cfg.root_dirichlet_alpha)
        if alpha <= 0.0 or len(root.priors) <= 1:
            return
        frac = float(self.macro_cfg.root_noise_frac)
        keys = list(root.priors)
        gammas = [self.rng.gammavariate(alpha, 1.0) for _ in keys]
        total = sum(gammas) or 1.0
        for k, g in zip(keys, gammas):
            root.priors[k] = (1.0 - frac) * root.priors[k] + frac * (g / total)

    def _puct(self, parent: _MacroNode, child: _MacroNode) -> float:
        q = child.q() if child.visits > 0 else 0.0
        prior = float(parent.priors.get(child.action_key, float(self.macro_cfg.min_prior)))
        explore = (
            float(self.macro_cfg.c_puct)
            * prior
            * math.sqrt(max(1.0, float(parent.visits)))
            / (1.0 + float(child.visits))
        )
        return float(q + explore)

    # ------------------------------------------------------------------ #
    # Rollout + hybrid leaf evaluation.
    # ------------------------------------------------------------------ #
    def _rollout(self, state: _TimedState) -> float:
        st = state
        guard = 0
        limit = 4 * max(1, len(self.players)) * max(1, int(self.horizon_turns))
        while int(st.turn) < int(self.horizon_turns) and guard < limit:
            guard += 1
            actions = self._macro_actions(st)
            non_pass = [a for a in actions if a.type is not MacroActionType.PASS]
            if non_pass and (
                not self.cfg.greedy_rollout
                or self.rng.random() <= float(self.cfg.rollout_randomness)
            ):
                action = self.rng.choice(non_pass)
            elif non_pass and self.cfg.greedy_rollout:
                action = max(non_pass, key=lambda a: a.heuristic)
            else:
                action = PASS_ACTION
            st = self._apply_macro(st, action)
        if int(st.turn) < int(self.horizon_turns):
            st = self._advance_to_horizon(st)
        return self._score(st)

    def _leaf_value(self, state: _TimedState) -> float:
        blend = float(self.macro_cfg.value_blend)
        if self.leaf_value_fn is not None and blend > 0.0:
            nn_v = float(self.leaf_value_fn(state, int(self.root_player)))
            if blend >= 1.0:
                return nn_v
            return (1.0 - blend) * self._rollout(state) + blend * nn_v
        return self._rollout(state)

    # ------------------------------------------------------------------ #
    # Public search.
    # ------------------------------------------------------------------ #
    def run(self) -> MacroSearchResult:
        """Run the macro MCTS and return root visit / Q statistics."""
        root = self._node_for_state(self._initial_state)
        empty = MacroSearchResult(
            baseline=float(self._baseline),
            root_value=float(self._baseline),
            visits={}, q_values={}, actions={}, priors={},
        )
        if int(self.cfg.simulations) <= 0:
            return empty
        # A lone forced Pass carries no decision signal.
        if not root.untried or (
            len(root.untried) == 1 and root.untried[0].type is MacroActionType.PASS
        ):
            return empty

        self._apply_root_noise(root)

        for _ in range(int(self.cfg.simulations)):
            node = root
            while not node.untried and node.children:
                node = max(node.children.values(), key=lambda c: self._puct(node, c))
            if node.untried:
                node = self._expand(node)
            value = self._leaf_value(node.state)
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent

        visits = {k: int(c.visits) for k, c in root.children.items()}
        q_values = {k: float(c.q()) for k, c in root.children.items()}
        return MacroSearchResult(
            baseline=float(self._baseline),
            root_value=float(root.q()),
            visits=visits,
            q_values=q_values,
            actions=dict(root.actions),
            priors=dict(root.priors),
        )

    def search(self) -> dict[int, float]:
        """Capture priority bonus for the scripted allocator (timed-MCTS API)."""
        if float(self.macro_cfg.bonus_weight) == 0.0:
            return {}
        result = self.run()
        return result.capture_bonus(
            weight=float(self.macro_cfg.bonus_weight),
            min_bonus=float(self.macro_cfg.min_bonus),
        )


class OrbitMacroMCTSAgent(OrbitScriptedAgent):
    """Scripted agent whose attack ranking is steered by macro MCTS.

    This is the safe Phase-2 integration from the plan: the scripted
    evacuate/defend/attack/support phases stay intact and the macro search only
    contributes a per-target capture priority bonus.  Reinforce decisions still
    influence the search value (they change which captures look good), while the
    scripted defense phase executes the actual rescues.
    """

    def __init__(
        self,
        cfg: OrbitScriptedConfig | None = None,
        macro_cfg: MacroMCTSConfig | None = None,
        prior_provider: Optional[MacroPriorProvider] = None,
        leaf_value_fn: Optional[LeafValueFn] = None,
    ):
        if cfg is None:
            cfg = OrbitScriptedConfig(
                attack_allocator="lp",
                reinforcement_allocator="lp",
            )
        super().__init__(cfg=cfg)
        self.macro_cfg = macro_cfg or MacroMCTSConfig()
        self.prior_provider = prior_provider
        self.leaf_value_fn = leaf_value_fn

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
        del plan, defense_states, omega
        if int(self.macro_cfg.timed.simulations) <= 0 or float(self.macro_cfg.bonus_weight) == 0.0:
            return None
        mcts = MacroTimedMCTS(
            og,
            self.macro_cfg,
            root_player=int(og.player),
            current_turn=int(current_turn),
            max_turns=int(max_turns),
            prior_provider=self.prior_provider,
            leaf_value_fn=self.leaf_value_fn,
        )
        bonus = mcts.search()
        return bonus or None


__all__ = [
    "MacroActionType",
    "MacroAction",
    "PASS_ACTION",
    "MACRO_FEATURE_NAMES",
    "MacroPriorRequest",
    "MacroPriorProvider",
    "LeafValueFn",
    "MacroMCTSConfig",
    "MacroSearchResult",
    "MacroTimedMCTS",
    "OrbitMacroMCTSAgent",
]
