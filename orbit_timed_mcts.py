"""Time-aware abstract capture MCTS for Orbit Wars.

The original :mod:`orbit_mcts` searches an instant-flip capture tree: choosing a
planet immediately changes its owner.  This module keeps that fast abstract
spirit but adds a real turn horizon:

* owned planets produce ships as time advances;
* capture actions consume source garrisons now and schedule an arrival event;
* arrivals resolve after their ETA and can flip ownership;
* terminal value maximises board metrics at ``horizon_turns``;
* capture income is valued from landing time until the actual game end.

It is still intentionally *not* a full env rollout.  The action space is a
macro "fund a capture wave to target T" action; exact launch headings and
multi-fleet routing remain the job of the scripted/hier allocator.  The public
output matches :class:`orbit_mcts.CaptureMCTS`: ``{planet_id: priority_bonus}``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from orbit_graph import OrbitGraph
from orbit_lookahead_agent import _candidate_target_ids


@dataclass(frozen=True)
class TimedMCTSConfig:
    """Hyperparameters for :class:`TimedCaptureMCTS`."""

    simulations: int = 96
    """Number of MCTS iterations per search."""
    horizon_turns: int = 10
    """Search horizon, in turns after the current decision."""
    exploration_c: float = 1.4
    """UCT exploration constant."""
    top_k_actions: int = 6
    """Branching cap per node, after heuristic action ordering."""
    eta_horizon: float = 60.0
    """Max ETA for a directed leg to be considered reachable."""
    bonus_weight: float = 1.0
    """Scale returned root improvements before they join attack ranking."""
    min_bonus: float = 1e-6
    """Drop tiny non-positive improvements."""
    capture_margin: float = 1.0
    """Extra ships above projected defender strength required for a wave."""
    source_reserve_frac: float = 0.25
    """Fraction of each source garrison kept home in the abstract planner."""
    min_send: int = 1
    """Minimum abstract ships to use from a source."""
    greedy_rollout: bool = True
    """Rollout policy picks the best heuristic capture unless it explores."""
    rollout_randomness: float = 0.10
    """Probability of a random legal capture during rollout."""
    gamma: float = 0.97
    """Per-turn discount for production income after a capture lands."""
    capture_income_weight: float = 1.0
    """Weight for discounted production income from successful captures."""
    denial_income_weight: float = 0.5
    """Extra reward when a capture is taken *from an enemy* (not a neutral): the
    captured world stops producing for the opponent, so a cheap enemy kill is
    worth more than an equivalent neutral grab.  0 restores neutral==enemy."""
    ship_cost_weight: float = 0.05
    """Penalty weight for ships spent by an abstract capture wave."""
    ship_cost_norm: float = 50.0
    """Normaliser for ship-cost penalty."""
    eta_cost_weight: float = 0.05
    """Penalty weight for late/slow capture routes in action ordering."""
    production_margin_weight: float = 1.0
    ship_margin_weight: float = 0.25
    planet_margin_weight: float = 0.10


@dataclass(frozen=True)
class _CaptureEvent:
    arrival_turn: int
    owner: int
    target: int
    ships: int


@dataclass(frozen=True)
class _TimedState:
    turn: int
    player_to_move: int
    owners: tuple[tuple[int, int], ...]
    ships: tuple[tuple[int, int], ...]
    pending: tuple[_CaptureEvent, ...] = ()
    reward: float = 0.0

    def owner_map(self) -> dict[int, int]:
        return dict(self.owners)

    def ship_map(self) -> dict[int, int]:
        return dict(self.ships)


@dataclass(frozen=True)
class _MacroAction:
    target: int
    arrival_turn: int
    ships: int
    sources: tuple[tuple[int, int], ...]
    heuristic: float


@dataclass
class _TimedNode:
    state: _TimedState
    untried: list[int] = field(default_factory=list)
    children: dict[int, "_TimedNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    parent: Optional["_TimedNode"] = None
    action: Optional[int] = None

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def uct(self, c: float) -> float:
        if self.parent is None or self.visits == 0:
            return float("inf")
        return self.q() + c * math.sqrt(math.log(max(1, self.parent.visits)) / self.visits)


class TimedCaptureMCTS:
    """Turn-horizon capture planner on top of a fixed :class:`OrbitGraph`."""

    def __init__(
        self,
        og: OrbitGraph,
        cfg: TimedMCTSConfig,
        *,
        root_player: int,
        current_turn: int = 0,
        max_turns: int = 500,
        rng: Optional[random.Random] = None,
    ):
        self.og = og
        self.cfg = cfg
        self.root_player = int(root_player)
        self.current_turn = max(0, int(current_turn))
        self.max_turns = max(self.current_turn, int(max_turns))
        self.horizon_turns = max(
            0,
            min(int(self.cfg.horizon_turns), self.max_turns - self.current_turn),
        )
        self.rng = rng or random.Random()
        self.candidates = _candidate_target_ids(og)
        self._non_comets = {
            int(p.id) for p in og.planets if not getattr(p, "is_comet", False)
        }
        self._prod = {
            int(p.id): int(p.production)
            for p in og.planets
            if int(p.id) in self._non_comets
        }
        self._owners0 = {
            int(p.id): int(p.owner)
            for p in og.planets
            if int(p.id) in self._non_comets
        }
        self._ships0 = {
            int(p.id): int(round(float(p.ships)))
            for p in og.planets
            if int(p.id) in self._non_comets
        }
        players = {self.root_player}
        players.update(o for o in self._owners0.values() if int(o) >= 0)
        self.players = tuple(sorted(int(p) for p in players))

        self._initial_state = _TimedState(
            turn=0,
            player_to_move=self.root_player,
            owners=tuple(sorted(self._owners0.items())),
            ships=tuple(sorted(self._ships0.items())),
            pending=self._initial_pending(),
            reward=0.0,
        )
        self._baseline = self._score(self._advance_to_horizon(self._initial_state))

    def _initial_pending(self) -> tuple[_CaptureEvent, ...]:
        events: list[_CaptureEvent] = []
        for _u, tgt, data in self.og.G.edges(data=True):
            if int(tgt) not in self._non_comets:
                continue
            for rec in data.get("fleets", []):
                if int(rec.owner) < 0:
                    continue
                eta = float(getattr(rec, "eta", math.inf))
                if not math.isfinite(eta):
                    continue
                events.append(_CaptureEvent(
                    arrival_turn=max(0, int(math.ceil(eta))),
                    owner=int(rec.owner),
                    target=int(tgt),
                    ships=max(0, int(getattr(rec.fleet, "ships", 0))),
                ))
        events.sort(key=lambda e: (e.arrival_turn, e.target, e.owner))
        return tuple(events)

    def _state(
        self,
        *,
        turn: int,
        player_to_move: int,
        owners: dict[int, int],
        ships: dict[int, int],
        pending: list[_CaptureEvent] | tuple[_CaptureEvent, ...],
        reward: float = 0.0,
    ) -> _TimedState:
        return _TimedState(
            turn=int(turn),
            player_to_move=int(player_to_move),
            owners=tuple(sorted((int(k), int(v)) for k, v in owners.items())),
            ships=tuple(sorted((int(k), max(0, int(v))) for k, v in ships.items())),
            pending=tuple(sorted(pending, key=lambda e: (e.arrival_turn, e.target, e.owner))),
            reward=float(reward),
        )

    def _discounted_hold_value(self, arrival_turn: int) -> float:
        """Discounted productive turns from landing until the game ends.

        ``arrival_turn`` is relative to this planning call.  The short MCTS
        ``horizon_turns`` still controls how far the abstract board is
        simulated; it no longer truncates a successful capture's economic
        value.
        """
        arrival = max(0, int(arrival_turn))
        absolute_arrival = self.current_turn + arrival
        hold = max(0, self.max_turns - absolute_arrival)
        if hold <= 0:
            return 0.0
        gamma = float(self.cfg.gamma)
        if abs(gamma - 1.0) < 1e-9:
            return float(hold)
        gamma = max(0.0, min(gamma, 0.999999))
        return float((gamma ** arrival) * (1.0 - gamma ** hold) / (1.0 - gamma))

    def _capture_income(self, target: int, arrival_turn: int) -> float:
        return float(self._prod.get(int(target), 0)) * self._discounted_hold_value(arrival_turn)

    def _next_player(self, player: int) -> tuple[int, bool]:
        if len(self.players) <= 1:
            return int(player), True
        idx = self.players.index(int(player)) if int(player) in self.players else 0
        nxt_idx = (idx + 1) % len(self.players)
        return int(self.players[nxt_idx]), bool(nxt_idx == 0)

    def _advance_one_turn(self, state: _TimedState) -> _TimedState:
        owners = state.owner_map()
        ships = state.ship_map()
        turn = int(state.turn) + 1
        reward = float(state.reward)

        for pid, owner in owners.items():
            if int(owner) >= 0:
                ships[pid] = int(ships.get(pid, 0)) + int(self._prod.get(pid, 0))

        remaining: list[_CaptureEvent] = []
        due = sorted(
            [e for e in state.pending if int(e.arrival_turn) <= turn],
            key=lambda e: (e.arrival_turn, e.target, e.owner),
        )
        for e in due:
            if int(e.target) not in owners or int(e.ships) <= 0:
                continue
            cur_owner = int(owners.get(e.target, -1))
            cur_ships = int(ships.get(e.target, 0))
            if cur_owner == int(e.owner):
                ships[e.target] = cur_ships + int(e.ships)
            elif int(e.ships) > cur_ships:
                owners[e.target] = int(e.owner)
                ships[e.target] = int(e.ships) - cur_ships
                income = self._capture_income(int(e.target), turn)
                denial = float(self.cfg.denial_income_weight) * income
                if int(e.owner) == self.root_player:
                    # Root captures; bonus denial when taken from an enemy world.
                    bonus = denial if cur_owner >= 0 else 0.0
                    reward += float(self.cfg.capture_income_weight) * income + bonus
                elif int(e.owner) >= 0:
                    # Enemy captures; extra penalty when it strips a root world.
                    penalty = denial if cur_owner == self.root_player else 0.0
                    reward -= float(self.cfg.capture_income_weight) * income + penalty
            else:
                ships[e.target] = cur_ships - int(e.ships)
        remaining.extend(e for e in state.pending if int(e.arrival_turn) > turn)

        return self._state(
            turn=turn,
            player_to_move=state.player_to_move,
            owners=owners,
            ships=ships,
            pending=remaining,
            reward=reward,
        )

    def _advance_to_horizon(self, state: _TimedState) -> _TimedState:
        out = state
        while int(out.turn) < int(self.horizon_turns):
            out = self._advance_one_turn(out)
        return out

    def _pass(self, state: _TimedState) -> _TimedState:
        nxt, wrapped = self._next_player(state.player_to_move)
        out = self._state(
            turn=state.turn,
            player_to_move=nxt,
            owners=state.owner_map(),
            ships=state.ship_map(),
            pending=state.pending,
            reward=state.reward,
        )
        return self._advance_one_turn(out) if wrapped else out

    def _capture_need(self, owners: dict[int, int], ships: dict[int, int], target: int, eta: float) -> int:
        owner = int(owners.get(int(target), -1))
        reinforce = 0.0 if owner == -1 else float(self._prod.get(int(target), 0)) * max(0.0, eta)
        need = float(ships.get(int(target), 0)) + reinforce + float(self.cfg.capture_margin)
        return max(1, int(math.ceil(need)))

    def _plan_capture(self, state: _TimedState, player: int, target: int) -> Optional[_MacroAction]:
        if int(state.turn) >= int(self.horizon_turns):
            return None
        owners = state.owner_map()
        ships = state.ship_map()
        target = int(target)
        player = int(player)
        if owners.get(target, -1) == player or target not in self._non_comets:
            return None
        if any(e.owner == player and e.target == target for e in state.pending):
            return None

        candidates: list[tuple[float, int, int]] = []
        for src, owner in owners.items():
            if int(owner) != player or int(src) == target:
                continue
            if not self.og.G.has_edge(int(src), target):
                continue
            data = self.og.G.edges[int(src), target]
            if not data.get("reachable", False):
                continue
            eta = float(data.get("eta", math.inf))
            if not math.isfinite(eta) or eta > float(self.cfg.eta_horizon):
                continue
            source_ships = int(ships.get(int(src), 0))
            surplus = int(math.floor(source_ships * (1.0 - float(self.cfg.source_reserve_frac))))
            if surplus < int(self.cfg.min_send):
                continue
            candidates.append((eta, int(src), surplus))
        candidates.sort(key=lambda x: x[0])

        chosen: list[tuple[float, int, int]] = []
        total_surplus = 0
        need = math.inf
        eta_max = math.inf
        for eta, src, surplus in candidates:
            chosen.append((eta, src, surplus))
            total_surplus += int(surplus)
            eta_max = float(eta)
            need = self._capture_need(owners, ships, target, eta_max)
            if total_surplus >= need:
                break
        if not chosen or total_surplus < need:
            return None

        arrival_turn = int(state.turn) + max(1, int(math.ceil(float(eta_max))))
        if arrival_turn > int(self.horizon_turns):
            return None

        remaining = int(need)
        sources: list[tuple[int, int]] = []
        for _eta, src, surplus in chosen:
            if remaining <= 0:
                break
            send = min(int(surplus), remaining)
            if send < int(self.cfg.min_send):
                continue
            sources.append((int(src), int(send)))
            remaining -= int(send)
        if remaining > 0 or not sources:
            return None

        prod = float(self._prod.get(target, 0))
        owner_bonus = 0.5 if owners.get(target, -1) >= 0 else 0.0
        income = self._capture_income(target, arrival_turn)
        ship_penalty = float(self.cfg.ship_cost_weight) * (
            float(need) / max(1e-6, float(self.cfg.ship_cost_norm))
        )
        eta_penalty = float(self.cfg.eta_cost_weight) * (
            float(eta_max) / max(1.0, float(self.horizon_turns))
        )
        heuristic = (
            float(self.cfg.capture_income_weight) * income
            + owner_bonus * prod
            - ship_penalty
            - eta_penalty
        )
        return _MacroAction(
            target=target,
            arrival_turn=arrival_turn,
            ships=sum(s for _src, s in sources),
            sources=tuple(sources),
            heuristic=float(heuristic),
        )

    def _rank_actions(self, state: _TimedState) -> list[int]:
        player = int(state.player_to_move)
        actions: list[_MacroAction] = []
        owners = state.owner_map()
        for target in self.candidates:
            if owners.get(int(target), -1) == player:
                continue
            action = self._plan_capture(state, player, int(target))
            if action is not None:
                actions.append(action)
        actions.sort(key=lambda a: a.heuristic, reverse=True)
        k = max(1, int(self.cfg.top_k_actions))
        return [int(a.target) for a in actions[:k]]

    def _apply_action(self, state: _TimedState, target: int) -> _TimedState:
        player = int(state.player_to_move)
        action = self._plan_capture(state, player, int(target))
        if action is None:
            return self._pass(state)

        owners = state.owner_map()
        ships = state.ship_map()
        for src, send in action.sources:
            ships[int(src)] = max(0, int(ships.get(int(src), 0)) - int(send))
        pending = list(state.pending)
        pending.append(_CaptureEvent(
            arrival_turn=int(action.arrival_turn),
            owner=player,
            target=int(action.target),
            ships=int(action.ships),
        ))

        nxt, wrapped = self._next_player(player)
        out = self._state(
            turn=state.turn,
            player_to_move=nxt,
            owners=owners,
            ships=ships,
            pending=pending,
            reward=state.reward,
        )
        return self._advance_one_turn(out) if wrapped else out

    def _metric_margin(self, values: dict[int, float], total: float) -> float:
        """Root-vs-best-opponent share over the whole board.

        ``total`` intentionally includes neutral planets/ships/production.  A
        no-action state with many rich neutrals should not look "perfect" just
        because no opponent owns those metrics yet; capturing them must improve
        the root player's share.
        """
        total = max(0.0, float(total))
        if total <= 0.0:
            return 0.0
        mine = values.get(self.root_player, 0.0)
        best_other = max((v for p, v in values.items() if p != self.root_player), default=0.0)
        return float((mine - best_other) / total)

    def _score(self, state: _TimedState) -> float:
        owners = state.owner_map()
        ships = state.ship_map()

        prod_by_owner: dict[int, float] = {}
        ships_by_owner: dict[int, float] = {}
        planets_by_owner: dict[int, float] = {}
        total_prod = 0.0
        total_ships = 0.0
        total_planets = 0.0
        for pid, owner in owners.items():
            total_prod += float(self._prod.get(pid, 0))
            total_ships += float(ships.get(pid, 0))
            total_planets += 1.0
            if int(owner) < 0:
                continue
            prod_by_owner[owner] = prod_by_owner.get(owner, 0.0) + float(self._prod.get(pid, 0))
            ships_by_owner[owner] = ships_by_owner.get(owner, 0.0) + float(ships.get(pid, 0))
            planets_by_owner[owner] = planets_by_owner.get(owner, 0.0) + 1.0

        return float(state.reward) + (
            float(self.cfg.production_margin_weight) * self._metric_margin(prod_by_owner, total_prod)
            + float(self.cfg.ship_margin_weight) * self._metric_margin(ships_by_owner, total_ships)
            + float(self.cfg.planet_margin_weight) * self._metric_margin(planets_by_owner, total_planets)
        )

    def _rollout(self, state: _TimedState) -> float:
        st = state
        guard = 0
        while int(st.turn) < int(self.horizon_turns) and guard < 4 * max(1, len(self.players)) * max(1, int(self.horizon_turns)):
            guard += 1
            legal = self._rank_actions(st)
            if legal:
                if self.cfg.greedy_rollout and self.rng.random() > float(self.cfg.rollout_randomness):
                    target = legal[0]
                else:
                    target = self.rng.choice(legal)
                st = self._apply_action(st, int(target))
            else:
                st = self._pass(st)
        if int(st.turn) < int(self.horizon_turns):
            st = self._advance_to_horizon(st)
        return self._score(st)

    def _expand(self, node: _TimedNode) -> _TimedNode:
        target = int(node.untried.pop(0))
        child_state = self._apply_action(node.state, target)
        child = _TimedNode(
            state=child_state,
            untried=self._rank_actions(child_state),
            parent=node,
            action=target,
        )
        node.children[target] = child
        return child

    def search(self) -> dict[int, float]:
        """Run timed MCTS; return root-target metric improvements."""
        if int(self.cfg.simulations) <= 0 or float(self.cfg.bonus_weight) == 0.0:
            return {}

        root_actions = self._rank_actions(self._initial_state)
        if not root_actions:
            return {}

        root = _TimedNode(
            state=self._initial_state,
            untried=list(root_actions),
        )

        for _ in range(int(self.cfg.simulations)):
            node = root
            while node.untried == [] and node.children:
                node = max(
                    node.children.values(),
                    key=lambda n: n.uct(float(self.cfg.exploration_c)),
                )
            if node.untried:
                node = self._expand(node)

            value = self._rollout(node.state)
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent

        out: dict[int, float] = {}
        for target, child in root.children.items():
            if child.visits <= 0:
                continue
            improvement = child.q() - float(self._baseline)
            val = float(self.cfg.bonus_weight) * improvement
            if val > float(self.cfg.min_bonus):
                out[int(target)] = float(val)
        return out


__all__ = ["TimedMCTSConfig", "TimedCaptureMCTS"]
