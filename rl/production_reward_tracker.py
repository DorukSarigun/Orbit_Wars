from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from feature_extraction import parse_planets


@dataclass
class ProductionCaptureLaunch:
    buffer_idx: int
    launch_turn: int
    player: int
    target_planet_id: int
    ships_sent: int
    est_arrival_turn: float
    launch_target_owner: int


@dataclass
class ProductionCaptureOutcome:
    ships_sent: int
    total_ships_sent: int
    planet_production: float
    capture_turn: int
    max_turns: int
    captured_from_enemy: bool
    num_opponents: int


@dataclass
class ProductionDefenseMissLaunch:
    buffer_idx: int
    launch_turn: int
    player: int
    threatened_planet_id: int
    ships_sent: int
    est_defense_arrival_turn: float


@dataclass
class ProductionPlanetLossOutcome:
    ships_sent: int
    total_ships_sent: int
    planet_production: float
    loss_turn: int
    max_turns: int
    num_opponents: int


class ProductionRewardTracker:
    def __init__(self, *, max_turns: int = 500, num_agents: int = 2):
        self.max_turns = int(max_turns)
        self.num_opponents = max(1, int(num_agents) - 1)
        self._capture_pending: dict[tuple[int, int], list[ProductionCaptureLaunch]] = {}
        self._loss_pending: dict[tuple[int, int], list[ProductionDefenseMissLaunch]] = {}
        self._resolved_captures: dict[int, deque] = {}
        self._resolved_losses: dict[int, deque] = {}

    def register_capture_launch(self, *, buffer_idx: int, launch_turn: int, player: int,
                                target_planet_id: int, ships_sent: int,
                                est_arrival_turn: float, launch_target_owner: int) -> None:
        if ships_sent <= 0 or launch_target_owner == player:
            return
        key = (int(player), int(target_planet_id))
        self._capture_pending.setdefault(key, []).append(ProductionCaptureLaunch(
            buffer_idx=int(buffer_idx),
            launch_turn=int(launch_turn),
            player=int(player),
            target_planet_id=int(target_planet_id),
            ships_sent=int(ships_sent),
            est_arrival_turn=float(est_arrival_turn),
            launch_target_owner=int(launch_target_owner),
        ))

    def register_defense_miss_candidate(self, *, buffer_idx: int, launch_turn: int,
                                        player: int, threatened_planet_id: int,
                                        ships_sent: int,
                                        est_defense_arrival_turn: float) -> None:
        if ships_sent <= 0:
            return
        key = (int(player), int(threatened_planet_id))
        self._loss_pending.setdefault(key, []).append(ProductionDefenseMissLaunch(
            buffer_idx=int(buffer_idx),
            launch_turn=int(launch_turn),
            player=int(player),
            threatened_planet_id=int(threatened_planet_id),
            ships_sent=int(ships_sent),
            est_defense_arrival_turn=float(est_defense_arrival_turn),
        ))

    def observe_turn(self, turn: int, prev_obs, curr_obs, player: int) -> None:
        prev_planets = {p.id: p for p in parse_planets(prev_obs) if not p.is_comet}
        curr_planets = {p.id: p for p in parse_planets(curr_obs) if not p.is_comet}
        for planet_id, curr in curr_planets.items():
            prev = prev_planets.get(planet_id)
            prev_owner = prev.owner if prev is not None else -1
            if prev_owner == player or curr.owner != player:
                continue
            key = (int(player), int(planet_id))
            launches = [
                fl for fl in self._capture_pending.pop(key, [])
                if fl.est_arrival_turn <= turn + 1e-6 and fl.launch_turn <= turn
            ]
            total_ships = sum(max(0, fl.ships_sent) for fl in launches)
            if total_ships <= 0:
                continue
            captured_from_enemy = prev_owner not in (-1, player)
            for fl in launches:
                outcome = ProductionCaptureOutcome(
                    ships_sent=fl.ships_sent,
                    total_ships_sent=total_ships,
                    planet_production=float(curr.production),
                    capture_turn=int(turn),
                    max_turns=self.max_turns,
                    captured_from_enemy=bool(captured_from_enemy),
                    num_opponents=self.num_opponents,
                )
                self._resolved_captures.setdefault(player, deque()).append((fl.buffer_idx, outcome))
        self._expire_arrived_non_captures(turn, player, curr_planets)
        for planet_id, prev in prev_planets.items():
            if prev.owner != player:
                continue
            curr = curr_planets.get(planet_id)
            if curr is not None and curr.owner == player:
                continue
            key = (int(player), int(planet_id))
            candidates = [
                fl for fl in self._loss_pending.pop(key, [])
                if fl.est_defense_arrival_turn <= turn + 1e-6 and fl.launch_turn <= turn
            ]
            total_ships = sum(max(0, fl.ships_sent) for fl in candidates)
            if total_ships <= 0:
                continue
            for fl in candidates:
                outcome = ProductionPlanetLossOutcome(
                    ships_sent=fl.ships_sent,
                    total_ships_sent=total_ships,
                    planet_production=float(prev.production),
                    loss_turn=int(turn),
                    max_turns=self.max_turns,
                    num_opponents=self.num_opponents,
                )
                self._resolved_losses.setdefault(player, deque()).append((fl.buffer_idx, outcome))

    def _expire_arrived_non_captures(self, turn: int, player: int, curr_planets: dict) -> None:
        for key in list(self._capture_pending):
            key_player, planet_id = key
            if key_player != player:
                continue
            curr = curr_planets.get(planet_id)
            if curr is not None and curr.owner == player:
                self._capture_pending.pop(key, None)
                continue
            launches = [
                fl for fl in self._capture_pending[key]
                if fl.est_arrival_turn > turn + 1e-6
            ]
            if launches:
                self._capture_pending[key] = launches
            else:
                self._capture_pending.pop(key, None)

    def pop_capture_resolved(self, player: int):
        q = self._resolved_captures.get(player)
        if q:
            while q:
                yield q.popleft()

    def pop_loss_resolved(self, player: int):
        q = self._resolved_losses.get(player)
        if q:
            while q:
                yield q.popleft()


def opponent_count_multiplier(num_opponents: int) -> float:
    n = max(1, int(num_opponents))
    return (n + 1.0) / n


def remaining_steps(max_turns: int, turn: int) -> int:
    return max(0, int(max_turns) - int(turn))


def compute_production_capture_reward(outcome: ProductionCaptureOutcome) -> float:
    if outcome.total_ships_sent <= 0:
        return 0.0
    reward = float(outcome.planet_production) * remaining_steps(outcome.max_turns, outcome.capture_turn)
    if outcome.captured_from_enemy:
        reward *= opponent_count_multiplier(outcome.num_opponents)
    return reward * max(0.0, outcome.ships_sent / outcome.total_ships_sent)


def compute_production_planet_loss_penalty(outcome: ProductionPlanetLossOutcome) -> float:
    if outcome.total_ships_sent <= 0:
        return 0.0
    penalty = -float(outcome.planet_production) * remaining_steps(outcome.max_turns, outcome.loss_turn)
    penalty *= opponent_count_multiplier(outcome.num_opponents)
    return penalty * max(0.0, outcome.ships_sent / outcome.total_ships_sent)
