"""Credit assignment for losing a planet after launching from it under threat."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from feature_extraction import parse_planets


@dataclass
class PlanetLossLaunch:
    buffer_idx: int
    launch_turn: int
    player: int
    planet_id: int
    ships_sent: int


@dataclass
class PlanetLossOutcome:
    ships_sent: int
    total_ships_sent: int
    production_lost: float


class PlanetLossTracker:
    def __init__(self, *, horizon: int = 30):
        self.horizon = int(horizon)
        self._pending: dict[tuple[int, int], list[PlanetLossLaunch]] = {}
        self._resolved: dict[int, deque] = {}

    def register_launch(self, *, buffer_idx: int, launch_turn: int, player: int,
                        planet_id: int, ships_sent: int,
                        source_had_enemy_incoming: bool) -> None:
        if not source_had_enemy_incoming or ships_sent <= 0:
            return
        key = (int(player), int(planet_id))
        self._pending.setdefault(key, []).append(PlanetLossLaunch(
            buffer_idx=int(buffer_idx),
            launch_turn=int(launch_turn),
            player=int(player),
            planet_id=int(planet_id),
            ships_sent=int(ships_sent),
        ))

    def observe_turn(self, turn: int, prev_obs, curr_obs, player: int) -> None:
        self._expire(turn)
        prev_planets = {p.id: p for p in parse_planets(prev_obs)}
        curr_planets = {p.id: p for p in parse_planets(curr_obs)}
        for planet_id, prev in prev_planets.items():
            if prev.owner != player:
                continue
            curr = curr_planets.get(planet_id)
            if curr is not None and curr.owner == player:
                continue
            key = (int(player), int(planet_id))
            launches = self._pending.pop(key, [])
            total_ships = sum(max(0, fl.ships_sent) for fl in launches)
            if total_ships <= 0:
                continue
            for fl in launches:
                outcome = PlanetLossOutcome(
                    ships_sent=fl.ships_sent,
                    total_ships_sent=total_ships,
                    production_lost=float(prev.production),
                )
                self._resolved.setdefault(player, deque()).append((fl.buffer_idx, outcome))

    def pop_resolved(self, player: int):
        q = self._resolved.get(player)
        if q:
            while q:
                yield q.popleft()

    def _expire(self, turn: int) -> None:
        if self.horizon <= 0:
            return
        for key in list(self._pending):
            launches = [
                fl for fl in self._pending[key]
                if turn - fl.launch_turn <= self.horizon
            ]
            if launches:
                self._pending[key] = launches
            else:
                self._pending.pop(key, None)


def compute_planet_loss_launch_penalty(outcome: PlanetLossOutcome, *,
                                       loss_weight: float = 0.05) -> float:
    if outcome.total_ships_sent <= 0:
        return 0.0
    share = max(0.0, outcome.ships_sent / outcome.total_ships_sent)
    return -float(loss_weight) * outcome.production_lost * share
