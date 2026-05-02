from __future__ import annotations

from dataclasses import dataclass

from feature_extraction import parse_fleets, parse_planets
from navigation import incoming_threats


@dataclass
class DefenseHold:
    buffer_idx: int
    player: int
    planet_id: int
    hold_turn: int
    ships_held: int


@dataclass
class DefenseOutcome:
    defended_planet: bool
    ships_absorbed: float = 0.0
    planet_production: float = 0.0


class DefenseTracker:
    def __init__(self, *, horizon: int = 15, arrival_window: float = 1.5):
        self.horizon = int(horizon)
        self.arrival_window = float(arrival_window)
        self._pending: dict[int, list[DefenseHold]] = {}
        self._prev_planets = {}
        self._prev_arriving_threats: dict[int, float] = {}
        self._resolved: list[tuple[int, DefenseOutcome]] = []

    def prime(self, env_obs, player: int) -> None:
        planets = parse_planets(env_obs)
        fleets = parse_fleets(env_obs)
        omega = env_obs.get("angular_velocity", 0.0) if isinstance(env_obs, dict) else 0.0
        self._prev_planets = {p.id: p for p in planets}
        threats: dict[int, float] = {}
        for row in incoming_threats(player, fleets, planets, omega):
            if float(row["eta"]) <= self.arrival_window:
                pid = int(row["target_id"])
                threats[pid] = threats.get(pid, 0.0) + float(row["ships"])
        self._prev_arriving_threats = threats

    def register_hold(self, *, buffer_idx: int, player: int, planet_id: int, hold_turn: int, ships_held: int) -> None:
        if self.horizon <= 0:
            return
        self._pending.setdefault(int(planet_id), []).append(
            DefenseHold(
                buffer_idx=int(buffer_idx),
                player=int(player),
                planet_id=int(planet_id),
                hold_turn=int(hold_turn),
                ships_held=int(ships_held),
            )
        )

    def cancel_planet(self, planet_id: int) -> None:
        self._pending.pop(int(planet_id), None)

    def observe_turn(self, turn: int, env_obs, player: int) -> None:
        curr_planets = {p.id: p for p in parse_planets(env_obs)}
        for planet_id, holds in list(self._pending.items()):
            holds = [h for h in holds if turn - h.hold_turn <= self.horizon]
            if not holds:
                self._pending.pop(planet_id, None)
                continue
            prev = self._prev_planets.get(planet_id)
            curr = curr_planets.get(planet_id)
            arriving = self._prev_arriving_threats.get(planet_id, 0.0)
            if prev is None or curr is None or prev.owner != player:
                self._pending.pop(planet_id, None)
                continue
            if curr.owner != player:
                self._pending.pop(planet_id, None)
                continue
            expected_ships = float(prev.ships + max(0, prev.production))
            absorbed = min(arriving, max(0.0, expected_ships - float(curr.ships)))
            if absorbed > 0.0:
                hold = max(holds, key=lambda h: h.hold_turn)
                self._resolved.append((
                    hold.buffer_idx,
                    DefenseOutcome(True, ships_absorbed=absorbed, planet_production=float(curr.production)),
                ))
                self._pending.pop(planet_id, None)
            else:
                self._pending[planet_id] = holds

    def pop_resolved(self):
        while self._resolved:
            yield self._resolved.pop(0)


def compute_defense_reward(outcome: DefenseOutcome, *, hold_weight: float = 0.001) -> float:
    if outcome.defended_planet:
        return hold_weight * outcome.ships_absorbed
    return 0.0
