"""Fleet-specific outcome tracking for delayed reward assignment."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from feature_extraction import Fleet, Planet, parse_fleets, parse_planets, fleet_speed


@dataclass
class FleetLaunch:
    launch_key: str
    buffer_idx: int
    launch_turn: int
    player: int
    src_planet_id: int
    tgt_planet_id: int
    ships_sent: int
    angle: float
    est_arrival_turn: float
    fleet_id: Optional[int] = None
    launch_tgt_owner: int = -2  # -1 neutral, player_id otherwise; -2 = unknown
    launch_tgt_ships: int = 0
    last_tgt_owner: int = -2
    last_tgt_ships: int = 0


@dataclass
class FleetOutcome:
    captured_planet: bool        # ownership flipped to me on arrival
    killed_enemy_ships: bool     # damaged enemy garrison without capture
    destroyed_enroute: bool      # target despawned / fleet lost
    self_reinforce: bool         # sent to a planet I already owned
    failed_neutral_attack: bool = False
    captured_from_enemy: bool = False
    planet_production: float = 0.0
    planet_radius: float = 0.0
    arrival_turn: int = 0
    enemy_ships_killed: float = 0.0
    own_waste: float = 0.0


class FleetTracker:
    def __init__(self):
        self._pending: dict[str, FleetLaunch] = {}
        self._resolved: dict[int, deque] = {}
        self._fleet_sig_map: dict[tuple[int, int, int, float], str] = {}

    def register_launch(self, *, buffer_idx: int, launch_turn: int, player: int,
                        src_planet_id: int, tgt_planet_id: int, ships_sent: int,
                        angle: float, est_distance: float,
                        launch_tgt_owner: int = -2, launch_tgt_ships: int = 0) -> str:
        key = (
            f"p{player}_t{launch_turn}_src{src_planet_id}_tgt{tgt_planet_id}"
            f"_s{ships_sent}_a{angle:.4f}_idx{buffer_idx}"
        )
        eta = est_distance / max(fleet_speed(ships_sent), 1e-6)
        fl = FleetLaunch(key, buffer_idx, launch_turn, player, src_planet_id,
                         tgt_planet_id, ships_sent, angle, launch_turn + eta,
                         launch_tgt_owner=launch_tgt_owner,
                         launch_tgt_ships=launch_tgt_ships)
        self._pending[key] = fl
        sig = (player, src_planet_id, ships_sent, angle)
        self._fleet_sig_map[sig] = key
        return key

    def observe_turn(self, turn: int, env_obs, player: int) -> None:
        planets = {p.id: p for p in parse_planets(env_obs)}
        fleets = parse_fleets(env_obs)
        current_by_id = {}
        current_sigs = {}
        for f in fleets:
            if f.owner == player:
                current_by_id[f.id] = f
                sig = (f.owner, f.from_planet_id, f.ships, f.angle)
                current_sigs.setdefault(sig, []).append(f)
        used_fleet_ids = set()
        to_resolve = []
        for key, fl in list(self._pending.items()):
            if fl.player != player:
                continue
            current_fleet = current_by_id.get(fl.fleet_id) if fl.fleet_id is not None else None
            if current_fleet is not None:
                used_fleet_ids.add(current_fleet.id)
            if current_fleet is None and fl.fleet_id is None:
                sig = (fl.player, fl.src_planet_id, fl.ships_sent, fl.angle)
                candidates = [f for f in current_sigs.get(sig, []) if f.id not in used_fleet_ids]
                if candidates:
                    current_fleet = min(candidates, key=lambda f: f.id)
                    fl.fleet_id = int(current_fleet.id)
                    used_fleet_ids.add(current_fleet.id)
                    self._fleet_sig_map.pop(sig, None)
            if current_fleet is not None:
                tgt = planets.get(fl.tgt_planet_id)
                if tgt is not None:
                    fl.last_tgt_owner = int(tgt.owner)
                    fl.last_tgt_ships = int(tgt.ships)
                continue
            outcome = self._resolve(fl, planets, turn)
            self._resolved.setdefault(player, deque()).append((fl.buffer_idx, outcome))
            to_resolve.append(key)
        for key in to_resolve:
            fl = self._pending.pop(key)
            sig = (fl.player, fl.src_planet_id, fl.ships_sent, fl.angle)
            self._fleet_sig_map.pop(sig, None)

    def pop_resolved(self, player: int):
        q = self._resolved.get(player)
        if q:
            while q:
                yield q.popleft()

    def _resolve(self, fl: FleetLaunch, planets: dict[int, Planet], turn: int) -> FleetOutcome:
        tgt = planets.get(fl.tgt_planet_id)
        if tgt is None:
            # Target despawned (e.g. comet) or fleet lost mid-flight.
            return FleetOutcome(False, False, True, False, own_waste=fl.ships_sent)
        # Capture iff ownership flipped FROM not-mine TO mine.
        prev_owner = fl.last_tgt_owner if fl.last_tgt_owner != -2 else fl.launch_tgt_owner
        prev_ships = fl.last_tgt_ships if fl.last_tgt_owner != -2 else fl.launch_tgt_ships
        was_mine = (prev_owner == fl.player)
        is_mine_now = (tgt.owner == fl.player)
        if was_mine and is_mine_now:
            # Reinforced own planet: no capture, no kill. This is the
            # "self-pinging" exploit we explicitly do NOT reward.
            return FleetOutcome(False, False, False, True, own_waste=fl.ships_sent)
        if (not was_mine) and is_mine_now:
            enemy_killed = prev_ships if prev_owner not in (-1, fl.player) else 0
            own_waste = max(0, fl.ships_sent - int(tgt.ships))
            return FleetOutcome(
                True,
                False,
                False,
                False,
                captured_from_enemy=prev_owner not in (-1, fl.player),
                planet_production=tgt.production,
                planet_radius=tgt.radius,
                arrival_turn=turn,
                enemy_ships_killed=enemy_killed,
                own_waste=own_waste,
            )
        # Arrived but didn't capture: we hit defended/neutral and bounced.
        killed = min(fl.ships_sent, max(0, prev_ships - int(tgt.ships)))
        if prev_owner == -1:
            return FleetOutcome(False, False, False, False,
                                failed_neutral_attack=True,
                                own_waste=fl.ships_sent)
        return FleetOutcome(False, True, False, False,
                            enemy_ships_killed=killed,
                            own_waste=fl.ships_sent)


def compute_fleet_reward(outcome: FleetOutcome, *, capture_weight: float = 0.05,
                         kill_weight: float = 0.001,
                         waste_weight: float = 0.0015,
                         enemy_capture_radius_weight: float = 0.02,
                         max_turns: int = 500,
                         time_decay_floor: float = 0.0) -> float:
    """Per-fleet credit assigned at arrival.

    Captures generate positive reward. Failed neutral attacks and destroyed
    fleets get a direct waste penalty on the original launch transition.
    Enemy garrison damage is rewarded, but netted against wasted own ships.

    capture_weight : reward per unit production for a real capture
                     (target was NOT mine at launch, IS mine on arrival).
    kill_weight    : reward per enemy ship destroyed when bouncing off
                     a defended planet.
    """
    if outcome.captured_planet:
        reward = capture_weight * outcome.planet_production
        if outcome.captured_from_enemy:
            raw_factor = max(0.0, (max_turns - outcome.arrival_turn) / max(1, max_turns))
            time_factor = max(time_decay_floor, raw_factor)
            reward += enemy_capture_radius_weight * outcome.planet_radius * time_factor
        return reward
    waste_penalty = waste_weight * outcome.own_waste
    if outcome.killed_enemy_ships:
        return kill_weight * outcome.enemy_ships_killed - waste_penalty
    if outcome.failed_neutral_attack or outcome.destroyed_enroute:
        return -waste_penalty
    return 0.0


def compute_production_simple_fleet_combat_reward(outcome: FleetOutcome) -> float:
    if outcome.self_reinforce:
        return 0.0
    reward = -float(outcome.own_waste)
    if outcome.captured_from_enemy or outcome.killed_enemy_ships:
        reward += float(outcome.enemy_ships_killed)
    return reward
