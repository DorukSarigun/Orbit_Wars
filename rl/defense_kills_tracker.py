"""Defense Kills Tracker — credit defending launches when enemy attacks fail.

Mechanism (per-arrival incremental resolution)
----------------------------------------------
1. Each turn, every enemy fleet in flight whose predicted target is one of
   MY planets tags that planet as 'under attack'. An *attack episode* is
   opened on first sighting and stays active as long as at least one enemy
   fleet is still en route.
2. While an episode is active, every launch I make TARGETING that planet
   is registered as a defender (with its est_arrival_turn).
3. As enemy fleets arrive during the episode:
     - Planet still mine after combat: that fleet's ships are credited
       IMMEDIATELY to all eligible defenders (defenders whose
       est_arrival_turn <= current turn). Reward is queued via
       `_resolved` for this turn.
     - Planet flipped to the enemy: the episode is dropped immediately
       (defense failed → no reward; the planet-loss tracker handles the
       loss penalty separately).
4. The episode stays in `_active` while any enemy fleet remains en route
   so that incoming defenders can still be registered. Once no more enemy
   fleets target this planet, the episode is dropped (no extra reward).

Reward formula (per arrival event)::

    share  = ships_sent / sum(eligible_defender_ships)
    reward = arrived_enemy_ships * share
             * defense_kills_weight * reward_scale

Eligible defenders at arrival turn t = those whose est_arrival_turn <= t
(i.e. they had already landed by the time the enemy fleet arrived).
Defenders that arrive after a given combat earn nothing for that combat
but remain eligible for any later combats in the same episode.

Why per-arrival (not episode-end)
---------------------------------
- In high-launch-rate rollouts, attack waves overlap continuously and an
  episode may never become "fully resolved" within an episode horizon,
  starving the reward channel.
- Crediting at each arrival ties the reward to the actual combat moment,
  giving denser learning signal.
- Kills are bounded and proportional to actual enemy losses (not arbitrary
  ships_sent like the legacy reinforcement reward).
- Solves the "blue team passes fleets back and forth" exploit: launches
  to safe planets earn zero credit (no enemy → no episode → no reward).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from feature_extraction import parse_planets, parse_fleets
from navigation import predict_all_fleet_targets
from .production_reward_tracker import production_reward_value
from .rewards import curved_time_factor


@dataclass
class _DefenderLaunch:
    buffer_idx: int
    launch_turn: int
    ships_sent: int
    est_arrival_turn: int


@dataclass
class _AttackEpisode:
    planet_id: int
    started_turn: int
    defenders: list[_DefenderLaunch] = field(default_factory=list)
    enemy_killed_total: float = 0.0
    last_combat_turn: int = -1


class DefenseKillsTracker:
    """Per-(player, planet) under-attack state machine + defender registry."""

    def __init__(self, *, max_turns: int = 500, angular_velocity: float = 0.0,
                 production_reward_beta: float = 0.0,
                 production_reward_reference: float = 3.0,
                 capture_time_curvature: float = 1.0) -> None:
        self.max_turns = int(max_turns)
        self.angular_velocity = float(angular_velocity)
        # Production-symmetric reward shaping (matches capture reward
        # formula so a successful defense at high-prod planet is comparable
        # in magnitude to a high-prod capture).
        self.production_reward_beta = float(production_reward_beta)
        self.production_reward_reference = float(production_reward_reference)
        self.capture_time_curvature = float(capture_time_curvature)
        # (player, planet_id) -> active episode
        self._active: dict[tuple[int, int], _AttackEpisode] = {}
        # player -> deque[(buffer_idx, reward, event_turn)]
        self._resolved: dict[int, deque] = {}

    # -------------------------------------------------------------- queries
    def is_under_attack(self, player: int, planet_id: int) -> bool:
        """True iff the player owns `planet_id` and at least one enemy
        fleet is currently in flight toward it (episode active).
        """
        return (int(player), int(planet_id)) in self._active

    # -------------------------------------------------------- registration
    def register_defender_launch(self, *, buffer_idx: int, player: int,
                                  target_planet_id: int, launch_turn: int,
                                  ships_sent: int, est_arrival_turn: int) -> None:
        """Record a launch that targets one of my planets currently under
        attack. Should be called from the rollout loop *after* the launch
        is decided but only when `is_under_attack(player, target) is True`.
        """
        if int(ships_sent) <= 0:
            return
        key = (int(player), int(target_planet_id))
        episode = self._active.get(key)
        if episode is None:
            return
        episode.defenders.append(_DefenderLaunch(
            buffer_idx=int(buffer_idx),
            launch_turn=int(launch_turn),
            ships_sent=int(ships_sent),
            est_arrival_turn=int(est_arrival_turn),
        ))

    # ---------------------------------------------------------- resolution
    def observe_turn(self, turn: int, prev_obs, curr_obs, player: int, *,
                     defense_kills_weight: float, reward_scale: float) -> None:
        """Update episodes for `player`, detect arrivals, and finalize
        completed episodes (paying out delayed rewards).
        """
        prev_planets_list = parse_planets(prev_obs)
        curr_planets_list = parse_planets(curr_obs)
        prev_planets = {p.id: p for p in prev_planets_list}
        curr_planets = {p.id: p for p in curr_planets_list}
        prev_fleets_list = parse_fleets(prev_obs)
        curr_fleets_list = parse_fleets(curr_obs)
        prev_fleets = {f.id: f for f in prev_fleets_list}
        curr_fleets = {f.id: f for f in curr_fleets_list}

        prev_preds = (
            predict_all_fleet_targets(
                prev_fleets_list, prev_planets_list, self.angular_velocity)
            if prev_fleets_list else {}
        )
        curr_preds = (
            predict_all_fleet_targets(
                curr_fleets_list, curr_planets_list, self.angular_velocity)
            if curr_fleets_list else {}
        )

        # 1. Determine which of my planets are currently under attack.
        my_under_attack: set[int] = set()
        for f_id, f in curr_fleets.items():
            if f.owner == player or f.owner == -1:
                continue
            pred = curr_preds.get(f_id)
            if pred is None:
                continue
            target_id = int(pred["target_id"])
            curr = curr_planets.get(target_id)
            if curr is None or curr.owner != player:
                continue
            my_under_attack.add(target_id)
            key = (int(player), target_id)
            if key not in self._active:
                self._active[key] = _AttackEpisode(
                    planet_id=target_id,
                    started_turn=int(turn),
                )

        # 2. Detect enemy fleet arrivals at my planets this step. Per-arrival
        #    credit: pay eligible defenders for THIS specific kill event
        #    immediately, instead of waiting for episode end.
        for f_id, prev_fl in prev_fleets.items():
            if prev_fl.owner == player or prev_fl.owner == -1:
                continue
            if f_id in curr_fleets:
                continue  # still in flight
            pred = prev_preds.get(f_id)
            if pred is None:
                continue
            target_id = int(pred["target_id"])
            key = (int(player), target_id)
            episode = self._active.get(key)
            if episode is None:
                continue
            curr_planet = curr_planets.get(target_id)
            if curr_planet is None:
                # Target vanished (comet etc.). Drop episode.
                self._active.pop(key, None)
                continue
            if curr_planet.owner != player:
                # Planet flipped during combat → defense failed; drop.
                self._active.pop(key, None)
                continue
            # Defense held against this fleet → distribute credit now.
            arrived_kills = float(prev_fl.ships)
            episode.enemy_killed_total += arrived_kills
            episode.last_combat_turn = int(turn)
            if arrived_kills <= 0.0:
                continue
            eligible = [
                d for d in episode.defenders
                if d.est_arrival_turn <= int(turn)
            ]
            total_ships = sum(d.ships_sent for d in eligible)
            if total_ships <= 0:
                continue
            # Production-symmetric reward (matches capture formula). A
            # successful defense at a hi-prod planet is comparable in
            # magnitude to a hi-prod capture.
            production_value = production_reward_value(
                float(curr_planet.production),
                production_reward_beta=self.production_reward_beta,
                production_reward_reference=self.production_reward_reference,
            )
            time_factor = curved_time_factor(
                self.max_turns, int(turn),
                curvature=self.capture_time_curvature,
            )
            for d in eligible:
                share = d.ships_sent / total_ships
                reward = (
                    production_value * time_factor * arrived_kills * share
                    * float(defense_kills_weight)
                    * float(reward_scale)
                )
                if reward != 0.0:
                    self._resolved.setdefault(int(player), deque()).append(
                        (int(d.buffer_idx),
                         float(reward),
                         int(turn))
                    )

        # 3. Drop episodes whose attack wave has fully passed (no more
        #    enemy fleets targeting this planet). No payout here — credit
        #    has already been distributed per arrival in step 2.
        for key in list(self._active.keys()):
            ep_player, planet_id = key
            if ep_player != player:
                continue
            if planet_id in my_under_attack:
                continue  # still under attack
            self._active.pop(key, None)

    def pop_resolved(self, player: int):
        """Yield (buffer_idx, reward, event_turn) tuples ready for credit."""
        q = self._resolved.get(int(player))
        if q:
            while q:
                yield q.popleft()
