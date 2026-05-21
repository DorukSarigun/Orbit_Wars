"""Planet-loss credit assignment via Defense Window (Tasarım 2).

Tracks per (player, planet_id) the recent DEPART (launches FROM the
planet) and HOLD (NO-OP / passive ship holding at the planet) actions
in a sliding window. When a planet's ownership transfers away from
the player, the loss penalty is distributed across these defensive
actions, weighted by ship counts:

    loss_value = production * (max_turns - t_loss)
                 * opponent_count_multiplier(num_opponents)
                 * loss_weight * reward_scale

    DEPART actions:  -loss_value * depart_share * (ships_diverted / total_diverted)
    HOLD actions:    -loss_value * hold_share   * (ships_held / total_held)

If only one of {DEPART, HOLD} has records in window, that side absorbs
the full penalty (depart_share + hold_share). If neither has records
(rare; agent never engaged with the planet), no penalty is emitted.

REINFORCE actions (launches *toward* my own planet) are intentionally
left out of v1; a future extension can give them a positive reward
when they arrive at a still-mine planet that was under threat.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator

from feature_extraction import parse_planets

from .rewards import curved_time_factor


@dataclass
class _PendingAction:
    buffer_idx: int
    turn: int
    ships: int
    kind: str  # "depart" or "hold"


def _opponent_count_multiplier(num_opponents: int) -> float:
    """Mirror of production_reward_tracker.opponent_count_multiplier so
    that capture and loss values use the same scaling."""
    total_players = max(2, int(num_opponents) + 1)
    return (total_players + 1.0) / total_players


class PlanetDefenseTracker:
    """Distributes planet-loss penalties across DEPART and HOLD actions."""

    def __init__(self, *, horizon: int = 30,
                 depart_share: float = 0.7,
                 hold_share: float = 0.3,
                 max_turns: int = 500,
                 num_opponents: int = 1,
                 capture_time_curvature: float = 1.0) -> None:
        self.horizon = int(horizon)
        self.depart_share = float(depart_share)
        self.hold_share = float(hold_share)
        self.max_turns = int(max_turns)
        self.num_opponents = max(1, int(num_opponents))
        # Symmetric with capture_reward: power-curved remaining-time
        # factor (curvature=1.0 reproduces the legacy linear (T-t)).
        self.capture_time_curvature = float(capture_time_curvature)
        # (player, planet_id) -> list of _PendingAction (chronological)
        self._actions: dict[tuple[int, int], list[_PendingAction]] = {}
        # player -> deque of (buffer_idx, signed_reward)
        self._resolved: dict[int, deque] = {}

    # ------------------------------------------------------------- registration
    def register_depart(self, *, buffer_idx: int, player: int,
                        src_planet_id: int, turn: int,
                        ships_sent: int) -> None:
        """Record a launch FROM `src_planet_id` (took ships away)."""
        if int(ships_sent) <= 0:
            return
        key = (int(player), int(src_planet_id))
        self._actions.setdefault(key, []).append(_PendingAction(
            buffer_idx=int(buffer_idx),
            turn=int(turn),
            ships=int(ships_sent),
            kind="depart",
        ))

    def register_hold(self, *, buffer_idx: int, player: int,
                      planet_id: int, turn: int,
                      ships_held: int) -> None:
        """Record passive holding of `ships_held` ships at `planet_id`."""
        if int(ships_held) <= 0:
            return
        key = (int(player), int(planet_id))
        self._actions.setdefault(key, []).append(_PendingAction(
            buffer_idx=int(buffer_idx),
            turn=int(turn),
            ships=int(ships_held),
            kind="hold",
        ))

    # -------------------------------------------------------------- resolution
    def observe_turn(self, turn: int, prev_obs, curr_obs, player: int,
                     *, reward_scale: float, loss_weight: float) -> None:
        """Detect ownership transitions away from `player` and emit
        per-action delayed rewards into the resolved queue.
        """
        self._expire(turn)
        prev_planets = {p.id: p for p in parse_planets(prev_obs)}
        curr_planets = {p.id: p for p in parse_planets(curr_obs)}
        for planet_id, prev in prev_planets.items():
            if prev.owner != player:
                continue
            curr = curr_planets.get(planet_id)
            if curr is not None and curr.owner == player:
                continue
            # Planet flipped away from `player` (lost or vanished).
            key = (int(player), int(planet_id))
            actions = [
                a for a in self._actions.pop(key, [])
                if turn - a.turn <= self.horizon
            ]
            if not actions:
                continue
            remaining = curved_time_factor(
                self.max_turns, int(turn),
                curvature=self.capture_time_curvature,
            )
            opp_mult = _opponent_count_multiplier(self.num_opponents)
            loss_value = (float(prev.production) * float(remaining)
                          * opp_mult * float(reward_scale)
                          * float(loss_weight))
            if loss_value <= 0.0:
                continue
            departs = [a for a in actions if a.kind == "depart"]
            holds = [a for a in actions if a.kind == "hold"]
            depart_total = sum(a.ships for a in departs)
            hold_total = sum(a.ships for a in holds)
            # Renormalize shares if one side is empty.
            if depart_total > 0 and hold_total > 0:
                d_share = self.depart_share
                h_share = self.hold_share
            elif depart_total > 0:
                d_share = self.depart_share + self.hold_share
                h_share = 0.0
            elif hold_total > 0:
                d_share = 0.0
                h_share = self.depart_share + self.hold_share
            else:
                continue
            q = self._resolved.setdefault(player, deque())
            if depart_total > 0:
                for a in departs:
                    penalty = -loss_value * d_share * (a.ships / depart_total)
                    q.append((a.buffer_idx, float(penalty)))
            if hold_total > 0:
                for a in holds:
                    penalty = -loss_value * h_share * (a.ships / hold_total)
                    q.append((a.buffer_idx, float(penalty)))

    def pop_resolved(self, player: int) -> Iterator[tuple[int, float]]:
        q = self._resolved.get(player)
        if q is None:
            return
        while q:
            yield q.popleft()

    # ------------------------------------------------------------------ helper
    def _expire(self, turn: int) -> None:
        if self.horizon <= 0:
            return
        for key in list(self._actions):
            kept = [
                a for a in self._actions[key]
                if turn - a.turn <= self.horizon
            ]
            if kept:
                self._actions[key] = kept
            else:
                self._actions.pop(key, None)
