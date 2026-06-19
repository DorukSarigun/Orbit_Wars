"""Target-local timeline simulation for final attack planning.

This module is intentionally small and independent.  It answers one question:
given a target planet and a set of known/planned arrivals, who owns the target
after the relevant horizon and which events were actually useful?

It does not model the whole game or opponent re-planning.  That makes it a
biased but stable oracle for the final micro-planner and, later, for local
credit assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


NEUTRAL_OWNER = -1


@dataclass(frozen=True)
class TimelineEvent:
    """One fleet arrival at a target-local ETA."""

    eta: float
    owner: int
    ships: float
    source_id: Optional[int] = None
    label: str = "known"


@dataclass(frozen=True)
class TimelineStep:
    """State transition recorded after one ETA group is resolved."""

    eta: float
    owner_before: int
    garrison_before: float
    owner_after: int
    garrison_after: float
    arrivals_by_owner: dict[int, float]


@dataclass(frozen=True)
class TimelineResult:
    """Summary of a target-local simulation."""

    owner_at_horizon: int
    garrison_at_horizon: float
    first_self_capture_eta: Optional[float]
    first_self_capture_surplus: float
    self_lost_after_capture: bool
    enemy_gift: bool
    stable_hold_turns: float
    steps: tuple[TimelineStep, ...] = field(default_factory=tuple)

    @property
    def self_captured(self) -> bool:
        return self.first_self_capture_eta is not None

    def self_holds_until_horizon(self, player: int) -> bool:
        return int(self.owner_at_horizon) == int(player) and self.self_captured


def _group_events(events: Iterable[TimelineEvent]) -> list[tuple[float, dict[int, float]]]:
    grouped: dict[float, dict[int, float]] = {}
    for event in events:
        if event.ships <= 0:
            continue
        eta = round(max(0.0, float(event.eta)), 6)
        by_owner = grouped.setdefault(eta, {})
        by_owner[int(event.owner)] = by_owner.get(int(event.owner), 0.0) + float(event.ships)
    return sorted(grouped.items(), key=lambda item: item[0])


def _advance_production(owner: int, garrison: float, production: float, dt: float) -> float:
    if int(owner) == NEUTRAL_OWNER:
        return float(garrison)
    return float(garrison) + max(0.0, float(production)) * max(0.0, float(dt))


def _resolve_arrivals(owner: int, garrison: float, arrivals_by_owner: dict[int, float]) -> tuple[int, float]:
    """Resolve one simultaneous ETA group.

    The env's exact multi-owner same-tick ordering is not exposed here.  We use a
    deterministic local rule: current-owner reinforcement is added first, then
    hostile owner groups are applied from largest to smallest.  This is stable
    for planning and exact for the common two-owner case.
    """

    cur_owner = int(owner)
    cur_garrison = float(garrison)

    own_arrival = arrivals_by_owner.get(cur_owner, 0.0)
    if cur_owner != NEUTRAL_OWNER and own_arrival > 0.0:
        cur_garrison += float(own_arrival)

    attackers = [
        (int(o), float(s))
        for o, s in arrivals_by_owner.items()
        if int(o) != cur_owner and float(s) > 0.0
    ]
    attackers.sort(key=lambda item: item[1], reverse=True)

    for attacker_owner, ships in attackers:
        if ships > cur_garrison:
            cur_owner = int(attacker_owner)
            cur_garrison = float(ships) - cur_garrison
        else:
            cur_garrison -= float(ships)
    return cur_owner, max(0.0, cur_garrison)


def simulate_target_timeline(
    *,
    initial_owner: int,
    initial_ships: float,
    production: float,
    events: Iterable[TimelineEvent],
    horizon: float,
    player: int,
) -> TimelineResult:
    """Simulate ownership and garrison on a single target until ``horizon``."""

    owner = int(initial_owner)
    garrison = max(0.0, float(initial_ships))
    horizon = max(0.0, float(horizon))
    last_eta = 0.0
    steps: list[TimelineStep] = []
    first_self_capture_eta: Optional[float] = None
    first_self_capture_surplus = 0.0
    self_lost_after_capture = False
    enemy_gift = False

    for eta, arrivals_by_owner in _group_events(events):
        if eta > horizon:
            break
        owner_before = owner
        garrison = _advance_production(owner, garrison, production, eta - last_eta)
        garrison_before = garrison
        owner, garrison = _resolve_arrivals(owner, garrison, arrivals_by_owner)

        if owner == int(player) and owner_before != int(player) and first_self_capture_eta is None:
            first_self_capture_eta = eta
            first_self_capture_surplus = garrison
        elif (
            first_self_capture_eta is not None
            and owner_before == int(player)
            and owner != int(player)
        ):
            self_lost_after_capture = True

        if (
            first_self_capture_eta is not None
            and owner != int(player)
            and owner != NEUTRAL_OWNER
            and owner_before == int(player)
        ):
            enemy_gift = True

        steps.append(TimelineStep(
            eta=eta,
            owner_before=owner_before,
            garrison_before=float(garrison_before),
            owner_after=int(owner),
            garrison_after=float(garrison),
            arrivals_by_owner=dict(arrivals_by_owner),
        ))
        last_eta = eta

    if horizon > last_eta:
        garrison = _advance_production(owner, garrison, production, horizon - last_eta)

    if first_self_capture_eta is None:
        stable_hold_turns = 0.0
    elif owner == int(player):
        stable_hold_turns = max(0.0, horizon - first_self_capture_eta)
    else:
        loss_eta = next(
            (s.eta for s in steps
             if s.owner_before == int(player) and s.owner_after != int(player)),
            horizon,
        )
        stable_hold_turns = max(0.0, float(loss_eta) - float(first_self_capture_eta))

    return TimelineResult(
        owner_at_horizon=int(owner),
        garrison_at_horizon=float(garrison),
        first_self_capture_eta=first_self_capture_eta,
        first_self_capture_surplus=float(first_self_capture_surplus),
        self_lost_after_capture=bool(self_lost_after_capture),
        enemy_gift=bool(enemy_gift),
        stable_hold_turns=float(stable_hold_turns),
        steps=tuple(steps),
    )

