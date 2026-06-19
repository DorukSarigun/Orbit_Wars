"""Target-local beam planner for final capture waves.

The planner is deliberately deterministic and small.  It enumerates source /
amount combinations for one target, simulates the target timeline, and returns
only plans that capture and hold the target through a local horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from feature_extraction import Planet
from navigation import compute_launch

from final_timeline import TimelineEvent, TimelineResult, simulate_target_timeline


@dataclass(frozen=True)
class SourceOption:
    source: Planet
    budget: int
    strip_cost: float = 0.0


@dataclass(frozen=True)
class PlannedLaunch:
    source_id: int
    target_id: int
    ships: int
    angle: float
    eta: float
    strip_cost: float = 0.0


@dataclass(frozen=True)
class AttackPlan:
    target_id: int
    launches: tuple[PlannedLaunch, ...]
    score: float
    timeline: TimelineResult
    value: float
    ships_sent: int
    true_overkill: float
    reason: str = "ok"

    @property
    def actions(self) -> list[list]:
        return [[l.source_id, float(l.angle), int(l.ships)] for l in self.launches]


@dataclass(frozen=True)
class MicroPlannerConfig:
    hold_horizon: float = 30.0
    beam_width: int = 16
    max_launches_per_target: int = 4
    top_m_sources: int = 6
    min_send: int = 1
    capture_margin: int = 1
    value_weight: float = 1.0
    enemy_denial_weight: float = 0.35
    ship_cost_weight: float = 0.025
    delay_weight: float = 0.01
    source_strip_weight: float = 1.0
    true_overkill_weight: float = 0.01
    enemy_gift_penalty: float = 8.0
    require_stable_hold: bool = True
    min_plan_score: float = -1.0e9


def _known_events_for_target(og, target_id: int) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for rec in og.fleets_to(int(target_id)):
        events.append(TimelineEvent(
            eta=float(rec.eta),
            owner=int(rec.owner),
            ships=float(rec.fleet.ships),
            source_id=int(rec.src_id),
            label=str(rec.mission),
        ))
    return events


def _projected_capture_need(target: Planet, eta_hint: float, cfg: MicroPlannerConfig) -> int:
    projected = float(target.ships)
    if int(target.owner) != -1:
        projected += float(target.production) * max(0.0, float(eta_hint))
    return int(math.ceil(projected)) + int(cfg.capture_margin)


def _candidate_amounts(
    *,
    budget: int,
    target: Planet,
    eta_hint: float,
    cfg: MicroPlannerConfig,
) -> tuple[int, ...]:
    budget = max(0, int(budget))
    if budget < int(cfg.min_send):
        return ()
    base_need = max(int(cfg.min_send), _projected_capture_need(target, eta_hint, cfg))
    raw = {
        int(cfg.min_send),
        min(budget, base_need),
        min(budget, max(base_need + 5, int(round(0.25 * budget)))),
        min(budget, int(round(0.50 * budget))),
        min(budget, int(round(0.75 * budget))),
        budget,
    }
    return tuple(sorted({x for x in raw if int(cfg.min_send) <= x <= budget}))


def _production_value(target: Planet, *, capture_eta: Optional[float], max_turns: int, current_turn: int) -> float:
    if capture_eta is None:
        return 0.0
    remaining = max(0.0, float(max_turns) - float(current_turn) - float(capture_eta))
    return float(target.production) * remaining


def _plan_value(
    *,
    target: Planet,
    player: int,
    timeline: TimelineResult,
    current_turn: int,
    max_turns: int,
    cfg: MicroPlannerConfig,
) -> float:
    capture_value = _production_value(
        target,
        capture_eta=timeline.first_self_capture_eta,
        max_turns=max_turns,
        current_turn=current_turn,
    )
    # Capturing an enemy planet already gives us the future production stream.
    # Denial is kept as a smaller strategic swing term, not another full copy
    # of the same value by default.
    denial = capture_value if target.owner not in (player, -1) else 0.0
    return float(cfg.value_weight) * capture_value + float(cfg.enemy_denial_weight) * denial


def _score_plan(
    *,
    target: Planet,
    player: int,
    launches: Sequence[PlannedLaunch],
    timeline: TimelineResult,
    current_turn: int,
    max_turns: int,
    cfg: MicroPlannerConfig,
) -> tuple[float, float, float]:
    value = _plan_value(
        target=target, player=player, timeline=timeline,
        current_turn=current_turn, max_turns=max_turns, cfg=cfg,
    )
    ships_sent = sum(int(l.ships) for l in launches)
    latest_eta = max((float(l.eta) for l in launches), default=0.0)
    strip = sum(float(l.strip_cost) * int(l.ships) for l in launches)

    hold_buffer = 0.0
    if timeline.self_holds_until_horizon(player):
        hold_buffer = max(0.0, float(timeline.garrison_at_horizon))
    surplus = max(0.0, float(timeline.first_self_capture_surplus))
    true_overkill = max(0.0, surplus - hold_buffer)

    score = (
        value
        - float(cfg.ship_cost_weight) * float(ships_sent)
        - float(cfg.delay_weight) * latest_eta
        - float(cfg.source_strip_weight) * strip
        - float(cfg.true_overkill_weight) * true_overkill
    )
    if timeline.enemy_gift:
        score -= float(cfg.enemy_gift_penalty)
    if cfg.require_stable_hold and not timeline.self_holds_until_horizon(player):
        score -= 1.0e6
    return float(score), float(value), float(true_overkill)


def _launch_options(
    *,
    target: Planet,
    sources: Sequence[SourceOption],
    planets: Sequence[Planet],
    omega: float,
    cfg: MicroPlannerConfig,
) -> list[PlannedLaunch]:
    options: list[PlannedLaunch] = []
    for opt in sources[: max(0, int(cfg.top_m_sources))]:
        if int(opt.budget) < int(cfg.min_send):
            continue
        amount_set = set(_candidate_amounts(
            budget=int(opt.budget),
            target=target,
            eta_hint=0.0,
            cfg=cfg,
        ))
        probe_ships = min(
            int(opt.budget),
            max(int(cfg.min_send), int(target.ships) + int(cfg.capture_margin)),
        )
        probe = compute_launch(opt.source, target, int(probe_ships), planets, omega)
        if probe is not None:
            amount_set.update(_candidate_amounts(
                budget=int(opt.budget),
                target=target,
                eta_hint=float(probe["eta"]),
                cfg=cfg,
            ))
        full_probe = compute_launch(opt.source, target, int(opt.budget), planets, omega)
        if full_probe is not None:
            amount_set.update(_candidate_amounts(
                budget=int(opt.budget),
                target=target,
                eta_hint=float(full_probe["eta"]),
                cfg=cfg,
            ))
        for ships in sorted(amount_set):
            launch = compute_launch(opt.source, target, int(ships), planets, omega)
            if launch is None:
                continue
            options.append(PlannedLaunch(
                source_id=int(opt.source.id),
                target_id=int(target.id),
                ships=int(launch["ships"]),
                angle=float(launch["angle"]),
                eta=float(launch["eta"]),
                strip_cost=float(opt.strip_cost),
            ))
    options.sort(key=lambda l: (float(l.eta), -int(l.ships), int(l.source_id)))
    return options


def plan_target_attack(
    *,
    og,
    target: Planet,
    sources: Sequence[SourceOption],
    player: int,
    current_turn: int,
    max_turns: int,
    omega: float,
    cfg: MicroPlannerConfig = MicroPlannerConfig(),
    extra_events: Iterable[TimelineEvent] = (),
) -> Optional[AttackPlan]:
    """Return the best stable capture plan for one target, or ``None``."""

    base_events = _known_events_for_target(og, int(target.id)) + list(extra_events)
    launch_options = _launch_options(
        target=target,
        sources=sources,
        planets=og.planets,
        omega=omega,
        cfg=cfg,
    )
    if not launch_options:
        return None

    horizon = min(float(cfg.hold_horizon), max(0.0, float(max_turns) - float(current_turn)))
    beam: list[tuple[tuple[PlannedLaunch, ...], set[int]]] = [((), set())]
    best: Optional[AttackPlan] = None

    for _depth in range(max(1, int(cfg.max_launches_per_target))):
        candidates: list[AttackPlan] = []
        next_beam: list[tuple[tuple[PlannedLaunch, ...], set[int]]] = []
        for launches, used_sources in beam:
            for launch in launch_options:
                if int(launch.source_id) in used_sources:
                    continue
                new_launches = tuple(sorted(launches + (launch,), key=lambda l: (l.eta, l.source_id)))
                planned_events = [
                    TimelineEvent(
                        eta=float(l.eta),
                        owner=int(player),
                        ships=float(l.ships),
                        source_id=int(l.source_id),
                        label="planned_attack",
                    )
                    for l in new_launches
                ]
                timeline = simulate_target_timeline(
                    initial_owner=int(target.owner),
                    initial_ships=float(target.ships),
                    production=float(target.production),
                    events=base_events + planned_events,
                    horizon=horizon,
                    player=int(player),
                )
                score, value, true_overkill = _score_plan(
                    target=target,
                    player=int(player),
                    launches=new_launches,
                    timeline=timeline,
                    current_turn=int(current_turn),
                    max_turns=int(max_turns),
                    cfg=cfg,
                )
                plan = AttackPlan(
                    target_id=int(target.id),
                    launches=new_launches,
                    score=float(score),
                    timeline=timeline,
                    value=float(value),
                    ships_sent=sum(int(l.ships) for l in new_launches),
                    true_overkill=float(true_overkill),
                )
                candidates.append(plan)
                next_beam.append((new_launches, set(used_sources) | {int(launch.source_id)}))

        valid = [
            p for p in candidates
            if p.score >= float(cfg.min_plan_score)
            and p.timeline.self_holds_until_horizon(player)
        ]
        if valid:
            local_best = max(valid, key=lambda p: p.score)
            if best is None or local_best.score > best.score:
                best = local_best

        candidates.sort(key=lambda p: p.score, reverse=True)
        allowed = {p.launches for p in candidates[: max(1, int(cfg.beam_width))]}
        beam = [(ls, used) for ls, used in next_beam if ls in allowed]
        if not beam:
            break

    return best
