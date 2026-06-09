"""Greedy multi-target capture allocation (nearest source first, no sync wait).

For each ranked target we walk owned sources in ascending ETA order and
commit surplus launch budget until the planet is capturable this wave (or
sources for that target are exhausted), then move to the next target until
launch cap or global budget runs out.

Neutral targets use production efficiency; **enemy** targets use
:mod:`orbit_enemy_weakness` (fragile local defense + thin empire support).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from navigation import compute_launch
from orbit_attack_targets import (
    AttackTargetConfig,
    AttackTargetPlan,
    EnemyTargetMetrics,
    NeutralTargetMetrics,
)
from orbit_expansion import estimated_defense_at_arrival, production_value_until_end
from orbit_graph import OrbitGraph
from orbit_enemy_weakness import compute_enemy_weak_spot, enemy_weak_spot_priority
from orbit_production_pressure import friendly_diffusion_potential_on_target


@dataclass(frozen=True)
class RankedCaptureTarget:
    """One non-owned planet in priority order for greedy allocation."""

    planet_id: int
    is_enemy: bool
    production: int
    priority: float
    friendly_influence: float
    diffusion_potential: float
    weakness_score: float
    attack_window: float
    min_eta: float

    @property
    def sort_key(self) -> tuple[float, ...]:
        return (self.priority, self.weakness_score, self.attack_window, -self.min_eta, self.production)


def friendly_influence_on_target(og: OrbitGraph, planet_id: int) -> float:
    """Our realized + potential pressure on ``planet_id`` (viewer-relative)."""
    p = og.node_pressure(int(planet_id))
    return float(p["potential_friendly_support"]) + float(p["realized_friendly_support"])


def _pressure_terms(
    og: OrbitGraph,
    planet_id: int,
    *,
    influence_weight: float,
    production_pressure_weight: float,
) -> tuple[float, float, float]:
    pid = int(planet_id)
    influence = friendly_influence_on_target(og, pid)
    diffusion = friendly_diffusion_potential_on_target(og, pid)
    window = float(og.attack_window_score(pid))
    combined = (
        float(influence_weight) * (influence + window)
        + float(production_pressure_weight) * diffusion
    )
    return influence, diffusion, combined


def neutral_capture_priority(
    og: OrbitGraph,
    metrics: NeutralTargetMetrics,
    *,
    influence_weight: float,
    production_pressure_weight: float,
) -> float:
    _infl, _prod, pressure = _pressure_terms(
        og, int(metrics.planet_id),
        influence_weight=influence_weight,
        production_pressure_weight=production_pressure_weight,
    )
    return float(metrics.score) + pressure


def _min_source_eta(
    og: OrbitGraph,
    tgt_id: int,
    launch_budgets: dict[int, int],
    omega: float,
) -> float:
    """Earliest reachable launch ETA from any owned source with budget."""
    tgt = og.planet_by_id.get(int(tgt_id))
    if tgt is None:
        return math.inf
    best = math.inf
    player = og.player
    for src_p in og.planets:
        if src_p.owner != player:
            continue
        sid = int(src_p.id)
        budget = int(launch_budgets.get(sid, 0))
        if budget <= 0:
            continue
        if not og.G.has_edge(sid, int(tgt_id)):
            continue
        if not og.G.edges[sid, int(tgt_id)].get("reachable", False):
            continue
        launch = compute_launch(src_p, tgt, budget, og.planets, omega)
        if launch is None:
            continue
        best = min(best, float(launch["eta"]))
    return best


def _total_reachable_budget(
    og: OrbitGraph,
    tgt_id: int,
    launch_budgets: dict[int, int],
) -> int:
    tid = int(tgt_id)
    total = 0
    for src_p in og.planets:
        if src_p.owner != og.player:
            continue
        sid = int(src_p.id)
        budget = int(launch_budgets.get(sid, 0))
        if budget <= 0:
            continue
        if not og.G.has_edge(sid, tid):
            continue
        if not og.G.edges[sid, tid].get("reachable", False):
            continue
        total += budget
    return int(total)


def _fallback_priority(
    og: OrbitGraph,
    planet_id: int,
    *,
    production: int,
    is_enemy: bool,
    influence_weight: float,
    production_pressure_weight: float,
    denial_weight: float,
    min_eta: float,
    attack_cfg: AttackTargetConfig,
    weakness_weight: float = 1.0,
    empire_support_weight: float = 1.0,
) -> tuple[float, float]:
    """Priority when only a multi-source wave can capture (no single-src plan)."""
    pid = int(planet_id)
    prod_val, _ = production_value_until_end(
        int(production),
        eta=float(min_eta),
        current_turn=int(attack_cfg.current_turn),
        max_turns=int(attack_cfg.max_turns),
        gamma=float(attack_cfg.gamma),
    )
    if is_enemy:
        spot = compute_enemy_weak_spot(
            og, pid,
            capture_eta=float(min_eta),
            empire_support_weight=empire_support_weight,
        )
        if spot is not None:
            return (
                float(weakness_weight) * float(spot.weakness_score),
                float(spot.weakness_score),
            )
        return 0.0, 0.0

    _infl, _prod, pressure = _pressure_terms(
        og, pid,
        influence_weight=influence_weight,
        production_pressure_weight=production_pressure_weight,
    )
    upside = float(prod_val) + float(production) - float(min_eta)
    return pressure + upside, 0.0


def rank_capture_targets(
    og: OrbitGraph,
    plan: AttackTargetPlan,
    *,
    attack_cfg: AttackTargetConfig,
    influence_weight: float = 1.0,
    production_pressure_weight: float = 1.0,
    denial_weight: float = 1.0,
    weakness_weight: float = 1.0,
    empire_support_weight: float = 1.0,
    prioritize_enemy_weakspots: bool = True,
) -> list[RankedCaptureTarget]:
    """Merge neutral + enemy metrics into one greedy priority list.

    Single-source-infeasible targets are still included when the **sum** of
    reachable launch budgets can cover capture at the earliest ETA.
    """
    omega = float(attack_cfg.angular_velocity)
    neutrals: list[RankedCaptureTarget] = []
    enemies: list[RankedCaptureTarget] = []
    seen: set[int] = set()

    for m in plan.neutrals:
        if not m.feasible:
            continue
        pid = int(m.planet_id)
        seen.add(pid)
        neutrals.append(RankedCaptureTarget(
            planet_id=pid,
            is_enemy=False,
            production=int(m.production),
            priority=neutral_capture_priority(
                og, m,
                influence_weight=influence_weight,
                production_pressure_weight=production_pressure_weight,
            ),
            friendly_influence=friendly_influence_on_target(og, pid),
            diffusion_potential=friendly_diffusion_potential_on_target(og, pid),
            weakness_score=0.0,
            attack_window=float(og.attack_window_score(pid)),
            min_eta=_min_source_eta(og, pid, plan.launch_budgets, omega),
        ))

    for m in plan.enemies:
        if not m.feasible:
            continue
        pid = int(m.planet_id)
        seen.add(pid)
        spot = compute_enemy_weak_spot(
            og, pid,
            capture_eta=float(m.capture_eta),
            reinf_inbound=float(m.enemy_reinforcement_ships),
            empire_support_weight=empire_support_weight,
        )
        enemies.append(RankedCaptureTarget(
            planet_id=pid,
            is_enemy=True,
            production=int(m.production),
            priority=enemy_weak_spot_priority(
                og, m,
                weakness_weight=weakness_weight,
                empire_support_weight=empire_support_weight,
            ),
            friendly_influence=friendly_influence_on_target(og, pid),
            diffusion_potential=friendly_diffusion_potential_on_target(og, pid),
            weakness_score=float(spot.weakness_score) if spot is not None else 0.0,
            attack_window=float(og.attack_window_score(pid)),
            min_eta=_min_source_eta(og, pid, plan.launch_budgets, omega),
        ))

    for p in og.planets:
        pid = int(p.id)
        if pid in seen:
            continue
        if p.owner == og.player or p.is_comet:
            continue
        if p.owner == -1:
            if not attack_cfg.include_neutrals:
                continue
        elif p.is_enemy(og.player):
            if not attack_cfg.include_enemies:
                continue
        else:
            continue

        min_eta = _min_source_eta(og, pid, plan.launch_budgets, omega)
        if not math.isfinite(min_eta):
            continue
        needed = _ships_needed_to_capture(og, pid, eta=min_eta)
        if needed <= 0:
            continue
        if _total_reachable_budget(og, pid, plan.launch_budgets) < needed:
            continue

        is_enemy = p.is_enemy(og.player)
        priority, weak = _fallback_priority(
            og,
            pid,
            production=int(p.production),
            is_enemy=is_enemy,
            influence_weight=influence_weight,
            production_pressure_weight=production_pressure_weight,
            denial_weight=denial_weight,
            min_eta=min_eta,
            attack_cfg=attack_cfg,
            weakness_weight=weakness_weight,
            empire_support_weight=empire_support_weight,
        )
        row = RankedCaptureTarget(
            planet_id=pid,
            is_enemy=is_enemy,
            production=int(p.production),
            priority=priority,
            friendly_influence=friendly_influence_on_target(og, pid),
            diffusion_potential=friendly_diffusion_potential_on_target(og, pid),
            weakness_score=float(weak),
            attack_window=float(og.attack_window_score(pid)),
            min_eta=min_eta,
        )
        if is_enemy:
            enemies.append(row)
        else:
            neutrals.append(row)

    neutrals.sort(key=lambda r: r.sort_key, reverse=True)
    enemies.sort(key=lambda r: r.sort_key, reverse=True)
    if prioritize_enemy_weakspots:
        return enemies + neutrals
    rows = neutrals + enemies
    rows.sort(key=lambda r: r.sort_key, reverse=True)
    return rows


def _inflight_already_captures(
    og: OrbitGraph,
    tgt_id: int,
) -> bool:
    """True when inbound own attack fleets are enough to take the planet."""
    attacks = og.friendly_attacks_inbound(planet_id=int(tgt_id))
    if not attacks:
        return False
    tgt = og.planet_by_id.get(int(tgt_id))
    if tgt is None:
        return False
    latest_eta = max(float(rec.eta) for rec in attacks)
    own_ships = sum(int(rec.fleet.ships) for rec in attacks)
    defense = estimated_defense_at_arrival(tgt, latest_eta, og.player, og)
    return own_ships > int(defense)


def _pending_own_attack_ships(og: OrbitGraph, tgt_id: int) -> int:
    return sum(
        int(rec.fleet.ships)
        for rec in og.friendly_attacks_inbound(planet_id=int(tgt_id))
    )


@dataclass(frozen=True)
class _EtaSource:
    eta: float
    src_id: int


def _sources_by_eta(
    og: OrbitGraph,
    tgt_id: int,
    remaining: dict[int, int],
    omega: float,
) -> list[_EtaSource]:
    """Owned sources with budget, sorted by launch ETA to ``tgt_id``."""
    tgt = og.planet_by_id.get(int(tgt_id))
    if tgt is None:
        return []
    rows: list[_EtaSource] = []
    tid = int(tgt_id)
    for src_p in og.planets:
        if src_p.owner != og.player:
            continue
        sid = int(src_p.id)
        budget = int(remaining.get(sid, 0))
        if budget <= 0:
            continue
        if not og.G.has_edge(sid, tid):
            continue
        if not og.G.edges[sid, tid].get("reachable", False):
            continue
        launch = compute_launch(src_p, tgt, budget, og.planets, omega)
        if launch is None:
            continue
        rows.append(_EtaSource(eta=float(launch["eta"]), src_id=sid))
    rows.sort(key=lambda r: (r.eta, r.src_id))
    return rows


def _ships_needed_to_capture(
    og: OrbitGraph,
    tgt_id: int,
    *,
    eta: float,
) -> int:
    tgt = og.planet_by_id.get(int(tgt_id))
    if tgt is None:
        return 0
    defense = estimated_defense_at_arrival(tgt, float(eta), og.player, og)
    pending = _pending_own_attack_ships(og, tgt_id)
    return max(0, int(defense) + 1 - int(pending))


def allocate_greedy_captures(
    og: OrbitGraph,
    plan: AttackTargetPlan,
    remaining: dict[int, int],
    used_sources: set[int],
    *,
    attack_cfg: AttackTargetConfig,
    influence_weight: float = 1.0,
    production_pressure_weight: float = 1.0,
    denial_weight: float = 1.0,
    weakness_weight: float = 1.0,
    empire_support_weight: float = 1.0,
    prioritize_enemy_weakspots: bool = True,
    eta_horizon: float = 60.0,
    max_launches: int = 8,
) -> list[list]:
    """Nearest-source greedy captures across ranked targets.

    No synchronized waiting: each source launches immediately this turn.
    One launch per source per turn (``used_sources``).
    """
    if max_launches <= 0:
        return []

    omega = float(attack_cfg.angular_velocity)
    targets = rank_capture_targets(
        og,
        plan,
        attack_cfg=attack_cfg,
        influence_weight=influence_weight,
        production_pressure_weight=production_pressure_weight,
        denial_weight=denial_weight,
        weakness_weight=weakness_weight,
        empire_support_weight=empire_support_weight,
        prioritize_enemy_weakspots=prioritize_enemy_weakspots,
    )
    actions: list[list] = []

    for target in targets:
        if len(actions) >= max_launches:
            break
        tid = int(target.planet_id)
        if _inflight_already_captures(og, tid):
            continue

        sources = _sources_by_eta(og, tid, remaining, omega)
        if not sources:
            continue

        min_eta = float(sources[0].eta)
        if min_eta > float(eta_horizon):
            continue

        needed = _ships_needed_to_capture(og, tid, eta=min_eta)
        if needed <= 0:
            continue

        committed = 0
        tgt = og.planet_by_id[tid]
        for row in sources:
            if len(actions) >= max_launches:
                break
            if committed >= needed:
                break
            sid = int(row.src_id)
            if sid in used_sources:
                continue
            budget = int(remaining.get(sid, 0))
            if budget <= 0:
                continue

            still_need = int(needed) - int(committed)
            ships = min(budget, still_need)
            if ships <= 0:
                continue

            src = og.planet_by_id[sid]
            launch = compute_launch(src, tgt, ships, og.planets, omega)
            if launch is None:
                continue
            if float(launch["eta"]) > float(eta_horizon):
                continue

            sent = int(launch["ships"])
            actions.append([sid, float(launch["angle"]), sent])
            committed += sent
            remaining[sid] = int(remaining.get(sid, 0)) - sent
            used_sources.add(sid)

    return actions


__all__ = [
    "RankedCaptureTarget",
    "allocate_greedy_captures",
    "friendly_influence_on_target",
    "neutral_capture_priority",
    "rank_capture_targets",
]
