"""Greedy multi-target capture allocation (nearest source first, no sync wait).

For each ranked target an **all-or-nothing wave** is planned: owned sources in
ascending ETA order commit surplus budget until the planet is capturable.  The
required ship count is evaluated at the wave's **latest** arrival ETA (enemy
garrisons keep producing until the slowest fleet lands) and padded by
``capture_margin``.  If the wave cannot fully fund the capture, **no ships are
sent** (partial waves on neutrals are pure loss).  A source may fund several
waves in one turn — only the global launch cap and its budget bound it.

Neutral targets use production efficiency; **enemy** targets use
:mod:`orbit_enemy_weakness` (fragile local defense + thin empire support).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

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
    priority_bonus: Optional[dict[int, float]] = None,
) -> list[RankedCaptureTarget]:
    """Merge neutral + enemy metrics into one greedy priority list.

    Single-source-infeasible targets are still included when the **sum** of
    reachable launch budgets can cover capture at the earliest ETA.

    ``priority_bonus`` (default ``None``) optionally adds a per-planet score to
    each target's ``priority`` — used by the lookahead agent to promote
    stepping-stone captures.  ``None``/empty leaves ranking byte-identical.
    """
    omega = float(attack_cfg.angular_velocity)
    bonus = priority_bonus or {}
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
            ) + float(bonus.get(pid, 0.0)),
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
            ) + float(bonus.get(pid, 0.0)),
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
            priority=priority + float(bonus.get(pid, 0.0)),
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
    margin: float = 0.0,
) -> int:
    """Ships required to flip ``tgt_id`` when the slowest fleet lands at ``eta``.

    ``margin`` pads the estimate against ETA / production rounding errors.
    Own attack fleets already in flight are subtracted.
    """
    tgt = og.planet_by_id.get(int(tgt_id))
    if tgt is None:
        return 0
    defense = estimated_defense_at_arrival(tgt, float(eta), og.player, og)
    base = int(defense) + 1
    padded = int(math.ceil(float(base) * (1.0 + max(0.0, float(margin)))))
    pending = _pending_own_attack_ships(og, tgt_id)
    return max(0, padded - int(pending))


def _plan_capture_wave(
    og: OrbitGraph,
    tgt_id: int,
    remaining: dict[int, int],
    *,
    omega: float,
    eta_horizon: float,
    margin: float,
    max_launches: int,
) -> Optional[list[list]]:
    """All-or-nothing nearest-source wave for one target.

    Iterates because adding a farther source raises the wave's latest arrival
    ETA, which raises the required ship count (enemy production).  Returns
    ``[[src_id, angle, ships], ...]`` or ``None`` when the capture cannot be
    fully funded.
    """
    if max_launches <= 0:
        return None
    tid = int(tgt_id)
    tgt = og.planet_by_id.get(tid)
    if tgt is None:
        return None
    sources = _sources_by_eta(og, tid, remaining, omega)
    if not sources:
        return None
    if float(sources[0].eta) > float(eta_horizon):
        return None

    target_eta = float(sources[0].eta)
    for _ in range(3):
        needed = _ships_needed_to_capture(og, tid, eta=target_eta, margin=margin)
        if needed <= 0:
            return None

        wave: list[list] = []
        committed = 0
        latest = 0.0
        for row in sources:
            if len(wave) >= int(max_launches) or committed >= needed:
                break
            sid = int(row.src_id)
            budget = int(remaining.get(sid, 0))
            if budget <= 0:
                continue
            ships = min(budget, needed - committed)
            launch = compute_launch(
                og.planet_by_id[sid], tgt, ships, og.planets, omega,
            )
            if launch is None:
                continue
            eta = float(launch["eta"])
            if eta > float(eta_horizon):
                continue
            wave.append([sid, float(launch["angle"]), int(launch["ships"])])
            committed += int(launch["ships"])
            latest = max(latest, eta)

        if not wave or committed < needed:
            return None

        # Authoritative check at the wave's actual latest arrival.
        final_needed = _ships_needed_to_capture(og, tid, eta=latest, margin=margin)
        if committed >= final_needed:
            return wave
        target_eta = max(latest, target_eta)
    return None


@dataclass(frozen=True)
class _LpEdge:
    src_id: int
    tgt_id: int
    eta: float
    capacity: int
    cost: float


@dataclass(frozen=True)
class _LpSolution:
    targets: tuple[RankedCaptureTarget, ...]
    demands: dict[int, int]
    edges: tuple[_LpEdge, ...]
    flow: tuple[float, ...]


def _source_strip_cost(
    og: OrbitGraph,
    src_id: int,
    remaining: dict[int, int],
    *,
    source_strip_weight: float,
) -> float:
    """Penalty for stripping a source planet.

    Sources with a small remaining launch budget relative to their garrison are
    probably frontline / constrained.  Interior surplus bases have cheaper
    budget, so the LP naturally prefers them when ETA is comparable.
    """
    if source_strip_weight <= 0.0:
        return 0.0
    src = og.planet_by_id.get(int(src_id))
    if src is None:
        return 0.0
    garrison = max(1.0, float(src.ships))
    budget = max(0.0, float(remaining.get(int(src_id), 0)))
    constrained = 1.0 - min(1.0, budget / garrison)
    return float(source_strip_weight) * constrained


def _lp_edges_for_targets(
    og: OrbitGraph,
    targets: Sequence[RankedCaptureTarget],
    remaining: dict[int, int],
    *,
    omega: float,
    eta_horizon: float,
    margin: float,
    source_strip_weight: float,
) -> tuple[list[_LpEdge], dict[int, int]]:
    """Build source-target LP edges and demands for a fixed target prefix."""
    edges: list[_LpEdge] = []
    demands: dict[int, int] = {}
    for target in targets:
        tid = int(target.planet_id)
        sources = _sources_by_eta(og, tid, remaining, omega)
        sources = [s for s in sources if float(s.eta) <= float(eta_horizon)]
        if not sources:
            return [], {}
        latest_eta = max(float(s.eta) for s in sources)
        demand = _ships_needed_to_capture(og, tid, eta=latest_eta, margin=margin)
        if demand <= 0:
            return [], {}
        demands[tid] = int(demand)
        for row in sources:
            sid = int(row.src_id)
            cap = int(remaining.get(sid, 0))
            if cap <= 0:
                continue
            # Graph reachability was already checked by _sources_by_eta; cost
            # stays in ETA units plus a source-risk term.
            cost = float(row.eta) + _source_strip_cost(
                og, sid, remaining, source_strip_weight=source_strip_weight)
            edges.append(_LpEdge(
                src_id=sid,
                tgt_id=tid,
                eta=float(row.eta),
                capacity=cap,
                cost=float(cost),
            ))
    return edges, demands


def _solve_lp_prefix(
    edges: Sequence[_LpEdge],
    demands: dict[int, int],
    remaining: dict[int, int],
) -> Optional[tuple[float, ...]]:
    """Solve the continuous logistics LP for one fixed target prefix."""
    if not edges or not demands:
        return None
    try:
        from scipy.optimize import linprog
    except Exception:
        return None

    n = len(edges)
    c = [float(e.cost) for e in edges]
    a_ub: list[list[float]] = []
    b_ub: list[float] = []

    source_ids = sorted({int(e.src_id) for e in edges})
    for sid in source_ids:
        row = [0.0] * n
        for i, e in enumerate(edges):
            if int(e.src_id) == sid:
                row[i] = 1.0
        a_ub.append(row)
        b_ub.append(float(remaining.get(int(sid), 0)))

    for tid, demand in demands.items():
        row = [0.0] * n
        for i, e in enumerate(edges):
            if int(e.tgt_id) == int(tid):
                row[i] = -1.0
        a_ub.append(row)
        b_ub.append(-float(demand))

    bounds = [(0.0, float(e.capacity)) for e in edges]
    res = linprog(
        c=c,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not getattr(res, "success", False):
        return None
    return tuple(float(x) for x in res.x)


def _flow_to_wave(
    og: OrbitGraph,
    solution: _LpSolution,
    remaining: dict[int, int],
    *,
    omega: float,
    eta_horizon: float,
    max_launches: int,
    min_send: int = 1,
) -> Optional[list[list]]:
    """Convert continuous LP flow to integer env launch actions.

    Floors flows first, then greedily fills any per-target rounding deficits on
    the cheapest edges while respecting source budgets.  The returned wave is
    all-or-nothing for every target in ``solution.targets``.
    """
    if max_launches <= 0:
        return None
    edges = list(solution.edges)
    flow = list(solution.flow)
    used_by_source: dict[int, int] = {}
    sent_by_target: dict[int, int] = {int(t): 0 for t in solution.demands}
    ships_by_edge = [0 for _ in edges]

    # Initial floor pass.
    for i, (edge, val) in enumerate(zip(edges, flow)):
        ships = int(math.floor(max(0.0, float(val))))
        if ships < int(min_send):
            continue
        sid = int(edge.src_id)
        tid = int(edge.tgt_id)
        room = int(remaining.get(sid, 0)) - int(used_by_source.get(sid, 0))
        ships = min(ships, max(0, room))
        if ships <= 0:
            continue
        ships_by_edge[i] = ships
        used_by_source[sid] = int(used_by_source.get(sid, 0)) + ships
        sent_by_target[tid] = int(sent_by_target.get(tid, 0)) + ships

    # Fill rounding deficits by cheapest available edges for each target.
    for tid, demand in solution.demands.items():
        deficit = int(demand) - int(sent_by_target.get(int(tid), 0))
        if deficit <= 0:
            continue
        candidates = sorted(
            [
                (edge.cost, i, edge)
                for i, edge in enumerate(edges)
                if int(edge.tgt_id) == int(tid)
            ],
            key=lambda x: (x[0], x[2].eta, x[2].src_id),
        )
        for _cost, i, edge in candidates:
            if deficit <= 0:
                break
            sid = int(edge.src_id)
            room = int(remaining.get(sid, 0)) - int(used_by_source.get(sid, 0))
            if room <= 0:
                continue
            add = min(room, deficit)
            ships_by_edge[i] += int(add)
            used_by_source[sid] = int(used_by_source.get(sid, 0)) + int(add)
            sent_by_target[int(tid)] = int(sent_by_target.get(int(tid), 0)) + int(add)
            deficit -= int(add)
        if deficit > 0:
            return None

    # Final all-or-nothing validation before building actions.
    for tid, demand in solution.demands.items():
        if int(sent_by_target.get(int(tid), 0)) < int(demand):
            return None

    nonzero = [(i, ships) for i, ships in enumerate(ships_by_edge) if int(ships) >= int(min_send)]
    if not nonzero or len(nonzero) > int(max_launches):
        return None
    nonzero.sort(key=lambda row: (
        next(t.sort_key for t in solution.targets if int(t.planet_id) == int(edges[row[0]].tgt_id)),
        -edges[row[0]].cost,
    ), reverse=True)

    actions: list[list] = []
    for i, ships in nonzero:
        edge = edges[i]
        src = og.planet_by_id.get(int(edge.src_id))
        tgt = og.planet_by_id.get(int(edge.tgt_id))
        if src is None or tgt is None:
            return None
        launch = compute_launch(src, tgt, int(ships), og.planets, omega)
        if launch is None or float(launch["eta"]) > float(eta_horizon):
            return None
        actions.append([int(edge.src_id), float(launch["angle"]), int(launch["ships"])])
    return actions or None


def _solve_lp_largest_prefix(
    og: OrbitGraph,
    targets: Sequence[RankedCaptureTarget],
    remaining: dict[int, int],
    *,
    omega: float,
    eta_horizon: float,
    margin: float,
    max_launches: int,
    top_k_targets: int,
    source_strip_weight: float,
) -> Optional[list[list]]:
    """Find the largest ranked target prefix with a valid LP allocation."""
    if max_launches <= 0:
        return None
    filtered = [
        t for t in targets
        if not _inflight_already_captures(og, int(t.planet_id))
    ][: max(1, int(top_k_targets))]
    if not filtered:
        return None

    for k in range(len(filtered), 0, -1):
        prefix = tuple(filtered[:k])
        edges, demands = _lp_edges_for_targets(
            og,
            prefix,
            remaining,
            omega=omega,
            eta_horizon=eta_horizon,
            margin=margin,
            source_strip_weight=source_strip_weight,
        )
        if not edges or not demands:
            continue
        flow = _solve_lp_prefix(edges, demands, remaining)
        if flow is None:
            continue
        actions = _flow_to_wave(
            og,
            _LpSolution(targets=prefix, demands=demands, edges=tuple(edges), flow=flow),
            remaining,
            omega=omega,
            eta_horizon=eta_horizon,
            max_launches=max_launches,
        )
        if actions:
            return actions
    return None


def _partial_enemy_wave(
    og: OrbitGraph,
    tgt_id: int,
    remaining: dict[int, int],
    *,
    omega: float,
    eta_horizon: float,
    margin: float,
    max_launches: int,
) -> Optional[list[list]]:
    """Underfunded pressure wave on an enemy planet (erodes garrison).

    Only used when ``allow_partial_enemy_waves`` is enabled; never applied to
    neutrals (partial neutral attacks are pure ship loss).
    """
    if max_launches <= 0:
        return None
    tid = int(tgt_id)
    tgt = og.planet_by_id.get(tid)
    if tgt is None:
        return None
    sources = _sources_by_eta(og, tid, remaining, omega)
    if not sources or float(sources[0].eta) > float(eta_horizon):
        return None

    needed = _ships_needed_to_capture(
        og, tid, eta=float(sources[0].eta), margin=margin,
    )
    if needed <= 0:
        return None

    wave: list[list] = []
    committed = 0
    for row in sources:
        if len(wave) >= int(max_launches) or committed >= needed:
            break
        sid = int(row.src_id)
        budget = int(remaining.get(sid, 0))
        if budget <= 0:
            continue
        ships = min(budget, needed - committed)
        launch = compute_launch(og.planet_by_id[sid], tgt, ships, og.planets, omega)
        if launch is None:
            continue
        if float(launch["eta"]) > float(eta_horizon):
            continue
        wave.append([sid, float(launch["angle"]), int(launch["ships"])])
        committed += int(launch["ships"])
    return wave or None


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
    capture_margin: float = 0.0,
    max_wave_launches: int = 4,
    allow_partial_enemy_waves: bool = False,
    priority_bonus: Optional[dict[int, float]] = None,
) -> list[list]:
    """All-or-nothing greedy captures across ranked targets.

    No synchronized waiting: every launch happens this turn.  A source may
    fund multiple waves (budget-bound); ``used_sources`` is updated for caller
    bookkeeping but no longer blocks reuse.  Underfunded waves are skipped
    entirely unless ``allow_partial_enemy_waves`` (enemy targets only).
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
        priority_bonus=priority_bonus,
    )
    actions: list[list] = []

    for target in targets:
        slots = int(max_launches) - len(actions)
        if slots <= 0:
            break
        tid = int(target.planet_id)
        if _inflight_already_captures(og, tid):
            continue

        wave = _plan_capture_wave(
            og, tid, remaining,
            omega=omega,
            eta_horizon=eta_horizon,
            margin=capture_margin,
            max_launches=min(int(max_wave_launches), slots),
        )
        if wave is None and allow_partial_enemy_waves and target.is_enemy:
            wave = _partial_enemy_wave(
                og, tid, remaining,
                omega=omega,
                eta_horizon=eta_horizon,
                margin=capture_margin,
                max_launches=min(int(max_wave_launches), slots),
            )
        if not wave:
            continue

        for sid, angle, ships in wave:
            actions.append([int(sid), float(angle), int(ships)])
            remaining[int(sid)] = int(remaining.get(int(sid), 0)) - int(ships)
            used_sources.add(int(sid))

    return actions


def allocate_lp_captures(
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
    capture_margin: float = 0.0,
    max_wave_launches: int = 4,
    allow_partial_enemy_waves: bool = False,
    priority_bonus: Optional[dict[int, float]] = None,
    lp_top_k_targets: int = 4,
    lp_source_strip_weight: float = 2.0,
) -> list[list]:
    """LP/simplex logistics allocation over a ranked target prefix.

    The target *choice* remains strategic and all-or-nothing: we try the largest
    prefix of the ranked target list (bounded by ``lp_top_k_targets``) whose
    demands can be fully funded.  The LP only decides source-target ship flow
    for that fixed prefix.  If SciPy is unavailable, the LP is infeasible, or
    integer launch conversion fails, this falls back to
    :func:`allocate_greedy_captures`.
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
        priority_bonus=priority_bonus,
    )
    actions = _solve_lp_largest_prefix(
        og,
        targets,
        remaining,
        omega=omega,
        eta_horizon=eta_horizon,
        margin=capture_margin,
        max_launches=max_launches,
        top_k_targets=lp_top_k_targets,
        source_strip_weight=lp_source_strip_weight,
    )
    if actions:
        for sid, _angle, ships in actions:
            remaining[int(sid)] = int(remaining.get(int(sid), 0)) - int(ships)
            used_sources.add(int(sid))
        return actions

    return allocate_greedy_captures(
        og,
        plan,
        remaining,
        used_sources,
        attack_cfg=attack_cfg,
        influence_weight=influence_weight,
        production_pressure_weight=production_pressure_weight,
        denial_weight=denial_weight,
        weakness_weight=weakness_weight,
        empire_support_weight=empire_support_weight,
        prioritize_enemy_weakspots=prioritize_enemy_weakspots,
        eta_horizon=eta_horizon,
        max_launches=max_launches,
        capture_margin=capture_margin,
        max_wave_launches=max_wave_launches,
        allow_partial_enemy_waves=allow_partial_enemy_waves,
        priority_bonus=priority_bonus,
    )


__all__ = [
    "RankedCaptureTarget",
    "allocate_greedy_captures",
    "allocate_lp_captures",
    "friendly_influence_on_target",
    "neutral_capture_priority",
    "rank_capture_targets",
]
