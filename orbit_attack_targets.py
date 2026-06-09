"""Production-based attack target metrics stored on graph nodes.

Each non-owned planet node carries either ``neutral_attack`` or
``enemy_attack`` (separate dataclasses) with capture timing, ship cost,
and production economics.  Owned planets get ``None`` for both.

Pipeline::

    1. :func:`orbit_defense.compute_launch_budgets` — surplus → launchable ships
    2. :func:`annotate_attack_metrics_on_graph` — write metrics on nodes
    3. :func:`plan_attack_targets` — budgets + ranked lists + node annotation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from navigation import compute_launch
from orbit_defense import compute_launch_budgets
from orbit_expansion import estimated_defense_at_arrival, production_value_until_end
from orbit_graph import OrbitGraph
from reworked_features import DEFAULT_GAMMA


class TargetKind(str, Enum):
    NEUTRAL = "neutral"
    ENEMY = "enemy"


@dataclass(frozen=True)
class NeutralTargetMetrics:
    """Capture economics for a neutral planet (stored on node)."""

    planet_id: int
    production: int
    best_src_id: int
    capture_eta: float
    capture_turn: float
    ships_required: int
    ships_sent: int
    ships_lost: int
    production_value: float
    hold_turns: float
    net_value: float
    attack_window: float
    efficiency: float
    score: float
    feasible: bool

    @property
    def score_tuple(self) -> tuple[float, ...]:
        return (self.score, self.efficiency, self.net_value, -self.capture_eta)


@dataclass(frozen=True)
class EnemyTargetMetrics:
    """Capture economics for an enemy planet (stored on node)."""

    planet_id: int
    production: int
    best_src_id: int
    capture_eta: float
    capture_turn: float
    ships_required: int
    ships_sent: int
    ships_lost: int
    acquisition_value: float
    denial_value: float
    combined_value: float
    hold_turns: float
    enemy_reinforcement_ships: float
    net_value: float
    attack_window: float
    efficiency: float
    score: float
    feasible: bool

    @property
    def score_tuple(self) -> tuple[float, ...]:
        return (self.score, self.efficiency, self.net_value, -self.capture_eta)


@dataclass(frozen=True)
class AttackTargetPlan:
    """Ranked neutral and enemy target lists for one turn."""

    neutrals: tuple[NeutralTargetMetrics, ...]
    enemies: tuple[EnemyTargetMetrics, ...]
    launch_budgets: dict[int, int]

    @property
    def best_neutral(self) -> Optional[NeutralTargetMetrics]:
        return self.neutrals[0] if self.neutrals else None

    @property
    def best_enemy(self) -> Optional[EnemyTargetMetrics]:
        return self.enemies[0] if self.enemies else None


@dataclass(frozen=True)
class AttackTargetConfig:
    current_turn: int = 0
    max_turns: int = 500
    gamma: float = DEFAULT_GAMMA
    angular_velocity: float = 0.0
    eta_horizon: float = 60.0
    denial_weight: float = 1.0
    include_neutrals: bool = True
    include_enemies: bool = True
    include_comets: bool = False


@dataclass(frozen=True)
class _CapturePlan:
    src_id: int
    capture_eta: float
    ships_required: int
    ships_sent: int
    angle: float


def _prod_value(
    production: int,
    *,
    capture_eta: float,
    cfg: AttackTargetConfig,
) -> tuple[float, float]:
    if capture_eta > cfg.eta_horizon:
        return 0.0, 0.0
    return production_value_until_end(
        int(production),
        eta=capture_eta,
        current_turn=cfg.current_turn,
        max_turns=cfg.max_turns,
        gamma=cfg.gamma,
    )


def _find_best_capture_plan(
    og: OrbitGraph,
    tgt_id: int,
    *,
    cfg: AttackTargetConfig,
    launch_budgets: dict[int, int],
) -> Optional[_CapturePlan]:
    """Best (source, ETA, ships) among owned planets with launch budget."""
    tgt = og.planet_by_id.get(int(tgt_id))
    if tgt is None:
        return None
    omega = float(cfg.angular_velocity)
    planets = og.planets
    player = og.player
    tid = int(tgt_id)

    best: Optional[_CapturePlan] = None
    best_key: Optional[tuple[float, ...]] = None

    for src_p in og.planets:
        if src_p.owner != player:
            continue
        sid = int(src_p.id)
        budget = int(launch_budgets.get(sid, 0))
        if budget <= 0:
            continue
        if not og.G.has_edge(sid, tid):
            continue
        if not og.G.edges[sid, tid].get("reachable", False):
            continue

        probe = compute_launch(src_p, tgt, budget, planets, omega)
        if probe is None:
            continue
        eta_probe = float(probe["eta"])
        defense = estimated_defense_at_arrival(tgt, eta_probe, player, og)
        needed = int(defense) + 1
        if needed > budget:
            continue

        launch = compute_launch(src_p, tgt, needed, planets, omega)
        if launch is None:
            continue
        eta = float(launch["eta"])
        if eta > cfg.eta_horizon:
            continue

        plan = _CapturePlan(
            src_id=sid,
            capture_eta=eta,
            ships_required=needed,
            ships_sent=needed,
            angle=float(launch["angle"]),
        )
        key = (-eta, needed, -sid)
        if best is None or key < best_key:  # type: ignore[operator]
            best = plan
            best_key = key
    return best


def score_neutral_target(
    og: OrbitGraph,
    planet_id: int,
    *,
    cfg: AttackTargetConfig,
    launch_budgets: dict[int, int],
) -> NeutralTargetMetrics:
    """Neutral capture metrics for one planet (also written to graph node)."""
    p = og.planet_by_id.get(int(planet_id))
    infeasible = NeutralTargetMetrics(
        planet_id=int(planet_id),
        production=int(p.production) if p else 0,
        best_src_id=-1,
        capture_eta=math.inf,
        capture_turn=math.inf,
        ships_required=0,
        ships_sent=0,
        ships_lost=0,
        production_value=0.0,
        hold_turns=0.0,
        net_value=0.0,
        attack_window=0.0,
        efficiency=0.0,
        score=0.0,
        feasible=False,
    )
    if p is None or p.owner != -1:
        return infeasible
    if p.is_comet and not cfg.include_comets:
        return infeasible

    plan = _find_best_capture_plan(
        og, p.id, cfg=cfg, launch_budgets=launch_budgets,
    )
    if plan is None:
        window = og.attack_window_score(int(p.id))
        return NeutralTargetMetrics(
            planet_id=int(p.id),
            production=int(p.production),
            best_src_id=-1,
            capture_eta=math.inf,
            capture_turn=math.inf,
            ships_required=0,
            ships_sent=0,
            ships_lost=0,
            production_value=0.0,
            hold_turns=0.0,
            net_value=0.0,
            attack_window=float(window),
            efficiency=0.0,
            score=0.0,
            feasible=False,
        )

    prod_val, hold = _prod_value(p.production, capture_eta=plan.capture_eta, cfg=cfg)
    ships_lost = int(plan.ships_required)
    net_val = float(prod_val) - float(ships_lost)
    window = og.attack_window_score(int(p.id))
    efficiency = float(prod_val) / max(float(ships_lost), 1.0)
    score = efficiency * max(0.0, 1.0 + window) if net_val > 0 else 0.0

    return NeutralTargetMetrics(
        planet_id=int(p.id),
        production=int(p.production),
        best_src_id=int(plan.src_id),
        capture_eta=float(plan.capture_eta),
        capture_turn=float(cfg.current_turn) + float(plan.capture_eta),
        ships_required=int(plan.ships_required),
        ships_sent=int(plan.ships_sent),
        ships_lost=ships_lost,
        production_value=float(prod_val),
        hold_turns=float(hold),
        net_value=float(net_val),
        attack_window=float(window),
        efficiency=float(efficiency),
        score=float(score),
        feasible=True,
    )


def score_enemy_target(
    og: OrbitGraph,
    planet_id: int,
    *,
    cfg: AttackTargetConfig,
    launch_budgets: dict[int, int],
) -> EnemyTargetMetrics:
    """Enemy capture metrics (acquisition + denial of enemy production)."""
    p = og.planet_by_id.get(int(planet_id))
    infeasible = EnemyTargetMetrics(
        planet_id=int(planet_id),
        production=int(p.production) if p else 0,
        best_src_id=-1,
        capture_eta=math.inf,
        capture_turn=math.inf,
        ships_required=0,
        ships_sent=0,
        ships_lost=0,
        acquisition_value=0.0,
        denial_value=0.0,
        combined_value=0.0,
        hold_turns=0.0,
        enemy_reinforcement_ships=0.0,
        net_value=0.0,
        attack_window=0.0,
        efficiency=0.0,
        score=0.0,
        feasible=False,
    )
    if p is None or not p.is_enemy(og.player):
        return infeasible
    if p.is_comet and not cfg.include_comets:
        return infeasible

    plan = _find_best_capture_plan(
        og, p.id, cfg=cfg, launch_budgets=launch_budgets,
    )
    reinf = sum(
        float(rec.fleet.ships)
        for rec in og.enemy_reinforcements_inbound(planet_id=p.id)
    )
    window = og.attack_window_score(int(p.id))

    if plan is None:
        return EnemyTargetMetrics(
            planet_id=int(p.id),
            production=int(p.production),
            best_src_id=-1,
            capture_eta=math.inf,
            capture_turn=math.inf,
            ships_required=0,
            ships_sent=0,
            ships_lost=0,
            acquisition_value=0.0,
            denial_value=0.0,
            combined_value=0.0,
            hold_turns=0.0,
            enemy_reinforcement_ships=float(reinf),
            net_value=0.0,
            attack_window=float(window),
            efficiency=0.0,
            score=0.0,
            feasible=False,
        )

    acq_val, hold = _prod_value(p.production, capture_eta=plan.capture_eta, cfg=cfg)
    denial_val = acq_val * float(cfg.denial_weight)
    combined = acq_val + denial_val
    ships_lost = int(plan.ships_required)
    net_val = float(combined) - float(ships_lost)
    efficiency = float(combined) / max(float(ships_lost) + reinf, 1.0)
    score = efficiency * max(0.0, 1.0 + window) if net_val > 0 else 0.0

    return EnemyTargetMetrics(
        planet_id=int(p.id),
        production=int(p.production),
        best_src_id=int(plan.src_id),
        capture_eta=float(plan.capture_eta),
        capture_turn=float(cfg.current_turn) + float(plan.capture_eta),
        ships_required=int(plan.ships_required),
        ships_sent=int(plan.ships_sent),
        ships_lost=ships_lost,
        acquisition_value=float(acq_val),
        denial_value=float(denial_val),
        combined_value=float(combined),
        hold_turns=float(hold),
        enemy_reinforcement_ships=float(reinf),
        net_value=float(net_val),
        attack_window=float(window),
        efficiency=float(efficiency),
        score=float(score),
        feasible=True,
    )


def annotate_attack_metrics_on_graph(
    og: OrbitGraph,
    *,
    cfg: AttackTargetConfig,
    launch_budgets: dict[int, int],
) -> None:
    """Compute and store ``neutral_attack`` / ``enemy_attack`` on every node."""
    for p in og.planets:
        nid = int(p.id)
        og.G.nodes[nid]["neutral_attack"] = None
        og.G.nodes[nid]["enemy_attack"] = None

    if cfg.include_neutrals:
        for p in og.planets:
            if p.owner != -1:
                continue
            m = score_neutral_target(
                og, p.id, cfg=cfg, launch_budgets=launch_budgets,
            )
            og.G.nodes[int(p.id)]["neutral_attack"] = m

    if cfg.include_enemies:
        for p in og.planets:
            if not p.is_enemy(og.player):
                continue
            m = score_enemy_target(
                og, p.id, cfg=cfg, launch_budgets=launch_budgets,
            )
            og.G.nodes[int(p.id)]["enemy_attack"] = m


def node_neutral_attack(og: OrbitGraph, planet_id: int) -> Optional[NeutralTargetMetrics]:
    raw = og.G.nodes[int(planet_id)].get("neutral_attack")
    return raw if isinstance(raw, NeutralTargetMetrics) else None


def node_enemy_attack(og: OrbitGraph, planet_id: int) -> Optional[EnemyTargetMetrics]:
    raw = og.G.nodes[int(planet_id)].get("enemy_attack")
    return raw if isinstance(raw, EnemyTargetMetrics) else None


def rank_neutral_targets(
    og: OrbitGraph,
    *,
    cfg: AttackTargetConfig | None = None,
    launch_budgets: dict[int, int] | None = None,
) -> list[NeutralTargetMetrics]:
    cfg = cfg or AttackTargetConfig()
    if launch_budgets is None:
        launch_budgets = {pid: b.launchable_ships for pid, b in compute_launch_budgets(og).items()}
    rows = [
        score_neutral_target(og, p.id, cfg=cfg, launch_budgets=launch_budgets)
        for p in og.planets
        if p.owner == -1
    ]
    rows = [r for r in rows if r.feasible]
    rows.sort(key=lambda r: r.score_tuple, reverse=True)
    return rows


def rank_enemy_targets(
    og: OrbitGraph,
    *,
    cfg: AttackTargetConfig | None = None,
    launch_budgets: dict[int, int] | None = None,
) -> list[EnemyTargetMetrics]:
    cfg = cfg or AttackTargetConfig()
    if launch_budgets is None:
        launch_budgets = {pid: b.launchable_ships for pid, b in compute_launch_budgets(og).items()}
    rows = [
        score_enemy_target(og, p.id, cfg=cfg, launch_budgets=launch_budgets)
        for p in og.planets
        if p.is_enemy(og.player)
    ]
    rows = [r for r in rows if r.feasible]
    rows.sort(key=lambda r: r.score_tuple, reverse=True)
    return rows


def plan_attack_targets(
    og: OrbitGraph,
    *,
    cfg: AttackTargetConfig | None = None,
    horizon_cap: float = 15.0,
    potential_weight: float = 1.0,
) -> AttackTargetPlan:
    """Defense budgets, node annotation, and ranked target lists."""
    cfg = cfg or AttackTargetConfig()
    budgets = compute_launch_budgets(
        og, horizon_cap=horizon_cap, potential_weight=potential_weight,
    )
    launch_map = {pid: b.launchable_ships for pid, b in budgets.items()}
    annotate_attack_metrics_on_graph(og, cfg=cfg, launch_budgets=launch_map)
    return AttackTargetPlan(
        neutrals=tuple(rank_neutral_targets(og, cfg=cfg, launch_budgets=launch_map)),
        enemies=tuple(rank_enemy_targets(og, cfg=cfg, launch_budgets=launch_map)),
        launch_budgets=launch_map,
    )


def _env_omega(env_obs: Any) -> float:
    if isinstance(env_obs, dict):
        return float(env_obs.get("angular_velocity", 0.0) or 0.0)
    return float(getattr(env_obs, "angular_velocity", 0.0) or 0.0)


def attack_config_from_env(
    env_obs: Any,
    *,
    current_turn: int = 0,
    max_turns: int = 500,
    **kwargs: Any,
) -> AttackTargetConfig:
    """Build :class:`AttackTargetConfig` with omega from env observation."""
    return AttackTargetConfig(
        current_turn=int(current_turn),
        max_turns=int(max_turns),
        angular_velocity=_env_omega(env_obs),
        **kwargs,
    )
