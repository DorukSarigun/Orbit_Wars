"""Graph-metric scripted agent: defense, backline support, production attacks.

Turn pipeline::

    1. :func:`orbit_graph.build_orbit_graph`
    2. :func:`orbit_defense.compute_owned_defense` + launch budgets
    3. :func:`orbit_attack_targets.plan_attack_targets`
    4. **act** — reinforce critical worlds, then
       :func:`orbit_attack_allocate.allocate_greedy_captures`, then backline
       support to high-influence frontline planets (remaining budget).

Public API::

    ``OrbitScriptedConfig``  — tunable hyperparameters
    ``OrbitScriptedAgent``   — ``act(env_obs, player, ...)``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from navigation import compute_launch
from orbit_attack_allocate import allocate_greedy_captures
from orbit_attack_targets import (
    AttackTargetConfig,
    attack_config_from_env,
    plan_attack_targets,
)
from orbit_defense import (
    DEFAULT_DEFENSE_HORIZON,
    DEFAULT_POTENTIAL_WEIGHT,
    DefenseStatus,
    PlanetDefenseState,
    compute_owned_defense,
)
from orbit_graph import OrbitGraph, build_orbit_graph


@dataclass(frozen=True)
class OrbitScriptedConfig:
    """Hyperparameters for :class:`OrbitScriptedAgent`."""

    max_launches: int = 8
    horizon_cap: float = DEFAULT_DEFENSE_HORIZON
    potential_weight: float = DEFAULT_POTENTIAL_WEIGHT
    strained_threshold: float = 0.0
    critical_threshold: float = -10.0

    # Backline → frontline support when enemy influence is high.
    support_threat_threshold: float = 8.0
    """Minimum realized + weighted potential threat to request support."""
    support_min_source_surplus: float = 5.0
    """Backline source must have at least this defense surplus."""
    support_max_source_threat: float = 4.0
    """Sources with higher local enemy influence are not support donors."""
    support_cushion: float = 3.0
    """Extra ships above liabilities we try to station on threatened worlds."""
    support_ship_fraction: float = 0.35
    """When surplus is thin but influence is high, cover this fraction of it."""

    include_neutrals: bool = True
    include_enemies: bool = True
    include_comets: bool = False
    eta_horizon: float = 60.0
    denial_weight: float = 1.0
    influence_weight: float = 1.0
    """Weight for ship-mass influence + attack-window terms on capture targets."""
    production_pressure_weight: float = 1.0
    """Weight for production diffusion potential on neutral capture targets."""
    enemy_weakness_weight: float = 1.0
    """Scale enemy weak-spot score (fragile + exposed planets first)."""
    enemy_empire_support_weight: float = 1.0
    """Penalty for enemy reinforcement / diffusion reach on the target."""
    prioritize_enemy_weakspots: bool = True
    """Rank enemy weak spots before neutral expansion targets."""


def _env_omega(env_obs: Any) -> float:
    if isinstance(env_obs, dict):
        return float(env_obs.get("angular_velocity", 0.0) or 0.0)
    return float(getattr(env_obs, "angular_velocity", 0.0) or 0.0)


def enemy_influence(state: PlanetDefenseState, *, potential_weight: float) -> float:
    """Combined realized + potential enemy pressure on an owned planet."""
    return (
        float(state.realized_enemy_ships)
        + float(potential_weight) * float(state.potential_enemy_ships_eq)
    )


def support_need(state: PlanetDefenseState, cfg: OrbitScriptedConfig) -> int:
    """Ships we would like to add at ``state`` via friendly reinforcement."""
    gap = max(0.0, float(state.liabilities) - float(state.assets))
    need = float(math.ceil(gap)) if gap > 0 else 0.0

    infl = enemy_influence(state, potential_weight=cfg.potential_weight)
    if infl < cfg.support_threat_threshold:
        return max(0, int(math.ceil(need)))

    # Military balance (exclude production buffer): backline support when
    # garrison + inbound help cannot comfortably absorb enemy influence.
    military_assets = float(state.garrison) + float(state.friendly_inbound_ships)
    military_liabilities = float(state.realized_enemy_ships) + (
        float(cfg.potential_weight) * float(state.potential_enemy_ships_eq)
    )
    mil_gap = military_liabilities + float(cfg.support_cushion) - military_assets
    if mil_gap > 0.0:
        need = max(need, mil_gap)

    if state.status != DefenseStatus.SECURE:
        bolster = max(0.0, float(state.liabilities) + float(cfg.support_cushion) - float(state.assets))
        need = max(need, bolster)

    return max(0, int(math.ceil(need)))


def _backline_score(state: PlanetDefenseState, cfg: OrbitScriptedConfig) -> float:
    threat = enemy_influence(state, potential_weight=cfg.potential_weight)
    if float(state.defense_surplus) < float(cfg.support_min_source_surplus):
        return -math.inf
    if threat > float(cfg.support_max_source_threat):
        return -math.inf
    return float(state.defense_surplus) - 0.25 * threat


@dataclass(frozen=True)
class _ReinforcementPick:
    src_id: int
    tgt_id: int
    angle: float
    ships: int
    eta: float
    score: float


def _best_reinforcement(
    og: OrbitGraph,
    tgt_id: int,
    need: int,
    defense_states: list[PlanetDefenseState],
    remaining: dict[int, int],
    *,
    cfg: OrbitScriptedConfig,
    used_sources: set[int],
    omega: float,
    relaxed_source: bool = False,
) -> Optional[_ReinforcementPick]:
    """Pick a backline (or any surplus) source to reinforce ``tgt_id``."""
    if need <= 0:
        return None
    tgt = og.planet_by_id.get(int(tgt_id))
    if tgt is None:
        return None
    tid = int(tgt_id)

    best: Optional[_ReinforcementPick] = None
    for state in defense_states:
        sid = int(state.planet_id)
        if sid == tid or sid in used_sources:
            continue
        budget = int(remaining.get(sid, 0))
        if budget <= 0:
            continue

        if relaxed_source:
            if float(state.defense_surplus) <= 0.0:
                continue
            score = float(state.defense_surplus)
        else:
            score = _backline_score(state, cfg)
            if not math.isfinite(score):
                continue

        if not og.G.has_edge(sid, tid):
            continue
        if not og.G.edges[sid, tid].get("reachable", False):
            continue

        src = og.planet_by_id[sid]
        ships = min(budget, need, int(state.garrison))
        if ships <= 0:
            continue
        launch = compute_launch(src, tgt, ships, og.planets, omega)
        if launch is None:
            continue
        eta = float(launch["eta"])
        pick = _ReinforcementPick(
            src_id=sid,
            tgt_id=tid,
            angle=float(launch["angle"]),
            ships=int(launch["ships"]),
            eta=eta,
            score=score,
        )
        if best is None or (pick.score, -pick.eta) > (best.score, -best.eta):
            best = pick
    return best


def _apply_launch(
    pick: _ReinforcementPick,
    remaining: dict[int, int],
    used_sources: set[int],
) -> list:
    remaining[pick.src_id] = int(remaining.get(pick.src_id, 0)) - int(pick.ships)
    used_sources.add(pick.src_id)
    return [pick.src_id, pick.angle, pick.ships]


class OrbitScriptedAgent:
    """Defense → attack → support scripted policy on the orbit graph."""

    def __init__(self, cfg: OrbitScriptedConfig | None = None):
        self.cfg = cfg or OrbitScriptedConfig()

    def act(
        self,
        env_obs: Any,
        player: int,
        *,
        current_turn: int = 0,
        max_turns: int = 500,
    ) -> list[list]:
        """Return env actions ``[[from_id, angle, ships], ...]``."""
        cfg = self.cfg
        og = build_orbit_graph(env_obs, player=player)
        if not any(p.owner == player and not p.is_comet for p in og.planets):
            return []

        attack_cfg = attack_config_from_env(
            env_obs,
            current_turn=int(current_turn),
            max_turns=int(max_turns),
            eta_horizon=float(cfg.eta_horizon),
            denial_weight=float(cfg.denial_weight),
            include_neutrals=cfg.include_neutrals,
            include_enemies=cfg.include_enemies,
            include_comets=cfg.include_comets,
        )
        defense_states = compute_owned_defense(
            og,
            horizon_cap=cfg.horizon_cap,
            potential_weight=cfg.potential_weight,
        )
        plan = plan_attack_targets(
            og,
            cfg=attack_cfg,
            horizon_cap=cfg.horizon_cap,
            potential_weight=cfg.potential_weight,
        )
        remaining = dict(plan.launch_budgets)
        used_sources: set[int] = set()
        actions: list[list] = []

        # Phase 1 — critical worlds: any positive-surplus donor.
        critical = [
            s for s in defense_states if s.status == DefenseStatus.CRITICAL
        ]
        critical.sort(key=lambda s: s.defense_surplus)
        for state in critical:
            if len(actions) >= cfg.max_launches:
                break
            need = max(1, int(math.ceil(-float(state.defense_surplus))))
            pick = _best_reinforcement(
                og,
                state.planet_id,
                need,
                defense_states,
                remaining,
                cfg=cfg,
                used_sources=used_sources,
                omega=float(attack_cfg.angular_velocity),
                relaxed_source=True,
            )
            if pick is not None:
                actions.append(_apply_launch(pick, remaining, used_sources))

        # Phase 2 — greedy multi-target captures (nearest source first, no sync).
        slots_left = int(cfg.max_launches) - len(actions)
        if slots_left > 0:
            actions.extend(
                allocate_greedy_captures(
                    og,
                    plan,
                    remaining,
                    used_sources,
                    attack_cfg=attack_cfg,
                    influence_weight=float(cfg.influence_weight),
                    production_pressure_weight=float(cfg.production_pressure_weight),
                    denial_weight=float(cfg.denial_weight),
                    weakness_weight=float(cfg.enemy_weakness_weight),
                    empire_support_weight=float(cfg.enemy_empire_support_weight),
                    prioritize_enemy_weakspots=bool(cfg.prioritize_enemy_weakspots),
                    eta_horizon=float(cfg.eta_horizon),
                    max_launches=slots_left,
                )
            )

        # Phase 3 — backline support toward high enemy-influence frontline.
        support_receivers = [s for s in defense_states if support_need(s, cfg) > 0]
        support_receivers.sort(
            key=lambda s: (
                enemy_influence(s, potential_weight=cfg.potential_weight),
                -float(s.defense_surplus),
            ),
            reverse=True,
        )
        for state in support_receivers:
            if len(actions) >= cfg.max_launches:
                break
            need = support_need(state, cfg)
            pick = _best_reinforcement(
                og,
                state.planet_id,
                need,
                defense_states,
                remaining,
                cfg=cfg,
                used_sources=used_sources,
                omega=float(attack_cfg.angular_velocity),
                relaxed_source=False,
            )
            if pick is not None:
                actions.append(_apply_launch(pick, remaining, used_sources))

        return actions


__all__ = [
    "OrbitScriptedAgent",
    "OrbitScriptedConfig",
    "enemy_influence",
    "support_need",
]
