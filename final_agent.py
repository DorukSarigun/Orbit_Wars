"""Final V1 agent: defense-aware micro-planned captures.

This is not a copy of the scripted agent.  It uses the existing public graph,
defense and target-ranking helpers, but the capture phase is driven by the new
target-local timeline simulator and beam micro-planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Optional

from navigation import compute_launch
from orbit_attack_allocate import rank_capture_targets
from orbit_attack_targets import attack_config_from_env, plan_attack_targets
from orbit_defense import (
    DEFAULT_DEFENSE_HORIZON,
    DEFAULT_POTENTIAL_WEIGHT,
    DefenseStatus,
    PlanetDefenseState,
    compute_owned_defense,
)
from orbit_graph import build_orbit_graph

from final_micro_planner import MicroPlannerConfig, SourceOption, plan_target_attack


@dataclass(frozen=True)
class FinalAgentConfig:
    max_launches: int = 8
    horizon_cap: float = DEFAULT_DEFENSE_HORIZON
    potential_weight: float = DEFAULT_POTENTIAL_WEIGHT
    strained_threshold: float = 0.0
    critical_threshold: float = -10.0

    # Defense first, but only for savable urgent deficits.
    max_defense_launches_per_target: int = 3
    defense_margin: int = 1

    # Attack ranking and micro-planner.
    include_neutrals: bool = True
    include_enemies: bool = True
    include_comets: bool = False
    eta_horizon: float = 60.0
    denial_weight: float = 1.0
    influence_weight: float = 1.0
    production_pressure_weight: float = 1.0
    enemy_weakness_weight: float = 1.0
    enemy_empire_support_weight: float = 1.0
    prioritize_enemy_weakspots: bool = True
    max_attack_targets: int = 6
    micro: MicroPlannerConfig = MicroPlannerConfig()


def _env_omega(env_obs: Any) -> float:
    if isinstance(env_obs, dict):
        return float(env_obs.get("angular_velocity", 0.0) or 0.0)
    return float(getattr(env_obs, "angular_velocity", 0.0) or 0.0)


def _source_strip_cost(state: Optional[PlanetDefenseState]) -> float:
    if state is None:
        return 0.0
    if state.garrison <= 0:
        return 1.0
    safe_ratio = max(0.0, float(state.defense_surplus)) / max(1.0, float(state.garrison))
    return max(0.0, 1.0 - min(1.0, safe_ratio))


def _source_options(og, remaining: dict[int, int], defense_by_id: dict[int, PlanetDefenseState]) -> list[SourceOption]:
    out: list[SourceOption] = []
    for planet in og.planets:
        if int(planet.owner) != int(og.player):
            continue
        budget = int(remaining.get(int(planet.id), 0))
        if budget <= 0:
            continue
        out.append(SourceOption(
            source=planet,
            budget=budget,
            strip_cost=_source_strip_cost(defense_by_id.get(int(planet.id))),
        ))
    out.sort(key=lambda s: (-int(s.budget), float(s.strip_cost), int(s.source.id)))
    return out


def _with_remaining_budget(state: PlanetDefenseState, remaining: dict[int, int]) -> PlanetDefenseState:
    """Approximate post-launch defense state for source strip costing."""

    budget = int(remaining.get(int(state.planet_id), int(state.garrison)))
    spent = max(0, int(state.garrison) - budget)
    if spent <= 0:
        return state
    return replace(
        state,
        garrison=max(0, int(state.garrison) - spent),
        assets=float(state.assets) - float(spent),
        defense_surplus=float(state.defense_surplus) - float(spent),
    )


def _refresh_defense_budget_view(
    defense_by_id: dict[int, PlanetDefenseState],
    remaining: dict[int, int],
) -> dict[int, PlanetDefenseState]:
    return {
        pid: _with_remaining_budget(state, remaining)
        for pid, state in defense_by_id.items()
    }


def _apply_action_budget(action: list, remaining: dict[int, int]) -> None:
    sid = int(action[0])
    ships = int(action[2])
    remaining[sid] = max(0, int(remaining.get(sid, 0)) - ships)


def _plan_simple_defense(
    *,
    og,
    target_state: PlanetDefenseState,
    remaining: dict[int, int],
    omega: float,
    cfg: FinalAgentConfig,
    slots: int,
) -> list[list]:
    """Small all-or-nothing reinforcement wave for one critical planet."""

    target = og.planet_by_id.get(int(target_state.planet_id))
    if target is None or slots <= 0:
        return []
    need = max(0, int(math.ceil(-float(target_state.defense_surplus))) + int(cfg.defense_margin))
    if need <= 0:
        return []

    eta_limit = (
        float(target_state.min_incoming_eta)
        if target_state.min_incoming_eta is not None
        else float("inf")
    )
    candidates: list[tuple[float, int, object]] = []
    for source in og.planets:
        if int(source.owner) != int(og.player) or int(source.id) == int(target.id):
            continue
        budget = int(remaining.get(int(source.id), 0))
        if budget <= 0:
            continue
        launch = compute_launch(source, target, max(1, min(budget, need)), og.planets, omega)
        if launch is None:
            continue
        eta = float(launch["eta"])
        if eta > eta_limit:
            continue
        candidates.append((eta, -budget, source))
    candidates.sort(key=lambda row: (row[0], row[1]))

    actions: list[list] = []
    sent = 0
    for _eta, _neg_budget, source in candidates:
        if len(actions) >= int(slots) or sent >= need:
            break
        budget = int(remaining.get(int(source.id), 0))
        ships = min(budget, need - sent)
        if ships <= 0:
            continue
        launch = compute_launch(source, target, ships, og.planets, omega)
        if launch is None:
            continue
        action = [int(source.id), float(launch["angle"]), int(launch["ships"])]
        actions.append(action)
        sent += int(action[2])

    if sent < need:
        return []
    return actions


class FinalMicroAgent:
    """Contest-facing V1 agent with target-local attack simulation."""

    def __init__(self, cfg: FinalAgentConfig | None = None):
        self.cfg = cfg or FinalAgentConfig()

    def act(
        self,
        env_obs: Any,
        player: int = 0,
        *,
        current_turn: int = 0,
        max_turns: int = 500,
    ) -> list[list]:
        cfg = self.cfg
        omega = _env_omega(env_obs)
        og = build_orbit_graph(env_obs, player=int(player))
        attack_cfg = attack_config_from_env(
            env_obs,
            current_turn=int(current_turn),
            max_turns=int(max_turns),
            eta_horizon=float(cfg.eta_horizon),
            denial_weight=float(cfg.denial_weight),
            include_neutrals=bool(cfg.include_neutrals),
            include_enemies=bool(cfg.include_enemies),
            include_comets=bool(cfg.include_comets),
        )
        target_plan = plan_attack_targets(
            og,
            cfg=attack_cfg,
            horizon_cap=float(cfg.horizon_cap),
            potential_weight=float(cfg.potential_weight),
        )
        remaining = dict(target_plan.launch_budgets)
        defense_states = compute_owned_defense(
            og,
            horizon_cap=float(cfg.horizon_cap),
            potential_weight=float(cfg.potential_weight),
        )
        defense_by_id = {int(s.planet_id): s for s in defense_states}
        actions: list[list] = []

        # Phase 1: urgent all-or-nothing defense.
        urgent = [
            s for s in defense_states
            if s.status in (DefenseStatus.CRITICAL, DefenseStatus.STRAINED)
            and float(s.defense_surplus) <= float(cfg.strained_threshold)
            and int(s.planet_id) in defense_by_id
        ]
        urgent.sort(key=lambda s: (s.status != DefenseStatus.CRITICAL, float(s.defense_surplus)))
        for state in urgent:
            slots = int(cfg.max_launches) - len(actions)
            if slots <= 0:
                break
            picks = _plan_simple_defense(
                og=og,
                target_state=state,
                remaining=remaining,
                omega=omega,
                cfg=cfg,
                slots=min(slots, int(cfg.max_defense_launches_per_target)),
            )
            for action in picks:
                actions.append(action)
                _apply_action_budget(action, remaining)
                defense_by_id = _refresh_defense_budget_view(defense_by_id, remaining)

        # Phase 2: micro-planned stable captures.
        slots = int(cfg.max_launches) - len(actions)
        if slots <= 0:
            return actions
        ranked = rank_capture_targets(
            og,
            target_plan,
            attack_cfg=attack_cfg,
            influence_weight=float(cfg.influence_weight),
            production_pressure_weight=float(cfg.production_pressure_weight),
            denial_weight=float(cfg.denial_weight),
            weakness_weight=float(cfg.enemy_weakness_weight),
            empire_support_weight=float(cfg.enemy_empire_support_weight),
            prioritize_enemy_weakspots=bool(cfg.prioritize_enemy_weakspots),
        )
        for ranked_target in ranked[: max(0, int(cfg.max_attack_targets))]:
            if len(actions) >= int(cfg.max_launches):
                break
            target = og.planet_by_id.get(int(ranked_target.planet_id))
            if target is None:
                continue
            sources = _source_options(og, remaining, defense_by_id)
            if not sources:
                break
            local_cfg = MicroPlannerConfig(
                hold_horizon=cfg.micro.hold_horizon,
                beam_width=cfg.micro.beam_width,
                max_launches_per_target=min(
                    int(cfg.micro.max_launches_per_target),
                    int(cfg.max_launches) - len(actions),
                ),
                top_m_sources=cfg.micro.top_m_sources,
                min_send=cfg.micro.min_send,
                capture_margin=cfg.micro.capture_margin,
                value_weight=cfg.micro.value_weight,
                enemy_denial_weight=cfg.micro.enemy_denial_weight,
                ship_cost_weight=cfg.micro.ship_cost_weight,
                delay_weight=cfg.micro.delay_weight,
                source_strip_weight=cfg.micro.source_strip_weight,
                true_overkill_weight=cfg.micro.true_overkill_weight,
                enemy_gift_penalty=cfg.micro.enemy_gift_penalty,
                require_stable_hold=cfg.micro.require_stable_hold,
                min_plan_score=cfg.micro.min_plan_score,
            )
            plan = plan_target_attack(
                og=og,
                target=target,
                sources=sources,
                player=int(player),
                current_turn=int(current_turn),
                max_turns=int(max_turns),
                omega=omega,
                cfg=local_cfg,
            )
            if plan is None:
                continue
            for action in plan.actions:
                if len(actions) >= int(cfg.max_launches):
                    break
                actions.append(action)
                _apply_action_budget(action, remaining)
                defense_by_id = _refresh_defense_budget_view(defense_by_id, remaining)

        return actions
