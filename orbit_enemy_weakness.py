"""Enemy weak-spot metrics for attack planning.

A weak enemy planet is one where **local defense is thin** and the **enemy
empire cannot easily reinforce** it (low ship-mass support and low production
diffusion from other enemy worlds), while **our** pressure already reaches
there favourably.

Use these scores only for enemy captures — neutral expansion keeps the
production-efficiency path in :mod:`orbit_attack_allocate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from orbit_attack_targets import EnemyTargetMetrics
from orbit_expansion import estimated_defense_at_arrival
from orbit_production_pressure import (
    enemy_diffusion_potential_on_target,
    friendly_diffusion_potential_on_target,
)

if TYPE_CHECKING:
    from orbit_graph import OrbitGraph


@dataclass(frozen=True)
class EnemyWeakSpotMetrics:
    """Weakness snapshot for one enemy-owned planet."""

    planet_id: int
    production: int
    garrison: int
    local_defense: float
    empire_ship_support: float
    empire_diffusion_support: float
    friendly_influence: float
    friendly_diffusion: float
    attack_window: float
    reinf_inbound: float
    fragility: float
    exposure: float
    weakness_score: float

    @property
    def sort_key(self) -> tuple[float, ...]:
        return (self.weakness_score, self.fragility, self.exposure, self.production)


def _enemy_pressure_on_node(og: OrbitGraph, planet_id: int) -> tuple[float, float]:
    """Enemy ship support and our ship pressure at ``planet_id``."""
    p = og.node_pressure(int(planet_id))
    enemy = float(p["potential_enemy_threat"]) + float(p["realized_enemy_threat"])
    friendly = float(p["potential_friendly_support"]) + float(p["realized_friendly_support"])
    return enemy, friendly


def compute_enemy_weak_spot(
    og: OrbitGraph,
    planet_id: int,
    *,
    capture_eta: float = 0.0,
    reinf_inbound: float = 0.0,
    local_defense_weight: float = 1.0,
    empire_support_weight: float = 1.0,
    exposure_weight: float = 1.0,
    value_weight: float = 1.0,
) -> Optional[EnemyWeakSpotMetrics]:
    """Score how attractive ``planet_id`` is as a weak-point strike."""
    p = og.planet_by_id.get(int(planet_id))
    if p is None or not p.is_enemy(og.player):
        return None

    pid = int(p.id)
    eta = max(0.0, float(capture_eta))
    garrison = int(p.ships)
    prod = int(p.production)

    defense_at_eta = float(
        estimated_defense_at_arrival(p, eta, og.player, og),
    )
    local_defense = defense_at_eta + float(reinf_inbound)

    empire_ship, friendly_infl = _enemy_pressure_on_node(og, pid)
    empire_diff = enemy_diffusion_potential_on_target(og, pid)
    friendly_diff = friendly_diffusion_potential_on_target(og, pid)
    window = float(og.attack_window_score(pid))

    empire_support = (
        float(local_defense_weight) * float(local_defense)
        + float(empire_support_weight) * (float(empire_ship) + float(empire_diff))
    )
    fragility = float(prod) / (1.0 + max(empire_support, 0.0))
    exposure = float(exposure_weight) * (
        max(0.0, window) + float(friendly_infl) + float(friendly_diff)
    )
    weakness_score = float(value_weight) * fragility * (1.0 + exposure)

    return EnemyWeakSpotMetrics(
        planet_id=pid,
        production=prod,
        garrison=garrison,
        local_defense=float(local_defense),
        empire_ship_support=float(empire_ship),
        empire_diffusion_support=float(empire_diff),
        friendly_influence=float(friendly_infl),
        friendly_diffusion=float(friendly_diff),
        attack_window=window,
        reinf_inbound=float(reinf_inbound),
        fragility=float(fragility),
        exposure=float(exposure),
        weakness_score=float(weakness_score),
    )


def enemy_weak_spot_priority(
    og: OrbitGraph,
    metrics: EnemyTargetMetrics,
    *,
    local_defense_weight: float = 1.0,
    empire_support_weight: float = 1.0,
    exposure_weight: float = 1.0,
    value_weight: float = 1.0,
    efficiency_weight: float = 0.5,
    weakness_weight: float = 1.0,
) -> float:
    """Rank key for enemy targets: strike fragile, exposed worlds first."""
    spot = compute_enemy_weak_spot(
        og,
        int(metrics.planet_id),
        capture_eta=float(metrics.capture_eta),
        reinf_inbound=float(metrics.enemy_reinforcement_ships),
        local_defense_weight=local_defense_weight,
        empire_support_weight=empire_support_weight,
        exposure_weight=exposure_weight,
        value_weight=value_weight,
    )
    if spot is None:
        return 0.0
    core = float(weakness_weight) * float(spot.weakness_score)
    eff = float(efficiency_weight) * float(metrics.efficiency)
    denial = float(metrics.combined_value) / max(float(metrics.ships_lost), 1.0)
    return core + eff + denial * 0.25


__all__ = [
    "EnemyWeakSpotMetrics",
    "compute_enemy_weak_spot",
    "enemy_weak_spot_priority",
]
