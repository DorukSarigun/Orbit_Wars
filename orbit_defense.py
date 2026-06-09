"""Per-turn defense threat and surplus for owned planets.

Two complementary threat channels (recomputed every ``build_orbit_graph`` call):

1. **Realized** — enemy fleets already in flight toward our planet (raw ship
   counts + flux-weighted sums within a defense horizon).
2. **Potential** — enemy garrisons that *could* reach us via **valid**
   trajectories only (``edge["reachable"] == True``; sun-blocked / planet-
   blocked legs are excluded).

Defense surplus compares local assets vs liabilities at a planet::

    H_max           = default planning cap (``DEFAULT_DEFENSE_HORIZON``)
    eta_enemy       = min incoming threat ETA, or ``H_max`` if none
    effective_horizon = eta_enemy when threats exist, else ``H_max``
    production_buffer = prod * min(H_max, eta_enemy)

    assets     = garrison + production_buffer + friendly_inbound_ships
    liabilities = realized_enemy_ships + potential_weight * potential_ships_eq
    surplus    = assets - liabilities

**Canonical potential kernel** (inverse-ETA, reachable legs only)::

    P(g) = sum_{h enemy, h->g reachable}  s_h / (min(eta(h,g), H_eff) + eps)

``eta(h,g)`` from :func:`navigation.compute_launch` at garnizon ``s_h``;
``eps = 1`` turn.  See :func:`inverse_eta_potential`.

Positive surplus ⇒ can spare ships; negative ⇒ underdefended.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from reworked_features import DEFAULT_THREAT_ETA_EPS
from orbit_graph import OrbitGraph

DEFAULT_DEFENSE_HORIZON = 15.0
DEFAULT_POTENTIAL_WEIGHT = 1.0


class DefenseStatus(str, Enum):
    SECURE = "secure"
    STRAINED = "strained"
    CRITICAL = "critical"


@dataclass(frozen=True)
class PlanetDefenseState:
    """Threat + surplus snapshot for one owned planet."""

    planet_id: int
    garrison: int
    production: int
    horizon: float
    horizon_cap: float
    eta_enemy: float

    # --- Realized (in-flight enemy fleets) ---
    realized_enemy_ships: float
    realized_enemy_flux: float
    incoming_threat_count: int
    min_incoming_eta: Optional[float]

    # --- Potential (reachable enemy garrisons) ---
    potential_enemy_threat: float
    potential_enemy_ships_eq: float

    # --- Friendly help already committed ---
    friendly_inbound_ships: float
    friendly_inbound_flux: float
    production_buffer: float

    # --- Surplus ---
    assets: float
    liabilities: float
    defense_surplus: float
    status: DefenseStatus


def _incoming_enemy_to_planet(
    og: OrbitGraph,
    planet_id: int,
    *,
    horizon_cap: float,
) -> tuple[float, float, int, Optional[float]]:
    """Raw ships, flux sum, count (within ``horizon_cap``), earliest ETA overall."""
    min_eta: Optional[float] = None
    for rec in og.incoming_threats(planet_id=planet_id):
        eta = float(rec.eta)
        min_eta = eta if min_eta is None else min(min_eta, eta)

    ships = 0.0
    flux = 0.0
    count = 0
    for rec in og.incoming_threats(planet_id=planet_id):
        eta = float(rec.eta)
        if eta > float(horizon_cap):
            continue
        ships += float(rec.fleet.ships)
        flux += float(rec.flux_weight)
        count += 1
    return ships, flux, count, min_eta


def _effective_horizon(
    min_incoming_eta: Optional[float],
    *,
    horizon_cap: float,
) -> tuple[float, float]:
    """Return ``(effective_horizon, eta_enemy)`` for surplus accounting.

    * ``effective_horizon`` = ``min_incoming_eta`` when threats exist, else
      ``horizon_cap`` (used for potential kernel + friendly inbound window).
    * ``eta_enemy`` = same as earliest threat ETA, or ``horizon_cap`` when
      none (so ``min(horizon_cap, eta_enemy)`` collapses correctly).
    """
    h_cap = float(horizon_cap)
    if min_incoming_eta is not None:
        eta = float(min_incoming_eta)
        return eta, eta
    return h_cap, h_cap


def _friendly_inbound_ships(
    og: OrbitGraph,
    planet_id: int,
    *,
    horizon: float,
) -> tuple[float, float]:
    ships = 0.0
    flux = 0.0
    for rec in og.friendly_reinforcements(planet_id=planet_id):
        if float(rec.eta) > float(horizon):
            continue
        ships += float(rec.fleet.ships)
        flux += float(rec.flux_weight)
    return ships, flux


def inverse_eta_potential(
    og: OrbitGraph,
    planet_id: int,
    *,
    horizon: float,
    eta_eps: float = DEFAULT_THREAT_ETA_EPS,
) -> float:
    r"""Inverse-ETA potential pressure on node ``planet_id`` (canonical kernel).

    .. math::

        P(g) = \sum_{h \in \text{enemy},\; h\to g\ \text{reachable}}
            \frac{s_h}{\min(\eta_{h\to g}, H_\text{eff}) + \varepsilon}

    * ``s_h`` — enemy garrison ship count at ``h``.
    * ``eta_{h\to g}`` — launch ETA from ``h`` using **that** garnizon's ships
      (speed baked in via :func:`reworked_features.fleet_speed`).
    * Unreachable legs (sun / blocking planet) contribute zero.
    """
    total = 0.0
    pid = int(planet_id)
    for src_id, tgt_id, data in og.G.in_edges(pid, data=True):
        if not data.get("reachable", False):
            continue
        src_node = og.G.nodes[int(src_id)]
        if int(src_node["owner"]) == int(og.player):
            continue
        if int(src_node["owner"]) == -1:
            continue
        eta = float(data.get("eta", math.inf))
        if not math.isfinite(eta):
            continue
        eta_eff = min(eta, float(horizon))
        ships_h = float(src_node.get("ships", 0))
        total += ships_h / (eta_eff + float(eta_eps))
    return total


# Back-compat alias.
potential_ships_equivalent = inverse_eta_potential


def compute_planet_defense(
    og: OrbitGraph,
    planet_id: int,
    *,
    horizon_cap: float = DEFAULT_DEFENSE_HORIZON,
    potential_weight: float = DEFAULT_POTENTIAL_WEIGHT,
    strained_threshold: float = 0.0,
    critical_threshold: float = -10.0,
) -> Optional[PlanetDefenseState]:
    """Build a defense snapshot for one planet (owned by ``og.player`` only)."""
    pid = int(planet_id)
    node = og.G.nodes[pid]
    if int(node["owner"]) != int(og.player):
        return None

    garrison = int(node.get("ships", 0))
    production = int(node.get("production", 0))
    pressure = og.node_pressure(pid)

    realized_ships, realized_flux, n_in, min_eta = _incoming_enemy_to_planet(
        og, pid, horizon_cap=horizon_cap,
    )
    h_eff, eta_enemy = _effective_horizon(min_eta, horizon_cap=horizon_cap)
    friendly_ships, friendly_flux = _friendly_inbound_ships(og, pid, horizon=h_eff)
    prod_buf = float(production) * min(float(horizon_cap), float(eta_enemy))
    pot_ships_eq = inverse_eta_potential(og, pid, horizon=h_eff)

    assets = float(garrison) + prod_buf + friendly_ships
    liabilities = realized_ships + float(potential_weight) * pot_ships_eq
    surplus = assets - liabilities

    if surplus <= critical_threshold:
        status = DefenseStatus.CRITICAL
    elif surplus <= strained_threshold:
        status = DefenseStatus.STRAINED
    else:
        status = DefenseStatus.SECURE

    return PlanetDefenseState(
        planet_id=pid,
        garrison=garrison,
        production=production,
        horizon=float(h_eff),
        horizon_cap=float(horizon_cap),
        eta_enemy=float(eta_enemy),
        realized_enemy_ships=realized_ships,
        realized_enemy_flux=realized_flux,
        incoming_threat_count=n_in,
        min_incoming_eta=min_eta,
        potential_enemy_threat=float(pressure["potential_enemy_threat"]),
        potential_enemy_ships_eq=pot_ships_eq,
        friendly_inbound_ships=friendly_ships,
        friendly_inbound_flux=friendly_flux,
        production_buffer=prod_buf,
        assets=assets,
        liabilities=liabilities,
        defense_surplus=surplus,
        status=status,
    )


def compute_owned_defense(
    og: OrbitGraph,
    *,
    horizon_cap: float = DEFAULT_DEFENSE_HORIZON,
    potential_weight: float = DEFAULT_POTENTIAL_WEIGHT,
) -> list[PlanetDefenseState]:
    """Defense snapshot for every planet owned by ``og.player``."""
    out: list[PlanetDefenseState] = []
    for p in og.planets:
        if p.owner != og.player or p.is_comet:
            continue
        state = compute_planet_defense(
            og, p.id, horizon_cap=horizon_cap, potential_weight=potential_weight,
        )
        if state is not None:
            out.append(state)
    out.sort(key=lambda s: s.defense_surplus)
    return out


@dataclass(frozen=True)
class LaunchBudget:
    """Ships a owned planet can launch after covering defense liabilities."""

    planet_id: int
    garrison: int
    defense_surplus: float
    launchable_ships: int


def launchable_from_surplus(state: PlanetDefenseState) -> int:
    """Convert defense surplus to an integral launch budget (capped by garrison)."""
    spare = max(0.0, float(state.defense_surplus))
    return max(0, min(int(state.garrison), int(math.floor(spare))))


def compute_launch_budgets(
    og: OrbitGraph,
    *,
    horizon_cap: float = DEFAULT_DEFENSE_HORIZON,
    potential_weight: float = DEFAULT_POTENTIAL_WEIGHT,
) -> dict[int, LaunchBudget]:
    """Per-owned-planet launch budget derived from defense surplus.

    ``launchable_ships = floor(max(0, defense_surplus))`` capped by current
    garrison.  Planets with negative surplus get zero.
    """
    out: dict[int, LaunchBudget] = {}
    for state in compute_owned_defense(
        og, horizon_cap=horizon_cap, potential_weight=potential_weight,
    ):
        launchable = launchable_from_surplus(state)
        out[int(state.planet_id)] = LaunchBudget(
            planet_id=int(state.planet_id),
            garrison=int(state.garrison),
            defense_surplus=float(state.defense_surplus),
            launchable_ships=int(launchable),
        )
    return out
