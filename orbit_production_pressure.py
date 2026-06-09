"""Production **diffusion potential** on the orbit graph.

Each owned planet is a source of production that *diffuses* along reachable
routes; travel time damps the contribution (inverse-ETA kernel, same geometry
as ship influence but mass = ``production``).  This parallels
``model_graph_pyg``'s ``friendly_production_potential`` / the RL graph metrics.

**Potential** (garrisons could launch — static field)::

    Φ_diff(g) = Σ_{h → g}  prod_h / (min(η_{h→g}, H) + ε)

**Realized** (production already flowing via inbound fleets)::

    Φ_real(g) = Σ_{f → g}  prod_{src(f)} / (η_f + ε)

On an enemy node, high ``friendly_diffusion_potential`` means our productive
empire projects sustained economic reach there — even when ship-mass
influence is thin.

Legacy names ``*_production_pressure`` are kept as aliases on nodes and in
the public API.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import networkx as nx

from feature_extraction import Planet
from orbit_graph import FleetSide
from reworked_features import DEFAULT_THREAT_ETA_EPS

if TYPE_CHECKING:
    from orbit_graph import OrbitGraph

DEFAULT_DIFFUSION_HORIZON = 15.0
DEFAULT_DIFFUSION_NORM = 1.0
# Back-compat aliases.
DEFAULT_PRODUCTION_HORIZON = DEFAULT_DIFFUSION_HORIZON
DEFAULT_PRODUCTION_NORM = DEFAULT_DIFFUSION_NORM


def _inverse_eta_production_sum(
    G: nx.DiGraph,
    tgt_id: int,
    *,
    horizon: float,
    eta_eps: float,
    source_owner_predicate,
) -> float:
    """Sum production / inverse-ETA from qualifying sources to ``tgt_id``."""
    total = 0.0
    tid = int(tgt_id)
    h_cap = float(horizon)
    eps = float(eta_eps)
    for src_id, _v, data in G.in_edges(tid, data=True):
        if not data.get("reachable", False):
            continue
        node = G.nodes[int(src_id)]
        if not source_owner_predicate(int(node["owner"])):
            continue
        prod = float(node.get("production", 0))
        if prod <= 0.0:
            continue
        eta = float(data.get("eta", math.inf))
        if not math.isfinite(eta):
            continue
        eta_eff = min(eta, h_cap)
        total += prod / (eta_eff + eps)
    return total


def _write_node_diffusion_channels(
    G: nx.DiGraph,
    nid: int,
    *,
    pot_friendly: float,
    pot_enemy: float,
    real_friendly: float,
    real_enemy: float,
) -> None:
    """Store diffusion potentials (and legacy production_pressure keys)."""
    node = G.nodes[int(nid)]
    node["potential_friendly_diffusion"] = float(pot_friendly)
    node["realized_friendly_diffusion"] = float(real_friendly)
    node["potential_enemy_diffusion"] = float(pot_enemy)
    node["realized_enemy_diffusion"] = float(real_enemy)
    # Legacy aliases (same values).
    node["potential_friendly_production_pressure"] = float(pot_friendly)
    node["realized_friendly_production_pressure"] = float(real_friendly)
    node["potential_enemy_production_pressure"] = float(pot_enemy)
    node["realized_enemy_production_pressure"] = float(real_enemy)


def annotate_diffusion_potential_on_graph(
    G: nx.DiGraph,
    planets: list[Planet],
    player: int,
    *,
    horizon: float = DEFAULT_DIFFUSION_HORIZON,
    eta_eps: float = DEFAULT_THREAT_ETA_EPS,
    norm: float = DEFAULT_DIFFUSION_NORM,
) -> None:
    """Write four diffusion-potential channels on every node."""
    scale = max(float(norm), 1e-6)
    pid = int(player)

    def is_own(owner: int) -> bool:
        return int(owner) == pid

    def is_enemy(owner: int) -> bool:
        return int(owner) != pid and int(owner) != -1

    for p in planets:
        nid = int(p.id)
        pot_friendly = _inverse_eta_production_sum(
            G, nid, horizon=horizon, eta_eps=eta_eps,
            source_owner_predicate=is_own,
        )
        pot_enemy = _inverse_eta_production_sum(
            G, nid, horizon=horizon, eta_eps=eta_eps,
            source_owner_predicate=is_enemy,
        )

        real_friendly = 0.0
        real_enemy = 0.0
        for _src, tgt_id, data in G.in_edges(nid, data=True):
            if int(tgt_id) != nid:
                continue
            for rec in data.get("fleets", []):
                src_node = G.nodes[int(rec.src_id)]
                prod = float(src_node.get("production", 0))
                if prod <= 0.0:
                    continue
                term = prod / (float(rec.eta) + float(eta_eps))
                if rec.side == FleetSide.SELF:
                    real_friendly += term
                elif rec.side == FleetSide.ENEMY:
                    real_enemy += term

        _write_node_diffusion_channels(
            G, nid,
            pot_friendly=float(pot_friendly / scale),
            pot_enemy=float(pot_enemy / scale),
            real_friendly=float(real_friendly / scale),
            real_enemy=float(real_enemy / scale),
        )


# Back-compat name used by orbit_graph.build_orbit_graph.
annotate_production_pressure_on_graph = annotate_diffusion_potential_on_graph


def node_diffusion_potential(og: OrbitGraph, planet_id: int) -> dict[str, float]:
    """All four diffusion-potential channels at ``planet_id``."""
    data = og.G.nodes[int(planet_id)]
    return {
        "potential_friendly_diffusion": float(
            data.get("potential_friendly_diffusion", 0.0),
        ),
        "realized_friendly_diffusion": float(
            data.get("realized_friendly_diffusion", 0.0),
        ),
        "potential_enemy_diffusion": float(
            data.get("potential_enemy_diffusion", 0.0),
        ),
        "realized_enemy_diffusion": float(
            data.get("realized_enemy_diffusion", 0.0),
        ),
    }


node_production_pressure = node_diffusion_potential


def friendly_diffusion_potential_on_target(og: OrbitGraph, planet_id: int) -> float:
    """Our potential + realized production diffusion at ``planet_id``."""
    p = node_diffusion_potential(og, int(planet_id))
    return (
        float(p["potential_friendly_diffusion"])
        + float(p["realized_friendly_diffusion"])
    )


friendly_production_pressure_on_target = friendly_diffusion_potential_on_target


def enemy_diffusion_potential_on_target(og: OrbitGraph, planet_id: int) -> float:
    """Enemy potential + realized production diffusion at ``planet_id``."""
    p = node_diffusion_potential(og, int(planet_id))
    return (
        float(p["potential_enemy_diffusion"])
        + float(p["realized_enemy_diffusion"])
    )


enemy_production_pressure_on_target = enemy_diffusion_potential_on_target


def diffusion_potential_balance(og: OrbitGraph, planet_id: int) -> float:
    """Net diffusion potential (friendly minus enemy) at ``planet_id``."""
    return (
        friendly_diffusion_potential_on_target(og, planet_id)
        - enemy_diffusion_potential_on_target(og, planet_id)
    )


production_pressure_balance = diffusion_potential_balance


__all__ = [
    "DEFAULT_DIFFUSION_HORIZON",
    "DEFAULT_DIFFUSION_NORM",
    "DEFAULT_PRODUCTION_HORIZON",
    "DEFAULT_PRODUCTION_NORM",
    "annotate_diffusion_potential_on_graph",
    "annotate_production_pressure_on_graph",
    "diffusion_potential_balance",
    "enemy_diffusion_potential_on_target",
    "enemy_production_pressure_on_target",
    "friendly_diffusion_potential_on_target",
    "friendly_production_pressure_on_target",
    "node_diffusion_potential",
    "node_production_pressure",
    "production_pressure_balance",
]
