"""NetworkX graph for Orbit Wars scripted agents.

Topology
--------
* **Nodes** = planets (attrs mirror :class:`feature_extraction.Planet`).
* **Edges** = one directed edge per ordered planet pair ``(src, tgt)``, ``src != tgt``.
  Static route attrs (``eta``, ``distance``, ``crosses_sun``) describe potential
  travel; the ``fleets`` list holds every in-flight fleet predicted to travel
  along that directed leg.

Each fleet is stored as a :class:`FleetEdgeRecord` with explicit ``owner``,
``side`` (self / enemy / neutral relative to the viewing player), ``flux_weight``
(ETA-discounted ship mass), and a :class:`FleetMission` class.

Every node carries the same four pressure channels (viewer-relative), so
scripted logic can compare *our* reach vs *enemy* reach on **any** planet —
including enemy-owned targets when picking an attack window:

* ``potential_enemy_threat`` / ``realized_enemy_threat`` — enemy garrison
  pressure (inverse-ETA kernel, reachable legs only) vs inbound fleets.
* ``potential_friendly_support`` / ``realized_friendly_support`` — own
  garrison reinforcement capacity vs own fleets already inbound.
* ``potential_*_diffusion`` / ``realized_*_diffusion`` — production
  **diffusion potential** (inverse-ETA reach, production-weighted; see
  :mod:`orbit_production_pressure`).  Legacy ``*_production_pressure`` keys
  mirror the same values.

On an enemy-owned node, high ``realized_enemy_threat`` means enemy
reinforcements are arriving soon (bad time to attack); high
``realized_friendly_support`` means our attack fleets are already en route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

import networkx as nx
import numpy as np

from feature_extraction import (
    BOARD_SIZE,
    Fleet,
    Planet,
    parse_fleets,
    parse_planets,
)
from navigation import compute_launch, crosses_sun, predict_all_fleet_targets
from reworked_features import (
    DEFAULT_FLUX_SHIP_NORM,
    DEFAULT_FLUX_TAU,
    FLUX_MODE_EXP_DECAY,
    potential_support,
    potential_threat,
)

# Reference fleet size for static route ETA (matches model_graph_pyg).
REFERENCE_SHIPS_FOR_ETA = 50
DEFAULT_THREAT_NORM = 50.0


class FleetSide(str, Enum):
    """Fleet ownership relative to the viewing player."""

    SELF = "self"
    ENEMY = "enemy"
    NEUTRAL = "neutral"


class FleetMission(str, Enum):
    """Tactical role of an in-flight fleet (relative to viewing player)."""

    REINFORCEMENT = "reinforcement"
    ATTACK = "attack"
    INCOMING_THREAT = "incoming_threat"
    ENEMY_REINFORCEMENT = "enemy_reinforcement"
    ENEMY_EXPANSION = "enemy_expansion"
    NEUTRAL = "neutral"


def fleet_side(owner: int, player: int) -> FleetSide:
    if int(owner) == int(player):
        return FleetSide.SELF
    if int(owner) == -1:
        return FleetSide.NEUTRAL
    return FleetSide.ENEMY


def classify_fleet_mission(
    side: FleetSide,
    fleet_owner: int,
    tgt_owner: int,
    player: int,
) -> FleetMission:
    """Label fleet intent from ownership + predicted target owner."""
    if side == FleetSide.NEUTRAL:
        return FleetMission.NEUTRAL
    if side == FleetSide.SELF:
        if int(tgt_owner) == int(player):
            return FleetMission.REINFORCEMENT
        return FleetMission.ATTACK
    if int(tgt_owner) == int(player):
        return FleetMission.INCOMING_THREAT
    if int(tgt_owner) == int(fleet_owner):
        return FleetMission.ENEMY_REINFORCEMENT
    return FleetMission.ENEMY_EXPANSION


def fleet_flux_weight(
    ships: int,
    eta: float,
    *,
    tau: float = DEFAULT_FLUX_TAU,
    ship_norm: float = DEFAULT_FLUX_SHIP_NORM,
    mode: str = FLUX_MODE_EXP_DECAY,
) -> float:
    """ETA-discounted ship mass for one fleet (``exp_decay`` by default)."""
    s = max(float(ships), 0.0)
    e = max(float(eta), 0.0)
    if mode == FLUX_MODE_EXP_DECAY:
        w = s * math.exp(-e / max(float(tau), 1e-6))
    else:
        w = s
    return w / max(float(ship_norm), 1e-6)


@dataclass(frozen=True)
class FleetEdgeRecord:
    """One in-flight fleet attached to a directed planet-pair edge."""

    fleet: Fleet
    owner: int
    side: FleetSide
    mission: FleetMission
    src_id: int
    tgt_id: int
    eta: float
    flux_weight: float


@dataclass
class OrbitGraph:
    """Planet graph with per-pair route edges and fleet lists."""

    G: nx.DiGraph
    player: int
    planets: list[Planet]
    planet_by_id: dict[int, Planet]

    def fleets_on_edge(self, src_id: int, tgt_id: int) -> list[FleetEdgeRecord]:
        if not self.G.has_edge(int(src_id), int(tgt_id)):
            return []
        return list(self.G.edges[int(src_id), int(tgt_id)].get("fleets", []))

    def fleets_to(self, tgt_id: int) -> list[FleetEdgeRecord]:
        tid = int(tgt_id)
        out: list[FleetEdgeRecord] = []
        for _u, v, data in self.G.in_edges(tid, data=True):
            out.extend(data.get("fleets", []))
        return out

    def fleets_from(self, src_id: int) -> list[FleetEdgeRecord]:
        sid = int(src_id)
        out: list[FleetEdgeRecord] = []
        for v, _u, data in self.G.out_edges(sid, data=True):
            out.extend(data.get("fleets", []))
        return out

    def fleets_by_side(
        self,
        *,
        side: FleetSide,
        tgt_id: Optional[int] = None,
    ) -> list[FleetEdgeRecord]:
        if tgt_id is None:
            records = [r for _u, _v, d in self.G.edges(data=True) for r in d.get("fleets", [])]
        else:
            records = self.fleets_to(tgt_id)
        return [r for r in records if r.side == side]

    def node_pressure(self, planet_id: int) -> dict[str, float]:
        """All four pressure channels at ``planet_id``."""
        data = self.G.nodes[int(planet_id)]
        return {
            "potential_enemy_threat": float(data.get("potential_enemy_threat", 0.0)),
            "realized_enemy_threat": float(data.get("realized_enemy_threat", 0.0)),
            "potential_friendly_support": float(data.get("potential_friendly_support", 0.0)),
            "realized_friendly_support": float(data.get("realized_friendly_support", 0.0)),
        }

    def potential_enemy_threat(self, planet_id: int) -> float:
        return self.node_pressure(planet_id)["potential_enemy_threat"]

    def realized_enemy_threat(self, planet_id: int) -> float:
        return self.node_pressure(planet_id)["realized_enemy_threat"]

    def potential_friendly_support(self, planet_id: int) -> float:
        return self.node_pressure(planet_id)["potential_friendly_support"]

    def realized_friendly_support(self, planet_id: int) -> float:
        return self.node_pressure(planet_id)["realized_friendly_support"]

    def friendly_diffusion_potential(self, planet_id: int) -> float:
        """Own production diffusion potential (Φ_pot + Φ_real) on a node."""
        from orbit_production_pressure import friendly_diffusion_potential_on_target

        return friendly_diffusion_potential_on_target(self, int(planet_id))

    def diffusion_potential_balance(self, planet_id: int) -> float:
        """Friendly minus enemy production diffusion at a node."""
        from orbit_production_pressure import diffusion_potential_balance

        return diffusion_potential_balance(self, int(planet_id))

    def friendly_production_pressure(self, planet_id: int) -> float:
        """Alias for :meth:`friendly_diffusion_potential`."""
        return self.friendly_diffusion_potential(int(planet_id))

    def production_pressure_balance(self, planet_id: int) -> float:
        """Alias for :meth:`diffusion_potential_balance`."""
        return self.diffusion_potential_balance(int(planet_id))

    def incoming_threats(self, planet_id: Optional[int] = None) -> list[FleetEdgeRecord]:
        records = self.fleets_by_side(side=FleetSide.ENEMY, tgt_id=planet_id)
        return [r for r in records if r.mission == FleetMission.INCOMING_THREAT]

    def friendly_reinforcements(
        self, planet_id: Optional[int] = None,
    ) -> list[FleetEdgeRecord]:
        records = self.fleets_by_side(side=FleetSide.SELF, tgt_id=planet_id)
        return [r for r in records if r.mission == FleetMission.REINFORCEMENT]

    def friendly_attacks_inbound(
        self, planet_id: Optional[int] = None,
    ) -> list[FleetEdgeRecord]:
        records = self.fleets_by_side(side=FleetSide.SELF, tgt_id=planet_id)
        return [r for r in records if r.mission == FleetMission.ATTACK]

    def enemy_reinforcements_inbound(
        self, planet_id: Optional[int] = None,
    ) -> list[FleetEdgeRecord]:
        records = self.fleets_by_side(side=FleetSide.ENEMY, tgt_id=planet_id)
        return [r for r in records if r.mission == FleetMission.ENEMY_REINFORCEMENT]

    def attack_window_score(self, planet_id: int) -> float:
        r"""Favourable attack moment on ``planet_id`` (higher = strike now).

        Net own pressure minus enemy pressure, potential plus realized::

            (pot_friendly + real_friendly) - (pot_enemy + real_enemy)

        Use on enemy / neutral nodes: low enemy ``realized`` reinforcement
        inbound and high own ``potential`` support ⇒ wide window.
        """
        p = self.node_pressure(planet_id)
        friendly = p["potential_friendly_support"] + p["realized_friendly_support"]
        enemy = p["potential_enemy_threat"] + p["realized_enemy_threat"]
        return friendly - enemy


def _env_omega(env_obs: Any) -> float:
    if isinstance(env_obs, dict):
        return float(env_obs.get("angular_velocity", 0.0) or 0.0)
    return float(getattr(env_obs, "angular_velocity", 0.0) or 0.0)


def _planet_distance(src: Planet, tgt: Planet) -> float:
    return math.hypot(src.x - tgt.x, src.y - tgt.y) / BOARD_SIZE


def _route_eta(
    src: Planet,
    tgt: Planet,
    planets: list[Planet],
    omega: float,
    *,
    reference_ships: int = REFERENCE_SHIPS_FOR_ETA,
) -> tuple[float, bool, bool]:
    """Return ``(eta_turns, crosses_sun_flag, reachable)`` for a directed leg.

  ``reachable`` is ``True`` only when :func:`navigation.compute_launch` finds a
    valid trajectory (no sun collision, no blocking planet).  Unreachable pairs
    get ``eta = inf`` and are excluded from potential-pressure sums.
    """
    if int(src.id) == int(tgt.id):
        return math.inf, False, False
    sun_xing = crosses_sun(src.x, src.y, tgt.x, tgt.y)
    launch = compute_launch(src, tgt, int(reference_ships), planets, omega)
    if launch is not None:
        return float(launch["eta"]), sun_xing, True
    return math.inf, sun_xing, False


def _attach_fleets(
    G: nx.DiGraph,
    fleets: Iterable[Fleet],
    planets: list[Planet],
    omega: float,
    player: int,
    planet_by_id: dict[int, Planet],
    *,
    flux_tau: float = DEFAULT_FLUX_TAU,
    flux_ship_norm: float = DEFAULT_FLUX_SHIP_NORM,
) -> None:
    if not fleets:
        return
    preds = predict_all_fleet_targets(fleets, planets, omega)
    for flt in fleets:
        pred = preds.get(int(flt.id))
        if pred is None:
            continue
        src_id = int(flt.from_planet_id)
        tgt_id = int(pred["target_id"])
        if src_id == tgt_id or not G.has_edge(src_id, tgt_id):
            continue
        if not G.edges[src_id, tgt_id].get("reachable", False):
            continue
        side = fleet_side(flt.owner, player)
        tgt = planet_by_id.get(tgt_id)
        tgt_owner = int(tgt.owner) if tgt is not None else -1
        eta = float(pred["eta"])
        rec = FleetEdgeRecord(
            fleet=flt,
            owner=int(flt.owner),
            side=side,
            mission=classify_fleet_mission(side, int(flt.owner), tgt_owner, player),
            src_id=src_id,
            tgt_id=tgt_id,
            eta=eta,
            flux_weight=fleet_flux_weight(
                flt.ships, eta, tau=flux_tau, ship_norm=flux_ship_norm,
            ),
        )
        G.edges[src_id, tgt_id]["fleets"].append(rec)


def _annotate_edge_flux_aggregates(G: nx.DiGraph) -> None:
    """Sum per-fleet ``flux_weight`` into edge-level friendly / enemy flux."""
    for _u, _v, data in G.edges(data=True):
        friendly = 0.0
        enemy = 0.0
        for rec in data.get("fleets", []):
            if rec.side == FleetSide.SELF:
                friendly += rec.flux_weight
            elif rec.side == FleetSide.ENEMY:
                enemy += rec.flux_weight
        data["friendly_flux"] = friendly
        data["enemy_flux"] = enemy
        data["realized_friendly_support"] = friendly
        data["realized_enemy_threat"] = enemy


def _distance_matrix(G: nx.DiGraph, ids: list[int], *, reachable_only: bool = True) -> np.ndarray:
    """Pairwise distances for pressure kernels.

    When ``reachable_only`` is ``True`` (default), unreachable directed legs
    stay at ``inf`` so they contribute zero threat/support.
    """
    n = len(ids)
    dist = np.full((n, n), np.inf, dtype=np.float32)
    for i, src_id in enumerate(ids):
        for j, tgt_id in enumerate(ids):
            if i == j:
                continue
            if not G.has_edge(src_id, tgt_id):
                continue
            data = G.edges[src_id, tgt_id]
            if reachable_only and not data.get("reachable", False):
                continue
            if not math.isfinite(float(data.get("eta", math.inf))):
                continue
            dist[i, j] = float(data["distance"])
    return dist


def _annotate_node_pressure_channels(
    G: nx.DiGraph,
    planets: list[Planet],
    player: int,
    *,
    support_norm: float = DEFAULT_THREAT_NORM,
) -> None:
    """Write all four potential / realized pressure channels on every node."""
    n = len(planets)
    if n <= 0:
        return

    ids = [int(p.id) for p in planets]
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    dist = _distance_matrix(G, ids)

    owner = np.array([int(p.owner) for p in planets], dtype=np.int64)
    ships = np.array([int(p.ships) for p in planets], dtype=np.float64)

    pot_enemy = potential_threat(
        dist, owner, ships, player=int(player), norm=support_norm,
    )
    pot_friendly = potential_support(
        dist, owner, ships, player=int(player), norm=support_norm,
    )

    real_enemy = np.zeros(n, dtype=np.float32)
    real_friendly = np.zeros(n, dtype=np.float32)
    for _src, tgt_id, data in G.in_edges(data=True):
        j = id_to_idx[int(tgt_id)]
        for rec in data.get("fleets", []):
            if rec.side == FleetSide.ENEMY:
                real_enemy[j] += rec.flux_weight
            elif rec.side == FleetSide.SELF:
                real_friendly[j] += rec.flux_weight

    for i, pid in enumerate(ids):
        G.nodes[pid]["potential_enemy_threat"] = float(pot_enemy[i])
        G.nodes[pid]["realized_enemy_threat"] = float(real_enemy[i])
        G.nodes[pid]["potential_friendly_support"] = float(pot_friendly[i])
        G.nodes[pid]["realized_friendly_support"] = float(real_friendly[i])


def build_orbit_graph(
    env_obs: Any,
    player: int,
    *,
    angular_velocity: Optional[float] = None,
    reference_ships: int = REFERENCE_SHIPS_FOR_ETA,
    flux_tau: float = DEFAULT_FLUX_TAU,
    flux_ship_norm: float = DEFAULT_FLUX_SHIP_NORM,
    threat_norm: float = DEFAULT_THREAT_NORM,
) -> OrbitGraph:
    """Build a fully-connected directed planet graph from a raw env observation.

    Parameters
    ----------
    env_obs:
        Kaggle ``orbit_wars`` observation dict (or compatible mapping).
    player:
        Viewing player id; fleet ``side`` is computed relative to this seat.
    angular_velocity:
        Board rotation rate.  Defaults to ``env_obs["angular_velocity"]``.
    reference_ships:
        Fleet size used when estimating static ``eta`` on route edges.
    flux_tau, flux_ship_norm:
        Parametric weighting for per-fleet ``flux_weight``.
    threat_norm:
        Normaliser for garrison-based potential pressure channels.
    """
    omega = float(angular_velocity if angular_velocity is not None else _env_omega(env_obs))
    planets = parse_planets(env_obs)
    planet_by_id = {int(p.id): p for p in planets}
    fleets = parse_fleets(env_obs)

    G = nx.DiGraph()
    for p in planets:
        G.add_node(
            int(p.id),
            planet=p,
            owner=int(p.owner),
            ships=int(p.ships),
            production=int(p.production),
            x=float(p.x),
            y=float(p.y),
            radius=float(p.radius),
            is_comet=bool(p.is_comet),
            is_rotating=bool(p.is_rotating),
        )

    for src in planets:
        for tgt in planets:
            if int(src.id) == int(tgt.id):
                continue
            eta, sun_xing, reachable = _route_eta(
                src, tgt, planets, omega, reference_ships=reference_ships,
            )
            G.add_edge(
                int(src.id),
                int(tgt.id),
                eta=eta,
                distance=_planet_distance(src, tgt),
                crosses_sun=bool(sun_xing),
                reachable=bool(reachable),
                fleets=[],
            )

    _attach_fleets(
        G, fleets, planets, omega, int(player), planet_by_id,
        flux_tau=flux_tau, flux_ship_norm=flux_ship_norm,
    )
    _annotate_edge_flux_aggregates(G)
    _annotate_node_pressure_channels(
        G, planets, int(player), support_norm=threat_norm,
    )
    from orbit_production_pressure import annotate_production_pressure_on_graph

    annotate_production_pressure_on_graph(
        G, planets, int(player), norm=threat_norm,
    )

    return OrbitGraph(
        G=G,
        player=int(player),
        planets=planets,
        planet_by_id=planet_by_id,
    )
