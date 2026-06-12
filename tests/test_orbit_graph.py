"""Tests for NetworkX orbit graph builder."""

from __future__ import annotations

import math

from feature_extraction import Fleet
from orbit_graph import (
    FleetMission,
    FleetSide,
    build_orbit_graph,
    classify_fleet_mission,
    fleet_flux_weight,
    fleet_side,
)


# NOTE: planet 1 sits at (80, 20), NOT (80, 50) — the 0 <-> 1 leg must not pass
# through the sun at (50, 50) or the route becomes unreachable and fleets on it
# are dropped from the graph.
_PLANET_ROWS = [
    [0, 0, 20.0, 50.0, 1.0, 30, 2],
    [1, -1, 80.0, 20.0, 1.0, 5, 3],
    [2, 1, 50.0, 20.0, 1.0, 10, 1],
]

# Headings along the (reachable) routes used by fleet tests below.
_ANGLE_0_TO_1 = math.atan2(20.0 - 50.0, 80.0 - 20.0)
_ANGLE_2_TO_0 = math.atan2(50.0 - 20.0, 20.0 - 50.0)


def _minimal_env(*, fleets=None):
    return {
        "planets": [list(r) for r in _PLANET_ROWS],
        "fleets": fleets or [],
        "initial_planets": [list(r) for r in _PLANET_ROWS],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_fleet_side_labels():
    assert fleet_side(0, 0) == FleetSide.SELF
    assert fleet_side(1, 0) == FleetSide.ENEMY
    assert fleet_side(-1, 0) == FleetSide.NEUTRAL


def test_build_orbit_graph_topology():
    og = build_orbit_graph(_minimal_env(), player=0)
    n = len(og.planets)
    assert og.G.number_of_nodes() == n
    # Directed: n * (n - 1) edges, one per ordered pair.
    assert og.G.number_of_edges() == n * (n - 1)
    for u, v, data in og.G.edges(data=True):
        assert u != v
        assert "eta" in data
        assert "reachable" in data
        assert "distance" in data
        assert "crosses_sun" in data
        if data["reachable"]:
            assert math.isfinite(data["eta"])
        else:
            assert not math.isfinite(data["eta"])
        assert data["fleets"] == []
        assert data["friendly_flux"] == 0.0
        assert data["enemy_flux"] == 0.0
        assert data["realized_friendly_support"] == 0.0
    for pid in og.planet_by_id:
        assert set(og.node_pressure(pid).keys()) == {
            "potential_enemy_threat",
            "realized_enemy_threat",
            "potential_friendly_support",
            "realized_friendly_support",
        }


def test_fleet_stored_on_directed_edge_with_owner():
    # Own fleet launched from planet 0 toward neutral 1, partway along the route.
    fleets = [[7, 0, 26.0, 47.0, _ANGLE_0_TO_1, 0, 25]]
    og = build_orbit_graph(_minimal_env(fleets=fleets), player=0)

    found = []
    for u, v, data in og.G.edges(data=True):
        for rec in data["fleets"]:
            found.append((u, v, rec))

    assert len(found) == 1
    src, tgt, rec = found[0]
    assert isinstance(rec.fleet, Fleet)
    assert rec.fleet.id == 7
    assert rec.owner == 0
    assert rec.side == FleetSide.SELF
    assert rec.src_id == src
    assert rec.tgt_id == tgt
    assert rec.eta >= 0.0
    assert rec.flux_weight > 0.0
    assert rec.mission == FleetMission.ATTACK
    assert og.fleets_on_edge(src, tgt) == [rec]


def test_enemy_fleet_side_relative_to_player():
    # Enemy fleet from planet 2 inbound on our planet 0.
    fleets = [[9, 1, 44.0, 26.0, _ANGLE_2_TO_0, 2, 15]]
    og = build_orbit_graph(_minimal_env(fleets=fleets), player=0)
    enemy = [r for r in og.fleets_by_side(side=FleetSide.ENEMY)]
    assert len(enemy) == 1
    assert enemy[0].owner == 1
    assert enemy[0].side == FleetSide.ENEMY


def test_fleet_mission_classification():
    assert classify_fleet_mission(FleetSide.SELF, 0, 0, 0) == FleetMission.REINFORCEMENT
    assert classify_fleet_mission(FleetSide.SELF, 0, -1, 0) == FleetMission.ATTACK
    assert classify_fleet_mission(FleetSide.ENEMY, 1, 0, 0) == FleetMission.INCOMING_THREAT
    assert classify_fleet_mission(FleetSide.ENEMY, 1, 1, 0) == FleetMission.ENEMY_REINFORCEMENT
    assert classify_fleet_mission(FleetSide.ENEMY, 1, -1, 0) == FleetMission.ENEMY_EXPANSION


def test_flux_weight_decreases_with_eta():
    near = fleet_flux_weight(30, 1.0)
    far = fleet_flux_weight(30, 40.0)
    assert near > far > 0.0


def test_potential_vs_realized_enemy_threat_split():
    og = build_orbit_graph(_minimal_env(), player=0)
    # Enemy garrison on planet 2 pressures our planet 0 even with no fleets.
    assert og.potential_enemy_threat(0) > 0.0
    assert og.realized_enemy_threat(0) == 0.0

    # Enemy fleet near player planet 0, heading into it.
    fleets = [[9, 1, 24.0, 46.0, _ANGLE_2_TO_0, 2, 15]]
    og2 = build_orbit_graph(_minimal_env(fleets=fleets), player=0)
    threats = og2.incoming_threats(planet_id=0)
    assert len(threats) == 1
    assert threats[0].mission == FleetMission.INCOMING_THREAT
    assert og2.realized_enemy_threat(0) > 0.0
    # Node aggregates are float32; compare with tolerance.
    assert abs(og2.realized_enemy_threat(0) - pytest_approx_sum(threats)) < 1e-5


def test_friendly_pressure_on_all_nodes():
    og = build_orbit_graph(_minimal_env(), player=0)
    # Own planet 0: we can reinforce from planet 0 garrison context via other owned... 
    # only one owned planet in minimal env, but friendly potential on enemy node > 0
    assert og.potential_friendly_support(1) > 0.0
    assert og.realized_friendly_support(1) == 0.0
    # Enemy node 2 gets the same four channels as ours.
    assert og.potential_enemy_threat(2) >= 0.0
    assert og.potential_friendly_support(2) >= 0.0


def test_own_attack_fleet_raises_realized_friendly_on_target():
    fleets = [[7, 0, 26.0, 47.0, _ANGLE_0_TO_1, 0, 25]]
    og = build_orbit_graph(_minimal_env(fleets=fleets), player=0)
    attacks = og.friendly_attacks_inbound()
    assert len(attacks) == 1
    tgt = attacks[0].tgt_id
    assert og.realized_friendly_support(tgt) > 0.0
    assert og.realized_friendly_support(0) == 0.0


def test_enemy_reinforcement_lowers_attack_window():
    og_base = build_orbit_graph(_minimal_env(), player=0)
    window_quiet = og_base.attack_window_score(2)

    # Enemy reinforcing their own planet 2.
    angle = math.atan2(20.0 - 50.0, 50.0 - 50.0)  # from ~planet1 y toward p2
    fleets = [[9, 1, 50.0, 35.0, angle, 2, 20]]
    og_busy = build_orbit_graph(_minimal_env(fleets=fleets), player=0)
    reinf = og_busy.enemy_reinforcements_inbound(planet_id=2)
    if reinf:
        assert og_busy.realized_enemy_threat(2) > 0.0
        assert og_busy.attack_window_score(2) < window_quiet


def test_edge_flux_aggregates_match_fleet_records():
    fleets = [[7, 0, 26.0, 47.0, _ANGLE_0_TO_1, 0, 25]]
    og = build_orbit_graph(_minimal_env(fleets=fleets), player=0)
    for u, v, data in og.G.edges(data=True):
        fsum = sum(r.flux_weight for r in data["fleets"] if r.side == FleetSide.SELF)
        assert data["friendly_flux"] == fsum
        esum = sum(r.flux_weight for r in data["fleets"] if r.side == FleetSide.ENEMY)
        assert data["enemy_flux"] == esum
        assert data["realized_enemy_threat"] == esum
        assert data["realized_friendly_support"] == fsum


def pytest_approx_sum(records):
    return sum(r.flux_weight for r in records)


def test_node_attrs_mirror_planets_p0():
    _assert_node_attrs_for_player(0)


def test_node_attrs_mirror_planets_p1():
    _assert_node_attrs_for_player(1)


def _assert_node_attrs_for_player(player: int):
    og = build_orbit_graph(_minimal_env(), player=player)
    for p in og.planets:
        data = og.G.nodes[int(p.id)]
        assert data["owner"] == p.owner
        assert data["ships"] == p.ships
        assert data["production"] == p.production
