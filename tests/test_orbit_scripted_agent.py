"""Tests for orbit graph scripted agent act()."""

from __future__ import annotations

from orbit_attack_targets import plan_attack_targets, AttackTargetConfig
from orbit_defense import compute_owned_defense
from orbit_graph import build_orbit_graph
from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig, support_need


def _env_vertical():
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 40, 2],
            [1, -1, 20.0, 70.0, 1.0, 5, 3],
            [2, 1, 80.0, 20.0, 1.0, 3, 1],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 40, 2],
            [1, -1, 20.0, 70.0, 1.0, 5, 3],
            [2, 1, 80.0, 20.0, 1.0, 3, 1],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def _env_support():
    """Backline (0) + threatened frontline (1) + reachable enemy garrison (2)."""
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 50, 2],
            [1, 0, 20.0, 50.0, 1.0, 3, 2],
            [2, 1, 20.0, 80.0, 1.0, 80, 3],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 50, 2],
            [1, 0, 20.0, 50.0, 1.0, 3, 2],
            [2, 1, 20.0, 80.0, 1.0, 80, 3],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_act_attacks_best_neutral():
    env = _env_vertical()
    agent = OrbitScriptedAgent()
    actions = agent.act(env, player=0, current_turn=10, max_turns=250)
    assert actions
    src, angle, ships = actions[-1]
    og = build_orbit_graph(env, player=0)
    plan = plan_attack_targets(
        og, cfg=AttackTargetConfig(current_turn=10, max_turns=250),
    )
    assert plan.best_neutral is not None
    assert src == plan.best_neutral.best_src_id
    assert ships >= plan.best_neutral.ships_required
    assert isinstance(angle, float)


def test_act_supports_lp_attack_allocator():
    env = _env_vertical()
    cfg = OrbitScriptedConfig(attack_allocator="lp", lp_top_k_targets=2)
    agent = OrbitScriptedAgent(cfg=cfg)
    actions = agent.act(env, player=0, current_turn=10, max_turns=250)
    assert actions
    assert all(len(a) == 3 for a in actions)


def test_support_need_rises_with_enemy_influence():
    og = build_orbit_graph(_env_support(), player=0)
    states = {s.planet_id: s for s in compute_owned_defense(og)}
    cfg = OrbitScriptedConfig(support_threat_threshold=5.0)
    front = states[1]
    back = states[0]
    assert support_need(back, cfg) == 0
    assert support_need(front, cfg) > 0


def test_act_sends_backline_support_when_frontline_threatened():
    env = _env_support()
    cfg = OrbitScriptedConfig(
        support_threat_threshold=5.0,
        support_min_source_surplus=3.0,
        max_launches=8,
        include_neutrals=False,
        include_enemies=False,
    )
    agent = OrbitScriptedAgent(cfg=cfg)
    actions = agent.act(env, player=0, current_turn=5, max_turns=250)
    assert actions
    reinforcements = [a for a in actions if int(a[0]) == 0 and int(a[2]) > 0]
    assert reinforcements, "expected backline planet 0 to reinforce frontline"


def test_act_supports_lp_reinforcement_allocator():
    env = _env_support()
    cfg = OrbitScriptedConfig(
        reinforcement_allocator="lp",
        support_threat_threshold=5.0,
        support_min_source_surplus=3.0,
        max_launches=8,
        include_neutrals=False,
        include_enemies=False,
    )
    agent = OrbitScriptedAgent(cfg=cfg)
    actions = agent.act(env, player=0, current_turn=5, max_turns=250)
    assert actions
    reinforcements = [a for a in actions if int(a[0]) == 0 and int(a[2]) > 0]
    assert reinforcements, "expected LP reinforcement from backline planet 0"


def _mk_env(planets, fleets=None):
    return {
        "planets": planets,
        "fleets": fleets or [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_no_partial_wave_on_unaffordable_neutral():
    """All-or-nothing: budget 10 vs neutral garrison 50 → zero launches."""
    env = _mk_env([
        [0, 0, 20.0, 20.0, 1.0, 10, 1],
        [1, -1, 20.0, 70.0, 1.0, 50, 3],
    ])
    agent = OrbitScriptedAgent()
    actions = agent.act(env, player=0, current_turn=10, max_turns=250)
    assert actions == [], f"expected no wasted partial wave, got {actions}"


def test_unsavable_planet_abandoned_and_evacuated():
    """Help arrives after the threat lands → no reinforcement; doomed garrison
    is evacuated to the backline instead."""
    import math as _math
    fleets = [[9, 2, 20.0, 45.0, _math.pi / 2, 2, 50]]  # 50 ships, ~1.5 turns out
    env = _mk_env(
        [
            [0, 0, 20.0, 50.0, 1.0, 3, 1],    # doomed frontline
            [1, 0, 20.0, 95.0, 1.0, 40, 2],   # backline, ~13 turns away
            [2, 1, 20.0, 5.0, 1.0, 30, 2],    # enemy source
        ],
        fleets=fleets,
    )
    cfg = OrbitScriptedConfig(include_neutrals=False, include_enemies=False)
    agent = OrbitScriptedAgent(cfg=cfg)
    actions = agent.act(env, player=0, current_turn=10, max_turns=250)
    # No reinforcement from backline (would land after the planet falls).
    assert not any(int(a[0]) == 1 for a in actions), actions
    # Garrison evacuated off the doomed planet.
    evac = [a for a in actions if int(a[0]) == 0]
    assert evac and int(evac[0][2]) == 3, actions


def test_lp_reinforcement_does_not_feed_unsavable_planet():
    import math as _math
    fleets = [[9, 2, 20.0, 45.0, _math.pi / 2, 2, 50]]
    env = _mk_env(
        [
            [0, 0, 20.0, 50.0, 1.0, 3, 1],
            [1, 0, 20.0, 95.0, 1.0, 40, 2],
            [2, 1, 20.0, 5.0, 1.0, 30, 2],
        ],
        fleets=fleets,
    )
    cfg = OrbitScriptedConfig(
        reinforcement_allocator="lp",
        include_neutrals=False,
        include_enemies=False,
    )
    agent = OrbitScriptedAgent(cfg=cfg)
    actions = agent.act(env, player=0, current_turn=10, max_turns=250)
    assert not any(int(a[0]) == 1 for a in actions), actions
    evac = [a for a in actions if int(a[0]) == 0]
    assert evac and int(evac[0][2]) == 3, actions


def test_consolidation_pushes_idle_backline_to_front():
    """No attack, no support need → idle backline surplus rallies forward."""
    env = _mk_env([
        [0, 0, 20.0, 20.0, 1.0, 50, 2],   # backline (enemy unreachable: blocked)
        [1, 0, 20.0, 50.0, 1.0, 30, 2],   # frontline (sees enemy)
        [2, 1, 20.0, 80.0, 1.0, 10, 2],   # enemy
    ])
    cfg = OrbitScriptedConfig(
        include_neutrals=False,
        include_enemies=False,
        support_threat_threshold=999.0,
        consolidation_min_surplus=10.0,
        consolidation_min_send=5,
    )
    agent = OrbitScriptedAgent(cfg=cfg)
    actions = agent.act(env, player=0, current_turn=5, max_turns=250)
    pushes = [a for a in actions if int(a[0]) == 0 and int(a[2]) >= 5]
    assert pushes, f"expected consolidation launch from backline, got {actions}"


def test_comet_garrison_evacuated_before_expiry():
    """Owned comet with 1 turn of life left → garrison flees to nearest planet."""
    path = [[20.0, 52.0 + 2.0 * i] for i in range(10)]
    planets = [
        [0, 0, 20.0, 20.0, 1.0, 40, 2],
        [3, 0, 20.0, 68.0, 1.0, 12, 0],
    ]
    env = {
        "planets": planets,
        "fleets": [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
        "comet_planet_ids": [3],
        "comets": [{"planet_ids": [3], "paths": [path], "path_index": 8}],
    }
    agent = OrbitScriptedAgent()
    actions = agent.act(env, player=0, current_turn=20, max_turns=250)
    evac = [a for a in actions if int(a[0]) == 3]
    assert evac and int(evac[0][2]) == 12, actions


def test_one_source_can_fund_two_waves():
    """Per-source single-launch restriction removed: one rich planet captures
    two cheap neutrals in the same turn."""
    env = _mk_env([
        [0, 0, 20.0, 20.0, 1.0, 60, 2],
        [1, -1, 20.0, 50.0, 1.0, 5, 3],
        [2, -1, 50.0, 20.0, 1.0, 5, 3],
    ])
    agent = OrbitScriptedAgent()
    actions = agent.act(env, player=0, current_turn=10, max_turns=250)
    attack_launches = [a for a in actions if int(a[0]) == 0]
    assert len(attack_launches) >= 2, actions
