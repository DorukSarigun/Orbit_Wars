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
