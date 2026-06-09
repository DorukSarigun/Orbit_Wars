"""Tests for enemy weak-spot attack metrics."""

from __future__ import annotations

from orbit_attack_allocate import rank_capture_targets
from orbit_attack_targets import AttackTargetConfig, plan_attack_targets
from orbit_enemy_weakness import compute_enemy_weak_spot
from orbit_graph import build_orbit_graph


def _env_two_enemies():
    """Us + weak enemy (few ships) + fortified enemy (many ships), both reachable."""
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 55, 3],
            [2, 1, 20.0, 50.0, 1.0, 4, 5],
            [3, 1, 50.0, 20.0, 1.0, 25, 5],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 55, 3],
            [2, 1, 20.0, 50.0, 1.0, 4, 5],
            [3, 1, 50.0, 20.0, 1.0, 25, 5],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_weak_enemy_has_higher_fragility():
    og = build_orbit_graph(_env_two_enemies(), player=0)
    weak = compute_enemy_weak_spot(og, 2)
    strong = compute_enemy_weak_spot(og, 3)
    assert weak is not None and strong is not None
    assert weak.fragility > strong.fragility
    assert weak.weakness_score > strong.weakness_score


def test_ranking_prefers_weak_enemy_first():
    og = build_orbit_graph(_env_two_enemies(), player=0)
    plan = plan_attack_targets(
        og,
        cfg=AttackTargetConfig(
            current_turn=10,
            max_turns=250,
            include_neutrals=False,
        ),
    )
    ranked = rank_capture_targets(
        og,
        plan,
        attack_cfg=AttackTargetConfig(current_turn=10, max_turns=250, include_neutrals=False),
        prioritize_enemy_weakspots=True,
    )
    enemies = [r for r in ranked if r.is_enemy]
    assert enemies
    assert enemies[0].planet_id == 2
    if len(enemies) >= 2:
        assert enemies[0].weakness_score >= enemies[1].weakness_score
