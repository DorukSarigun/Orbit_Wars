from orbit_graph import build_orbit_graph

from final_micro_planner import MicroPlannerConfig, SourceOption, plan_target_attack


def _env_simple_neutral():
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 80, 3],
            [1, -1, 30.0, 20.0, 1.0, 20, 2],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 80, 3],
            [1, -1, 30.0, 20.0, 1.0, 20, 2],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def _env_contested_neutral():
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 80, 3],
            [1, -1, 30.0, 20.0, 1.0, 20, 2],
            [2, 1, 90.0, 20.0, 1.0, 40, 3],
        ],
        "fleets": [
            [0, 1, 90.0, 20.0, 3.14159265, 2, 30],
        ],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 80, 3],
            [1, -1, 30.0, 20.0, 1.0, 20, 2],
            [2, 1, 90.0, 20.0, 1.0, 40, 3],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_micro_planner_captures_simple_neutral():
    og = build_orbit_graph(_env_simple_neutral(), player=0)
    src = og.planet_by_id[0]
    tgt = og.planet_by_id[1]

    plan = plan_target_attack(
        og=og,
        target=tgt,
        sources=[SourceOption(source=src, budget=80)],
        player=0,
        current_turn=0,
        max_turns=100,
        omega=0.0,
        cfg=MicroPlannerConfig(hold_horizon=20, beam_width=8),
    )

    assert plan is not None
    assert plan.timeline.self_holds_until_horizon(0)
    assert plan.ships_sent > 20


def test_micro_planner_accounts_for_known_enemy_inbound_to_neutral():
    og = build_orbit_graph(_env_contested_neutral(), player=0)
    src = og.planet_by_id[0]
    tgt = og.planet_by_id[1]

    plan = plan_target_attack(
        og=og,
        target=tgt,
        sources=[SourceOption(source=src, budget=80)],
        player=0,
        current_turn=0,
        max_turns=100,
        omega=0.0,
        cfg=MicroPlannerConfig(hold_horizon=30, beam_width=8),
    )

    assert plan is not None
    assert plan.timeline.self_holds_until_horizon(0)
    assert plan.ships_sent > 25


def test_micro_planner_keeps_stable_capture_even_when_score_is_negative():
    og = build_orbit_graph(_env_simple_neutral(), player=0)
    src = og.planet_by_id[0]
    tgt = og.planet_by_id[1]

    plan = plan_target_attack(
        og=og,
        target=tgt,
        sources=[SourceOption(source=src, budget=80)],
        player=0,
        current_turn=0,
        max_turns=100,
        omega=0.0,
        cfg=MicroPlannerConfig(
            hold_horizon=20,
            beam_width=8,
            value_weight=0.0,
            enemy_denial_weight=0.0,
            ship_cost_weight=10.0,
        ),
    )

    assert plan is not None
    assert plan.score < 0.0
    assert plan.timeline.self_holds_until_horizon(0)
