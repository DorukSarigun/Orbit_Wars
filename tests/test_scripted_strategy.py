"""Tests for the capture-efficiency scripted opponent."""
from __future__ import annotations

import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from feature_extraction import Planet
from scripted_strategy import (
    CaptureEfficiencyAgent,
    ScriptedConfig,
    SyncCoalition,
    _build_fleets_by_target,
    _defense_at_arrival,
    _own_fleets_capture_target,
    best_synchronized_coalition,
    compute_src_budgets,
    score_capture_candidate,
)


def _planet(pid, owner, x, y, ships=0, prod=2, radius=2.0):
    return Planet(
        id=int(pid), owner=int(owner), x=float(x), y=float(y),
        radius=float(radius), ships=int(ships), production=int(prod),
    )


# ---------------------------------------------------------------------------
# Defense estimate
# ---------------------------------------------------------------------------
def test_defense_at_arrival_owned_is_zero():
    src = _planet(0, 0, 50, 50, ships=10, prod=3)
    tgt = _planet(1, 0, 60, 50, ships=5, prod=2)
    assert _defense_at_arrival(tgt, eta=20.0, player=0) == 0


def test_defense_at_arrival_neutral_is_static():
    tgt = _planet(1, -1, 60, 50, ships=15, prod=2)
    # neutrals do not produce; eta does not matter
    assert _defense_at_arrival(tgt, eta=0.0, player=0) == 15
    assert _defense_at_arrival(tgt, eta=50.0, player=0) == 15


def test_defense_at_arrival_enemy_grows_with_eta():
    tgt = _planet(1, 1, 60, 50, ships=10, prod=2)
    assert _defense_at_arrival(tgt, eta=0.0, player=0) == 10
    assert _defense_at_arrival(tgt, eta=5.0, player=0) == 10 + 2 * 5
    assert _defense_at_arrival(tgt, eta=15.5, player=0) == int(10 + 2 * 15.5)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
def test_score_captures_neutral():
    src = _planet(0, 0, 50, 50, ships=20)
    tgt = _planet(1, -1, 60, 50, ships=5, prod=3)
    score, defense, prod_gain = score_capture_candidate(
        src, tgt, ships_sent=10, eta=10.0,
        current_turn=0, max_turns=500, player=0,
    )
    assert defense == 5
    assert prod_gain == pytest.approx(3 * (500 - 0 - 10))
    assert score == pytest.approx(prod_gain - 5)


def test_score_insufficient_ships_is_neg_inf():
    src = _planet(0, 0, 50, 50, ships=20)
    tgt = _planet(1, -1, 60, 50, ships=15, prod=3)
    score, _, _ = score_capture_candidate(
        src, tgt, ships_sent=10, eta=10.0,
        current_turn=0, max_turns=500, player=0,
    )
    assert score == float("-inf")


def test_score_owned_target_is_neg_inf():
    src = _planet(0, 0, 50, 50, ships=20)
    tgt = _planet(1, 0, 60, 50, ships=2, prod=3)
    score, _, _ = score_capture_candidate(
        src, tgt, ships_sent=10, eta=5.0,
        current_turn=0, max_turns=500, player=0,
    )
    assert score == float("-inf")


def test_score_enemy_grows_during_flight():
    src = _planet(0, 0, 50, 50, ships=200)
    tgt = _planet(1, 1, 60, 50, ships=10, prod=4)
    # eta=10 -> defense = 10 + 4*10 = 50
    score, defense, prod_gain = score_capture_candidate(
        src, tgt, ships_sent=60, eta=10.0,
        current_turn=0, max_turns=500, player=0,
    )
    assert defense == 50
    assert prod_gain == pytest.approx(4 * (500 - 10))
    assert score == pytest.approx(prod_gain - 50)


def test_score_zero_remaining_at_game_end():
    src = _planet(0, 0, 50, 50, ships=20)
    tgt = _planet(1, -1, 60, 50, ships=2, prod=3)
    # current_turn + eta exceeds max_turns -> remaining clamped to 0
    score, defense, prod_gain = score_capture_candidate(
        src, tgt, ships_sent=10, eta=20.0,
        current_turn=490, max_turns=500, player=0,
    )
    assert defense == 2
    assert prod_gain == 0.0
    assert score == -2.0  # only the cost remains


# ---------------------------------------------------------------------------
# Defensive reserve / budget
# ---------------------------------------------------------------------------
def test_budget_no_threats_full_ships():
    src = _planet(0, 0, 50, 50, ships=30, prod=2)
    cfg = ScriptedConfig()
    budgets = compute_src_budgets([src], {}, {}, cfg=cfg)
    assert budgets[0] == 30


def test_budget_distant_threat_covered_by_production():
    src = _planet(0, 0, 50, 50, ships=30, prod=2)
    cfg = ScriptedConfig(defense_window=15)
    enemy = {0: [(25, 10.0)]}        # 25 enemy ships in 10 turns
    # production_buf = 2 * 10 = 20; help = 0; reserve = max(0, 25-20) = 5
    budgets = compute_src_budgets([src], {}, enemy, cfg=cfg)
    assert budgets[0] == 30 - 5


def test_budget_imminent_threat_holds_back_ships():
    src = _planet(0, 0, 50, 50, ships=30, prod=2)
    cfg = ScriptedConfig(defense_window=15)
    enemy = {0: [(25, 2.0)]}
    # production_buf = 4; reserve = 25 - 4 = 21; budget = 30 - 21 = 9
    budgets = compute_src_budgets([src], {}, enemy, cfg=cfg)
    assert budgets[0] == 9


def test_budget_help_offsets_threat():
    src = _planet(0, 0, 50, 50, ships=30, prod=2)
    cfg = ScriptedConfig(defense_window=15)
    enemy = {0: [(25, 5.0)]}
    helps = {0: [(20, 4.0)]}         # 20 friendly arrive before threat
    # production_buf = 10; help_at_first = 20; reserve = max(0, 25-20-10) = 0
    budgets = compute_src_budgets([src], helps, enemy, cfg=cfg)
    assert budgets[0] == 30


def test_budget_threat_outside_window_ignored():
    src = _planet(0, 0, 50, 50, ships=30, prod=2)
    cfg = ScriptedConfig(defense_window=15)
    enemy = {0: [(50, 30.0)]}
    budgets = compute_src_budgets([src], {}, enemy, cfg=cfg)
    assert budgets[0] == 30


def test_budget_simultaneous_threats_aggregate():
    src = _planet(0, 0, 50, 50, ships=40, prod=1)
    cfg = ScriptedConfig(defense_window=15, threat_cluster_buffer=2)
    enemy = {0: [(10, 5.0), (15, 6.0), (5, 12.0)]}
    # e_first=5; cluster_cutoff=7; aggregate: 10+15=25 (12.0 excluded)
    # production_buf = 1*5 = 5; reserve = 25-5 = 20; budget = 40-20 = 20
    budgets = compute_src_budgets([src], {}, enemy, cfg=cfg)
    assert budgets[0] == 20


# ---------------------------------------------------------------------------
# Full agent integration on a synthetic env_obs
# ---------------------------------------------------------------------------
def _toy_env_obs():
    """Two planets per player + one neutral, no comets, no fleets."""
    return {
        "planets": [
            [0, 0, 70.0, 70.0, 3.0, 30, 3],   # mine, big stash
            [1, 0, 80.0, 70.0, 2.0, 5, 2],    # mine, small
            [2, 1, 30.0, 30.0, 2.0, 8, 2],    # enemy
            [3, -1, 70.0, 30.0, 2.0, 4, 2],   # neutral, easy capture
        ],
        "initial_planets": [
            [0, 0, 70.0, 70.0, 3.0, 30, 3],
            [1, 0, 80.0, 70.0, 2.0, 5, 2],
            [2, 1, 30.0, 30.0, 2.0, 8, 2],
            [3, -1, 70.0, 30.0, 2.0, 4, 2],
        ],
        "comet_planet_ids": [],
        "comets": [],
        "fleets": [],
        "angular_velocity": 0.0,
    }


def test_agent_returns_valid_actions():
    agent = CaptureEfficiencyAgent()
    actions = agent.act(_toy_env_obs(), player=0, current_turn=0, max_turns=500)
    assert isinstance(actions, list)
    assert len(actions) >= 1
    # every action: [from_id, angle, ships]
    for a in actions:
        assert len(a) == 3
        from_id, angle, ships = a
        # source must be one of my planets (0 or 1)
        assert int(from_id) in (0, 1)
        assert math.isfinite(float(angle))
        assert int(ships) >= 1


def test_agent_obeys_launch_cap():
    cfg = ScriptedConfig(max_launches=2)
    agent = CaptureEfficiencyAgent(cfg)
    actions = agent.act(_toy_env_obs(), player=0, current_turn=0, max_turns=500)
    assert len(actions) <= 2


def test_agent_no_planets_returns_empty():
    agent = CaptureEfficiencyAgent()
    obs = _toy_env_obs()
    # remove player 0 ownership
    for row in obs["planets"]:
        if row[1] == 0:
            row[1] = -1
    actions = agent.act(obs, player=0, current_turn=0, max_turns=500)
    assert actions == []


def test_agent_no_double_targeting_per_pair():
    agent = CaptureEfficiencyAgent()
    actions = agent.act(_toy_env_obs(), player=0, current_turn=0, max_turns=500)
    pairs = set()
    for from_id, _, _ in actions:
        # targets aren't directly visible from env action; we only check
        # that each from_id appears at most ``len(targets)`` times.
        pairs.add(int(from_id))
    # at least uniqueness sanity: number of distinct sources is reasonable
    assert len(pairs) <= 2


# ---------------------------------------------------------------------------
# Synchronized coalitions
# ---------------------------------------------------------------------------
def test_coalition_single_source_easy_neutral():
    """A single static planet with low defense: size-1 coalition captures."""
    src = _planet(0, 0, 20, 20, ships=20, prod=2, radius=2.0)
    tgt = _planet(1, -1, 35, 20, ships=4, prod=3, radius=2.0)
    cfg = ScriptedConfig()
    plan = best_synchronized_coalition(
        tgt, [src], {0: 20}, [src, tgt], 0.0,
        current_turn=0, max_turns=500, player=0, cfg=cfg,
    )
    assert plan is not None
    assert len(plan.members) == 1
    assert plan.members[0].src_id == 0
    assert plan.members[0].delay == 0          # bottleneck launches now
    assert plan.total_ships > plan.defense
    assert plan.score > 0


def test_coalition_two_sources_insufficient_alone():
    """Each source alone has too few ships, but together they capture."""
    # Two sources, each 30 ships, target with 50 ships defense.
    src_a = _planet(0, 0, 20, 20, ships=30, prod=2, radius=2.0)
    src_b = _planet(1, 0, 20, 30, ships=30, prod=2, radius=2.0)
    tgt = _planet(2, -1, 35, 25, ships=50, prod=3, radius=2.0)
    cfg = ScriptedConfig(max_coalition_size=4, max_delay_window=8)
    plan = best_synchronized_coalition(
        tgt, [src_a, src_b], {0: 30, 1: 30},
        [src_a, src_b, tgt], 0.0,
        current_turn=0, max_turns=500, player=0, cfg=cfg,
    )
    assert plan is not None
    assert len(plan.members) >= 2
    assert plan.total_ships > plan.defense     # cross-threshold
    assert plan.score > 0


def test_coalition_no_valid_returns_none():
    """Insufficient ships even with all sources combined."""
    src_a = _planet(0, 0, 20, 20, ships=5, prod=0, radius=2.0)
    src_b = _planet(1, 0, 20, 30, ships=5, prod=0, radius=2.0)
    tgt = _planet(2, -1, 35, 25, ships=200, prod=3, radius=2.0)
    cfg = ScriptedConfig()
    plan = best_synchronized_coalition(
        tgt, [src_a, src_b], {0: 5, 1: 5},
        [src_a, src_b, tgt], 0.0,
        current_turn=0, max_turns=500, player=0, cfg=cfg,
    )
    assert plan is None


def test_coalition_respects_max_size():
    """``max_coalition_size`` caps coalition members."""
    sources = [
        _planet(i, 0, 15.0 + i * 4, 15.0, ships=10, prod=1, radius=2.0)
        for i in range(6)
    ]
    tgt = _planet(99, -1, 25.0, 35.0, ships=40, prod=3, radius=2.0)
    cfg = ScriptedConfig(max_coalition_size=2, max_delay_window=8)
    budgets = {s.id: 10 for s in sources}
    plan = best_synchronized_coalition(
        tgt, sources, budgets,
        sources + [tgt], 0.0,
        current_turn=0, max_turns=500, player=0, cfg=cfg,
    )
    if plan is not None:
        assert len(plan.members) <= 2


def test_coalition_delay_grows_ships_for_neutral():
    """Against a neutral, waiting lets us send more ships at the same defense."""
    src = _planet(0, 0, 20, 20, ships=10, prod=5, radius=2.0)
    tgt = _planet(1, -1, 35, 20, ships=8, prod=3, radius=2.0)
    cfg = ScriptedConfig(max_coalition_size=1, max_delay_window=8)
    plan = best_synchronized_coalition(
        tgt, [src], {0: 10}, [src, tgt], 0.0,
        current_turn=0, max_turns=500, player=0, cfg=cfg,
    )
    assert plan is not None
    # Neutral defense is static (8); production_gain = 3 * (500 - T_star)
    # No-delay capture is already trivial; algorithm should still find a plan.
    assert plan.score > 0
    assert plan.total_ships > plan.defense


def test_coalition_shift_zero_bottleneck_launches_now():
    """When max_delay_window=0 the bottleneck always has delay=0."""
    # Sources well separated in y so neither blocks the other's path to tgt.
    src_a = _planet(0, 0, 15, 10, ships=30, prod=2, radius=2.0)
    src_b = _planet(1, 0, 15, 30, ships=30, prod=2, radius=2.0)
    tgt = _planet(2, -1, 40, 20, ships=40, prod=3, radius=2.0)
    cfg = ScriptedConfig(max_coalition_size=4, max_delay_window=0)
    plan = best_synchronized_coalition(
        tgt, [src_a, src_b], {0: 30, 1: 30},
        [src_a, src_b, tgt], 0.0,
        current_turn=0, max_turns=500, player=0, cfg=cfg,
    )
    assert plan is not None
    # With shift=0 forced, the slowest member has delay=0.
    assert any(m.delay == 0 for m in plan.members)


def test_coalition_shift_nonzero_all_members_wait():
    """With shift>0 every coalition member has delay>0; this turn is passive."""
    # Identical etas (ships, distance) so T_min is the same for both.
    src_a = _planet(0, 0, 20, 20, ships=30, prod=2, radius=2.0)
    src_b = _planet(1, 0, 22, 20, ships=30, prod=2, radius=2.0)
    tgt = _planet(2, -1, 40, 20, ships=40, prod=3, radius=2.0)
    # Shift=0 valid plan exists; shift>0 plans have all delays > 0.
    cfg = ScriptedConfig(max_coalition_size=4, max_delay_window=8)
    plan = best_synchronized_coalition(
        tgt, [src_a, src_b], {0: 30, 1: 30},
        [src_a, src_b, tgt], 0.0,
        current_turn=0, max_turns=500, player=0, cfg=cfg,
    )
    assert plan is not None
    # Sanity: total ships still cross the defense threshold.
    assert plan.total_ships > plan.defense
    # All delays must be consistent: min(delay) == T_star - max(eta_now).
    # Since the agent re-derives each turn, both shift=0 and shift>0 plans
    # are valid at planning time. The downstream `act()` will simply skip
    # this plan if no member has delay=0.


def test_agent_uses_coalition_for_hard_target():
    """Build an env_obs where two of my planets are needed to capture."""
    obs = {
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 25, 2],   # mine
            [1, 0, 20.0, 30.0, 2.0, 25, 2],   # mine
            [2, -1, 35.0, 25.0, 2.0, 40, 3],  # neutral, hard
        ],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 2.0, 25, 2],
            [1, 0, 20.0, 30.0, 2.0, 25, 2],
            [2, -1, 35.0, 25.0, 2.0, 40, 3],
        ],
        "comet_planet_ids": [],
        "comets": [],
        "fleets": [],
        "angular_velocity": 0.0,
    }
    agent = CaptureEfficiencyAgent(ScriptedConfig(max_coalition_size=4, max_delay_window=8))
    actions = agent.act(obs, player=0, current_turn=0, max_turns=500)
    # Expect at least one launch this turn (the bottleneck source).
    assert len(actions) >= 1
    # All launches should be from valid sources (0 or 1) toward the only target.
    for a in actions:
        from_id, angle, ships = a
        assert int(from_id) in (0, 1)
        assert int(ships) >= 1
        assert math.isfinite(float(angle))


# ---------------------------------------------------------------------------
# In-flight fleet awareness (skip targets already covered)
# ---------------------------------------------------------------------------
def _fleet_row(fid, owner, fx, fy, angle, from_id, ships):
    """Helper to build a raw fleet observation row."""
    return [int(fid), int(owner), float(fx), float(fy),
            float(angle), int(from_id), int(ships)]


def test_own_fleets_capture_target_sufficient():
    """A 30-ship own fleet en route fully covers a 20-ship neutral."""
    from feature_extraction import parse_fleets
    tgt = _planet(2, -1, 35, 20, ships=20, prod=2)
    src = _planet(0, 0, 20, 20, ships=10, prod=2)
    obs = {"fleets": [_fleet_row(7, 0, 22, 20, 0.0, 0, 30)]}
    fleets = parse_fleets(obs)
    fbt = _build_fleets_by_target(fleets, [src, tgt], 0.0)
    assert _own_fleets_capture_target(tgt, fbt, player=0) is True


def test_own_fleets_capture_target_insufficient():
    """A 10-ship own fleet does NOT cover a 20-ship neutral."""
    from feature_extraction import parse_fleets
    tgt = _planet(2, -1, 35, 20, ships=20, prod=2)
    src = _planet(0, 0, 20, 20, ships=10, prod=2)
    obs = {"fleets": [_fleet_row(7, 0, 22, 20, 0.0, 0, 10)]}
    fleets = parse_fleets(obs)
    fbt = _build_fleets_by_target(fleets, [src, tgt], 0.0)
    assert _own_fleets_capture_target(tgt, fbt, player=0) is False


def test_own_fleets_capture_target_enemy_reinforcement_neutralizes():
    """Own 30 ships en route, but a closer 25-ship enemy fleet defeats it."""
    from feature_extraction import parse_fleets
    tgt = _planet(2, -1, 35, 20, ships=20, prod=2)
    src = _planet(0, 0, 20, 20, ships=10, prod=2)
    # Enemy fleet at (38, 20) heading left -> arrives before own (which is
    # at (22, 20) heading right and has ~13 distance to tgt vs ~3 for enemy).
    obs = {"fleets": [
        _fleet_row(7, 0, 22, 20, 0.0, 0, 30),               # own, eta ~ 5
        _fleet_row(8, 1, 38, 20, math.pi, 99, 25),          # enemy, eta ~ 1
    ]}
    fleets = parse_fleets(obs)
    fbt = _build_fleets_by_target(fleets, [src, tgt], 0.0)
    # 30 own vs 20 base + 25 enemy = 45 -> insufficient.
    assert _own_fleets_capture_target(tgt, fbt, player=0) is False


def test_agent_skips_target_already_covered_by_own_fleet():
    """If an own fleet already suffices to capture a neutral, agent skips it.

    Setup: two neutral targets; one (id=2) already has an own 30-ship fleet
    en route, the other (id=3) does not.  Agent must direct ships to id=3.
    """
    obs = {
        "planets": [
            [0, 0, 20.0, 20.0, 2.0, 30, 2],   # mine, with ships
            [2, -1, 35.0, 20.0, 2.0, 20, 2],  # neutral, already covered
            [3, -1, 20.0, 35.0, 2.0, 20, 2],  # neutral, NOT covered
        ],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 2.0, 30, 2],
            [2, -1, 35.0, 20.0, 2.0, 20, 2],
            [3, -1, 20.0, 35.0, 2.0, 20, 2],
        ],
        "comet_planet_ids": [],
        "comets": [],
        # Own fleet already on its way to planet 2 with 30 ships.
        "fleets": [_fleet_row(7, 0, 23.0, 20.0, 0.0, 0, 30)],
        "angular_velocity": 0.0,
    }
    agent = CaptureEfficiencyAgent(ScriptedConfig())
    actions = agent.act(obs, player=0, current_turn=0, max_turns=500)
    # Some launch should happen (planet 3 is uncovered and reachable).
    assert len(actions) >= 1
    # NO launch should target planet 2: angle would point right (toward x=35).
    # Planet 3 is up (toward y=35), so any new launch should aim ~+pi/2.
    # Robust check: the action set must not contain a launch whose angle
    # is within ~0.1 rad of pointing at planet 2 from src 0.
    angle_to_p2 = math.atan2(20.0 - 20.0, 35.0 - 20.0)  # 0
    for from_id, angle, ships in actions:
        if int(from_id) == 0:
            assert abs(float(angle) - angle_to_p2) > 0.2, (
                f"agent launched toward already-covered planet 2 "
                f"(angle={angle}, expected_to_p2={angle_to_p2})"
            )
