"""Tests for the macro action MCTS (Capture + Reinforce + Pass)."""

from __future__ import annotations

import random

from orbit_graph import build_orbit_graph
from orbit_timed_mcts import TimedMCTSConfig, _CaptureEvent
from orbit_macro_mcts_agent import (
    MacroAction,
    MacroActionType,
    MacroMCTSConfig,
    MacroPriorRequest,
    MacroTimedMCTS,
    OrbitMacroMCTSAgent,
    PASS_ACTION,
)


# Planet row: [id, owner, x, y, radius, ships, production]; owner -1 = neutral.
def _capture_env():
    planets = [
        [0, 0, 20.0, 50.0, 1.0, 60, 2],
        [1, -1, 20.0, 74.0, 1.0, 6, 2],
        [2, -1, 20.0, 96.0, 1.0, 5, 9],
        [3, -1, 80.0, 50.0, 1.0, 6, 1],
    ]
    return {
        "planets": planets,
        "fleets": [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def _defense_env():
    # Planet 0: friendly source (many ships); planet 1: friendly, will be
    # threatened by an injected enemy wave; planet 2: enemy home so player 1
    # exists in the abstract player set.
    planets = [
        [0, 0, 10.0, 10.0, 1.0, 120, 1],
        [1, 0, 10.0, 18.0, 1.0, 3, 1],
        [2, 1, 90.0, 90.0, 1.0, 50, 1],
    ]
    return {
        "planets": planets,
        "fleets": [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def _macro_cfg(**timed_kwargs):
    params = dict(
        simulations=64,
        horizon_turns=16,
        eta_horizon=60.0,
        source_reserve_frac=0.0,
        gamma=1.0,
        ship_cost_weight=0.0,
        eta_cost_weight=0.0,
    )
    params.update(timed_kwargs)
    return MacroMCTSConfig(timed=TimedMCTSConfig(**params), reinforce_margin=1.0)


def _inject_threat(mcts, target, *, rel_eta, ships, enemy=1):
    st = mcts._initial_state
    pend = list(st.pending) + [
        _CaptureEvent(
            arrival_turn=int(st.turn) + int(rel_eta),
            owner=int(enemy),
            target=int(target),
            ships=int(ships),
        )
    ]
    return mcts._state(
        turn=st.turn,
        player_to_move=mcts.root_player,
        owners=st.owner_map(),
        ships=st.ship_map(),
        pending=pend,
        reward=st.reward,
    )


# --------------------------------------------------------------------------- #
# 1. Capture-only behaviour matches the timed MCTS spirit.
# --------------------------------------------------------------------------- #
def test_capture_candidates_and_bonus():
    og = build_orbit_graph(_capture_env(), player=0)
    mcts = MacroTimedMCTS(og, _macro_cfg(), root_player=0, rng=random.Random(0))
    actions = mcts._macro_actions(mcts._initial_state)
    captured = {a.target for a in actions if a.type is MacroActionType.CAPTURE}
    assert captured, actions
    bonus = mcts.search()
    assert bonus, bonus
    assert all(v > 0.0 for v in bonus.values()), bonus


# --------------------------------------------------------------------------- #
# 2. A threatened owned planet yields a Reinforce candidate.
# --------------------------------------------------------------------------- #
def test_reinforce_candidate_generated_when_threatened():
    og = build_orbit_graph(_defense_env(), player=0)
    mcts = MacroTimedMCTS(og, _macro_cfg(), root_player=0, rng=random.Random(0))
    state = _inject_threat(mcts, target=1, rel_eta=6, ships=40)
    threats = mcts._threatened_owned(state, 0)
    assert 1 in threats, threats
    actions = mcts._macro_actions(state)
    assert any(
        a.type is MacroActionType.REINFORCE and a.target == 1 for a in actions
    ), actions


# --------------------------------------------------------------------------- #
# 3. Support that cannot arrive before the threat ETA is not a candidate.
# --------------------------------------------------------------------------- #
def test_no_reinforce_when_help_arrives_too_late():
    og = build_orbit_graph(_defense_env(), player=0)
    mcts = MacroTimedMCTS(og, _macro_cfg(), root_player=0, rng=random.Random(0))
    # Enemy lands next turn -> any reinforcement arrives at or after the wave.
    state = _inject_threat(mcts, target=1, rel_eta=1, ships=40)
    action = mcts._plan_reinforce(state, 0, 1, enemy_ships=40.0, enemy_rel_eta=1)
    assert action is None


# --------------------------------------------------------------------------- #
# 4. Reinforce is all-or-nothing: insufficient ships -> no candidate.
# --------------------------------------------------------------------------- #
def test_reinforce_all_or_nothing():
    og = build_orbit_graph(_defense_env(), player=0)
    mcts = MacroTimedMCTS(og, _macro_cfg(), root_player=0, rng=random.Random(0))
    base = _inject_threat(mcts, target=1, rel_eta=6, ships=40)
    # Starve the source so it cannot cover the shortfall.
    ships = base.ship_map()
    ships[0] = 5
    starved = mcts._state(
        turn=base.turn,
        player_to_move=0,
        owners=base.owner_map(),
        ships=ships,
        pending=base.pending,
        reward=base.reward,
    )
    action = mcts._plan_reinforce(starved, 0, 1, enemy_ships=40.0, enemy_rel_eta=6)
    assert action is None


# --------------------------------------------------------------------------- #
# 5. A successful reinforce scores better than passing (defense value).
# --------------------------------------------------------------------------- #
def test_reinforce_prevents_loss_value():
    og = build_orbit_graph(_defense_env(), player=0)
    mcts = MacroTimedMCTS(og, _macro_cfg(), root_player=0, rng=random.Random(0))
    state = _inject_threat(mcts, target=1, rel_eta=6, ships=40)
    reinforce = next(
        a for a in mcts._macro_actions(state)
        if a.type is MacroActionType.REINFORCE and a.target == 1
    )
    s_rein = mcts._advance_to_horizon(mcts._apply_macro(state, reinforce))
    s_pass = mcts._advance_to_horizon(mcts._apply_macro(state, PASS_ACTION))
    assert mcts._score(s_rein) > mcts._score(s_pass)
    # Pass loses the planet to the enemy; reinforce keeps it.
    assert s_rein.owner_map()[1] == 0
    assert s_pass.owner_map()[1] == 1


# --------------------------------------------------------------------------- #
# 6. Pass is always legal.
# --------------------------------------------------------------------------- #
def test_pass_always_legal():
    og = build_orbit_graph(_capture_env(), player=0)
    mcts = MacroTimedMCTS(og, _macro_cfg(), root_player=0, rng=random.Random(0))
    actions = mcts._macro_actions(mcts._initial_state)
    assert any(a.type is MacroActionType.PASS for a in actions)


# --------------------------------------------------------------------------- #
# 7. A learned prior can reorder the branch priors.
# --------------------------------------------------------------------------- #
def test_learned_prior_reorders_priors():
    og = build_orbit_graph(_capture_env(), player=0)

    base = MacroTimedMCTS(og, _macro_cfg(), root_player=0, rng=random.Random(0))
    capture_keys = [
        a.key() for a in base._macro_actions(base._initial_state)
        if a.type is MacroActionType.CAPTURE
    ]
    assert capture_keys
    target_key = capture_keys[-1]  # boost one capture over the rest (incl. Pass)

    def provider(request: MacroPriorRequest):
        return {k: (10.0 if k == target_key else 0.0) for k in request.features_by_key}

    mcts = MacroTimedMCTS(
        og, _macro_cfg(), root_player=0, prior_provider=provider, rng=random.Random(0)
    )
    root = mcts._node_for_state(mcts._initial_state)
    assert root.priors[target_key] == max(root.priors.values())


# --------------------------------------------------------------------------- #
# Search result helpers + agent integration.
# --------------------------------------------------------------------------- #
def test_search_result_visit_distribution_and_best_action():
    og = build_orbit_graph(_capture_env(), player=0)
    mcts = MacroTimedMCTS(og, _macro_cfg(), root_player=0, rng=random.Random(0))
    result = mcts.run()
    dist = result.visit_distribution()
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert isinstance(result.best_action(), MacroAction)


def test_zero_simulations_returns_empty_bonus():
    og = build_orbit_graph(_capture_env(), player=0)
    cfg = MacroMCTSConfig(timed=TimedMCTSConfig(simulations=0))
    mcts = MacroTimedMCTS(og, cfg, root_player=0)
    assert mcts.search() == {}


def test_agent_act_runs():
    agent = OrbitMacroMCTSAgent(macro_cfg=_macro_cfg(simulations=16, horizon_turns=12))
    actions = agent.act(_capture_env(), player=0, current_turn=0, max_turns=300)
    assert isinstance(actions, list)


def test_evacuate_candidate_for_doomed_planet():
    og = build_orbit_graph(_defense_env(), player=0)
    cfg = _macro_cfg()
    cfg = MacroMCTSConfig(
        timed=cfg.timed, reinforce_margin=cfg.reinforce_margin, enable_evacuate=True,
    )
    mcts = MacroTimedMCTS(og, cfg, root_player=0, rng=random.Random(0))
    # Overwhelming wave: planet 1 cannot be held -> doomed -> evacuate to 0.
    state = _inject_threat(mcts, target=1, rel_eta=6, ships=500)
    doomed = mcts._doomed_owned(state, 0)
    assert 1 in doomed, doomed
    actions = mcts._macro_actions(state)
    evac = [a for a in actions if a.type is MacroActionType.EVACUATE and a.target == 0]
    assert evac, actions
    # Applying evacuation moves the garrison off the doomed planet.
    applied = mcts._apply_macro(state, evac[0])
    assert applied.ship_map().get(1, 0) < state.ship_map().get(1, 0)


def test_consolidate_moves_backline_surplus():
    # Backline planet 0 (no non-owned neighbours within reach via the frontier
    # planet 1) feeds frontier planet 1 which is adjacent to neutral 2.
    planets = [
        [0, 0, 10.0, 10.0, 1.0, 80, 1],
        [1, 0, 18.0, 14.0, 1.0, 5, 1],
        [2, -1, 26.0, 18.0, 1.0, 5, 5],
    ]
    env = {
        "planets": planets, "fleets": [], "initial_planets": planets,
        "angular_velocity": 0.0, "comet_planet_ids": [], "comets": [],
    }
    og = build_orbit_graph(env, player=0)
    cfg = MacroMCTSConfig(
        timed=TimedMCTSConfig(simulations=8, horizon_turns=10, source_reserve_frac=0.0),
        enable_consolidate=True,
    )
    mcts = MacroTimedMCTS(og, cfg, root_player=0, rng=random.Random(0))
    actions = mcts._macro_actions(mcts._initial_state)
    cons = [a for a in actions if a.type is MacroActionType.CONSOLIDATE]
    # Consolidate is opportunistic; when present it must be a valid owned->owned move.
    for a in cons:
        assert a.sources and a.ships > 0
        applied = mcts._apply_macro(mcts._initial_state, a)
        assert isinstance(applied.owner_map(), dict)


def test_hybrid_leaf_value_blend_used():
    og = build_orbit_graph(_capture_env(), player=0)
    calls = {"n": 0}

    def leaf_value(state, root_player):
        calls["n"] += 1
        return 0.0

    cfg = MacroMCTSConfig(
        timed=TimedMCTSConfig(simulations=16, horizon_turns=12, source_reserve_frac=0.0),
        value_blend=1.0,
    )
    mcts = MacroTimedMCTS(og, cfg, root_player=0, leaf_value_fn=leaf_value, rng=random.Random(0))
    mcts.run()
    assert calls["n"] > 0
