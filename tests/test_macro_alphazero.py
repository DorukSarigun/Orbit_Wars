"""Tests for the AlphaZero-style macro net, self-play and training loop."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from orbit_graph import build_orbit_graph
from orbit_timed_mcts import TimedMCTSConfig
from orbit_macro_mcts_agent import (
    MacroAction,
    MacroActionType,
    MacroMCTSConfig,
    MacroSearchResult,
    MacroTimedMCTS,
)
from macro_alphazero_net import (
    ACTION_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    PLANET_FEATURE_DIM,
    MacroAlphaZeroAgent,
    MacroAlphaZeroNet,
    MacroAlphaZeroNetMLP,
    abstract_edge_arrays,
    abstract_state_arrays,
    abstract_state_ids,
    action_feature_vector,
    attach_net,
    build_net,
    load_net,
    save_net,
)
from macro_selfplay import (
    REAL_PLANETS_MAX,
    REAL_PLANETS_MIN,
    SelfPlayConfig,
    generate_selfplay,
    random_board,
    sample_env_board,
    sample_n_planets,
)
from train_macro_alphazero import (
    collate, default_min_decisions, eval_gate, scheduled_max_decisions, train, train_epoch,
)


def _cfg(sims=24, horizon=10, **kw):
    return MacroMCTSConfig(
        timed=TimedMCTSConfig(simulations=sims, horizon_turns=horizon,
                              source_reserve_frac=0.0, gamma=1.0),
        **kw,
    )


def _board():
    planets = [
        [0, 0, 20.0, 50.0, 1.0, 60, 2],
        [1, -1, 20.0, 74.0, 1.0, 6, 2],
        [2, -1, 20.0, 96.0, 1.0, 5, 9],
        [3, 1, 80.0, 50.0, 1.0, 30, 1],
    ]
    return {
        "planets": planets, "fleets": [], "initial_planets": planets,
        "angular_velocity": 0.0, "comet_planet_ids": [], "comets": [],
    }


def test_net_forward_shapes():
    net = MacroAlphaZeroNet(d_model=16)
    pf = torch.randn(2, 5, PLANET_FEATURE_DIM)
    ef = torch.randn(2, 5, 5, EDGE_FEATURE_DIM)
    gf = torch.randn(2, GLOBAL_FEATURE_DIM)
    mask = torch.ones(2, 5)
    v = net.forward_value(pf, ef, gf, mask)
    assert v.shape == (2,)
    af = torch.randn(2, 4, ACTION_FEATURE_DIM)
    am = torch.ones(2, 4, dtype=torch.bool)
    target_idx = torch.tensor([[0, 1, -1, 2], [3, -1, 1, 0]], dtype=torch.long)
    logits = net.forward_prior(pf, ef, gf, mask, af, target_idx, am)
    assert logits.shape == (2, 4)


def test_value_mask_invariance_to_padding():
    net = MacroAlphaZeroNet(d_model=16).eval()
    pf = torch.randn(1, 3, PLANET_FEATURE_DIM)
    ef = torch.randn(1, 3, 3, EDGE_FEATURE_DIM)
    gf = torch.randn(1, GLOBAL_FEATURE_DIM)
    mask = torch.ones(1, 3)
    v_small = net.forward_value(pf, ef, gf, mask)
    # Pad with two dummy planets that are masked out -> value must be identical.
    pf_pad = torch.cat([pf, torch.randn(1, 2, PLANET_FEATURE_DIM)], dim=1)
    ef_pad = torch.randn(1, 5, 5, EDGE_FEATURE_DIM)
    ef_pad[:, :3, :3] = ef
    mask_pad = torch.cat([mask, torch.zeros(1, 2)], dim=1)
    v_pad = net.forward_value(pf_pad, ef_pad, gf, mask_pad)
    assert torch.allclose(v_small, v_pad, atol=1e-5)


def test_abstract_state_arrays_dims():
    og = build_orbit_graph(_board(), player=0)
    mcts = MacroTimedMCTS(og, _cfg(), root_player=0, rng=random.Random(0))
    pf, gf = abstract_state_arrays(mcts, mcts._initial_state)
    assert pf.shape == (4, PLANET_FEATURE_DIM)
    assert gf.shape == (GLOBAL_FEATURE_DIM,)
    ids = abstract_state_ids(mcts._initial_state)
    ef = abstract_edge_arrays(mcts, ids)
    assert ef.shape == (4, 4, EDGE_FEATURE_DIM)


def test_attach_net_changes_priors():
    og = build_orbit_graph(_board(), player=0)
    net = MacroAlphaZeroNet(d_model=16)
    mcts = MacroTimedMCTS(og, _cfg(value_blend=0.5), root_player=0, rng=random.Random(0))
    attach_net(net, mcts)
    result = mcts.run()
    assert result.actions
    assert abs(sum(result.priors.values()) - 1.0) < 1e-5


def test_selfplay_generates_labelled_samples():
    cfg = _cfg(sims=16, horizon=8, root_dirichlet_alpha=0.5)
    sp = SelfPlayConfig(max_decisions=20, temperature_moves=4)
    samples = generate_selfplay(2, cfg, selfplay_cfg=sp, seed=3, n_planets=6)
    assert samples
    for s in samples:
        assert abs(float(s.policy_target.sum()) - 1.0) < 1e-4
        assert s.action_feats.shape[1] == ACTION_FEATURE_DIM
        assert s.planet_feats.shape[1] == PLANET_FEATURE_DIM
        p = s.planet_feats.shape[0]
        assert s.edge_feats.shape == (p, p, EDGE_FEATURE_DIM)
        assert s.action_target_idx.shape[0] == s.action_feats.shape[0]


def test_collate_and_train_epoch_runs():
    cfg = _cfg(sims=16, horizon=8, root_dirichlet_alpha=0.5)
    sp = SelfPlayConfig(max_decisions=16, temperature_moves=4)
    samples = generate_selfplay(2, cfg, selfplay_cfg=sp, seed=4, n_planets=6)
    assert samples
    device = torch.device("cpu")
    batch = collate(samples[: min(8, len(samples))], device)
    assert batch["planet_feats"].dim() == 3
    assert batch["edge_feats"].dim() == 4
    assert batch["edge_feats"].shape[1] == batch["edge_feats"].shape[2]
    assert batch["action_target_idx"].shape == batch["action_mask"].shape
    assert batch["policy_target"].shape == batch["action_mask"].shape

    net = MacroAlphaZeroNet(d_model=16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    stats = train_epoch(net, opt, samples, device=device, batch_size=8,
                        value_coef=5e-3, grad_clip=5.0, rng=random.Random(0))
    assert stats.n > 0 and np.isfinite(stats.loss)


def test_eval_gate_returns_winrate_in_range():
    cfg = _cfg(sims=12, horizon=8)
    a = MacroAlphaZeroNet(d_model=16)
    b = MacroAlphaZeroNet(d_model=16)
    wr = eval_gate(a, b, cfg, n_games=2, seed=1, max_decisions=16,
                   device=torch.device("cpu"))
    assert 0.0 <= wr <= 1.0


def test_scheduled_max_decisions_ramps_to_cap():
    assert default_min_decisions(20) == 80
    assert scheduled_max_decisions(0, min_decisions=80, max_decisions=500, ramp_iters=200) == 80
    mid = scheduled_max_decisions(100, min_decisions=80, max_decisions=500, ramp_iters=200)
    assert 250 < mid < 310
    assert scheduled_max_decisions(199, min_decisions=80, max_decisions=500, ramp_iters=200) == 500
    assert scheduled_max_decisions(500, min_decisions=80, max_decisions=500, ramp_iters=200) == 500


def test_sample_n_planets_in_real_range():
    rng = random.Random(0)
    counts = {sample_n_planets(rng) for _ in range(200)}
    assert min(counts) >= 20
    assert max(counts) <= 40
    assert len(counts) > 10


def test_sample_env_board_uses_live_generator():
    pytest.importorskip("kaggle_environments")
    obs = sample_env_board(seed=42)
    n = len(obs["planets"])
    assert REAL_PLANETS_MIN <= n <= REAL_PLANETS_MAX
    assert obs.get("initial_planets")
    assert "angular_velocity" in obs


def test_train_loop_smoke():
    net = train(
        iterations=1, games_per_iter=2, simulations=12, horizon_turns=6,
        batch_size=16, epochs_per_iter=1, gate_games=2,
        buffer_games=4, seed=0, checkpoint_dir=None,
    )
    assert sum(p.numel() for p in net.parameters()) > 0


def test_save_load_roundtrip(tmp_path):
    net = MacroAlphaZeroNet(d_model=16)
    path = str(tmp_path / "net.pt")
    save_net(net, path, meta={"x": 1})
    loaded = load_net(path)
    pf = torch.randn(1, 4, PLANET_FEATURE_DIM)
    ef = torch.randn(1, 4, 4, EDGE_FEATURE_DIM)
    gf = torch.randn(1, GLOBAL_FEATURE_DIM)
    mask = torch.ones(1, 4)
    with torch.no_grad():
        assert torch.allclose(net.forward_value(pf, ef, gf, mask),
                              loaded.forward_value(pf, ef, gf, mask), atol=1e-6)


def test_mlp_arch_roundtrip_and_detection(tmp_path):
    net = build_net("mlp", d_model=16)
    assert isinstance(net, MacroAlphaZeroNetMLP)
    assert net.arch == "mlp"
    path = str(tmp_path / "mlp.pt")
    save_net(net, path)
    loaded = load_net(path)
    assert loaded.arch == "mlp"
    pf = torch.randn(1, 4, PLANET_FEATURE_DIM)
    gf = torch.randn(1, GLOBAL_FEATURE_DIM)
    mask = torch.ones(1, 4)
    with torch.no_grad():
        assert torch.allclose(net.forward_value(pf, gf, mask),
                              loaded.forward_value(pf, gf, mask), atol=1e-6)


def test_legacy_checkpoint_without_arch_field_detected_as_mlp(tmp_path):
    net = build_net("mlp", d_model=16)
    path = str(tmp_path / "legacy.pt")
    # Mimic an old checkpoint payload (no "arch" field).
    torch.save({"state_dict": net.state_dict(), "d_model": 16, "meta": {}}, path)
    loaded = load_net(path)
    assert loaded.arch == "mlp"


def test_mlp_arch_trains_and_plays():
    cfg = _cfg(sims=16, horizon=8, root_dirichlet_alpha=0.5, value_blend=0.5)
    sp = SelfPlayConfig(max_decisions=16, temperature_moves=4)
    samples = generate_selfplay(2, cfg, selfplay_cfg=sp, seed=7, n_planets=6)
    assert samples
    net = build_net("mlp", d_model=16)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    stats = train_epoch(net, opt, samples, device=torch.device("cpu"), batch_size=8,
                        value_coef=5e-3, grad_clip=5.0, rng=random.Random(0))
    assert stats.n > 0 and np.isfinite(stats.loss)

    og = build_orbit_graph(_board(), player=0)
    mcts = MacroTimedMCTS(og, cfg, root_player=0, rng=random.Random(0))
    attach_net(net, mcts)
    result = mcts.run()
    assert result.actions


# --------------------------------------------------------------------------- #
# Real-env self-play (slow path: plays the actual orbit_wars game).
# --------------------------------------------------------------------------- #
def test_game_outcome_elimination_and_margin():
    from macro_selfplay_real import _game_outcome

    # Player 1 has no planets -> clean win/loss.
    elim = {"planets": [[0, 0, 10.0, 10.0, 1.0, 30, 5],
                        [2, 0, 20.0, 20.0, 1.0, 10, 3]], "fleets": []}
    out = _game_outcome(elim, num_agents=2)
    assert out == [1.0, -1.0]

    # Both alive: bounded, anti-symmetric ship-count margin.
    both = {"planets": [[0, 0, 10.0, 10.0, 1.0, 80, 5],
                        [1, 1, 20.0, 20.0, 1.0, 20, 5]], "fleets": []}
    out = _game_outcome(both, num_agents=2)
    assert -1.0 <= out[1] < 0.0 < out[0] <= 1.0
    assert out[0] == pytest.approx(-out[1])


def test_real_selfplay_produces_labelled_samples():
    from macro_selfplay_real import play_real_game

    cfg = _cfg(sims=6, horizon=8, root_dirichlet_alpha=0.5, value_blend=0.5)
    net = build_net("gnn", d_model=16)
    sp = SelfPlayConfig(max_decisions=12, temperature=1.0, temperature_moves=4)
    samples = play_real_game(cfg, net=net, selfplay_cfg=sp, max_steps=12, seed=3,
                             device=torch.device("cpu"))
    assert samples, "real-env self-play produced no samples"
    for s in samples:
        assert s.player in (0, 1)
        assert -1.0 <= s.z <= 1.0
        assert s.planet_feats.shape[1] == PLANET_FEATURE_DIM
        assert s.edge_feats.shape[2] == EDGE_FEATURE_DIM
        assert s.action_feats.shape[1] == ACTION_FEATURE_DIM
        assert s.policy_target.shape[0] == s.action_feats.shape[0]
    # Samples must collate + train without shape errors.
    net2 = build_net("gnn", d_model=16)
    opt = torch.optim.Adam(net2.parameters(), lr=1e-3)
    stats = train_epoch(net2, opt, samples, device=torch.device("cpu"), batch_size=8,
                        value_coef=1.0, grad_clip=5.0, rng=random.Random(0))
    assert stats.n > 0 and np.isfinite(stats.loss)


# --------------------------------------------------------------------------- #
# Defense steering: reinforce_bonus + single-search sharing.
# --------------------------------------------------------------------------- #
def _result_with(actions_qs):
    """Build a MacroSearchResult from {(action, q): visits} with baseline 0."""
    visits, q_values, actions, priors = {}, {}, {}, {}
    for action, q, v in actions_qs:
        k = action.key()
        visits[k] = v
        q_values[k] = q
        actions[k] = action
        priors[k] = 1.0 / len(actions_qs)
    return MacroSearchResult(
        baseline=0.0, root_value=0.0, visits=visits,
        q_values=q_values, actions=actions, priors=priors,
    )


def test_reinforce_bonus_separates_from_capture_bonus():
    cap = MacroAction(type=MacroActionType.CAPTURE, target=7)
    reinf = MacroAction(type=MacroActionType.REINFORCE, target=3)
    result = _result_with([(cap, 2.0, 5), (reinf, 1.5, 4)])

    rb = result.reinforce_bonus(weight=1.0, min_bonus=1e-6)
    cb = result.capture_bonus(weight=1.0, min_bonus=1e-6)
    assert rb == {3: pytest.approx(1.5)}        # only the owned/defended planet
    assert cb == {7: pytest.approx(2.0)}        # only the capture target
    # Unvisited / non-improving reinforce yields no bonus.
    assert _result_with([(reinf, -1.0, 0)]).reinforce_bonus(weight=1.0, min_bonus=1e-6) == {}


def test_macro_agent_shares_single_search_for_attack_and_defense():
    cfg = _cfg(sims=16, horizon=8)
    net = build_net("gnn", d_model=16)
    agent = MacroAlphaZeroAgent(net, macro_cfg=cfg)

    from orbit_graph import build_orbit_graph
    og = build_orbit_graph(_board(), player=0)

    agent.act(_board(), 0, current_turn=0, max_turns=80)
    # One act() => one cached search exposed for self-play recording.
    assert agent.last_mcts is not None
    assert agent.last_result is not None

    # Within a single turn the attack + defense hooks reuse one search object.
    agent._macro_cache = None
    b1 = agent._attack_priority_bonus(og, None, [], omega=0.0, current_turn=0, max_turns=80)
    mcts_after_attack = agent.last_mcts
    b2 = agent._defense_priority_bonus(og, [], omega=0.0, current_turn=0, max_turns=80)
    assert agent.last_mcts is mcts_after_attack  # no second search was run
    del b1, b2


def test_defense_bonus_weight_zero_disables_defense_hook():
    cfg = _cfg(sims=16, horizon=8, defense_bonus_weight=0.0)
    net = build_net("gnn", d_model=16)
    agent = MacroAlphaZeroAgent(net, macro_cfg=cfg)
    from orbit_graph import build_orbit_graph
    og = build_orbit_graph(_board(), player=0)
    assert agent._defense_priority_bonus(og, [], omega=0.0, current_turn=0, max_turns=80) is None


# --------------------------------------------------------------------------- #
# Capture economics: features (B) + denial reward (A) + warm-start partial load.
# --------------------------------------------------------------------------- #
def test_macro_features_expose_capture_economics():
    from orbit_graph import build_orbit_graph
    # P0 home (lots of ships) next to a weakly-held enemy planet -> cheap kill.
    planets = [
        [0, 0, 20.0, 50.0, 1.0, 90, 3],
        [1, 1, 30.0, 50.0, 1.0, 4, 8],    # enemy, weak garrison, high prod
        [2, -1, 20.0, 80.0, 1.0, 50, 1],  # neutral, well garrisoned, low prod
    ]
    board = {"planets": planets, "fleets": [], "initial_planets": planets,
             "angular_velocity": 0.0, "comet_planet_ids": [], "comets": []}
    og = build_orbit_graph(board, player=0)
    cfg = _cfg(sims=8, horizon=8)
    mcts = MacroTimedMCTS(og, cfg, root_player=0, rng=random.Random(0))
    st = mcts._initial_state
    actions = mcts._macro_actions(st)
    cap = next((a for a in actions if a.type is MacroActionType.CAPTURE
                and int(a.target) == 1), None)
    assert cap is not None, "expected a capture action on the weak enemy planet"
    feats = mcts._macro_features(st, cap)
    # Enemy capture must flag denial; cheap target => low defender ratio.
    assert feats["enemy_denial_norm"] > 0.0
    assert feats["defender_ratio"] >= 0.0
    assert feats["capture_efficiency"] >= 0.0
    # All three names are part of the packed action vector.
    vec = action_feature_vector(feats)
    assert vec.shape[0] == ACTION_FEATURE_DIM


def test_denial_weight_makes_enemy_capture_score_higher_than_neutral():
    from orbit_timed_mcts import TimedCaptureMCTS, _TimedState, _CaptureEvent
    from orbit_graph import build_orbit_graph
    # Two identical-production targets reachable from root: one enemy, one neutral.
    planets = [
        [0, 0, 20.0, 50.0, 1.0, 200, 1],
        [1, 1, 26.0, 50.0, 1.0, 1, 5],     # enemy
        [2, -1, 20.0, 56.0, 1.0, 1, 5],    # neutral, same production
    ]
    board = {"planets": planets, "fleets": [], "initial_planets": planets,
             "angular_velocity": 0.0, "comet_planet_ids": [], "comets": []}
    og = build_orbit_graph(board, player=0)

    def _score_capture(target, denial_w):
        cfg = _cfg(sims=8, horizon=10)
        cfg = replace(cfg, timed=replace(cfg.timed, denial_income_weight=denial_w))
        mcts = MacroTimedMCTS(og, cfg, root_player=0, rng=random.Random(0))
        st = mcts._initial_state
        # Resolve an immediate capture of `target` by root and score the result.
        owners = st.owner_map(); ships = st.ship_map()
        # Overwhelming wave so the capture certainly resolves (defender garrison
        # plus one turn of production must be strictly exceeded).
        ev = _CaptureEvent(arrival_turn=int(st.turn) + 1, owner=0,
                           target=int(target), ships=int(ships[int(target)]) + 200)
        nxt = mcts._state(turn=st.turn, player_to_move=0, owners=owners,
                          ships=ships, pending=(ev,), reward=0.0)
        return mcts._score(mcts._advance_one_turn(nxt))

    # With denial off, enemy and neutral capture of equal-prod targets score the
    # same; with denial on, the enemy capture scores strictly higher.
    enemy_off = _score_capture(1, 0.0)
    neutral_off = _score_capture(2, 0.0)
    enemy_on = _score_capture(1, 0.5)
    assert enemy_on > enemy_off
    assert enemy_on > neutral_off


def test_warm_start_partial_load_across_action_dim_change(tmp_path):
    # Save a net, then load it back: identical schema loads cleanly (no skips).
    net = build_net("gnn", d_model=16)
    path = str(tmp_path / "warm.pt")
    save_net(net, path)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # would raise if anything were skipped
        loaded = load_net(path)
    assert loaded.arch == "gnn"

    # Simulate an old checkpoint whose prior head used a *smaller* action dim:
    # the encoder/value tensors load, the prior head is re-initialised.
    from macro_alphazero_net import _load_state_dict_compatible
    fresh = build_net("gnn", d_model=16)
    sd = dict(fresh.state_dict())
    prior_key = next(k for k in sd if k.startswith("prior_head") and k.endswith("weight"))
    sd[prior_key] = sd[prior_key][:, :-3].clone()  # drop the 3 new feature cols
    target = build_net("gnn", d_model=16)
    skipped = _load_state_dict_compatible(target, sd)
    assert any(k.startswith("prior_head") for k in skipped)
    assert not any(k.startswith("encoder") for k in skipped)
