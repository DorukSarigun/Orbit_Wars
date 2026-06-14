"""Tests for the AlphaZero-style macro net, self-play and training loop."""

from __future__ import annotations

import random

import numpy as np
import torch

from orbit_graph import build_orbit_graph
from orbit_timed_mcts import TimedMCTSConfig
from orbit_macro_mcts_agent import MacroMCTSConfig, MacroTimedMCTS
from macro_alphazero_net import (
    ACTION_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    PLANET_FEATURE_DIM,
    MacroAlphaZeroNet,
    abstract_state_arrays,
    attach_net,
    load_net,
    save_net,
)
from macro_selfplay import SelfPlayConfig, generate_selfplay, random_board
from train_macro_alphazero import collate, eval_gate, train, train_epoch


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
    gf = torch.randn(2, GLOBAL_FEATURE_DIM)
    mask = torch.ones(2, 5)
    v = net.forward_value(pf, gf, mask)
    assert v.shape == (2,)
    af = torch.randn(2, 4, ACTION_FEATURE_DIM)
    am = torch.ones(2, 4, dtype=torch.bool)
    logits = net.forward_prior(af, am)
    assert logits.shape == (2, 4)


def test_value_mask_invariance_to_padding():
    net = MacroAlphaZeroNet(d_model=16).eval()
    pf = torch.randn(1, 3, PLANET_FEATURE_DIM)
    gf = torch.randn(1, GLOBAL_FEATURE_DIM)
    mask = torch.ones(1, 3)
    v_small = net.forward_value(pf, gf, mask)
    # Pad with two dummy planets that are masked out -> value must be identical.
    pf_pad = torch.cat([pf, torch.randn(1, 2, PLANET_FEATURE_DIM)], dim=1)
    mask_pad = torch.cat([mask, torch.zeros(1, 2)], dim=1)
    v_pad = net.forward_value(pf_pad, gf, mask_pad)
    assert torch.allclose(v_small, v_pad, atol=1e-5)


def test_abstract_state_arrays_dims():
    og = build_orbit_graph(_board(), player=0)
    mcts = MacroTimedMCTS(og, _cfg(), root_player=0, rng=random.Random(0))
    pf, gf = abstract_state_arrays(mcts, mcts._initial_state)
    assert pf.shape == (4, PLANET_FEATURE_DIM)
    assert gf.shape == (GLOBAL_FEATURE_DIM,)


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


def test_collate_and_train_epoch_runs():
    cfg = _cfg(sims=16, horizon=8, root_dirichlet_alpha=0.5)
    sp = SelfPlayConfig(max_decisions=16, temperature_moves=4)
    samples = generate_selfplay(2, cfg, selfplay_cfg=sp, seed=4, n_planets=6)
    assert samples
    device = torch.device("cpu")
    batch = collate(samples[: min(8, len(samples))], device)
    assert batch["planet_feats"].dim() == 3
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
                   n_planets=6, device=torch.device("cpu"))
    assert 0.0 <= wr <= 1.0


def test_train_loop_smoke():
    net = train(
        iterations=1, games_per_iter=2, simulations=12, horizon_turns=6,
        n_planets=6, batch_size=16, epochs_per_iter=1, gate_games=2,
        buffer_games=4, seed=0, checkpoint_dir=None,
    )
    assert sum(p.numel() for p in net.parameters()) > 0


def test_save_load_roundtrip(tmp_path):
    net = MacroAlphaZeroNet(d_model=16)
    path = str(tmp_path / "net.pt")
    save_net(net, path, meta={"x": 1})
    loaded = load_net(path)
    pf = torch.randn(1, 4, PLANET_FEATURE_DIM)
    gf = torch.randn(1, GLOBAL_FEATURE_DIM)
    mask = torch.ones(1, 4)
    with torch.no_grad():
        assert torch.allclose(net.forward_value(pf, gf, mask),
                              loaded.forward_value(pf, gf, mask), atol=1e-6)
