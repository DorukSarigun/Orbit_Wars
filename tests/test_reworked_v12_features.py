"""Tests for the v12 feature expansion (node 16->23, edge 4->9) + migration."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from obs_builder import build_observation, PF_IDX_COMET_LIFE, PLANET_FEATURE_DIM
from reworked_features import (
    edge_features,
    EDGE_IDX_ETA,
    EDGE_IDX_DPHI_SIN,
    EDGE_IDX_DPHI_COS,
    EDGE_IDX_SHIPS_AT_ARRIVAL,
    EDGE_IDX_RACE_MARGIN,
    REWORKED_EDGE_FEATURE_DIM,
)
from reworked_model import (
    REWORKED_NODE_FEATURE_DIM,
    NUM_EXTRA_NODE_METRICS,
    ReworkedEdgePolicy,
)
from reworked_pipeline import pack_reworked_batch
from reworked_rollout import _policy_feature_views, _recapture_array


def make_obs(*, planets, fleets=None):
    """[id, owner, x, y, radius, ships, production] / fleet rows
    [id, owner, x, y, angle, from_planet_id, ships]."""
    return {
        "planets": planets,
        "fleets": fleets or [],
        "angular_velocity": 0.02,
        "next_fleet_id": 100,
    }


def _board():
    return make_obs(planets=[
        [0, 0, 30.0, 50.0, 3.0, 40, 3],    # mine
        [1, 1, 70.0, 50.0, 3.0, 30, 4],    # enemy
        [2, -1, 50.0, 70.0, 2.0, 10, 2],   # neutral
    ])


def _packed(obs_dict, *, player=0, recap=None):
    obs = build_observation(obs_dict, player=player, angular_velocity=0.02,
                            p_max=6, f_max=4, turn=10, max_turns=300)
    return obs, pack_reworked_batch(
        [(obs, player)], p_max=6,
        recapture_counts=None if recap is None else [recap])


def test_packed_shapes_are_v12():
    _, batch = _packed(_board())
    assert batch["node_feats"].shape[-1] == REWORKED_NODE_FEATURE_DIM == 23
    assert batch["edge_feats"].shape[-1] == REWORKED_EDGE_FEATURE_DIM == 9
    assert PLANET_FEATURE_DIM == 35 and PF_IDX_COMET_LIFE == 34


def test_edge_eta_and_phase_channels():
    _, batch = _packed(_board())
    ef = batch["edge_feats"][0].numpy()
    # ETA between distinct real planets is positive and normalised.
    assert ef[0, 1, EDGE_IDX_ETA] > 0.0
    assert ef[0, 1, EDGE_IDX_ETA] <= 2.0
    # Orbital phase difference is antisymmetric in sin, symmetric in cos.
    assert ef[0, 1, EDGE_IDX_DPHI_SIN] == pytest.approx(
        -ef[1, 0, EDGE_IDX_DPHI_SIN], abs=1e-5)
    assert ef[0, 1, EDGE_IDX_DPHI_COS] == pytest.approx(
        ef[1, 0, EDGE_IDX_DPHI_COS], abs=1e-5)


def test_edge_ships_at_arrival_grows_with_production():
    _, batch = _packed(_board())
    ef = batch["edge_feats"][0].numpy()
    # Owned target (planet 1: 30 ships, prod 4) grows over the flight;
    # neutral target (planet 2: 10 ships) does not.
    assert ef[0, 1, EDGE_IDX_SHIPS_AT_ARRIVAL] > 30.0 / 50.0
    assert ef[0, 2, EDGE_IDX_SHIPS_AT_ARRIVAL] == pytest.approx(10.0 / 50.0, abs=1e-4)


def test_edge_race_margin_defaults_to_one_without_inbound():
    _, batch = _packed(_board())
    ef = batch["edge_feats"][0].numpy()
    pm = batch["planet_mask"][0].numpy()
    n = int(pm.sum())
    # No enemy fleets in flight -> no race to lose anywhere on the board.
    assert np.allclose(ef[:n, :n, EDGE_IDX_RACE_MARGIN], 1.0)


def test_node_recapture_channel_routed_from_pack():
    recap = np.zeros(6, dtype=np.float32)
    recap[1] = 2.0   # captured planet slot 1 twice already
    _, batch = _packed(_board(), recap=recap)
    nf = batch["node_feats"][0].numpy()
    recap_col = REWORKED_NODE_FEATURE_DIM - 3   # [.., recap, comet_life, frontier]
    assert nf[1, recap_col] == pytest.approx(2.0 / 4.0)
    assert nf[0, recap_col] == 0.0


def test_node_frontier_channel():
    _, batch = _packed(_board())
    nf = batch["node_feats"][0].numpy()
    frontier_col = REWORKED_NODE_FEATURE_DIM - 1
    # The enemy planet itself sits on the frontier (distance 0 to an enemy).
    assert nf[1, frontier_col] == pytest.approx(0.0, abs=1e-5)
    assert nf[0, frontier_col] > 0.0


def test_feature_views_slice_for_legacy_policy():
    _, batch = _packed(_board())
    legacy = ReworkedEdgePolicy(node_dim=16, edge_dim=4, d_model=32,
                                n_layers=1, n_heads=2)
    nf, ef = _policy_feature_views(legacy, batch)
    assert nf.shape[-1] == 16 and ef.shape[-1] == 4
    # New-layout policy gets the full tensors.
    fresh = ReworkedEdgePolicy(d_model=32, n_layers=1, n_heads=2)
    nf2, ef2 = _policy_feature_views(fresh, batch)
    assert nf2.shape[-1] == 23 and ef2.shape[-1] == 9


def test_recapture_array_builds_from_planet_ids():
    obs, _ = _packed(_board())
    arr = _recapture_array({1: 3}, obs, 6)
    assert arr is not None
    slot = int(np.where(np.asarray(obs.planet_ids).reshape(-1) == 1)[0][0])
    assert arr[slot] == 3.0
    assert _recapture_array({}, obs, 6) is None


def test_migration_expands_legacy_checkpoint(tmp_path):
    from reworked_migrate_features import migrate_checkpoint
    from reworked_train import load_checkpoint

    old = ReworkedEdgePolicy(node_dim=16, edge_dim=4, d_model=32,
                             n_layers=1, n_heads=2)
    src = tmp_path / "old.pt"
    torch.save({
        "iteration": 7,
        "policy_state_dict": old.state_dict(),
        "optimizer_state_dict": {"dummy": 1},
        "arch": {"node_dim": 16, "edge_dim": 4, "global_dim": 14,
                 "d_model": 32, "n_layers": 1, "n_heads": 2,
                 "aux_owner": True, "eta_value_feature": False,
                 "policy_kind": "edge"},
    }, src)

    dst = tmp_path / "new.pt"
    migrate_checkpoint(src, dst)
    policy, ckpt = load_checkpoint(dst)
    assert ckpt["arch"]["node_dim"] == 23
    assert ckpt["arch"]["edge_dim"] == 9
    assert "optimizer_state_dict" not in ckpt

    # Forward pass on the new dims works.
    B, P = 2, 5
    out = policy.forward(
        torch.randn(B, P, 23), torch.randn(B, P, P, 9),
        torch.randn(B, 14), torch.ones(B, P, dtype=torch.bool))
    assert out["gate_logit"].shape == (B, P, P)

    # Zero-padded Linear columns: new channels do not change the stem's
    # linear response (LayerNorm statistics aside).
    lin = policy.encoder.node_embed[1].weight
    assert torch.all(lin[:, 16:] == 0)
    lin_e = policy.encoder.edge_embed[1].weight
    assert torch.all(lin_e[:, 4:] == 0)


def test_act_runs_end_to_end_on_v12_features():
    _, batch = _packed(_board())
    policy = ReworkedEdgePolicy(d_model=32, n_layers=1, n_heads=2)
    P = batch["node_feats"].shape[1]
    em = batch["planet_mask"][:, :, None] & batch["planet_mask"][:, None, :]
    em = em & ~torch.eye(P, dtype=torch.bool)[None]
    action, tlp, tv = policy.act(
        node_feats=batch["node_feats"], edge_feats=batch["edge_feats"],
        global_feats=batch["global_feats"], planet_mask=batch["planet_mask"],
        edge_mask=em)
    assert action["gate"].shape == (1, P, P)
    assert tlp.shape == (1, P) and tv.shape == (1, P)
    assert NUM_EXTRA_NODE_METRICS == 7
