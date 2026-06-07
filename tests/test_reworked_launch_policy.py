"""Unit tests for the v8 source-centric launch policy (allocator-free).

Covers the masked Dirichlet helpers, the source/dest masks, and the
:class:`ReworkedLaunchPolicy` act/evaluate consistency.  End-to-end episode
collection + PPO is exercised separately as a smoke run.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reworked_model import (  # noqa: E402
    ReworkedLaunchPolicy,
    masked_dirichlet_sample_log_prob, masked_dirichlet_log_prob,
    masked_dirichlet_mean, masked_dirichlet_entropy,
    REWORKED_NODE_FEATURE_DIM, REWORKED_EDGE_FEATURE_DIM, REWORKED_GLOBAL_FEATURE_DIM,
)
from reworked_pipeline import BoardState  # noqa: E402
from reworked_rollout import _source_dest_masks, MIN_SEND, P_MAX  # noqa: E402


def _board(*, owner, ships):
    n = len(owner)
    return BoardState(
        n=n, player=0,
        owner=np.asarray(owner, dtype=np.int64),
        ships=np.asarray(ships, dtype=np.float32),
        production=np.ones(n, dtype=np.float32),
        radius=np.ones(n, dtype=np.float32),
        is_rotating=np.zeros(n, dtype=np.float32),
        is_comet=np.zeros(n, dtype=np.float32),
        distance_matrix=np.ones((n, n), dtype=np.float32),
        crosses_sun=np.zeros((n, n), dtype=np.float32),
        remaining_turns=100,
    )


def _masks(B, P, source_rows):
    src = torch.zeros(B, P, dtype=torch.bool)
    dest = torch.zeros(B, P, P, dtype=torch.bool)
    for b in range(B):
        for h in source_rows:
            src[b, h] = True
            for t in range(P):
                if t != h:
                    dest[b, h, t] = True
    return src, dest


# --------------------------------------------------------------------------- masks
def test_source_dest_masks_basic():
    # player 0 owns planets 0 and 2; planet 2 has too few ships to send.
    state = _board(owner=[0, 1, 0, -1], ships=[10.0, 5.0, 0.0, 3.0])
    src, dest = _source_dest_masks(state, P_MAX, 0, min_send=MIN_SEND)
    assert src.shape == (P_MAX,)
    assert dest.shape == (P_MAX, P_MAX)
    assert src[0] and not src[1] and not src[2] and not src[3]  # 2 has 0 ships
    # source row 0: every real planet except itself is a valid target
    assert dest[0, 1] and dest[0, 2] and dest[0, 3] and not dest[0, 0]
    # non-source rows have no targets
    assert not dest[1].any() and not dest[2].any() and not dest[3].any()
    # padded region is all False
    assert not src[4:].any() and not dest[4:, :].any()


# ----------------------------------------------------------------- masked Dirichlet
def test_masked_dirichlet_sample_is_simplex_on_valid_cols():
    torch.manual_seed(0)
    B, P = 3, 5
    conc = torch.rand(B, P, P) + 0.5
    _, dest = _masks(B, P, source_rows=[0, 2])
    w, logp = masked_dirichlet_sample_log_prob(conc, dest)
    # valid source rows sum to 1; everything else is 0
    for b in range(B):
        for h in range(P):
            s = float(w[b, h].sum())
            if h in (0, 2):
                assert abs(s - 1.0) < 1e-4
            else:
                assert s < 1e-6
    # invalid columns are exactly zero
    assert float((w * (~dest)).abs().sum()) < 1e-6
    assert torch.isfinite(logp).all()
    # re-evaluated log-prob of the same sample matches the sampling log-prob
    logp2 = masked_dirichlet_log_prob(conc, w, dest)
    assert torch.allclose(logp, logp2, atol=1e-5)


def test_masked_dirichlet_mean_and_entropy_finite():
    B, P = 2, 4
    conc = torch.rand(B, P, P) + 0.5
    _, dest = _masks(B, P, source_rows=[1])
    mean = masked_dirichlet_mean(conc, dest)
    assert abs(float(mean[0, 1].sum()) - 1.0) < 1e-5
    assert float(mean[0, 0].sum()) < 1e-6  # non-source row -> 0
    ent = masked_dirichlet_entropy(conc, dest)
    assert ent.shape == (B,) and torch.isfinite(ent).all()


# ------------------------------------------------------------------- launch policy
def test_launch_policy_act_evaluate_consistency():
    torch.manual_seed(0)
    B, P = 2, 6
    pol = ReworkedLaunchPolicy(d_model=32, n_layers=2)
    nf = torch.randn(B, P, REWORKED_NODE_FEATURE_DIM)
    ef = torch.randn(B, P, P, REWORKED_EDGE_FEATURE_DIM)
    gf = torch.randn(B, REWORKED_GLOBAL_FEATURE_DIM)
    pm = torch.ones(B, P, dtype=torch.bool)
    src, dest = _masks(B, P, source_rows=[0, 1])

    action, logp, value = pol.act(
        node_feats=nf, edge_feats=ef, global_feats=gf, planet_mask=pm,
        source_mask=src, dest_mask=dest)
    assert action["commit"].shape == (B, P)
    assert action["split_w"].shape == (B, P, P)
    assert logp.shape == (B,) and value.shape == (B,)
    assert torch.isfinite(logp).all()

    ev = pol.evaluate_actions(
        node_feats=nf, edge_feats=ef, global_feats=gf, planet_mask=pm,
        source_mask=src, dest_mask=dest, action=action)
    assert torch.allclose(logp, ev["log_prob"], atol=1e-4)
    assert torch.isfinite(ev["entropy"]).all()
    assert ev["value"].shape == (B,)


def test_launch_policy_deterministic_reproducible():
    torch.manual_seed(1)
    B, P = 1, 5
    pol = ReworkedLaunchPolicy(d_model=16, n_layers=2)
    nf = torch.randn(B, P, REWORKED_NODE_FEATURE_DIM)
    ef = torch.randn(B, P, P, REWORKED_EDGE_FEATURE_DIM)
    gf = torch.randn(B, REWORKED_GLOBAL_FEATURE_DIM)
    pm = torch.ones(B, P, dtype=torch.bool)
    src, dest = _masks(B, P, source_rows=[0])
    a1, lp1, _ = pol.act(node_feats=nf, edge_feats=ef, global_feats=gf, planet_mask=pm,
                         source_mask=src, dest_mask=dest, deterministic=True)
    a2, lp2, _ = pol.act(node_feats=nf, edge_feats=ef, global_feats=gf, planet_mask=pm,
                         source_mask=src, dest_mask=dest, deterministic=True)
    assert torch.allclose(a1["commit"], a2["commit"])
    assert torch.allclose(a1["split_w"], a2["split_w"])
    assert torch.allclose(lp1, lp2)


if __name__ == "__main__":
    test_source_dest_masks_basic()
    test_masked_dirichlet_sample_is_simplex_on_valid_cols()
    test_masked_dirichlet_mean_and_entropy_finite()
    test_launch_policy_act_evaluate_consistency()
    test_launch_policy_deterministic_reproducible()
    print("ok")
