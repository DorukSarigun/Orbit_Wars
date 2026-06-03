"""Permutation-equivariance / invariance check for the reworked policy.

The hierarchical policy must be **permutation equivariant** over planet slots:
relabelling planets by a permutation π must permute every *per-planet* output
(attack/defend/priority logits, commit Beta params, node tokens) by π, while
every *graph-level* scalar output (value, global posture, graph summary) stays
**invariant**.  No output may depend on the arbitrary planet ordering.

Run::

    python -m pytest tests/test_reworked_permutation.py -q
    # or directly:
    python tests/test_reworked_permutation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reworked_model import (  # noqa: E402
    ReworkedHierarchicalPolicy,
    ReworkedPlanetGraphEncoder,
    REWORKED_NODE_FEATURE_DIM,
    REWORKED_EDGE_FEATURE_DIM,
    REWORKED_GLOBAL_FEATURE_DIM,
)


def _random_inputs(B=2, P=9, n_valid=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    node = torch.randn(B, P, REWORKED_NODE_FEATURE_DIM, generator=g)
    edge = torch.randn(B, P, P, REWORKED_EDGE_FEATURE_DIM, generator=g)
    glob = torch.randn(B, REWORKED_GLOBAL_FEATURE_DIM, generator=g)
    mask = torch.ones(B, P, dtype=torch.bool)
    if n_valid is not None:
        mask[:, n_valid:] = False
        node[:, n_valid:] = 0.0
        edge[:, n_valid:, :] = 0.0
        edge[:, :, n_valid:] = 0.0
    # attack = subset of valid, defend = disjoint subset of valid
    attack = torch.zeros(B, P, dtype=torch.bool)
    defend = torch.zeros(B, P, dtype=torch.bool)
    for b in range(B):
        valid = torch.where(mask[b])[0].tolist()
        for i, idx in enumerate(valid):
            if i % 2 == 0:
                attack[b, idx] = True
            else:
                defend[b, idx] = True
    return node, edge, glob, mask, attack, defend


def _permute(perm, node, edge, glob, mask, attack, defend):
    node_p = node[:, perm]
    edge_p = edge[:, perm][:, :, perm]
    mask_p = mask[:, perm]
    attack_p = attack[:, perm]
    defend_p = defend[:, perm]
    return node_p, edge_p, glob, mask_p, attack_p, defend_p


def _finite(x):
    # Replace the masked_fill sentinel (finfo.min) with 0 so it doesn't blow up
    # allclose; masked positions are compared structurally via the permutation.
    return torch.where(torch.isfinite(x) & (x > -1e30), x, torch.zeros_like(x))


def check(B=2, P=9, n_valid=None, seed=0, atol=1e-5):
    torch.manual_seed(123)
    policy = ReworkedHierarchicalPolicy(d_model=64, n_layers=3, aux_owner=True).eval()

    node, edge, glob, mask, attack, defend = _random_inputs(B, P, n_valid, seed)
    perm = torch.randperm(P)

    with torch.no_grad():
        o1 = policy.forward(node, edge, glob, mask, attack, defend)
        np_, ep_, gp_, mp_, ap_, dp_ = _permute(perm, node, edge, glob, mask, attack, defend)
        o2 = policy.forward(np_, ep_, gp_, mp_, ap_, dp_)

    results = {}
    # Per-planet outputs must permute by `perm`.
    for key in ["attack_logits", "defend_logits", "priority_logits",
                "commit_alpha", "commit_beta", "node_tokens", "aux_owner_logits"]:
        a = _finite(o1[key])[:, perm]
        b = _finite(o2[key])
        results[key] = torch.allclose(a, b, atol=atol)
    # Graph-level scalars must be invariant.
    for key in ["value", "posture_pw_alpha", "posture_pw_beta",
                "posture_h_alpha", "posture_h_beta"]:
        results[key] = torch.allclose(o1[key], o2[key], atol=atol)
    return results


def test_permutation_equivariance_full():
    res = check(n_valid=None)
    assert all(res.values()), f"equivariance failed (full): {res}"


def test_permutation_equivariance_padded():
    # With padded (invalid) slots the property must still hold.
    res = check(P=12, n_valid=7)
    assert all(res.values()), f"equivariance failed (padded): {res}"


if __name__ == "__main__":
    for tag, kw in [("full", dict(n_valid=None)), ("padded", dict(P=12, n_valid=7))]:
        r = check(**kw)
        ok = all(r.values())
        print(f"[{tag}] {'PASS' if ok else 'FAIL'}")
        for k, v in r.items():
            print(f"    {'ok ' if v else 'BAD'} {k}")
    print("done")
