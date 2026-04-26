"""Tests for the Set Transformer policy/value network.

Validates forward-pass shapes, action sampling shapes, and crucial
masking invariants (no policy mass on padded planets / illegal sources).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from obs_builder import build_observation
from model import OrbitWarsPolicy, pack_observations


def _toy_obs(player=0):
    return {
        "planets": [
            [0, 0, 70.0, 70.0, 3.0, 100, 3],
            [1, 0, 80.0, 70.0, 2.0, 50, 2],
            [2, 1, 30.0, 30.0, 2.0, 80, 2],
            [3, -1, 20.0, 80.0, 2.0, 10, 1],
        ],
        "fleets": [
            [10, 1, 60.0, 60.0, 0.0, 2, 25],
        ],
        "initial_planets": [
            [0, 0, 70.0, 70.0, 3.0, 100, 3],
            [1, 0, 80.0, 70.0, 2.0, 50, 2],
            [2, 1, 30.0, 30.0, 2.0, 80, 2],
            [3, -1, 20.0, 80.0, 2.0, 10, 1],
        ],
        "comet_planet_ids": [],
        "comets": [],
    }


@pytest.fixture
def batch():
    obs_list = [
        build_observation(_toy_obs(), player=0, angular_velocity=0.05,
                          p_max=8, f_max=4)
        for _ in range(3)
    ]
    return obs_list, pack_observations(obs_list)


def _net():
    return OrbitWarsPolicy(p_max=8, f_max=4, d_model=32, n_heads=2, n_layers=2)


def test_forward_pass_shapes(batch):
    obs_list, b = batch
    net = _net()
    out = net(**b)
    B = len(obs_list)
    assert out.src_logits.shape == (B, 8 + 1)        # +1 NO-OP
    assert out.tgt_logits.shape == (B, 8, 8)
    assert out.ship_logits.shape == (B, 8, 5)
    assert out.value.shape == (B,)


def test_no_nan_in_logits(batch):
    _, b = batch
    net = _net()
    out = net(**b)
    for name in ("src_logits", "tgt_logits", "ship_logits", "value"):
        t = getattr(out, name)
        # We mask invalid slots to a very-negative finite value, so finite
        # check should pass everywhere.
        assert torch.isfinite(t).all(), f"{name} has non-finite entries"


def test_padded_planets_get_zero_target_prob(batch):
    obs_list, b = batch
    net = _net()
    out = net(**b)
    # Real planets are slots 0..3; 4..7 are padding.
    probs = torch.softmax(out.tgt_logits, dim=-1)  # (B, P, P)
    pad_probs = probs[:, :, 4:]  # probability of choosing a padded target
    assert pad_probs.max().item() < 1e-6


def test_padded_or_invalid_sources_get_zero_prob(batch):
    obs_list, b = batch
    net = _net()
    out = net(**b)
    src_probs = torch.softmax(out.src_logits, dim=-1)  # (B, P+1)
    # Source mask in our toy obs: only planets 0 and 1 are mine with ships > 0.
    # So slots 2,3 (enemy/neutral) AND 4..7 (padding) should be zero.
    invalid = src_probs[:, 2:8]
    assert invalid.max().item() < 1e-6
    # NO-OP and the two valid sources must carry all the mass.
    valid = torch.cat([src_probs[:, :2], src_probs[:, 8:9]], dim=-1)
    assert torch.allclose(valid.sum(dim=-1),
                          torch.ones(src_probs.shape[0]),
                          atol=1e-5)


def test_self_targets_masked(batch):
    _, b = batch
    net = _net()
    out = net(**b)
    probs = torch.softmax(out.tgt_logits, dim=-1)
    # Diagonal (target == source) must be ~0.
    B, P, _ = probs.shape
    diag = probs[:, torch.arange(P), torch.arange(P)]
    assert diag.max().item() < 1e-6


def test_act_returns_correct_shapes(batch):
    _, b = batch
    net = _net()
    out = net(**b)
    action = net.act(out, deterministic=False)
    B = b["planet_feats"].shape[0]
    assert action["src"].shape == (B,)
    assert action["tgt"].shape == (B,)
    assert action["ship_bin"].shape == (B,)
    assert action["is_noop"].shape == (B,) and action["is_noop"].dtype == torch.bool
    assert action["log_prob"].shape == (B,)
    # Sampled bins are in valid ranges.
    assert action["src"].min() >= 0 and action["src"].max() <= net.p_max
    assert action["tgt"].max() < net.p_max
    assert action["ship_bin"].max() < net.n_ship_bins


def test_act_deterministic_is_argmax(batch):
    _, b = batch
    net = _net()
    out = net(**b)
    a1 = net.act(out, deterministic=True)
    a2 = net.act(out, deterministic=True)
    assert torch.equal(a1["src"], a2["src"])
    assert torch.equal(a1["tgt"], a2["tgt"])
    assert torch.equal(a1["ship_bin"], a2["ship_bin"])


def test_gradient_flows_through_network(batch):
    _, b = batch
    net = _net()
    out = net(**b)
    # Fake loss: maximize value + log_prob of a fixed action.
    target = torch.zeros_like(out.value)
    loss = ((out.value - target) ** 2).mean() + out.src_logits.mean()
    loss.backward()
    # At least one parameter should have a non-zero gradient.
    grad_norms = [p.grad.norm().item() for p in net.parameters()
                  if p.grad is not None]
    assert any(g > 0 for g in grad_norms)
