from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import pack_observations
from model_grouped_planet_edge import (
    EdgeAwareGroupedPlanetOrbitWarsPolicy,
    PairwiseEdgeGroupedPlanetOrbitWarsPolicy,
)
from obs_builder import build_observation


PLANETS = [
    [0, 0, 70.0, 70.0, 3.0, 100, 3],
    [1, 0, 82.0, 68.0, 2.0, 50, 2],
    [2, 1, 30.0, 30.0, 2.0, 80, 2],
    [3, -1, 20.0, 80.0, 2.0, 10, 1],
    [4, 1, 60.0, 25.0, 4.0, 120, 3],
]

FLEETS = [
    [10, 1, 46.0, 42.0, 0.2, 2, 25],
    [11, 0, 76.0, 69.0, -0.1, 1, 12],
]

PERMUTATION = [2, 0, 4, 3, 1]


POLICIES = [
    EdgeAwareGroupedPlanetOrbitWarsPolicy,
    PairwiseEdgeGroupedPlanetOrbitWarsPolicy,
]


def _make_env_obs(planets, fleets):
    return {
        "planets": copy.deepcopy(planets),
        "initial_planets": copy.deepcopy(planets),
        "fleets": copy.deepcopy(fleets),
        "comet_planet_ids": [],
        "comets": [],
        "step": 17,
    }


def _alignment(original_ids, shuffled_ids):
    original_real_ids = [int(pid) for pid in original_ids if int(pid) >= 0]
    shuffled_slot_by_id = {
        int(pid): slot
        for slot, pid in enumerate(shuffled_ids)
        if int(pid) >= 0
    }
    return original_real_ids, torch.tensor(
        [shuffled_slot_by_id[pid] for pid in original_real_ids],
        dtype=torch.long,
    )


def _capture_head_io(policy, batch):
    names = [
        "global_feat_proj",
        "summary_head",
        "src_head",
        "noop_head",
        "pair_edge_proj",
        "ship_head",
        "value_head",
        "aux_owner_head",
    ]
    if hasattr(policy, "tgt_query"):
        names.extend(["tgt_query", "tgt_key", "pair_edge_score"])
    if hasattr(policy, "pair_score_head"):
        names.append("pair_score_head")

    captured = {}
    hooks = []
    modules = dict(policy.named_modules())
    for name in names:
        module = modules[name]

        def _hook(_module, inputs, output, *, _name=name):
            captured[_name] = {
                "input": inputs[0].detach().clone(),
                "output": output.detach().clone(),
            }

        hooks.append(module.register_forward_hook(_hook))
    with torch.no_grad():
        out = policy(**batch)
    for hook in hooks:
        hook.remove()
    return out, captured


def _assert_node_tensor_equivariant(original, shuffled, align):
    assert torch.allclose(original[0, : align.numel()], shuffled[0, align], atol=1e-5, rtol=1e-5)


def _assert_pair_tensor_equivariant(original, shuffled, align):
    p_real = align.numel()
    assert torch.allclose(
        original[0, :p_real, :p_real],
        shuffled[0][align][:, align],
        atol=1e-5,
        rtol=1e-5,
    )


def _assert_invariant(original, shuffled):
    assert torch.allclose(original, shuffled, atol=1e-5, rtol=1e-5)


def _assert_head_io_equivariant(name, original, shuffled, align):
    if name in {"src_head", "noop_head", "tgt_query", "tgt_key", "aux_owner_head"}:
        _assert_node_tensor_equivariant(original[name]["input"], shuffled[name]["input"], align)
        _assert_node_tensor_equivariant(original[name]["output"], shuffled[name]["output"], align)
    elif name in {"pair_edge_proj", "pair_edge_score", "ship_head", "pair_score_head"}:
        _assert_pair_tensor_equivariant(original[name]["input"], shuffled[name]["input"], align)
        _assert_pair_tensor_equivariant(original[name]["output"], shuffled[name]["output"], align)
    elif name in {"global_feat_proj", "summary_head", "value_head"}:
        _assert_invariant(original[name]["input"], shuffled[name]["input"])
        _assert_invariant(original[name]["output"], shuffled[name]["output"])
    else:
        raise AssertionError(f"Unhandled head: {name}")


@pytest.mark.parametrize("policy_cls", POLICIES)
@pytest.mark.parametrize("fleets", [[], FLEETS])
def test_grouped_edge_models_are_planet_order_equivariant(policy_cls, fleets):
    shuffled_planets = [PLANETS[i] for i in PERMUTATION]
    original_obs = build_observation(
        _make_env_obs(PLANETS, fleets),
        player=0,
        angular_velocity=0.05,
        p_max=8,
        f_max=4,
    )
    shuffled_obs = build_observation(
        _make_env_obs(shuffled_planets, fleets),
        player=0,
        angular_velocity=0.05,
        p_max=8,
        f_max=4,
    )
    planet_ids, align = _alignment(original_obs.planet_ids, shuffled_obs.planet_ids)
    p_real = len(planet_ids)
    align_np = align.numpy()

    assert np.allclose(original_obs.planet_feats[:p_real], shuffled_obs.planet_feats[align_np])
    assert np.allclose(
        original_obs.edge_bias[:p_real, :p_real],
        shuffled_obs.edge_bias[np.ix_(align_np, align_np)],
    )
    assert np.allclose(original_obs.global_feats, shuffled_obs.global_feats)

    torch.manual_seed(123)
    policy = policy_cls(p_max=8, f_max=4, d_model=64, n_heads=4, n_layers=2)
    policy.eval()
    original_out, original_heads = _capture_head_io(policy, pack_observations([original_obs]))
    shuffled_out, shuffled_heads = _capture_head_io(policy, pack_observations([shuffled_obs]))

    assert original_heads.keys() == shuffled_heads.keys()
    for head_name in original_heads:
        _assert_head_io_equivariant(head_name, original_heads, shuffled_heads, align)

    assert torch.allclose(
        original_out.src_logits[0, :p_real],
        shuffled_out.src_logits[0, align],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        original_out.src_logits[0, policy.p_max],
        shuffled_out.src_logits[0, policy.p_max],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        original_out.tgt_logits[0, :p_real, :p_real],
        shuffled_out.tgt_logits[0][align][:, align],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        original_out.ship_params[0, :p_real, :p_real],
        shuffled_out.ship_params[0][align][:, align],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        original_out.tokens[0, :p_real],
        shuffled_out.tokens[0, align],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        original_out.aux_owner_logits[0, :p_real],
        shuffled_out.aux_owner_logits[0, align],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(original_out.value, shuffled_out.value, atol=1e-5, rtol=1e-5)
