import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import pack_observations
from model_graph_pyg_edgeonly import PygEdgeNativePolicy
from obs_builder import build_observation


def _toy_obs():
    return {
        "planets": [
            [0, 0, 70.0, 70.0, 3.0, 100, 3],
            [1, 0, 80.0, 70.0, 2.0, 50, 2],
            [2, 1, 30.0, 30.0, 2.0, 80, 2],
            [3, -1, 20.0, 80.0, 2.0, 10, 1],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 70.0, 70.0, 3.0, 100, 3],
            [1, 0, 80.0, 70.0, 2.0, 50, 2],
            [2, 1, 30.0, 30.0, 2.0, 80, 2],
            [3, -1, 20.0, 80.0, 2.0, 10, 1],
        ],
        "comet_planet_ids": [],
        "comets": [],
    }


def test_edge_native_forward_shapes():
    obs_list = [
        build_observation(_toy_obs(), player=0, angular_velocity=0.05, p_max=8, f_max=4)
        for _ in range(2)
    ]
    batch = pack_observations(obs_list)
    net = PygEdgeNativePolicy(p_max=8, d_model=32, n_layers=2)
    out = net(**batch)

    assert out.src_logits.shape == (2, 9)        # 8 + 1 noop
    assert out.tgt_logits.shape == (2, 8, 8)
    assert out.ship_params.shape == (2, 8, 8, 2)
    assert out.value.shape == (2,)
    assert out.tokens.shape == (2, 8, 32)
    assert out.aux_owner_logits.shape == (2, 8, 5)

    assert torch.isfinite(out.src_logits).all()
    assert torch.isfinite(out.tgt_logits).all()


def test_edge_native_act_eval():
    obs = build_observation(_toy_obs(), player=0, angular_velocity=0.05, p_max=8, f_max=4)
    batch = pack_observations([obs])
    net = PygEdgeNativePolicy(p_max=8, d_model=32, n_layers=2)
    out = net(**batch)

    action = net.act(out, deterministic=True)
    assert action["src"].shape == (1,)
    assert action["tgt"].shape == (1,)
    assert action["ship_value"].shape == (1,)
    assert action["is_noop"].shape == (1,)
    assert action["log_prob"].shape == (1,)

    # evaluate_actions backward compat
    ev = net.evaluate_actions(out, action["src"], action["tgt"], action["ship_value"])
    assert "log_prob" in ev
    assert "entropy" in ev
