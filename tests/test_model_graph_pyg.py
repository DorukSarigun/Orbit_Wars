from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import pack_observations
from model_graph_pyg import (
    GRAPH_NODE_METRIC_DIM,
    PYG_EDGE_FEATURE_DIM,
    PYG_PLANET_FEATURE_DIM,
    PygPlanetGraphPolicy,
    observation_to_pyg_data,
)
from obs_builder import EDGE_FEATURE_DIM, PLANET_FEATURE_DIM, build_observation


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


def test_observation_to_pyg_data_appends_graph_node_metrics():
    obs = build_observation(_toy_obs(), player=0, angular_velocity=0.05, p_max=8, f_max=4)
    data = observation_to_pyg_data(obs)
    assert PYG_PLANET_FEATURE_DIM == PLANET_FEATURE_DIM + GRAPH_NODE_METRIC_DIM
    assert data.x.shape == (4, PYG_PLANET_FEATURE_DIM)
    assert data.edge_index.shape == (2, 12)
    assert PYG_EDGE_FEATURE_DIM == EDGE_FEATURE_DIM + 9
    assert data.edge_attr.shape == (12, PYG_EDGE_FEATURE_DIM)
    assert data.x[:, -GRAPH_NODE_METRIC_DIM:].isfinite().all()
    assert data.edge_attr[:, -9:].isfinite().all()
    assert data.edge_attr[0, -9] > 0.0
    assert data.edge_attr[0, -7] == 1.0
    assert data.edge_attr[0, -6] == 1.0
    assert data.edge_attr[0, -5] == 1.0
    assert data.edge_attr[1, -4] == 1.0


def test_pyg_planet_graph_policy_forward_shapes():
    obs_list = [
        build_observation(_toy_obs(), player=0, angular_velocity=0.05, p_max=8, f_max=4)
        for _ in range(2)
    ]
    batch = pack_observations(obs_list)
    net = PygPlanetGraphPolicy(p_max=8, d_model=32, n_layers=2)
    out = net(**batch)
    assert net.model_variant == "pyg_planet_graph"
    assert out.src_logits.shape == (2, 9)
    assert out.tgt_logits.shape == (2, 8, 8)
    assert out.ship_params.shape == (2, 8, 8, 2)
    assert out.value.shape == (2,)
    assert out.tokens.shape == (2, 8, 32)
    assert out.aux_owner_logits.shape == (2, 8, 5)
