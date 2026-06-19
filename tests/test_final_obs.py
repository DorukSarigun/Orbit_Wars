from __future__ import annotations

import math

import numpy as np

from final_obs import build_final_observation


def _obs_with_multiple_owners():
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 10, 3],
            [1, 1, 40.0, 20.0, 1.0, 20, 4],
            [2, -1, 60.0, 20.0, 1.0, 30, 2],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 10, 3],
            [1, 1, 40.0, 20.0, 1.0, 20, 4],
            [2, -1, 60.0, 20.0, 1.0, 30, 2],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def _obs_with_enemy_inbound():
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 10, 3],
            [1, 1, 40.0, 20.0, 1.0, 20, 4],
        ],
        "fleets": [
            [0, 1, 34.0, 20.0, math.pi, 1, 10],
        ],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 10, 3],
            [1, 1, 40.0, 20.0, 1.0, 20, 4],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_node_global_production_features_are_not_prefix_sums() -> None:
    obs = build_final_observation(_obs_with_multiple_owners(), player=0)
    valid_rows = obs.node[obs.valid_mask]

    expected = obs.global_features[:3]
    assert np.allclose(valid_rows[:, 15:18], expected[None, :])


def test_incoming_enemy_feature_is_populated_from_predicted_target() -> None:
    obs = build_final_observation(_obs_with_enemy_inbound(), player=0)
    target_slot = int(np.where(obs.planet_ids == 0)[0][0])

    assert obs.node[target_slot, 14] > 0.0
    assert obs.node[target_slot, 13] == 0.0
