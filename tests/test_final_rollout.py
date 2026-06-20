from __future__ import annotations

from types import SimpleNamespace

import torch

from final_obs import build_final_observation
from final_rollout import _action_to_launch


def _env_two_source_capture():
    return {
        "planets": [
            [0, 0, 20.0, 20.0, 1.0, 12, 2],
            [1, -1, 32.0, 25.0, 1.0, 20, 2],
            [2, 0, 20.0, 32.0, 1.0, 12, 2],
        ],
        "fleets": [],
        "initial_planets": [
            [0, 0, 20.0, 20.0, 1.0, 12, 2],
            [1, -1, 32.0, 25.0, 1.0, 20, 2],
            [2, 0, 20.0, 32.0, 1.0, 12, 2],
        ],
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


def test_action_decoder_allows_multi_source_capture_for_selected_target() -> None:
    env_obs = _env_two_source_capture()
    obs = build_final_observation(env_obs, player=0)
    source_slot = int((obs.planet_ids == 0).nonzero()[0][0])
    target_slot = int((obs.planet_ids == 1).nonzero()[0][0])
    action = SimpleNamespace(
        source_idx=torch.tensor([source_slot]),
        target_idx=torch.tensor([target_slot]),
        amount=torch.tensor([1.0]),
    )

    launches, reason = _action_to_launch(
        env_obs,
        obs,
        action,
        player=0,
        current_turn=0,
        max_turns=100,
        planner_top_m_sources=3,
        planner_max_launches_per_target=3,
        planner_budget_floor=1.0,
    )

    assert reason == "micro_capture"
    assert len(launches) >= 2
    assert sum(int(row[2]) for row in launches) > 20
