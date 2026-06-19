from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from final_model import FinalNodePolicy
from final_obs import EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM, P_MAX, FinalObservation
from final_rollout import RolloutBuffer


def _toy_obs() -> FinalObservation:
    node = np.zeros((P_MAX, NODE_FEATURE_DIM), dtype=np.float32)
    node[0, 0] = 1.0
    node[1, 1] = 1.0
    edge = np.zeros((P_MAX, P_MAX, EDGE_FEATURE_DIM), dtype=np.float32)
    edge[0, 1, 0] = 1.0
    edge[1, 0, 0] = 1.0
    global_features = np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32)
    valid = np.zeros(P_MAX, dtype=bool)
    valid[:2] = True
    edge_mask = np.zeros((P_MAX, P_MAX), dtype=bool)
    edge_mask[0, 1] = True
    edge_mask[1, 0] = True
    source = np.zeros(P_MAX, dtype=bool)
    source[0] = True
    target = np.zeros(P_MAX, dtype=bool)
    target[1] = True
    ids = np.full(P_MAX, -1, dtype=np.int64)
    ids[:2] = [10, 11]
    return FinalObservation(node, edge, global_features, valid, edge_mask, source, target, ids)


def test_final_node_policy_action_and_evaluate_are_finite() -> None:
    torch.manual_seed(0)
    obs = _toy_obs()
    policy = FinalNodePolicy(d_model=32, n_heads=4, n_layers=1)
    tensors = {
        "node": torch.tensor(obs.node[None, ...]),
        "edge": torch.tensor(obs.edge[None, ...]),
        "global_features": torch.tensor(obs.global_features[None, ...]),
        "valid_mask": torch.tensor(obs.valid_mask[None, ...]),
        "edge_mask": torch.tensor(obs.edge_mask[None, ...]),
        "source_mask": torch.tensor(obs.source_mask[None, ...]),
        "target_mask": torch.tensor(obs.target_mask[None, ...]),
    }

    action = policy.act(**tensors, greedy=True)
    assert int(action.source_idx.item()) == 0
    assert int(action.target_idx.item()) == 1
    assert torch.isfinite(action.logprob).all()
    assert torch.isfinite(action.value).all()

    logprob, entropy, value = policy.evaluate_actions(
        tensors["node"],
        tensors["edge"],
        tensors["global_features"],
        tensors["valid_mask"],
        tensors["edge_mask"],
        tensors["source_mask"],
        tensors["target_mask"],
        action.target_idx,
        action.source_idx,
        action.amount,
    )
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()


def test_policy_falls_back_when_masks_are_empty() -> None:
    obs = _toy_obs()
    policy = FinalNodePolicy(d_model=32, n_heads=4, n_layers=1)
    action = policy.act(
        torch.tensor(obs.node[None, ...]),
        torch.tensor(obs.edge[None, ...]),
        torch.tensor(obs.global_features[None, ...]),
        torch.tensor(obs.valid_mask[None, ...]),
        torch.tensor(obs.edge_mask[None, ...]),
        torch.zeros((1, P_MAX), dtype=torch.bool),
        torch.zeros((1, P_MAX), dtype=torch.bool),
        greedy=True,
    )
    assert int(action.source_idx.item()) == 0
    assert int(action.target_idx.item()) == 0
    assert torch.isfinite(action.logprob).all()


def test_rollout_buffer_single_item_advantage_is_finite() -> None:
    obs = _toy_obs()
    action = SimpleNamespace(
        target_idx=torch.tensor([1]),
        source_idx=torch.tensor([0]),
        amount=torch.tensor([0.5]),
        logprob=torch.tensor([0.0]),
        value=torch.tensor([0.0]),
    )
    buffer = RolloutBuffer()
    buffer.add(obs, action, reward=1.0, done=True)
    buffer.compute_gae(gamma=0.99, lam=0.95)
    tensors = buffer.tensors(torch.device("cpu"))
    assert torch.isfinite(tensors["advantages"]).all()
    assert torch.isfinite(tensors["returns"]).all()
