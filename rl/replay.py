"""Off-policy experience replay buffer for DQN.

Differs from `RolloutBuffer` (PPO):
  - circular FIFO with capped capacity
  - stores *both* current and next observation tensors so DQN can
    bootstrap targets without re-running the env
  - uniform random sampling (no prioritized variant yet)
  - no GAE / advantage computation; just (s, a, r, s', done)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class DQNTransition:
    # Current obs.
    planet_feats: np.ndarray
    fleet_feats: np.ndarray
    global_feats: np.ndarray
    edge_bias: np.ndarray
    planet_mask: np.ndarray
    fleet_mask: np.ndarray
    source_mask: np.ndarray
    # Action.
    src: int
    tgt: int
    ship_bin: int
    is_noop: bool
    # Reward + terminal flag.
    reward: float
    done: bool
    # Next obs (zero arrays if done; consumer ignores them via `done`).
    next_planet_feats: np.ndarray
    next_fleet_feats: np.ndarray
    next_global_feats: np.ndarray
    next_edge_bias: np.ndarray
    next_planet_mask: np.ndarray
    next_fleet_mask: np.ndarray
    next_source_mask: np.ndarray


class ReplayBuffer:
    """Simple FIFO experience replay with uniform sampling."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.data: list[Optional[DQNTransition]] = [None] * capacity
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, t: DQNTransition) -> None:
        self.data[self._idx] = t
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def sample(self, batch_size: int, *, device: Optional[torch.device] = None,
               rng: Optional[np.random.Generator] = None) -> dict:
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self._size, size=batch_size)
        sub = [self.data[i] for i in idx]

        def stk(field, dtype=None):
            arr = np.stack([getattr(t, field) for t in sub])
            return torch.from_numpy(arr if dtype is None else arr.astype(dtype))

        batch = {
            "planet_feats": stk("planet_feats"),
            "fleet_feats": stk("fleet_feats"),
            "global_feats": stk("global_feats"),
            "edge_bias": stk("edge_bias"),
            "planet_mask": stk("planet_mask"),
            "fleet_mask": stk("fleet_mask"),
            "source_mask": stk("source_mask"),
            "next_planet_feats": stk("next_planet_feats"),
            "next_fleet_feats": stk("next_fleet_feats"),
            "next_global_feats": stk("next_global_feats"),
            "next_edge_bias": stk("next_edge_bias"),
            "next_planet_mask": stk("next_planet_mask"),
            "next_fleet_mask": stk("next_fleet_mask"),
            "next_source_mask": stk("next_source_mask"),
            "src":      torch.tensor([t.src for t in sub], dtype=torch.long),
            "tgt":      torch.tensor([t.tgt for t in sub], dtype=torch.long),
            "ship_bin": torch.tensor([t.ship_bin for t in sub], dtype=torch.long),
            "is_noop":  torch.tensor([t.is_noop for t in sub], dtype=torch.bool),
            "reward":   torch.tensor([t.reward for t in sub], dtype=torch.float32),
            "done":     torch.tensor([t.done for t in sub], dtype=torch.float32),
        }
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch
