"""On-policy rollout buffer with GAE advantage estimation.

Despite the user-facing name "replay buffer", PPO is on-policy: data is
collected with the *current* policy, used for K epochs of minibatch SGD,
then thrown away. This buffer is the standard PPO data structure.

We store one transition per `add()` call. Each transition is a single
(state, action, reward) tuple from a single agent perspective. In
self-play we add TWO transitions per env.step() call (one for each
side), which doubles the data collected per environment step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import numpy as np
import torch


@dataclass
class Transition:
    player: int
    # Observation tensors (already on CPU as numpy; we tensorize on flush).
    planet_feats: np.ndarray
    fleet_feats: np.ndarray
    global_feats: np.ndarray
    edge_bias: np.ndarray
    planet_mask: np.ndarray
    fleet_mask: np.ndarray
    source_mask: np.ndarray
    # Action triple + flags. ship_value is the continuous Beta sample
    # in [0, 1]; the absolute ship request is computed downstream as
    # ship_value * SHIP_REQUEST_CAP (see obs_builder.decode_action_with_launch).
    src: int
    tgt: int
    ship_value: float
    is_noop: bool
    # Old-policy log-prob and value estimate (used by PPO).
    log_prob: float
    value: float
    # Per-step reward and terminal flag.
    reward: float
    done: bool
    turn: int = 0
    # Optional per-step flux features ``(P_max, 4)`` used by PyG variants
    # built with ``enable_flux_features=True``.  Stored as ``None`` for
    # legacy policies so the buffer stays compact.
    flux_features: Optional[np.ndarray] = None
    # Optional per-step edge flux features ``(P_max, P_max, 2)`` carrying
    # the directed [friendly_flux_ij, enemy_flux_ij] channels.  Paired
    # with ``flux_features`` -- when the policy was built with
    # ``enable_flux_features=True`` both fields are populated together
    # so the PPO update can rebuild the same encoder input.
    edge_flux_features: Optional[np.ndarray] = None


class RolloutBuffer:
    """Variable-length on-policy buffer.

    Usage:
        buf = RolloutBuffer()
        # ... fill via buf.add(...)
        buf.compute_gae(last_value=v_bootstrap, gamma=0.99, lam=0.95)
        for batch in buf.iter_minibatches(batch_size=256, shuffle=True):
            # PPO update
    """

    def __init__(self) -> None:
        self.data: list[Transition] = []
        self._advantages: np.ndarray | None = None
        self._returns: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.data)

    def add(self, t: Transition) -> int:
        idx = len(self.data)
        self.data.append(t)
        return idx

    def add_reward(self, idx: int, reward: float) -> None:
        self.data[idx].reward += float(reward)

    def add_delayed_reward(self, idx: int, reward: float, *, event_turn: int, discount: float) -> None:
        delay = max(0, int(event_turn) - int(self.data[idx].turn))
        self.add_reward(idx, float(reward) * (float(discount) ** delay))

    # --------------------------------------------------------------- GAE
    def compute_gae(self, last_value: float | dict[int, float], *, gamma: float = 0.99, lam: float = 0.95) -> None:
        """Compute Generalized Advantage Estimation for all transitions.

        `last_value` is V(s_T) for bootstrapping at the final stored step
        (set to 0.0 if the rollout ends on a terminal observation).
        """
        n = len(self.data)
        adv = np.zeros(n, dtype=np.float32)
        players = sorted({tr.player for tr in self.data})
        for player in players:
            indices = [i for i, tr in enumerate(self.data) if tr.player == player]
            last_gae = 0.0
            next_value = (
                float(last_value.get(player, 0.0))
                if isinstance(last_value, dict)
                else float(last_value)
            )
            for t in reversed(indices):
                tr = self.data[t]
                mask = 0.0 if tr.done else 1.0
                delta = tr.reward + gamma * next_value * mask - tr.value
                last_gae = delta + gamma * lam * mask * last_gae
                adv[t] = last_gae
                next_value = tr.value
        self._advantages = adv
        self._returns = adv + np.array([tr.value for tr in self.data], dtype=np.float32)

    # ----------------------------------------------------------- Iteration
    def iter_minibatches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        device: torch.device | None = None,
    ) -> Iterator[dict]:
        """Yield minibatch dicts ready for the PPO update step.

        Each batch is a dict with stacked tensors keyed by:
          planet_feats, fleet_feats, global_feats, edge_bias,
          planet_mask, fleet_mask, source_mask,
          src, tgt, ship_value, is_noop,
          old_log_prob, old_value, returns, advantages
        """
        assert self._advantages is not None, "Call compute_gae() first"
        n = len(self.data)
        idx = np.arange(n)
        if shuffle:
            np.random.shuffle(idx)

        # Normalize advantages per buffer (standard PPO trick).
        adv = self._advantages.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, batch_size):
            chunk = idx[start:start + batch_size]
            batch = self._stack(chunk, adv)
            if device is not None:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            yield batch

    def batch_from_indices(self, indices: Sequence[int] | np.ndarray, *,
                           normalize_advantages: bool = True) -> dict:
        assert self._advantages is not None, "Call compute_gae() first"
        idx = np.asarray(indices, dtype=np.int64)
        adv = self._advantages.copy()
        if normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return self._stack(idx, adv)

    # ----------------------------------------------------------- helpers
    def _stack(self, idx: np.ndarray, adv_norm: np.ndarray) -> dict:
        sub = [self.data[i] for i in idx]
        out = {
            "planet_feats": torch.from_numpy(np.stack([t.planet_feats for t in sub])),
            "fleet_feats":  torch.from_numpy(np.stack([t.fleet_feats for t in sub])),
            "global_feats": torch.from_numpy(np.stack([t.global_feats for t in sub])),
            "edge_bias":    torch.from_numpy(np.stack([t.edge_bias for t in sub])),
            "planet_mask":  torch.from_numpy(np.stack([t.planet_mask for t in sub])),
            "fleet_mask":   torch.from_numpy(np.stack([t.fleet_mask for t in sub])),
            "source_mask":  torch.from_numpy(np.stack([t.source_mask for t in sub])),
            "src":          torch.tensor([t.src for t in sub], dtype=torch.long),
            "tgt":          torch.tensor([t.tgt for t in sub], dtype=torch.long),
            "ship_value":   torch.tensor([t.ship_value for t in sub], dtype=torch.float32),
            "is_noop":      torch.tensor([t.is_noop for t in sub], dtype=torch.bool),
            "player":       torch.tensor([t.player for t in sub], dtype=torch.long),
            "old_log_prob": torch.tensor([t.log_prob for t in sub], dtype=torch.float32),
            "old_value":    torch.tensor([t.value for t in sub], dtype=torch.float32),
            "returns":      torch.tensor([self._returns[i] for i in idx], dtype=torch.float32),
            "advantages":   torch.tensor([adv_norm[i] for i in idx], dtype=torch.float32),
        }
        # Optional flux features (PyG variants only).  Only stack when every
        # sampled transition recorded them; mixing flux/no-flux transitions
        # in a single batch is unsupported (and indicates a rollout bug).
        if any(t.flux_features is not None for t in sub):
            flux = [t.flux_features for t in sub]
            if any(f is None for f in flux):
                missing = sum(1 for f in flux if f is None)
                raise ValueError(
                    f"flux_features present on {len(flux) - missing}/{len(flux)} "
                    "sampled transitions; either all or none must carry them."
                )
            out["flux_features"] = torch.from_numpy(np.stack(flux)).float()
        if any(t.edge_flux_features is not None for t in sub):
            edge_flux = [t.edge_flux_features for t in sub]
            if any(f is None for f in edge_flux):
                missing = sum(1 for f in edge_flux if f is None)
                raise ValueError(
                    f"edge_flux_features present on {len(edge_flux) - missing}/{len(edge_flux)} "
                    "sampled transitions; either all or none must carry them."
                )
            out["edge_flux_features"] = torch.from_numpy(np.stack(edge_flux)).float()
        return out

    def clear(self) -> None:
        self.data.clear()
        self._advantages = None
        self._returns = None
