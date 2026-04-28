"""Per-parameter training metrics.

Three things we want to log every iteration:

  - **grad_norms** : per-top-module L2 grad norm. If a block's grad is
    ~0 the encoder is starving; if it's huge we may need clipping.
  - **weight_norms** : per-top-module parameter L2. Useful to see if a
    block's params are drifting.
  - **update_ratios** : ||delta_w|| / ||w||. The de-facto sanity check
    for learning-rate scaling (~1e-3 to 1e-2 is healthy; <1e-4 means
    you're not learning, >1e-1 means lr is too high).

All metrics are grouped by the first segment of `named_parameters()`
so we get one number per logical sub-module
(`planet_proj`, `blocks.0`, `blocks.1`, `src_head`, ...).
"""

from __future__ import annotations

import math
from typing import Dict

import torch.nn as nn


def _top_key(name: str) -> str:
    """Group `blocks.0.attn.qkv.weight` -> `blocks.0`, leave leaf heads alone."""
    parts = name.split(".")
    if parts[0] == "blocks" and len(parts) >= 2:
        return f"blocks.{parts[1]}"
    return parts[0]


def grad_norms_by_module(model: nn.Module) -> Dict[str, float]:
    """L2 norm of `param.grad` aggregated per logical sub-module.

    Returns {} if no params have grads (call AFTER `loss.backward()` and
    BEFORE `optimizer.zero_grad()`).
    """
    groups: Dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        key = _top_key(name)
        groups[key] = groups.get(key, 0.0) + p.grad.detach().pow(2).sum().item()
    return {k: math.sqrt(v) for k, v in groups.items()}


def weight_norms_by_module(model: nn.Module) -> Dict[str, float]:
    """L2 norm of `param.data` aggregated per logical sub-module."""
    groups: Dict[str, float] = {}
    for name, p in model.named_parameters():
        key = _top_key(name)
        groups[key] = groups.get(key, 0.0) + p.data.detach().pow(2).sum().item()
    return {k: math.sqrt(v) for k, v in groups.items()}


class WeightSnapshot:
    """Capture model weights, then later compute per-module update norms.

    Usage:
        snap = WeightSnapshot(model)
        # ... run optimizer.step() (potentially many times) ...
        deltas = snap.compute_update_norms(model)         # ||Δw||
        ratios = snap.compute_update_ratios(model)        # ||Δw|| / ||w||
    """

    def __init__(self, model: nn.Module) -> None:
        # Per-param L2 norm at snapshot time, plus per-param flat copy of weights.
        self._snapshot = {
            name: p.data.detach().clone()
            for name, p in model.named_parameters()
        }

    def compute_update_norms(self, model: nn.Module) -> Dict[str, float]:
        groups: Dict[str, float] = {}
        for name, p in model.named_parameters():
            if name not in self._snapshot:
                continue
            delta = (p.data - self._snapshot[name])
            key = _top_key(name)
            groups[key] = groups.get(key, 0.0) + delta.pow(2).sum().item()
        return {k: math.sqrt(v) for k, v in groups.items()}

    def compute_update_ratios(self, model: nn.Module) -> Dict[str, float]:
        update = self.compute_update_norms(model)
        before = self._snapshot_norms()
        # ratio = ||Δw|| / ||w_before||  (use snapshot, not current)
        return {
            k: (update[k] / before[k]) if before.get(k, 0.0) > 0 else 0.0
            for k in update
        }

    def _snapshot_norms(self) -> Dict[str, float]:
        groups: Dict[str, float] = {}
        for name, w in self._snapshot.items():
            key = _top_key(name)
            groups[key] = groups.get(key, 0.0) + w.pow(2).sum().item()
        return {k: math.sqrt(v) for k, v in groups.items()}


def round_metrics(d: Dict[str, float], digits: int = 4) -> Dict[str, float]:
    """Round dict floats for compact JSON logging."""
    return {k: round(float(v), digits) for k, v in d.items()}
