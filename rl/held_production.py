"""Held discounted production metric for graph-native rewards.

For each self-owned planet *j* with per-turn production ``prod_j`` and
``R = max(0, max_turns - turn)`` remaining steps, the contribution is

    V_j = prod_j * (1 + x_j + x_j^2 + ... + x_j^{R-1})
        = prod_j * (1 - x_j^R) / (1 - x_j)

where ``x_j`` is a per-turn survival / hold-uncertainty factor in
``(0, 1)``.  Lower ``x_j`` (more threatened planets) shrinks the effective
future production horizon; losing ownership zeroes ``is_self`` so ``V_j``
drops immediately.

Capture turn does not enter: only current ownership, production, threat,
and remaining horizon matter.
"""

from __future__ import annotations

from typing import Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

Number = Union[float, int]
ArrayLike = Union[np.ndarray, "torch.Tensor"]


def geometric_hold_multiplier(
    remaining_turns: Union[int, ArrayLike],
    survival_x: ArrayLike,
    *,
    eps: float = 1e-6,
) -> ArrayLike:
    """Return ``sum_{k=0}^{R-1} x^k`` (vectorized numpy or torch).

    ``remaining_turns`` may be a scalar or, for torch, a per-batch vector
    ``(B,)`` broadcast over planet dimension ``(B, P)``.
    """
    if isinstance(survival_x, np.ndarray):
        r = int(max(0, int(remaining_turns)))
        x = np.clip(np.asarray(survival_x, dtype=np.float64), 0.0, 1.0 - eps)
        if r == 0:
            return np.zeros_like(x, dtype=np.float64)
        one_minus_xr = 1.0 - np.power(x, r)
        denom = np.maximum(1.0 - x, eps)
        near_one = np.abs(x - 1.0) < eps
        out = np.where(near_one, float(r), one_minus_xr / denom)
        return out.astype(np.float32)

    if torch is None:
        raise ImportError("torch is required for tensor survival_x")
    x = survival_x.clamp(0.0, 1.0 - eps).to(dtype=survival_x.dtype)
    if isinstance(remaining_turns, torch.Tensor):
        r = remaining_turns.to(dtype=x.dtype, device=x.device).clamp(min=0)
        if r.dim() == 0:
            r = r.view(1)
        if r.dim() == 1 and x.dim() == 2 and r.shape[0] == x.shape[0]:
            r = r.view(-1, 1)
        zero_mask = r <= 0
        if bool(zero_mask.all()):
            return survival_x.new_zeros(survival_x.shape)
        one_minus_xr = 1.0 - x.pow(r)
        denom = (1.0 - x).clamp_min(eps)
        near_one = (x - 1.0).abs() < eps
        geom = torch.where(near_one, r.expand_as(x), one_minus_xr / denom)
        return torch.where(zero_mask.view(-1, 1).expand_as(geom), geom.new_zeros(()), geom)

    r = int(max(0, int(remaining_turns)))
    if r == 0:
        return survival_x.new_zeros(survival_x.shape)
    one_minus_xr = 1.0 - x.pow(r)
    denom = (1.0 - x).clamp_min(eps)
    near_one = (x - 1.0).abs() < eps
    return torch.where(near_one, x.new_tensor(float(r)), one_minus_xr / denom)


def per_planet_hold_survival(
    vulnerability: ArrayLike,
    *,
    base: float = 0.99,
    from_vulnerability: bool = True,
    survival_min: float = 0.5,
    survival_max: float = 0.99,
) -> ArrayLike:
    """Map normalized vulnerability ``[0,1]`` to per-turn survival ``x_j``."""
    lo = float(min(survival_min, survival_max))
    hi = float(max(survival_min, survival_max))
    if not from_vulnerability:
        if isinstance(vulnerability, np.ndarray):
            return np.full(vulnerability.shape, float(base), dtype=np.float32)
        return vulnerability.new_full(vulnerability.shape, float(base))

    if isinstance(vulnerability, np.ndarray):
        v = np.clip(np.asarray(vulnerability, dtype=np.float64), 0.0, 1.0)
        return (lo + (1.0 - v) * (hi - lo)).astype(np.float32)

    if torch is None:
        raise ImportError("torch is required for tensor vulnerability")
    v = vulnerability.clamp(0.0, 1.0)
    return lo + (1.0 - v) * (hi - lo)


def held_discounted_production_value(
    production: ArrayLike,
    is_self: ArrayLike,
    vulnerability: ArrayLike,
    *,
    remaining_turns: Union[int, ArrayLike],
    norm: float = 500.0,
    base: float = 0.99,
    from_vulnerability: bool = True,
    survival_min: float = 0.5,
    survival_max: float = 0.99,
) -> ArrayLike:
    """Per-planet held production metric (0 for non-self slots)."""
    x = per_planet_hold_survival(
        vulnerability,
        base=base,
        from_vulnerability=from_vulnerability,
        survival_min=survival_min,
        survival_max=survival_max,
    )
    geom = geometric_hold_multiplier(remaining_turns, x)
    scale = max(float(norm), 1e-6)
    if isinstance(production, np.ndarray):
        prod = np.clip(np.asarray(production, dtype=np.float64), 0.0, None)
        owned = (np.asarray(is_self, dtype=np.float64) > 0.5).astype(np.float64)
        return (prod * owned * geom / scale).astype(np.float32)

    if torch is None:
        raise ImportError("torch is required for tensor production")
    prod = production.clamp_min(0.0)
    owned = is_self.clamp(0.0, 1.0)
    return prod * owned * geom / scale
