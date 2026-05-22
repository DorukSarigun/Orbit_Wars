"""rl/flux.py — Flux helpers for credit assignment.

Computes ship-flow ("flux") matrices from in-flight fleet data that
``obs_builder.build_observation`` records on each
:class:`obs_builder.Observation` (fields
``fleet_src_slot``, ``fleet_tgt_slot``, ``fleet_eta_raw``,
``fleet_ships_raw``, ``fleet_is_self``, ``fleet_is_enemy``).

Three parametric weighting modes are supported:

    - ``"raw"``       : weight = ships
    - ``"linear"``    : weight = ships * max(0, 1 - eta / eta_norm)
    - ``"exp_decay"`` : weight = ships * exp(-eta / tau)

``linear`` and ``exp_decay`` give more credit to fleets that are
close to arrival, which is what we want for short-horizon
credit-assignment shaping.  ``raw`` is a useful sanity baseline.

The directed flux matrix ``M`` has shape ``(P_max, P_max)`` where
``M[i, j]`` is the total weighted ship flow originating at planet
slot ``i`` and predicted to land at slot ``j``.  Fleets whose
source/target slot is unknown (-1) or out of range are skipped.

For node features we expose four channels per planet slot::

    [friendly_in, friendly_out, enemy_in, enemy_out]

Values are divided by ``ship_norm`` (default 50.0) so they sit in
the same magnitude range as the other graph metrics in
``model_graph_pyg``.  Padded planet slots are zeroed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


FluxMode = Literal["raw", "linear", "exp_decay"]

DEFAULT_FLUX_MODE: FluxMode = "exp_decay"
DEFAULT_TAU: float = 10.0
DEFAULT_ETA_NORM: float = 50.0
DEFAULT_SHIP_NORM: float = 50.0

NODE_FLUX_CHANNELS: int = 4  # [friendly_in, friendly_out, enemy_in, enemy_out]


@dataclass
class FluxMatrices:
    """Per-owner directed flux matrices, shape ``(P_max, P_max)`` each."""

    friendly: np.ndarray
    enemy: np.ndarray


def _weight(
    ships: np.ndarray,
    eta: np.ndarray,
    mode: FluxMode,
    tau: float,
    eta_norm: float,
) -> np.ndarray:
    """Per-fleet scalar weight under the requested parametric mode."""
    if mode == "raw":
        return ships.astype(np.float32, copy=False)
    if mode == "linear":
        denom = max(float(eta_norm), 1e-6)
        return (ships * np.maximum(0.0, 1.0 - eta / denom)).astype(np.float32, copy=False)
    if mode == "exp_decay":
        denom = max(float(tau), 1e-6)
        return (ships * np.exp(-eta / denom)).astype(np.float32, copy=False)
    raise ValueError(f"unknown flux mode: {mode!r}")


def compute_flux_matrices(
    obs,
    p_max: Optional[int] = None,
    *,
    mode: FluxMode = DEFAULT_FLUX_MODE,
    tau: float = DEFAULT_TAU,
    eta_norm: float = DEFAULT_ETA_NORM,
) -> FluxMatrices:
    """Build directed ship-flow matrices for friendly and enemy fleets.

    Parameters
    ----------
    obs : Observation
        Must expose ``planet_feats``, ``fleet_mask``, ``fleet_src_slot``,
        ``fleet_tgt_slot``, ``fleet_eta_raw``, ``fleet_ships_raw``,
        ``fleet_is_self``, and ``fleet_is_enemy`` (populated by
        :func:`obs_builder.build_observation`).
    p_max : int, optional
        Output matrix size.  Defaults to ``obs.planet_feats.shape[0]``.
    mode, tau, eta_norm
        Parametric flux weighting; see the module docstring.

    Returns
    -------
    FluxMatrices
        Two ``(P_max, P_max)`` ``float32`` matrices.  Padded slots and
        fleets with unknown src/tgt contribute zero.
    """
    if p_max is None:
        p_max = int(obs.planet_feats.shape[0])

    friendly = np.zeros((p_max, p_max), dtype=np.float32)
    enemy = np.zeros((p_max, p_max), dtype=np.float32)

    fleet_mask = getattr(obs, "fleet_mask", None)
    if fleet_mask is None or fleet_mask.size == 0:
        return FluxMatrices(friendly=friendly, enemy=enemy)

    mask = np.asarray(fleet_mask, dtype=bool)
    if not mask.any():
        return FluxMatrices(friendly=friendly, enemy=enemy)

    # All routing arrays are expected to be present on Observation; fall
    # back to safe defaults if a legacy obs is supplied.
    src = np.asarray(getattr(obs, "fleet_src_slot", np.full(mask.shape, -1, dtype=np.int32)), dtype=np.int64)
    tgt = np.asarray(getattr(obs, "fleet_tgt_slot", np.full(mask.shape, -1, dtype=np.int32)), dtype=np.int64)
    eta = np.asarray(getattr(obs, "fleet_eta_raw", np.zeros(mask.shape, dtype=np.float32)), dtype=np.float32)
    ships = np.asarray(getattr(obs, "fleet_ships_raw", np.zeros(mask.shape, dtype=np.float32)), dtype=np.float32)
    is_self = np.asarray(getattr(obs, "fleet_is_self", np.zeros(mask.shape, dtype=bool)), dtype=bool)
    is_enemy = np.asarray(getattr(obs, "fleet_is_enemy", np.zeros(mask.shape, dtype=bool)), dtype=bool)

    valid = mask & (src >= 0) & (tgt >= 0) & (src < p_max) & (tgt < p_max)
    if not valid.any():
        return FluxMatrices(friendly=friendly, enemy=enemy)

    w = _weight(ships[valid], np.maximum(eta[valid], 0.0), mode, tau, eta_norm)
    s = src[valid]
    t = tgt[valid]
    sf = is_self[valid]
    en = is_enemy[valid]

    # Scatter-add per owner.  np.add.at handles repeated (i, j) indices
    # which can legitimately occur when several fleets share an edge.
    if sf.any():
        np.add.at(friendly, (s[sf], t[sf]), w[sf])
    if en.any():
        np.add.at(enemy, (s[en], t[en]), w[en])

    return FluxMatrices(friendly=friendly, enemy=enemy)


def compute_node_flux_features(
    obs,
    p_max: Optional[int] = None,
    *,
    mode: FluxMode = DEFAULT_FLUX_MODE,
    tau: float = DEFAULT_TAU,
    eta_norm: float = DEFAULT_ETA_NORM,
    ship_norm: float = DEFAULT_SHIP_NORM,
    matrices: Optional[FluxMatrices] = None,
) -> np.ndarray:
    """Summarise flux into four node-level channels.

    Returns
    -------
    np.ndarray, shape ``(P_max, 4)``, dtype ``float32``
        Columns are ``[friendly_in, friendly_out, enemy_in, enemy_out]``,
        i.e. the column-sum (inbound) and row-sum (outbound) of the
        friendly and enemy flux matrices, divided by ``ship_norm`` to
        keep values comparable to the other graph metrics.  Padded
        planet slots are zeroed.

    Notes
    -----
    The convention is ``M[i, j] = flow from i to j``; therefore
    ``inbound[j] = M[:, j].sum()`` (axis=0) and
    ``outbound[i] = M[i, :].sum()`` (axis=1).
    """
    if matrices is None:
        matrices = compute_flux_matrices(
            obs, p_max=p_max, mode=mode, tau=tau, eta_norm=eta_norm
        )
    fr = matrices.friendly
    en = matrices.enemy
    p = fr.shape[0]

    scale = 1.0 / max(float(ship_norm), 1e-6)
    out = np.zeros((p, NODE_FLUX_CHANNELS), dtype=np.float32)
    out[:, 0] = fr.sum(axis=0) * scale  # friendly_in  (column-sum)
    out[:, 1] = fr.sum(axis=1) * scale  # friendly_out (row-sum)
    out[:, 2] = en.sum(axis=0) * scale  # enemy_in
    out[:, 3] = en.sum(axis=1) * scale  # enemy_out

    planet_mask = getattr(obs, "planet_mask", None)
    if planet_mask is not None:
        mask = np.asarray(planet_mask, dtype=bool)
        if mask.shape[0] == p:
            out[~mask] = 0.0

    return out


__all__ = [
    "FluxMode",
    "FluxMatrices",
    "NODE_FLUX_CHANNELS",
    "DEFAULT_FLUX_MODE",
    "DEFAULT_TAU",
    "DEFAULT_ETA_NORM",
    "DEFAULT_SHIP_NORM",
    "compute_flux_matrices",
    "compute_node_flux_features",
]
