"""End-to-end adapter: ``obs_builder.Observation`` -> reworked feature tensors.

This is the glue that wires the redesigned, compact feature set
(:mod:`reworked_features`) and encoder (:mod:`reworked_model`) onto the **real**
observation pipeline produced by :func:`obs_builder.build_observation`, plus the
inputs the scripted logistics layer (:mod:`reworked_allocator`) needs.

Why an adapter is needed
------------------------
The live ``Observation`` stores everything the *legacy* policy used:

* ``planet_feats`` -- a 34-D **normalised** per-planet block (ownership one-hot,
  ``ships/100``, ``production/5``, ``radius/5``, flags, incoming buckets, ...).
* ``edge_bias`` -- a ``(T, T, 6)`` token-pair tensor whose channel 0 is the
  **normalised** distance ``||Δpos|| / BOARD_SIZE`` and channel 5 is
  ``crosses_sun``.
* fleet routing arrays (``fleet_src_slot`` / ``fleet_tgt_slot`` /
  ``fleet_eta_raw`` / ``fleet_ships_raw`` / ``fleet_is_self`` /
  ``fleet_is_enemy``) describing in-flight fleets as planet-slot pointers.

The reworked feature functions instead want clean *physical* inputs
(``owner``, ``ships``, ``production``, ``radius``, ``distance_matrix``, ...).
This module derives those by:

1. **De-normalising** the scalar channels (``× 100`` ships, ``× 5`` production /
   radius).
2. **Reconstructing an absolute ``owner`` array** from the perspective-relative
   ownership one-hot (see :func:`effective_owner_from_obs`).  The reconstruction
   round-trips exactly through ``reworked_features.standard_node_features`` (same
   enemy-slot convention), so it is faithful for every metric that only needs
   the self / enemy / neutral partition.
3. **Slicing** the planet sub-block ``edge_bias[:n, :n, {0,5}]`` for the
   distance matrix and ``crosses_sun`` mask.

The output node features follow the ``REWORKED_NODE_FEATURE_DIM = 15`` layout
expected by :class:`reworked_model.ReworkedPlanetGraphEncoder`:

    [ standard_node_features (10) ,
      discounted_production_potential , closeness_centrality ,
      potential_threat , potential_support , friendly_compactness ]   (5)

and edge features the ``REWORKED_EDGE_FEATURE_DIM = 4`` layout
``[distance, crosses_sun, friendly_flux, enemy_flux]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:  # package layout
    from .reworked_features import (
        standard_node_features,
        discounted_production_potential,
        closeness_centrality,
        potential_threat,
        potential_support,
        friendly_compactness,
        edge_features,
        incoming_enemy_estimate,
        STANDARD_NODE_FEATURE_DIM,
        REWORKED_EDGE_FEATURE_DIM,
        DEFAULT_BOARD_SIZE,
        DEFAULT_MAX_SHIP_SPEED,
        MAX_ENEMY_OWNER_SLOTS,
    )
    from .reworked_model import (
        REWORKED_NODE_FEATURE_DIM,
        REWORKED_GLOBAL_FEATURE_DIM,
    )
    from .obs_builder import (
        PF_IDX_OWNER_SELF,
        PF_IDX_OWNER_ENEMY_0,
        PF_IDX_OWNER_NEUTRAL,
        PF_IDX_SHIPS,
        PF_IDX_PRODUCTION,
        PF_IDX_RADIUS,
        PF_IDX_IS_ROTATING,
        PF_IDX_IS_COMET,
    )
except ImportError:  # pragma: no cover - flat layout / direct execution
    from reworked_features import (  # type: ignore
        standard_node_features,
        discounted_production_potential,
        closeness_centrality,
        potential_threat,
        potential_support,
        friendly_compactness,
        edge_features,
        incoming_enemy_estimate,
        STANDARD_NODE_FEATURE_DIM,
        REWORKED_EDGE_FEATURE_DIM,
        DEFAULT_BOARD_SIZE,
        DEFAULT_MAX_SHIP_SPEED,
        MAX_ENEMY_OWNER_SLOTS,
    )
    from reworked_model import (  # type: ignore
        REWORKED_NODE_FEATURE_DIM,
        REWORKED_GLOBAL_FEATURE_DIM,
    )
    from obs_builder import (  # type: ignore
        PF_IDX_OWNER_SELF,
        PF_IDX_OWNER_ENEMY_0,
        PF_IDX_OWNER_NEUTRAL,
        PF_IDX_SHIPS,
        PF_IDX_PRODUCTION,
        PF_IDX_RADIUS,
        PF_IDX_IS_ROTATING,
        PF_IDX_IS_COMET,
    )

# De-normalisation factors that mirror obs_builder._planet_features.
SHIPS_SCALE = 100.0
PRODUCTION_SCALE = 5.0
RADIUS_SCALE = 5.0

# edge_bias channel layout (obs_builder.EDGE_FEATURE_DIM == 6).
EDGE_BIAS_DISTANCE_IDX = 0
EDGE_BIAS_CROSSES_SUN_IDX = 5

# Scale-parity divisor for the two speed-weighted gravity metrics so they land
# in roughly the same O(1) range as the other node channels before the
# encoder's input LayerNorm.  (The exact value is not critical because the stem
# normalises, but keeping channels comparable avoids one feature dominating the
# pre-norm mean/variance.)
THREAT_SUPPORT_NORM = 50.0


@dataclass
class BoardState:
    """De-normalised per-planet board state sliced to real planets ``[:n]``.

    Shared input for both the feature builders and the allocator so the two
    layers see an identical view of the world.
    """

    n: int
    player: int
    owner: np.ndarray            # (n,) int64; -1 neutral, else (reconstructed) owner id
    ships: np.ndarray            # (n,) float32, real units
    production: np.ndarray       # (n,) float32, ships / turn
    radius: np.ndarray           # (n,) float32
    is_rotating: np.ndarray      # (n,) float32 0/1
    is_comet: np.ndarray         # (n,) float32 0/1
    distance_matrix: np.ndarray  # (n, n) float32, normalised (× BOARD_SIZE -> raw)
    crosses_sun: np.ndarray      # (n, n) float32 0/1
    remaining_turns: int


def effective_owner_from_obs(planet_feats_n: np.ndarray, player: int) -> np.ndarray:
    """Reconstruct an absolute ``owner`` array from the ownership one-hot block.

    ``Observation.planet_feats`` only stores ownership *relative to the viewing
    player*: a self flag, three enemy slots, and a neutral flag.  We invert that
    into a synthetic absolute owner id that round-trips exactly through
    :func:`reworked_features.standard_node_features` (which derives the enemy
    slot as ``(owner - player - 1) mod (MAX_ENEMY_OWNER_SLOTS + 1)``):

    * self planet  -> ``owner = player``
    * neutral      -> ``owner = -1``
    * enemy slot j -> ``owner = player + 1 + j``  (so ``(owner-player-1) mod 4 == j``)

    This is sufficient for every reworked metric, which only needs the
    self / enemy / neutral partition (and consistent enemy slots), not the true
    cross-perspective player ids.

    Parameters
    ----------
    planet_feats_n:
        ``(n, PLANET_FEATURE_DIM)`` slice of real planets.
    player:
        The viewing player's id.

    Returns
    -------
    ``(n,)`` int64 owner array.
    """
    pf = np.asarray(planet_feats_n)
    n = pf.shape[0]
    owner = np.full(n, -1, dtype=np.int64)
    is_self = pf[:, PF_IDX_OWNER_SELF] > 0.5
    owner[is_self] = int(player)
    for j in range(MAX_ENEMY_OWNER_SLOTS):
        sel = pf[:, PF_IDX_OWNER_ENEMY_0 + j] > 0.5
        owner[sel] = int(player) + 1 + j
    return owner


def board_state_from_obs(
    obs,
    player: int,
    *,
    remaining_turns: Optional[int] = None,
) -> BoardState:
    """De-normalise a live :class:`Observation` into a :class:`BoardState`."""
    n = int(np.asarray(obs.planet_mask).sum())
    pf = np.asarray(obs.planet_feats)[:n]
    eb = np.asarray(obs.edge_bias)

    if remaining_turns is None:
        meta = getattr(obs, "meta", {}) or {}
        remaining_turns = int(meta.get("remaining_turns", 0))

    return BoardState(
        n=n,
        player=int(player),
        owner=effective_owner_from_obs(pf, player),
        ships=(pf[:, PF_IDX_SHIPS] * SHIPS_SCALE).astype(np.float32),
        production=(pf[:, PF_IDX_PRODUCTION] * PRODUCTION_SCALE).astype(np.float32),
        radius=(pf[:, PF_IDX_RADIUS] * RADIUS_SCALE).astype(np.float32),
        is_rotating=pf[:, PF_IDX_IS_ROTATING].astype(np.float32),
        is_comet=pf[:, PF_IDX_IS_COMET].astype(np.float32),
        distance_matrix=eb[:n, :n, EDGE_BIAS_DISTANCE_IDX].astype(np.float32),
        crosses_sun=eb[:n, :n, EDGE_BIAS_CROSSES_SUN_IDX].astype(np.float32),
        remaining_turns=int(remaining_turns),
    )


def build_reworked_node_features(
    state: BoardState,
    *,
    board_size: float = DEFAULT_BOARD_SIZE,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    threat_support_norm: float = THREAT_SUPPORT_NORM,
) -> np.ndarray:
    """Assemble the ``(n, REWORKED_NODE_FEATURE_DIM)`` node feature matrix.

    Layout: 10 ``standard_node_features`` channels followed by the 5 derived
    metrics ``[discounted_prod, closeness, potential_threat, potential_support,
    friendly_compactness]``.
    """
    n = state.n
    player = state.player
    dist = state.distance_matrix

    std = standard_node_features(
        state.owner, state.ships,
        player=player,
        production=state.production,
        radius=state.radius,
        is_rotating=state.is_rotating,
        is_comet=state.is_comet,
    )  # (n, 10)

    dpp = np.asarray(
        discounted_production_potential(state.production, state.remaining_turns),
        dtype=np.float32,
    ).reshape(n, 1)
    clo = np.asarray(closeness_centrality(dist), dtype=np.float32).reshape(n, 1)
    thr = np.asarray(
        potential_threat(dist, state.owner, state.ships, player=player,
                         board_size=board_size, max_speed=max_speed,
                         norm=threat_support_norm),
        dtype=np.float32,
    ).reshape(n, 1)
    sup = np.asarray(
        potential_support(dist, state.owner, state.ships, player=player,
                          board_size=board_size, max_speed=max_speed,
                          norm=threat_support_norm),
        dtype=np.float32,
    ).reshape(n, 1)
    cmp_ = np.asarray(
        friendly_compactness(dist, state.owner, player=player),
        dtype=np.float32,
    ).reshape(n, 1)

    node = np.concatenate([std, dpp, clo, thr, sup, cmp_], axis=1)
    return node.astype(np.float32)


def build_reworked_edge_features(
    obs,
    state: BoardState,
    **flux_kwargs,
) -> np.ndarray:
    """Assemble the ``(n, n, REWORKED_EDGE_FEATURE_DIM)`` edge feature tensor.

    Channels: ``[distance, crosses_sun, friendly_flux, enemy_flux]``.  Fleet
    routing arrays come straight off the observation (planet-slot pointers).
    """
    return edge_features(
        state.distance_matrix,
        fleet_src=obs.fleet_src_slot,
        fleet_tgt=obs.fleet_tgt_slot,
        fleet_eta=obs.fleet_eta_raw,
        fleet_ships=obs.fleet_ships_raw,
        fleet_is_self=obs.fleet_is_self,
        fleet_is_enemy=obs.fleet_is_enemy,
        fleet_mask=obs.fleet_mask,
        crosses_sun=state.crosses_sun,
        **flux_kwargs,
    )


def build_reworked_features(
    obs,
    player: int,
    *,
    remaining_turns: Optional[int] = None,
    board_size: float = DEFAULT_BOARD_SIZE,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    threat_support_norm: float = THREAT_SUPPORT_NORM,
    flux_kwargs: Optional[dict] = None,
) -> dict:
    """Full single-observation feature bundle for the reworked encoder.

    Returns a dict with NumPy arrays:

    * ``node_feats``   ``(n, 15)``
    * ``edge_feats``   ``(n, n, 4)``
    * ``global_feats`` ``(14,)``  (re-uses the existing global block)
    * ``planet_mask``  ``(n,)`` bool (all True; real planets only)
    * ``state``        the :class:`BoardState` (handy for the allocator)
    """
    state = board_state_from_obs(obs, player, remaining_turns=remaining_turns)
    node = build_reworked_node_features(
        state, board_size=board_size, max_speed=max_speed,
        threat_support_norm=threat_support_norm,
    )
    edge = build_reworked_edge_features(obs, state, **(flux_kwargs or {}))
    glob = np.asarray(obs.global_feats, dtype=np.float32).reshape(-1)
    return {
        "node_feats": node,
        "edge_feats": edge,
        "global_feats": glob,
        "planet_mask": np.ones(state.n, dtype=bool),
        "state": state,
    }


def pack_reworked_batch(
    obs_player_pairs: Sequence[tuple],
    *,
    device=None,
    p_max: Optional[int] = None,
    remaining_turns: Optional[int] = None,
    board_size: float = DEFAULT_BOARD_SIZE,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    threat_support_norm: float = THREAT_SUPPORT_NORM,
    flux_kwargs: Optional[dict] = None,
) -> dict:
    """Pack ``(obs, player)`` pairs into batched torch tensors for the encoder.

    Each observation may hold a different number of real planets ``n_b``; all
    are right-padded to a common ``P`` (``max_b n_b`` unless ``p_max`` given),
    with padded node / edge rows zeroed and ``planet_mask`` set ``False`` so the
    encoder masks them out of message passing and pooling.

    Returns a dict with keys matching
    :meth:`reworked_model.ReworkedPlanetGraphEncoder.forward`:
    ``node_feats (B,P,15)``, ``edge_feats (B,P,P,4)``, ``global_feats (B,14)``,
    ``planet_mask (B,P)`` -- plus ``states`` (list of :class:`BoardState`).
    """
    import torch

    bundles = [
        build_reworked_features(
            obs, player,
            remaining_turns=remaining_turns,
            board_size=board_size, max_speed=max_speed,
            threat_support_norm=threat_support_norm,
            flux_kwargs=flux_kwargs,
        )
        for (obs, player) in obs_player_pairs
    ]

    B = len(bundles)
    ns = [b["state"].n for b in bundles]
    P = int(p_max) if p_max is not None else max(ns + [1])

    node = np.zeros((B, P, REWORKED_NODE_FEATURE_DIM), dtype=np.float32)
    edge = np.zeros((B, P, P, REWORKED_EDGE_FEATURE_DIM), dtype=np.float32)
    glob = np.zeros((B, REWORKED_GLOBAL_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((B, P), dtype=bool)

    for b, bundle in enumerate(bundles):
        nb = bundle["state"].n
        node[b, :nb] = bundle["node_feats"]
        edge[b, :nb, :nb] = bundle["edge_feats"]
        glob[b] = bundle["global_feats"][:REWORKED_GLOBAL_FEATURE_DIM]
        mask[b, :nb] = True

    out = {
        "node_feats": torch.from_numpy(node),
        "edge_feats": torch.from_numpy(edge),
        "global_feats": torch.from_numpy(glob),
        "planet_mask": torch.from_numpy(mask),
    }
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    out["states"] = [b["state"] for b in bundles]
    return out


# Routing-cost weights in ETA (turn) units: how many "equivalent turns" the
# most-threatened source is penalised / the most-interior base is discounted.
# Kept well below typical ETA spreads so routing preference is a tie-breaker,
# never a priority-breaker.
DEFAULT_ROUTING_W_THREAT = 20.0
DEFAULT_ROUTING_W_COMPACT = 12.0


def build_routing_cost_matrix(
    state: BoardState,
    *,
    w_threat: float = DEFAULT_ROUTING_W_THREAT,
    w_compact: float = DEFAULT_ROUTING_W_COMPACT,
    w_crosses_sun: float = 0.0,
    board_size: float = DEFAULT_BOARD_SIZE,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
) -> np.ndarray:
    r"""Metric-based ``(n, n)`` routing-preference cost for the allocator.

    Builds the additive ``cost_matrix`` consumed by
    :func:`reworked_allocator.allocate_launches`.  It is expressed in **ETA
    (turn) units** so it sits on the same scale as the transport ETA and acts as
    a *tie-breaker among sources*, never overriding the RL priority order.

    Per-**source** penalty (broadcast across all targets):

    .. math::

        \text{penalty}(h) = w_\text{threat}\cdot \widehat{\text{threat}}(h)
                          - w_\text{compact}\cdot \widehat{\text{compact}}(h)

    where ``threat`` is :func:`reworked_features.potential_threat` (don't strip a
    source that is itself under enemy pressure) and ``compact`` is
    :func:`reworked_features.friendly_compactness` (prefer drawing from
    deep-interior bases that are cheap to leave thin).  Both metrics are
    min-max normalised to ``[0, 1]`` across planets so the weights are
    interpretable turn-equivalents.

    Optionally a per-**pair** ``w_crosses_sun`` penalty discourages routes whose
    straight path is blocked by the sun (``state.crosses_sun``), demonstrating
    the full ``(n, n)`` generality of the hook.

    Returns
    -------
    ``(n, n)`` ``float32``; entry ``[h, T]`` adds to ``ETA(h, T)``.
    """
    n = state.n
    threat = np.asarray(
        potential_threat(state.distance_matrix, state.owner, state.ships,
                         player=state.player, board_size=board_size, max_speed=max_speed),
        dtype=np.float64,
    ).reshape(n)
    compact = np.asarray(
        friendly_compactness(state.distance_matrix, state.owner, player=state.player),
        dtype=np.float64,
    ).reshape(n)

    tn = threat / (float(threat.max()) + 1e-9)
    cn = compact / (float(compact.max()) + 1e-9)
    source_penalty = w_threat * tn - w_compact * cn                  # (n,) per source

    cost = np.repeat(source_penalty[:, None], n, axis=1).astype(np.float32)
    if w_crosses_sun != 0.0:
        cost = cost + np.float32(w_crosses_sun) * state.crosses_sun.astype(np.float32)
    return cost


def build_allocator_inputs(
    obs,
    player: int,
    *,
    state: Optional[BoardState] = None,
    horizon: float = 15.0,
    potential_weight: float = 0.0,
    routing_cost: bool = False,
    w_threat: float = DEFAULT_ROUTING_W_THREAT,
    w_compact: float = DEFAULT_ROUTING_W_COMPACT,
    w_crosses_sun: float = 0.0,
    board_size: float = DEFAULT_BOARD_SIZE,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
) -> dict:
    """Build the board-state + ``incoming_enemy`` inputs for the allocator.

    Bridges the feature pipeline to :func:`reworked_allocator.allocate_launches`:
    everything except the RL decisions (``priority_order`` / ``attack_targets`` /
    ``defend_targets`` / ``commit_fraction``) is derived here from the live
    observation, including the ``incoming_enemy`` estimate
    (:func:`reworked_features.incoming_enemy_estimate`).

    If ``routing_cost=True`` a metric-based ``cost_matrix`` is also attached
    (see :func:`build_routing_cost_matrix`) so the allocator prefers stripping
    safe interior bases over threatened / frontier ones.

    Returns a dict ready to splat into ``allocate_launches(**inputs, player=...,
    priority_order=..., attack_targets=..., defend_targets=..., commit_fraction=...)``.
    """
    if state is None:
        state = board_state_from_obs(obs, player)

    incoming = incoming_enemy_estimate(
        state.distance_matrix,
        state.owner,
        state.ships,
        player=player,
        fleet_src=obs.fleet_src_slot,
        fleet_tgt=obs.fleet_tgt_slot,
        fleet_eta=obs.fleet_eta_raw,
        fleet_ships=obs.fleet_ships_raw,
        fleet_is_self=obs.fleet_is_self,
        fleet_is_enemy=obs.fleet_is_enemy,
        fleet_mask=obs.fleet_mask,
        horizon=horizon,
        potential_weight=potential_weight,
        board_size=board_size,
        max_speed=max_speed,
    )

    inputs = {
        "distance_matrix": state.distance_matrix,
        "owner": state.owner,
        "ships": state.ships,
        "production": state.production,
        "incoming_enemy": incoming,
        "board_size": board_size,
        "max_speed": max_speed,
    }
    if routing_cost:
        inputs["cost_matrix"] = build_routing_cost_matrix(
            state, w_threat=w_threat, w_compact=w_compact,
            w_crosses_sun=w_crosses_sun, board_size=board_size, max_speed=max_speed,
        )
    return inputs
