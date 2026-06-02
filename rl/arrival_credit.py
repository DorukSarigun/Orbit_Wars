"""ETA-weighted arrival-credit eligibility trace (Option B).

For each successful fleet launch we snapshot the launch-time per-planet
graph-metric vector at the target slot.  When the fleet arrives — i.e.
``FleetTracker.pop_resolved`` yields ``(buffer_idx, outcome)`` for the
corresponding launch — we recompute the same per-planet vector under
the current state and credit the *original launch transition* with the
ETA-discounted weighted delta:

    credit = w(eta) * arrival_credit_weight *
             Σ_c channel_weight[c] · (curr_vec[c] - launch_vec[c])

where ``channel_weight`` mirrors the 14-vector built by
:func:`rl.arrival_credit.build_channel_weights`.
``buffer.add_delayed_reward`` so it shares the same delay-discount
machinery as fleet/planet-loss credit.

This is a *backward* credit assignment: launches that take τ turns
to land receive an extra reward (or penalty) at the launch step
proportional to what the fleet ultimately accomplishes on the graph,
weighted by ETA so close-by launches are credited more heavily.

The module is **opt-in**: nothing fires unless
``RewardConfig.enable_arrival_credit=True``.

All operations are O(1) per snapshot / per arrival; the only non-trivial
cost is the once-per-turn ``compute_planet_metric_vectors`` pass which
the rollout already runs anyway when graph_native shaping is active.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from obs_builder import (
    INCOMING_ETA_BUCKETS,
    PF_IDX_INCOMING_ENEMY_B0,
    PF_IDX_INCOMING_ENEMY_MIN_ETA,
    PF_IDX_INCOMING_ENEMY_SOONEST_SHIPS,
    PF_IDX_INCOMING_ENEMY_TOTAL,
    PF_IDX_LOG_SHIPS,
    PF_IDX_OWNER_ENEMY_0,
    PF_IDX_OWNER_ENEMY_1,
    PF_IDX_OWNER_ENEMY_2,
    PF_IDX_OWNER_NEUTRAL,
    PF_IDX_OWNER_SELF,
    PF_IDX_PRODUCTION,
    PF_IDX_SHIPS,
)


__all__ = [
    "ArrivalCreditSnapshot",
    "ArrivalCreditTracker",
    "build_channel_weights",
    "eta_weight",
]


# Length of the channel-weight vector used by arrival-credit.  Mirrors
# ``model_graph_pyg.GRAPH_NODE_METRIC_DIM_FLUX`` (= 10 base + 4 flux).
NUM_METRIC_CHANNELS: int = 14


@dataclass
class ArrivalCreditSnapshot:
    """One snapshot taken at fleet-launch time.

    ``target_vec`` and (optionally) ``source_vec`` are float32 arrays of
    length ``NUM_METRIC_CHANNELS``.  Slots are pinned by *planet id*
    because the slot index can in principle shift across turns; the
    arrival callback resolves the current slot via
    ``obs.planet_ids`` lookup.
    """

    buffer_idx: int
    player: int
    target_planet_id: int
    eta_at_launch: float
    target_vec: np.ndarray
    ships_sent: int = 0
    source_planet_id: int = -1
    source_vec: Optional[np.ndarray] = None


def eta_weight(
    eta: float,
    *,
    mode: str = "exp_decay",
    tau: float = 10.0,
    eta_norm: float = 50.0,
) -> float:
    """ETA-decay weight applied to the realized arrival impact.

    Mirrors the parametric modes of :mod:`rl.flux._weight` so a single
    set of knobs governs both flux-feature decay and arrival-credit
    decay when the user wants them coupled.
    """
    e = max(0.0, float(eta))
    if mode == "raw":
        return 1.0
    if mode == "linear":
        denom = max(float(eta_norm), 1e-6)
        return max(0.0, 1.0 - e / denom)
    if mode == "exp_decay":
        denom = max(float(tau), 1e-6)
        return float(np.exp(-e / denom))
    raise ValueError(f"unknown arrival_credit mode: {mode!r}")


def build_channel_weights(cfg) -> np.ndarray:
    """Build the 14-channel weight vector from a :class:`RewardConfig`.

    Order mirrors ``compute_planet_metric_vectors`` / GNN node metrics:

        [friendly_influence, enemy_influence, influence_balance,
         friendly_support, supportability, vulnerability,
         production_potential, friendly_production_potential,
         strategic_potential, held_discounted_production,
         flux_friendly_in, flux_friendly_out,
         flux_enemy_in, flux_enemy_out]

    Uses ``arrival_credit_*_weight`` fields — independent of any removed
    step-level graph shaping.
    """
    return np.array(
        [
            float(getattr(cfg, "arrival_credit_friendly_influence_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_enemy_influence_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_influence_balance_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_friendly_support_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_supportability_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_vulnerability_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_production_potential_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_friendly_production_potential_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_strategic_potential_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_held_discounted_production_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_flux_friendly_in_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_flux_friendly_out_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_flux_enemy_in_weight", 0.0)),
            float(getattr(cfg, "arrival_credit_flux_enemy_out_weight", 0.0)),
        ],
        dtype=np.float32,
    )


def _align_to_channels(vec: np.ndarray) -> np.ndarray:
    """Right-pad or truncate ``vec`` to length :data:`NUM_METRIC_CHANNELS`."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    d = v.shape[0]
    if d == NUM_METRIC_CHANNELS:
        return v
    if d > NUM_METRIC_CHANNELS:
        return v[:NUM_METRIC_CHANNELS]
    out = np.zeros(NUM_METRIC_CHANNELS, dtype=np.float32)
    out[:d] = v
    return out


class ArrivalCreditTracker:
    """Per-player snapshot store keyed by launch-transition buffer index.

    Lifecycle:

    1.  At launch time the rollout calls :meth:`register` with the
        launch-time per-planet metric matrix and the target / source
        planet ids.
    2.  Each turn, for every ``(buffer_idx, outcome)`` returned by
        :meth:`rl.fleet_tracker.FleetTracker.pop_resolved`, the rollout
        calls :meth:`compute_credit` with the current per-planet metric
        matrix.  The matching snapshot is consumed (removed) regardless
        of whether the credit is non-zero, so memory stays bounded.
    3.  At episode end :meth:`clear` (or simply garbage-collecting the
        tracker) drops any unresolved launches.

    The tracker holds **no torch tensors** and no env references — only
    numpy arrays — so it is safe to instantiate per-episode without
    extra cleanup.

    Default behaviour is *no-op* if either ``cfg.enable_arrival_credit``
    is False or the launch-time / current metric matrix is ``None``.
    """

    __slots__ = ("_snapshots", "credits_emitted", "credits_total")

    def __init__(self) -> None:
        self._snapshots: dict[int, ArrivalCreditSnapshot] = {}
        # Diagnostic counters: how many arrivals were credited and the
        # signed running sum of the credits.  Useful for unit tests and
        # training-time logging (the rollout layer can expose these to
        # the iteration log to see whether the trace is firing at all).
        self.credits_emitted: int = 0
        self.credits_total: float = 0.0

    # ----------------------------------------------------------- registration
    def register(
        self,
        *,
        buffer_idx: int,
        player: int,
        target_planet_id: int,
        eta_at_launch: float,
        launch_metric_matrix: Optional[np.ndarray],
        launch_planet_ids: np.ndarray,
        ships_sent: int = 0,
        source_planet_id: int = -1,
        include_source: bool = False,
    ) -> bool:
        """Record a snapshot for one launch.  Returns ``True`` if stored.

        The snapshot is silently skipped (returns ``False``) when:

        - ``launch_metric_matrix`` is ``None`` (torch unavailable);
        - the target planet id cannot be located in ``launch_planet_ids``;
        - the located slot is outside the rows present in
          ``launch_metric_matrix``.

        The metric matrix is sliced (not retained as a reference) so the
        caller is free to overwrite the source array.
        """
        if launch_metric_matrix is None:
            return False
        tgt_slot = _planet_id_to_slot(launch_planet_ids, int(target_planet_id))
        if tgt_slot < 0 or tgt_slot >= launch_metric_matrix.shape[0]:
            return False
        target_vec = _align_to_channels(launch_metric_matrix[tgt_slot])

        source_vec: Optional[np.ndarray] = None
        if include_source and source_planet_id >= 0:
            src_slot = _planet_id_to_slot(launch_planet_ids, int(source_planet_id))
            if 0 <= src_slot < launch_metric_matrix.shape[0]:
                source_vec = _align_to_channels(launch_metric_matrix[src_slot])

        self._snapshots[int(buffer_idx)] = ArrivalCreditSnapshot(
            buffer_idx=int(buffer_idx),
            player=int(player),
            target_planet_id=int(target_planet_id),
            eta_at_launch=float(eta_at_launch),
            target_vec=target_vec,
            ships_sent=int(ships_sent),
            source_planet_id=int(source_planet_id),
            source_vec=source_vec,
        )
        return True

    # ------------------------------------------------------- counterfactual obs
    def build_counterfactual_observations(
        self,
        *,
        buffer_idx: int,
        curr_obs,
        curr_planet_ids: np.ndarray,
        actual_target_planet_id: int = -1,
        previous_target_owner: int = -2,
        return_to_source: bool = True,
        horizon: float = 0.0,
    ):
        """Build actual/counterfactual observations for an arrival.

        Counterfactual semantics:

        * target: remove this arriving fleet's ships from the actual
          arrival state.  If that makes a self-owned target negative,
          the target is marked non-self with ``abs(negative)`` ships, so
          friendly production/influence metrics are disabled naturally.
        * source: optionally put the fleet back at the source.  If the
          source is currently enemy-owned, those returned ships can flip
          it back to self when they would have prevented the loss.
        * horizon: if positive, both actual and counterfactual branches
          resolve the soonest enemy arrival at the affected self-owned
          planets when its ETA is within the horizon cap.

        Returns ``(actual_obs, counterfactual_obs)`` or ``None`` when the
        snapshot / target cannot be resolved.
        """
        snap = self._snapshots.get(int(buffer_idx))
        if snap is None or curr_obs is None or curr_planet_ids is None:
            return None
        ships = float(max(0, int(snap.ships_sent)))
        if ships <= 0.0:
            return None

        curr_feats = np.asarray(curr_obs.planet_feats, dtype=np.float32)
        actual_feats = curr_feats.copy()
        cf_feats = curr_feats.copy()
        tgt_id = (
            int(actual_target_planet_id)
            if int(actual_target_planet_id) >= 0
            else int(snap.target_planet_id)
        )
        tgt_slot = _planet_id_to_slot(curr_planet_ids, tgt_id)
        if tgt_slot < 0 or tgt_slot >= cf_feats.shape[0]:
            return None

        _remove_arriving_fleet_from_target(
            cf_feats,
            tgt_slot,
            ships=ships,
            player=int(snap.player),
            previous_owner=int(previous_target_owner),
        )

        src_slot = -1
        if return_to_source and int(snap.source_planet_id) >= 0:
            src_slot = _planet_id_to_slot(curr_planet_ids, int(snap.source_planet_id))
            if 0 <= src_slot < cf_feats.shape[0]:
                _return_fleet_to_source(cf_feats, src_slot, ships=ships)

        horizon_f = max(0.0, float(horizon))
        if horizon_f > 0.0:
            _resolve_next_enemy_arrival_if_due(actual_feats, tgt_slot, horizon_f)
            _resolve_next_enemy_arrival_if_due(cf_feats, tgt_slot, horizon_f)
            if 0 <= src_slot < cf_feats.shape[0]:
                _resolve_next_enemy_arrival_if_due(actual_feats, src_slot, horizon_f)
                _resolve_next_enemy_arrival_if_due(cf_feats, src_slot, horizon_f)

        return (
            _replace_planet_feats(curr_obs, actual_feats),
            _replace_planet_feats(curr_obs, cf_feats),
        )

    # --------------------------------------------------------------- arrival
    def compute_credit(
        self,
        *,
        buffer_idx: int,
        curr_metric_matrix: Optional[np.ndarray],
        curr_planet_ids: np.ndarray,
        channel_weights: np.ndarray,
        cfg,
        actual_target_planet_id: int = -1,
        counterfactual_metric_matrix: Optional[np.ndarray] = None,
        counterfactual_planet_ids: Optional[np.ndarray] = None,
        owner_self_mask: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Resolve one arrival.  Returns the signed credit, or ``None``.

        The snapshot is consumed (removed from the tracker) whether or
        not the credit is non-zero, since the launch has now been
        resolved.

        ``actual_target_planet_id`` lets the caller use the *realized*
        target slot when the fleet was redirected en route (e.g. sun
        deflection).  When negative, falls back to the snapshot's
        intended target.

        Returns ``None`` if no snapshot exists for ``buffer_idx`` or the
        current per-planet metric matrix is unavailable.
        """
        snap = self._snapshots.pop(int(buffer_idx), None)
        if snap is None:
            return None
        if curr_metric_matrix is None:
            return None

        tgt_id = (
            int(actual_target_planet_id)
            if int(actual_target_planet_id) >= 0
            else int(snap.target_planet_id)
        )
        tgt_slot = _planet_id_to_slot(curr_planet_ids, tgt_id)
        if tgt_slot < 0 or tgt_slot >= curr_metric_matrix.shape[0]:
            return 0.0

        if counterfactual_metric_matrix is not None:
            cf_planet_ids = (
                curr_planet_ids
                if counterfactual_planet_ids is None
                else counterfactual_planet_ids
            )
            scope = str(
                getattr(cfg, "arrival_credit_counterfactual_scope", "full_graph")
            )
            weighted = _weighted_counterfactual_delta(
                curr_metric_matrix=curr_metric_matrix,
                curr_planet_ids=curr_planet_ids,
                counterfactual_metric_matrix=counterfactual_metric_matrix,
                counterfactual_planet_ids=cf_planet_ids,
                channel_weights=channel_weights,
                target_planet_id=tgt_id,
                source_planet_id=int(snap.source_planet_id),
                scope=scope,
                owner_self_mask=(
                    owner_self_mask
                    if str(getattr(cfg, "arrival_credit_owner_filter", "self")) == "self"
                    else None
                ),
            )
        else:
            curr_target_vec = _align_to_channels(curr_metric_matrix[tgt_slot])
            delta = curr_target_vec - snap.target_vec
            weighted = float(np.dot(channel_weights, delta))

        # Optional source-side credit for the legacy launch→arrival delta.
        # Counterfactual mode compares actual-vs-counterfactual directly,
        # so source cost is represented by the counterfactual source edit.
        if counterfactual_metric_matrix is None and snap.source_vec is not None:
            src_slot = _planet_id_to_slot(curr_planet_ids, int(snap.source_planet_id))
            if 0 <= src_slot < curr_metric_matrix.shape[0]:
                curr_source_vec = _align_to_channels(curr_metric_matrix[src_slot])
                # Sign convention: a launch *costs* ships at the source.
                # A drop in friendly_influence / supportability at src is
                # therefore expected and shouldn't be punished.  We add
                # the source delta with a *minus* so the net signal is
                # "improvement at tgt minus deterioration at src".  This
                # encodes the trade-off the agent is making.
                weighted += -float(np.dot(channel_weights, curr_source_vec - snap.source_vec))

        w_eta = eta_weight(
            snap.eta_at_launch,
            mode=str(getattr(cfg, "arrival_credit_mode", "exp_decay")),
            tau=float(getattr(cfg, "arrival_credit_tau", 10.0)),
            eta_norm=float(getattr(cfg, "arrival_credit_eta_norm", 50.0)),
        )
        scale = float(getattr(cfg, "arrival_credit_weight", 1.0))
        reward_scale = float(getattr(cfg, "reward_scale", 1.0))
        credit = w_eta * scale * reward_scale * weighted
        # Diagnostic accounting: count every arrival we *resolved*, even
        # when the resulting credit is exactly 0 (e.g. zero deltas).
        # The signed running sum lets training logs show whether the
        # trace is net-positive / net-negative for the agent.
        self.credits_emitted += 1
        self.credits_total += float(credit)
        return credit

    # --------------------------------------------------------------- helpers
    def discard(self, buffer_idx: int) -> bool:
        """Drop a snapshot without computing credit (e.g. fleet destroyed).

        Returns ``True`` if a snapshot was actually present.
        """
        return self._snapshots.pop(int(buffer_idx), None) is not None

    def clear(self) -> None:
        self._snapshots.clear()

    def __len__(self) -> int:
        return len(self._snapshots)

    def __contains__(self, buffer_idx: int) -> bool:
        return int(buffer_idx) in self._snapshots


# --------------------------------------------------------------------- utils
def _planet_id_to_slot(planet_ids: np.ndarray, target_id: int) -> int:
    """Linear scan; ``planet_ids`` is tiny (P_max ~ 44)."""
    if planet_ids is None or target_id < 0:
        return -1
    arr = np.asarray(planet_ids)
    matches = np.where(arr == int(target_id))[0]
    if matches.size == 0:
        return -1
    return int(matches[0])


def _replace_planet_feats(obs, planet_feats: np.ndarray):
    """Dataclass-preserving copy for Observation-like objects."""
    try:
        return replace(obs, planet_feats=planet_feats)
    except TypeError:
        import copy

        out = copy.copy(obs)
        out.planet_feats = planet_feats
        return out


def _ship_count(feats: np.ndarray, slot: int) -> float:
    return float(max(0.0, feats[slot, PF_IDX_SHIPS] * 100.0))


def _set_ship_count(feats: np.ndarray, slot: int, ships: float) -> None:
    ships_f = float(max(0.0, ships))
    feats[slot, PF_IDX_SHIPS] = ships_f / 100.0
    feats[slot, PF_IDX_LOG_SHIPS] = float(np.log(ships_f + 1.0) / np.log(1000.0))


def _owner_kind(feats: np.ndarray, slot: int) -> str:
    if float(feats[slot, PF_IDX_OWNER_SELF]) > 0.5:
        return "self"
    if float(feats[slot, PF_IDX_OWNER_NEUTRAL]) > 0.5:
        return "neutral"
    return "enemy"


def _owner_kind_from_owner_id(owner: int, player: int, *, fallback: str = "enemy") -> str:
    if int(owner) == int(player):
        return "self"
    if int(owner) == -1:
        return "neutral"
    if int(owner) >= 0:
        return "enemy"
    return fallback


def _set_owner_kind(feats: np.ndarray, slot: int, kind: str) -> None:
    feats[
        slot,
        PF_IDX_OWNER_SELF:PF_IDX_OWNER_NEUTRAL + 1,
    ] = 0.0
    if kind == "self":
        feats[slot, PF_IDX_OWNER_SELF] = 1.0
    elif kind == "neutral":
        feats[slot, PF_IDX_OWNER_NEUTRAL] = 1.0
    else:
        feats[slot, PF_IDX_OWNER_ENEMY_0] = 1.0
        feats[slot, PF_IDX_OWNER_ENEMY_1] = 0.0
        feats[slot, PF_IDX_OWNER_ENEMY_2] = 0.0


def _remove_arriving_fleet_from_target(
    feats: np.ndarray,
    slot: int,
    *,
    ships: float,
    player: int,
    previous_owner: int,
) -> None:
    current_owner = _owner_kind(feats, slot)
    current_ships = _ship_count(feats, slot)
    if current_owner == "self":
        without_arrival = current_ships - float(ships)
        if without_arrival >= 0.0:
            _set_ship_count(feats, slot, without_arrival)
            return
        # The arriving fleet was necessary.  If the target was formerly
        # neutral/enemy, restore that ownership; if it was already ours,
        # the negative branch means a later/same-turn enemy attack would
        # have taken it without this fleet.
        prev_kind = _owner_kind_from_owner_id(
            previous_owner,
            player,
            fallback="enemy",
        )
        if prev_kind == "self":
            prev_kind = "enemy"
        _set_owner_kind(feats, slot, prev_kind)
        _set_ship_count(feats, slot, -without_arrival)
        return

    # Failed attack / partial damage: without this fleet, the non-self
    # defender keeps the ships that were killed by the arriving fleet.
    _set_ship_count(feats, slot, current_ships + float(ships))


def _return_fleet_to_source(feats: np.ndarray, slot: int, *, ships: float) -> None:
    current_owner = _owner_kind(feats, slot)
    current_ships = _ship_count(feats, slot)
    if current_owner == "self":
        _set_ship_count(feats, slot, current_ships + float(ships))
        return
    # If the source is no longer ours, treat the stayed-home fleet as
    # additional defense against the force that captured/neutralized it.
    attacker_remainder = current_ships - float(ships)
    if attacker_remainder >= 0.0:
        _set_ship_count(feats, slot, attacker_remainder)
    else:
        _set_owner_kind(feats, slot, "self")
        _set_ship_count(feats, slot, -attacker_remainder)


def _resolve_next_enemy_arrival_if_due(
    feats: np.ndarray,
    slot: int,
    horizon: float,
) -> None:
    if _owner_kind(feats, slot) != "self":
        return
    eta = float(feats[slot, PF_IDX_INCOMING_ENEMY_MIN_ETA]) * 50.0
    enemy_ships = float(feats[slot, PF_IDX_INCOMING_ENEMY_SOONEST_SHIPS]) * 100.0
    if eta <= 0.0 or enemy_ships <= 0.0 or eta > float(horizon):
        return

    production = float(max(0.0, feats[slot, PF_IDX_PRODUCTION] * 5.0))
    garrison = _ship_count(feats, slot) + production * eta
    remaining = garrison - enemy_ships
    if remaining >= 0.0:
        _set_ship_count(feats, slot, remaining)
    else:
        _set_owner_kind(feats, slot, "enemy")
        _set_ship_count(feats, slot, -remaining)

    _subtract_soonest_enemy_arrival(feats, slot, eta=eta, ships=enemy_ships)


def _subtract_soonest_enemy_arrival(
    feats: np.ndarray,
    slot: int,
    *,
    eta: float,
    ships: float,
) -> None:
    remaining_total = max(0.0, float(feats[slot, PF_IDX_INCOMING_ENEMY_TOTAL]) * 100.0 - ships)
    feats[slot, PF_IDX_INCOMING_ENEMY_TOTAL] = remaining_total / 100.0
    for i, upper in enumerate(INCOMING_ETA_BUCKETS):
        if eta <= float(upper):
            bucket_idx = PF_IDX_INCOMING_ENEMY_B0 + i
            bucket_ships = float(feats[slot, bucket_idx]) * 100.0
            feats[slot, bucket_idx] = max(0.0, bucket_ships - ships) / 100.0
            break
    feats[slot, PF_IDX_INCOMING_ENEMY_MIN_ETA] = 0.0
    feats[slot, PF_IDX_INCOMING_ENEMY_SOONEST_SHIPS] = 0.0


def _weighted_counterfactual_delta(
    *,
    curr_metric_matrix: np.ndarray,
    curr_planet_ids: np.ndarray,
    counterfactual_metric_matrix: np.ndarray,
    counterfactual_planet_ids: np.ndarray,
    channel_weights: np.ndarray,
    target_planet_id: int,
    source_planet_id: int,
    scope: str,
    owner_self_mask: Optional[np.ndarray] = None,
) -> float:
    if scope == "target":
        ids = [int(target_planet_id)]
    elif scope in ("target_source", "local"):
        ids = [int(target_planet_id)]
        if int(source_planet_id) >= 0 and int(source_planet_id) != int(target_planet_id):
            ids.append(int(source_planet_id))
    else:
        total = np.zeros(NUM_METRIC_CHANNELS, dtype=np.float32)
        max_rows = min(len(curr_planet_ids), curr_metric_matrix.shape[0])
        for curr_slot in range(max_rows):
            pid = int(curr_planet_ids[curr_slot])
            if pid < 0:
                continue
            # Ownership filter: only sum (curr - cf) on planets *I* own
            # in the actual current state.  The intent is "what happened
            # to MY empire's metrics, vs what would have happened if I
            # hadn't launched".  Enemy-side metric changes (e.g. their
            # supportability) are not directly part of MY reward.
            if owner_self_mask is not None:
                if curr_slot >= owner_self_mask.shape[0]:
                    continue
                if not bool(owner_self_mask[curr_slot]):
                    continue
            cf_slot = _planet_id_to_slot(counterfactual_planet_ids, pid)
            if 0 <= cf_slot < counterfactual_metric_matrix.shape[0]:
                total += (
                    _align_to_channels(curr_metric_matrix[curr_slot])
                    - _align_to_channels(counterfactual_metric_matrix[cf_slot])
                )
        return float(np.dot(channel_weights, total))

    total = 0.0
    for pid in ids:
        curr_slot = _planet_id_to_slot(curr_planet_ids, pid)
        cf_slot = _planet_id_to_slot(counterfactual_planet_ids, pid)
        if (
            0 <= curr_slot < curr_metric_matrix.shape[0]
            and 0 <= cf_slot < counterfactual_metric_matrix.shape[0]
        ):
            total += float(
                np.dot(
                    channel_weights,
                    _align_to_channels(curr_metric_matrix[curr_slot])
                    - _align_to_channels(counterfactual_metric_matrix[cf_slot]),
                )
            )
    return total
