"""ETA-weighted arrival-credit eligibility trace (Option B).

For each successful fleet launch we snapshot the launch-time per-planet
graph-metric vector at the target slot.  When the fleet arrives — i.e.
``FleetTracker.pop_resolved`` yields ``(buffer_idx, outcome)`` for the
corresponding launch — we recompute the same per-planet vector under
the current state and credit the *original launch transition* with the
ETA-discounted weighted delta:

    credit = w(eta) * arrival_credit_weight *
             Σ_c channel_weight[c] · (curr_vec[c] - launch_vec[c])

where ``channel_weight`` mirrors the 12-vector consumed by
:func:`rl.rewards.compute_graph_native_reward` (8 base channels +
4 flux channels).  The credit is added via
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

from dataclasses import dataclass
from typing import Optional

import numpy as np


__all__ = [
    "ArrivalCreditSnapshot",
    "ArrivalCreditTracker",
    "build_channel_weights",
    "eta_weight",
]


# Length of the channel-weight vector used by arrival-credit.  Mirrors
# ``model_graph_pyg.GRAPH_NODE_METRIC_DIM_FLUX`` so the snapshot vector
# can be sliced/padded consistently regardless of flux being enabled.
NUM_METRIC_CHANNELS: int = 12


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
    """Build the 12-channel weight vector from a :class:`RewardConfig`.

    Order must mirror :func:`rl.rewards.compute_graph_native_reward`:

        [friendly_influence, enemy_influence, influence_balance,
         friendly_support, supportability, vulnerability,
         production_potential, strategic_potential,
         flux_friendly_in, flux_friendly_out,
         flux_enemy_in, flux_enemy_out]

    Returns a ``(NUM_METRIC_CHANNELS,)`` float32 array.  All-zero is a
    valid configuration (the caller treats it as "credit disabled").
    """
    return np.array(
        [
            float(getattr(cfg, "graph_friendly_influence_weight", 0.0)),
            float(getattr(cfg, "graph_enemy_influence_weight", 0.0)),
            float(getattr(cfg, "graph_influence_balance_weight", 0.0)),
            float(getattr(cfg, "graph_friendly_support_weight", 0.0)),
            float(getattr(cfg, "graph_supportability_weight", 0.0)),
            float(getattr(cfg, "graph_vulnerability_weight", 0.0)),
            float(getattr(cfg, "graph_production_potential_weight", 0.0)),
            float(getattr(cfg, "graph_strategic_potential_weight", 0.0)),
            float(getattr(cfg, "graph_flux_friendly_in_weight", 0.0)),
            float(getattr(cfg, "graph_flux_friendly_out_weight", 0.0)),
            float(getattr(cfg, "graph_flux_enemy_in_weight", 0.0)),
            float(getattr(cfg, "graph_flux_enemy_out_weight", 0.0)),
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
            source_planet_id=int(source_planet_id),
            source_vec=source_vec,
        )
        return True

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

        curr_target_vec = _align_to_channels(curr_metric_matrix[tgt_slot])
        delta = curr_target_vec - snap.target_vec
        weighted = float(np.dot(channel_weights, delta))

        # Optional source-side credit.  Source vector is captured at
        # launch time only when ``include_source`` was True at register.
        if snap.source_vec is not None:
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
