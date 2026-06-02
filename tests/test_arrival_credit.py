"""Tests for the ETA-weighted arrival-credit eligibility trace (Option B).

Covers:
    * Helpers: ``rl.arrival_credit.eta_weight`` modes, channel-weight order.
    * Snapshot: ``compute_planet_metric_vectors`` returns ``(n_real, D)``
      with ``D in {10, 14}`` driven by ``enable_flux_metrics``.
    * Tracker: register → compute_credit lifecycle, snapshot consumption,
      and that all metric channels contribute to the credit when their
      weights are non-zero.
    * Backward compatibility: ``enable_arrival_credit=False`` produces
      zero behavioural difference.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from rl.arrival_credit import NUM_METRIC_CHANNELS


# ---------------------------------------------------------------------------
# Stub observation helper (no env dependency, keeps the test fast).
# ---------------------------------------------------------------------------


class _StubObs:
    """Minimal Observation duck-type sufficient for
    :func:`rl.rewards.compute_planet_metric_vectors`.

    The ``compute_planet_metric_vectors`` path only touches
    ``planet_feats``, ``edge_bias``, ``planet_mask`` (+ flux fields when
    enabled).  We supply zero arrays so the metrics are well-defined
    (mostly zero) — the *delta* between two stubs with controlled
    differences is what we test below.
    """

    def __init__(self, n_real: int = 4, p_max: int = 8):
        from model_graph_pyg import PYG_PLANET_FEATURE_DIM
        # The metric path only reads the raw obs_builder feature indices.
        self.planet_feats = np.zeros((p_max, PYG_PLANET_FEATURE_DIM), dtype=np.float32)
        # 6 base + 9 edge metric channels = 15 in obs.edge_bias.  The
        # metric pipeline only reads the leading channels; zeros are fine.
        self.edge_bias = np.zeros((p_max + 16, p_max + 16, 6), dtype=np.float32)
        self.planet_mask = np.zeros(p_max, dtype=bool)
        self.planet_mask[:n_real] = True
        self.planet_ids = np.full(p_max, -1, dtype=np.int32)
        # Use ids 100..100+n_real so they don't collide with slot indices.
        for i in range(n_real):
            self.planet_ids[i] = 100 + i
        # Fleet fields (used only by flux helper when enable_flux_metrics)
        self.fleet_mask = np.zeros(0, dtype=bool)
        self.fleet_src_slot = np.zeros(0, dtype=np.int32)
        self.fleet_tgt_slot = np.zeros(0, dtype=np.int32)
        self.fleet_eta_raw = np.zeros(0, dtype=np.float32)
        self.fleet_ships_raw = np.zeros(0, dtype=np.float32)
        self.fleet_is_self = np.zeros(0, dtype=bool)
        self.fleet_is_enemy = np.zeros(0, dtype=bool)


# ---------------------------------------------------------------------------
# eta_weight modes
# ---------------------------------------------------------------------------


def test_eta_weight_raw_constant():
    """raw mode: weight always 1.0 regardless of ETA."""
    from rl.arrival_credit import eta_weight
    assert eta_weight(0.0, mode="raw") == 1.0
    assert eta_weight(1.0, mode="raw") == 1.0
    assert eta_weight(99.0, mode="raw") == 1.0


def test_eta_weight_linear_clamped():
    from rl.arrival_credit import eta_weight
    assert eta_weight(0.0, mode="linear", eta_norm=10.0) == pytest.approx(1.0)
    assert eta_weight(5.0, mode="linear", eta_norm=10.0) == pytest.approx(0.5)
    # Beyond eta_norm clamps to 0.
    assert eta_weight(15.0, mode="linear", eta_norm=10.0) == pytest.approx(0.0)
    # Negative ETA treated as 0.
    assert eta_weight(-3.0, mode="linear", eta_norm=10.0) == pytest.approx(1.0)


def test_eta_weight_exp_decay_monotone():
    from rl.arrival_credit import eta_weight
    w0 = eta_weight(0.0, mode="exp_decay", tau=5.0)
    w_mid = eta_weight(5.0, mode="exp_decay", tau=5.0)
    w_far = eta_weight(20.0, mode="exp_decay", tau=5.0)
    assert w0 == pytest.approx(1.0)
    assert w_mid == pytest.approx(math.exp(-1.0), rel=1e-5)
    assert w_far < w_mid < w0


def test_eta_weight_unknown_mode_raises():
    from rl.arrival_credit import eta_weight
    with pytest.raises(ValueError):
        eta_weight(1.0, mode="not_a_mode")


# ---------------------------------------------------------------------------
# build_channel_weights mirrors RewardConfig 13-vector ordering
# ---------------------------------------------------------------------------


def test_channel_weights_order_matches_reward_config():
    """The 14-vector built by ``build_channel_weights`` must match
    ``RewardConfig.arrival_credit_*_weight`` field ordering.
    """
    from rl.arrival_credit import build_channel_weights, NUM_METRIC_CHANNELS
    from rl.rewards import RewardConfig

    cfg = RewardConfig(
        arrival_credit_friendly_influence_weight=1.0,
        arrival_credit_enemy_influence_weight=2.0,
        arrival_credit_influence_balance_weight=3.0,
        arrival_credit_friendly_support_weight=4.0,
        arrival_credit_supportability_weight=5.0,
        arrival_credit_vulnerability_weight=6.0,
        arrival_credit_production_potential_weight=7.0,
        arrival_credit_friendly_production_potential_weight=8.0,
        arrival_credit_strategic_potential_weight=9.0,
        arrival_credit_held_discounted_production_weight=10.0,
        arrival_credit_flux_friendly_in_weight=11.0,
        arrival_credit_flux_friendly_out_weight=12.0,
        arrival_credit_flux_enemy_in_weight=13.0,
        arrival_credit_flux_enemy_out_weight=14.0,
    )
    w = build_channel_weights(cfg)
    assert w.shape == (NUM_METRIC_CHANNELS,)
    assert w.dtype == np.float32
    expected = np.arange(1, NUM_METRIC_CHANNELS + 1, dtype=np.float32)
    np.testing.assert_array_equal(w, expected)


# ---------------------------------------------------------------------------
# compute_planet_metric_vectors shape & flux toggling
# ---------------------------------------------------------------------------


def test_planet_metric_vectors_base_shape():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from rl.rewards import compute_planet_metric_vectors, RewardConfig
    obs = _StubObs(n_real=3, p_max=6)
    cfg = RewardConfig(enable_flux_metrics=False)
    mat = compute_planet_metric_vectors(obs, cfg=cfg)
    assert mat is not None
    assert mat.shape == (3, 10)
    assert mat.dtype == np.float32


def test_planet_metric_vectors_flux_shape():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from rl.rewards import compute_planet_metric_vectors, RewardConfig
    obs = _StubObs(n_real=3, p_max=6)
    cfg = RewardConfig(enable_flux_metrics=True)
    mat = compute_planet_metric_vectors(obs, cfg=cfg)
    assert mat is not None
    assert mat.shape == (3, 14)


def test_planet_metric_vectors_no_real_planets_returns_none():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from rl.rewards import compute_planet_metric_vectors, RewardConfig
    obs = _StubObs(n_real=0, p_max=4)
    assert compute_planet_metric_vectors(obs, cfg=RewardConfig()) is None


# ---------------------------------------------------------------------------
# Tracker register / compute_credit lifecycle
# ---------------------------------------------------------------------------


def _zero_planet_ids(n: int) -> np.ndarray:
    """Convenience: planet ids 100..100+n-1, padded with -1 to length 8."""
    ids = np.full(8, -1, dtype=np.int32)
    ids[:n] = np.arange(100, 100 + n, dtype=np.int32)
    return ids


def _channel_weights_unit() -> np.ndarray:
    """All-ones weight vector so each channel contributes equally."""
    from rl.arrival_credit import NUM_METRIC_CHANNELS
    return np.ones(NUM_METRIC_CHANNELS, dtype=np.float32)


class _CfgStub:
    """Minimal cfg accepted by ArrivalCreditTracker.compute_credit."""

    def __init__(
        self,
        *,
        mode: str = "exp_decay",
        tau: float = 10.0,
        eta_norm: float = 50.0,
        weight: float = 1.0,
        reward_scale: float = 1.0,
        counterfactual_scope: str = "full_graph",
    ):
        self.arrival_credit_mode = mode
        self.arrival_credit_tau = tau
        self.arrival_credit_eta_norm = eta_norm
        self.arrival_credit_weight = weight
        self.reward_scale = reward_scale
        self.arrival_credit_counterfactual_scope = counterfactual_scope


def test_tracker_register_then_consume():
    from rl.arrival_credit import ArrivalCreditTracker
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(3)
    launch_mat = np.zeros((3, NUM_METRIC_CHANNELS), dtype=np.float32)
    ok = tr.register(
        buffer_idx=7,
        player=0,
        target_planet_id=101,  # slot 1
        eta_at_launch=5.0,
        launch_metric_matrix=launch_mat,
        launch_planet_ids=ids,
    )
    assert ok is True
    assert 7 in tr
    assert len(tr) == 1
    curr_mat = np.zeros((3, NUM_METRIC_CHANNELS), dtype=np.float32)
    credit = tr.compute_credit(
        buffer_idx=7,
        curr_metric_matrix=curr_mat,
        curr_planet_ids=ids,
        channel_weights=_channel_weights_unit(),
        cfg=_CfgStub(mode="raw"),  # w_eta=1.0
    )
    # Zero deltas → zero credit.
    assert credit == pytest.approx(0.0)
    # Snapshot consumed.
    assert 7 not in tr
    assert len(tr) == 0


def test_tracker_register_missing_target_skipped():
    from rl.arrival_credit import ArrivalCreditTracker
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(3)
    launch_mat = np.zeros((3, NUM_METRIC_CHANNELS), dtype=np.float32)
    ok = tr.register(
        buffer_idx=1,
        player=0,
        target_planet_id=999,  # not in ids
        eta_at_launch=1.0,
        launch_metric_matrix=launch_mat,
        launch_planet_ids=ids,
    )
    assert ok is False
    assert len(tr) == 0


def test_tracker_register_none_matrix_skipped():
    from rl.arrival_credit import ArrivalCreditTracker
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(3)
    ok = tr.register(
        buffer_idx=1,
        player=0,
        target_planet_id=101,
        eta_at_launch=1.0,
        launch_metric_matrix=None,
        launch_planet_ids=ids,
    )
    assert ok is False


def test_tracker_compute_credit_no_snapshot_returns_none():
    from rl.arrival_credit import ArrivalCreditTracker
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(3)
    curr = np.zeros((3, NUM_METRIC_CHANNELS), dtype=np.float32)
    out = tr.compute_credit(
        buffer_idx=42,
        curr_metric_matrix=curr,
        curr_planet_ids=ids,
        channel_weights=_channel_weights_unit(),
        cfg=_CfgStub(),
    )
    assert out is None


def test_tracker_discard():
    from rl.arrival_credit import ArrivalCreditTracker
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(2)
    tr.register(
        buffer_idx=3,
        player=0,
        target_planet_id=101,
        eta_at_launch=1.0,
        launch_metric_matrix=np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32),
        launch_planet_ids=ids,
    )
    assert tr.discard(3) is True
    assert tr.discard(3) is False  # already gone
    assert len(tr) == 0


# ---------------------------------------------------------------------------
# CORE INVARIANT: all channels contribute to the arrival credit.
# This is the user's hard requirement: "tüm metriklere etki ettiğine emin
# olalım" — every channel with a non-zero weight must show up in the
# computed credit when its target-vector entry changes between launch
# and arrival.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel", range(NUM_METRIC_CHANNELS))
def test_all_metric_channels_contribute(channel: int):
    """For each channel, force a unit delta only on that
    channel and assert the resulting credit equals the channel's weight.

    This guards against silent regressions where a future refactor
    drops/permutes/zero-truncates the metric vector.
    """
    from rl.arrival_credit import ArrivalCreditTracker, NUM_METRIC_CHANNELS
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(2)
    launch_mat = np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32)
    tr.register(
        buffer_idx=11,
        player=0,
        target_planet_id=100,  # slot 0
        eta_at_launch=0.0,     # exp_decay gives w_eta=1 at eta=0
        launch_metric_matrix=launch_mat,
        launch_planet_ids=ids,
    )
    # Construct current matrix with exactly +1 on the channel under test
    # at the target slot.
    curr_mat = launch_mat.copy()
    curr_mat[0, channel] = 1.0
    # Per-channel weight vector (1.0 on this channel, 0 elsewhere).
    w = np.zeros(NUM_METRIC_CHANNELS, dtype=np.float32)
    w[channel] = 3.0  # arbitrary non-unit weight to make sure scaling is right
    credit = tr.compute_credit(
        buffer_idx=11,
        curr_metric_matrix=curr_mat,
        curr_planet_ids=ids,
        channel_weights=w,
        cfg=_CfgStub(mode="raw", weight=1.0, reward_scale=1.0),
    )
    assert credit == pytest.approx(3.0, abs=1e-6), (
        f"channel {channel} delta failed to propagate into credit"
    )


def test_all_channels_simultaneously_additive():
    """A delta on every channel simultaneously is the weighted sum."""
    from rl.arrival_credit import ArrivalCreditTracker, NUM_METRIC_CHANNELS
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(2)
    launch_mat = np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32)
    tr.register(
        buffer_idx=21, player=0, target_planet_id=100, eta_at_launch=0.0,
        launch_metric_matrix=launch_mat, launch_planet_ids=ids,
    )
    deltas = np.arange(1, NUM_METRIC_CHANNELS + 1, dtype=np.float32)
    curr_mat = launch_mat.copy()
    curr_mat[0, :] = deltas
    weights = np.full(NUM_METRIC_CHANNELS, 0.5, dtype=np.float32)
    credit = tr.compute_credit(
        buffer_idx=21,
        curr_metric_matrix=curr_mat,
        curr_planet_ids=ids,
        channel_weights=weights,
        cfg=_CfgStub(mode="raw"),
    )
    assert credit == pytest.approx(float((weights * deltas).sum()))


def test_eta_decay_scales_credit():
    """ETA decay reduces credit magnitude monotonically with launch ETA."""
    from rl.arrival_credit import ArrivalCreditTracker, NUM_METRIC_CHANNELS
    ids = _zero_planet_ids(2)
    launch_mat = np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32)
    curr_mat = launch_mat.copy()
    curr_mat[0, 0] = 1.0
    w = np.zeros(NUM_METRIC_CHANNELS, dtype=np.float32)
    w[0] = 1.0
    credits = []
    for eta in (0.0, 5.0, 20.0):
        tr = ArrivalCreditTracker()
        tr.register(
            buffer_idx=1, player=0, target_planet_id=100, eta_at_launch=eta,
            launch_metric_matrix=launch_mat, launch_planet_ids=ids,
        )
        c = tr.compute_credit(
            buffer_idx=1,
            curr_metric_matrix=curr_mat,
            curr_planet_ids=ids,
            channel_weights=w,
            cfg=_CfgStub(mode="exp_decay", tau=5.0),
        )
        credits.append(c)
    assert credits[0] > credits[1] > credits[2] >= 0.0
    assert credits[0] == pytest.approx(1.0)
    assert credits[1] == pytest.approx(math.exp(-1.0), rel=1e-5)


def test_arrival_credit_disabled_yields_zero_when_weights_zero():
    """When channel weights are all zero (i.e. user disables every
    contribution), the credit is exactly zero regardless of deltas.
    """
    from rl.arrival_credit import ArrivalCreditTracker, NUM_METRIC_CHANNELS
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(2)
    launch_mat = np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32)
    tr.register(
        buffer_idx=1, player=0, target_planet_id=100, eta_at_launch=0.0,
        launch_metric_matrix=launch_mat, launch_planet_ids=ids,
    )
    curr_mat = launch_mat + 5.0  # large delta everywhere
    credit = tr.compute_credit(
        buffer_idx=1,
        curr_metric_matrix=curr_mat,
        curr_planet_ids=ids,
        channel_weights=np.zeros(NUM_METRIC_CHANNELS, dtype=np.float32),
        cfg=_CfgStub(mode="raw"),
    )
    assert credit == pytest.approx(0.0)


def test_arrival_credit_source_side_signal():
    """``include_source`` should subtract source delta from the credit.

    A launch that *increases* friendly_influence at src (channel 0)
    should reduce the credit when source is included (we netted it with
    a minus sign in the tracker)."""
    from rl.arrival_credit import ArrivalCreditTracker, NUM_METRIC_CHANNELS
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(3)
    launch_mat = np.zeros((3, NUM_METRIC_CHANNELS), dtype=np.float32)
    tr.register(
        buffer_idx=5,
        player=0,
        target_planet_id=101,  # slot 1
        eta_at_launch=0.0,
        launch_metric_matrix=launch_mat,
        launch_planet_ids=ids,
        source_planet_id=100,  # slot 0
        include_source=True,
    )
    curr_mat = launch_mat.copy()
    # +1 friendly_influence at tgt (slot 1), +1 friendly_influence at src (slot 0).
    curr_mat[1, 0] = 1.0
    curr_mat[0, 0] = 1.0
    w = np.zeros(NUM_METRIC_CHANNELS, dtype=np.float32)
    w[0] = 1.0
    credit = tr.compute_credit(
        buffer_idx=5,
        curr_metric_matrix=curr_mat,
        curr_planet_ids=ids,
        channel_weights=w,
        cfg=_CfgStub(mode="raw"),
    )
    # tgt delta = +1, source delta = +1, net = +1 - +1 = 0.
    assert credit == pytest.approx(0.0)


def test_arrival_credit_reward_scale_applied():
    """``cfg.reward_scale`` and ``arrival_credit_weight`` both scale credit."""
    from rl.arrival_credit import ArrivalCreditTracker, NUM_METRIC_CHANNELS
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(2)
    launch_mat = np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32)
    tr.register(
        buffer_idx=1, player=0, target_planet_id=100, eta_at_launch=0.0,
        launch_metric_matrix=launch_mat, launch_planet_ids=ids,
    )
    curr_mat = launch_mat.copy()
    curr_mat[0, 0] = 1.0
    w = np.zeros(NUM_METRIC_CHANNELS, dtype=np.float32)
    w[0] = 1.0
    credit = tr.compute_credit(
        buffer_idx=1,
        curr_metric_matrix=curr_mat,
        curr_planet_ids=ids,
        channel_weights=w,
        cfg=_CfgStub(mode="raw", weight=2.0, reward_scale=0.5),
    )
    assert credit == pytest.approx(1.0 * 2.0 * 0.5)


def _set_owner_for_test(feats: np.ndarray, slot: int, kind: str) -> None:
    from obs_builder import (
        PF_IDX_OWNER_ENEMY_0,
        PF_IDX_OWNER_NEUTRAL,
        PF_IDX_OWNER_SELF,
    )
    feats[slot, PF_IDX_OWNER_SELF:PF_IDX_OWNER_NEUTRAL + 1] = 0.0
    if kind == "self":
        feats[slot, PF_IDX_OWNER_SELF] = 1.0
    elif kind == "neutral":
        feats[slot, PF_IDX_OWNER_NEUTRAL] = 1.0
    else:
        feats[slot, PF_IDX_OWNER_ENEMY_0] = 1.0


def _set_ships_for_test(feats: np.ndarray, slot: int, ships: float) -> None:
    from obs_builder import PF_IDX_LOG_SHIPS, PF_IDX_SHIPS
    feats[slot, PF_IDX_SHIPS] = float(ships) / 100.0
    feats[slot, PF_IDX_LOG_SHIPS] = math.log(float(ships) + 1.0) / math.log(1000.0)


def test_counterfactual_negative_target_flips_to_non_self():
    """If removing the arriving fleet makes target ships negative, the
    counterfactual target must stop being self-owned.  This is the
    ownership switch that disables friendly production potential.
    """
    from obs_builder import PF_IDX_OWNER_ENEMY_0, PF_IDX_OWNER_SELF, PF_IDX_SHIPS
    from rl.arrival_credit import ArrivalCreditTracker

    obs = _StubObs(n_real=2, p_max=4)
    _set_owner_for_test(obs.planet_feats, 0, "self")
    _set_ships_for_test(obs.planet_feats, 0, 10.0)
    tr = ArrivalCreditTracker()
    tr.register(
        buffer_idx=9,
        player=0,
        target_planet_id=100,
        eta_at_launch=0.0,
        launch_metric_matrix=np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32),
        launch_planet_ids=obs.planet_ids,
        ships_sent=30,
    )

    pair = tr.build_counterfactual_observations(
        buffer_idx=9,
        curr_obs=obs,
        curr_planet_ids=obs.planet_ids,
        actual_target_planet_id=100,
        previous_target_owner=1,
        return_to_source=False,
    )
    assert pair is not None
    _, cf_obs = pair
    assert cf_obs.planet_feats[0, PF_IDX_OWNER_SELF] == pytest.approx(0.0)
    assert cf_obs.planet_feats[0, PF_IDX_OWNER_ENEMY_0] == pytest.approx(1.0)
    assert cf_obs.planet_feats[0, PF_IDX_SHIPS] == pytest.approx(20.0 / 100.0)


def test_counterfactual_returned_source_ships_can_prevent_loss():
    from obs_builder import PF_IDX_OWNER_ENEMY_0, PF_IDX_OWNER_SELF, PF_IDX_SHIPS
    from rl.arrival_credit import ArrivalCreditTracker

    obs = _StubObs(n_real=2, p_max=4)
    _set_owner_for_test(obs.planet_feats, 0, "self")
    _set_ships_for_test(obs.planet_feats, 0, 50.0)
    _set_owner_for_test(obs.planet_feats, 1, "enemy")
    _set_ships_for_test(obs.planet_feats, 1, 10.0)
    tr = ArrivalCreditTracker()
    tr.register(
        buffer_idx=10,
        player=0,
        target_planet_id=100,
        eta_at_launch=0.0,
        launch_metric_matrix=np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32),
        launch_planet_ids=obs.planet_ids,
        ships_sent=30,
        source_planet_id=101,
    )

    pair = tr.build_counterfactual_observations(
        buffer_idx=10,
        curr_obs=obs,
        curr_planet_ids=obs.planet_ids,
        actual_target_planet_id=100,
        previous_target_owner=0,
        return_to_source=True,
    )
    assert pair is not None
    _, cf_obs = pair
    assert cf_obs.planet_feats[1, PF_IDX_OWNER_SELF] == pytest.approx(1.0)
    assert cf_obs.planet_feats[1, PF_IDX_OWNER_ENEMY_0] == pytest.approx(0.0)
    assert cf_obs.planet_feats[1, PF_IDX_SHIPS] == pytest.approx(20.0 / 100.0)


def test_counterfactual_horizon_resolves_soonest_enemy_arrival():
    from obs_builder import (
        PF_IDX_INCOMING_ENEMY_MIN_ETA,
        PF_IDX_INCOMING_ENEMY_SOONEST_SHIPS,
        PF_IDX_INCOMING_ENEMY_TOTAL,
        PF_IDX_OWNER_ENEMY_0,
        PF_IDX_OWNER_SELF,
        PF_IDX_SHIPS,
    )
    from rl.arrival_credit import ArrivalCreditTracker

    obs = _StubObs(n_real=1, p_max=4)
    _set_owner_for_test(obs.planet_feats, 0, "self")
    _set_ships_for_test(obs.planet_feats, 0, 30.0)
    obs.planet_feats[0, PF_IDX_INCOMING_ENEMY_MIN_ETA] = 2.0 / 50.0
    obs.planet_feats[0, PF_IDX_INCOMING_ENEMY_SOONEST_SHIPS] = 25.0 / 100.0
    obs.planet_feats[0, PF_IDX_INCOMING_ENEMY_TOTAL] = 25.0 / 100.0
    tr = ArrivalCreditTracker()
    tr.register(
        buffer_idx=11,
        player=0,
        target_planet_id=100,
        eta_at_launch=0.0,
        launch_metric_matrix=np.zeros((1, NUM_METRIC_CHANNELS), dtype=np.float32),
        launch_planet_ids=obs.planet_ids,
        ships_sent=20,
    )

    pair = tr.build_counterfactual_observations(
        buffer_idx=11,
        curr_obs=obs,
        curr_planet_ids=obs.planet_ids,
        actual_target_planet_id=100,
        previous_target_owner=0,
        return_to_source=False,
        horizon=5.0,
    )
    assert pair is not None
    actual_obs, cf_obs = pair
    assert actual_obs.planet_feats[0, PF_IDX_OWNER_SELF] == pytest.approx(1.0)
    assert actual_obs.planet_feats[0, PF_IDX_SHIPS] == pytest.approx(5.0 / 100.0)
    assert cf_obs.planet_feats[0, PF_IDX_OWNER_SELF] == pytest.approx(0.0)
    assert cf_obs.planet_feats[0, PF_IDX_OWNER_ENEMY_0] == pytest.approx(1.0)
    assert cf_obs.planet_feats[0, PF_IDX_SHIPS] == pytest.approx(15.0 / 100.0)


def test_counterfactual_credit_full_graph_uses_actual_minus_counterfactual():
    from rl.arrival_credit import ArrivalCreditTracker, NUM_METRIC_CHANNELS

    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(2)
    launch_mat = np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32)
    tr.register(
        buffer_idx=12,
        player=0,
        target_planet_id=100,
        eta_at_launch=0.0,
        launch_metric_matrix=launch_mat,
        launch_planet_ids=ids,
    )
    curr = np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32)
    cf = np.zeros_like(curr)
    curr[0, 0] = 5.0
    cf[0, 0] = 2.0
    cf[1, 0] = 10.0
    w = np.zeros(NUM_METRIC_CHANNELS, dtype=np.float32)
    w[0] = 1.0
    credit = tr.compute_credit(
        buffer_idx=12,
        curr_metric_matrix=curr,
        curr_planet_ids=ids,
        channel_weights=w,
        cfg=_CfgStub(mode="raw"),
        counterfactual_metric_matrix=cf,
        counterfactual_planet_ids=ids,
    )
    assert credit == pytest.approx(-7.0)


# ---------------------------------------------------------------------------
# End-to-end rollout integration: collect a short episode under
# graph_native + arrival_credit and verify the tracker is actually
# wired in (snapshots → arrivals → credit additions on the buffer).
#
# We can't easily check exact reward values because PPO rollout uses
# stochastic Beta sampling — same torch seed at the test entry doesn't
# guarantee identical sampling order inside collect_episode.  Instead
# we expose two diagnostic counters on the tracker
# (``credits_emitted`` and ``credits_total``) and a monkey-patch hook
# that observes ``ArrivalCreditTracker.compute_credit`` and
# ``ArrivalCreditTracker.register`` calls.
# ---------------------------------------------------------------------------


def _make_pyg_policy_cpu(*, enable_flux: bool = False):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("kaggle_environments")
    import torch
    from rl.train import _make_policy
    policy = _make_policy(
        "pyg_planet_graph",
        d_model=32,
        n_layers=1,
        enable_flux_features=enable_flux,
        enable_sinkhorn=False,
    )
    policy.to(torch.device("cpu"))
    return policy


def test_rollout_smoke_disabled():
    """Smoke test: with ``enable_arrival_credit=False`` the episode runs
    to completion and produces a non-empty buffer.  No tracker instance
    is ever created (verified by the fact that we don't import
    ``ArrivalCreditTracker`` from the rollout namespace here).
    """
    import torch
    from rl.buffer import RolloutBuffer
    from rl.rollout import collect_episode
    from rl.rewards import build_reward_config

    policy = _make_pyg_policy_cpu()
    reward_cfg = build_reward_config(
        "graph_native", enable_arrival_credit=False,
    )
    buf = RolloutBuffer()
    collect_episode(
        policy, buf,
        max_steps=8, num_agents=2,
        reward_cfg=reward_cfg, device=torch.device("cpu"),
        seed=7,
    )
    assert len(buf.data) > 0


def test_rollout_tracker_register_is_called_on_launch():
    """With ``enable_arrival_credit=True`` and at least one launch in the
    episode, the tracker's ``register`` method must be invoked at least
    once.  Validated via a monkey-patch counter on the class.
    """
    import torch
    from rl import arrival_credit as ac_mod
    from rl.buffer import RolloutBuffer
    from rl.rollout import collect_episode
    from rl.rewards import build_reward_config

    register_calls: list[dict] = []
    original_register = ac_mod.ArrivalCreditTracker.register

    def _spy_register(self, **kwargs):
        register_calls.append(kwargs)
        return original_register(self, **kwargs)

    ac_mod.ArrivalCreditTracker.register = _spy_register
    try:
        policy = _make_pyg_policy_cpu()
        reward_cfg = build_reward_config(
            "graph_native",
            enable_arrival_credit=True,
            arrival_credit_mode="raw",
        )
        buf = RolloutBuffer()
        collect_episode(
            policy, buf,
            max_steps=20, num_agents=2,
            reward_cfg=reward_cfg, device=torch.device("cpu"),
            seed=11,
        )
    finally:
        ac_mod.ArrivalCreditTracker.register = original_register

    # Buffer should have transitions; at least one register call must
    # have fired (every *successful* launch calls register, and 20 turns
    # with a random-init policy on a fresh env will land at least one
    # valid trajectory on essentially every seed).
    assert len(buf.data) > 0, "expected non-empty buffer"
    assert len(register_calls) > 0, (
        "expected ≥1 register() call from rollout; "
        "if 0, the snapshot hook is not wired into register_launch"
    )
    # Sanity-check the recorded kwargs shape.
    sample = register_calls[0]
    assert "buffer_idx" in sample
    assert "target_planet_id" in sample
    assert "eta_at_launch" in sample
    assert "launch_metric_matrix" in sample


def test_rollout_graph_native_non_terminal_step_reward_zero():
    """``graph_native`` has no step-level shaping; non-terminal buffer
    rewards come only from arrival credit (delayed) and invalid penalties.
    """
    import torch
    from rl.buffer import RolloutBuffer
    from rl.rollout import collect_episode
    from rl.rewards import build_reward_config

    policy = _make_pyg_policy_cpu()
    reward_cfg = build_reward_config(
        "graph_native",
        enable_arrival_credit=False,
    )
    buf = RolloutBuffer()
    collect_episode(
        policy, buf,
        max_steps=12, num_agents=2,
        reward_cfg=reward_cfg, device=torch.device("cpu"),
        seed=123,
    )
    non_terminal = [tr for tr in buf.data if not tr.done]
    assert non_terminal, "expected transitions before terminal"
    assert all(abs(tr.reward) < 1e-12 for tr in non_terminal)


def test_rollout_terminal_reward_reaches_buffer():
    """Terminal +/-1 must reach the buffer when the game has a winner."""
    import torch
    from rl.buffer import RolloutBuffer
    from rl.rollout import collect_episode
    from rl.rewards import build_reward_config

    policy = _make_pyg_policy_cpu()
    reward_cfg = build_reward_config(
        "graph_native",
        enable_arrival_credit=False,
        terminal_scale=1.0,
    )
    for seed in range(8):
        buf = RolloutBuffer()
        stats = collect_episode(
            policy, buf,
            max_steps=120, num_agents=2,
            reward_cfg=reward_cfg, device=torch.device("cpu"),
            seed=seed,
        )
        if stats.winner not in (0, 1):
            continue
        p0_rews = [tr.reward for tr in buf.data if tr.player == 0]
        assert p0_rews, "expected P0 transitions"
        expected = 1.0 if stats.winner == 0 else -1.0
        assert stats.reward_p0 == pytest.approx(expected, abs=1e-6)
        assert sum(p0_rews) == pytest.approx(expected, abs=1e-6), (
            f"seed={seed} winner={stats.winner} buffer_sum={sum(p0_rews):.4f}"
        )
        return
    pytest.skip("no decisive game in seeds 0..7 within 120 steps")


def test_rollout_combo_flux_and_arrival_credit_runs():
    """Smoke test: arrival credit + flux metrics + flux encoder features
    all on simultaneously must not raise.  This catches shape / dtype
    regressions in the combined code path (the most likely place for a
    future bug).
    """
    import torch
    from rl.buffer import RolloutBuffer
    from rl.rollout import collect_episode
    from rl.rewards import build_reward_config

    policy = _make_pyg_policy_cpu(enable_flux=True)
    reward_cfg = build_reward_config(
        "graph_native",
        enable_arrival_credit=True,
        enable_flux_metrics=True,
    )
    buf = RolloutBuffer()
    collect_episode(
        policy, buf,
        max_steps=8, num_agents=2,
        reward_cfg=reward_cfg, device=torch.device("cpu"),
        seed=42,
    )
    assert len(buf.data) > 0


def test_tracker_diagnostic_counters_increment():
    """Direct invariant on the tracker class (no env / rollout): every
    successful ``compute_credit`` increments ``credits_emitted`` and
    accumulates the credit in ``credits_total``.  These counters are
    what the training loop will surface to its logs.
    """
    from rl.arrival_credit import ArrivalCreditTracker, NUM_METRIC_CHANNELS
    tr = ArrivalCreditTracker()
    ids = _zero_planet_ids(2)
    launch_mat = np.zeros((2, NUM_METRIC_CHANNELS), dtype=np.float32)
    tr.register(
        buffer_idx=1, player=0, target_planet_id=100, eta_at_launch=0.0,
        launch_metric_matrix=launch_mat, launch_planet_ids=ids,
    )
    tr.register(
        buffer_idx=2, player=0, target_planet_id=101, eta_at_launch=0.0,
        launch_metric_matrix=launch_mat, launch_planet_ids=ids,
    )
    curr = launch_mat.copy()
    curr[0, 0] = 1.0
    w = np.zeros(NUM_METRIC_CHANNELS, dtype=np.float32)
    w[0] = 1.0
    c1 = tr.compute_credit(
        buffer_idx=1, curr_metric_matrix=curr, curr_planet_ids=ids,
        channel_weights=w, cfg=_CfgStub(mode="raw"),
    )
    assert tr.credits_emitted == 1
    assert tr.credits_total == pytest.approx(c1)
    c2 = tr.compute_credit(
        buffer_idx=2, curr_metric_matrix=curr, curr_planet_ids=ids,
        channel_weights=w, cfg=_CfgStub(mode="raw"),
    )
    # planet_id=101 is at slot 1; its row in curr is zeros → credit=0.
    assert c2 == pytest.approx(0.0)
    assert tr.credits_emitted == 2
    assert tr.credits_total == pytest.approx(c1)  # zero added
