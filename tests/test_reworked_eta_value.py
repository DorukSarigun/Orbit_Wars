"""Unit tests for the ETA-aware capture-value node channel.

Covers both the standalone metric (:func:`eta_aware_capture_value`) and its
opt-in wiring into :func:`build_reworked_node_features`, including the
backward-compatibility guarantee that the default layout is byte-for-byte
unchanged (no extra channel unless explicitly requested).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reworked_features import eta_aware_capture_value  # noqa: E402
from reworked_pipeline import (  # noqa: E402
    BoardState, build_reworked_node_features,
)
from reworked_model import REWORKED_NODE_FEATURE_DIM  # noqa: E402


def _board(*, owner, ships, production, dist, remaining_turns=100):
    n = len(owner)
    return BoardState(
        n=n,
        player=0,
        owner=np.asarray(owner, dtype=np.int64),
        ships=np.asarray(ships, dtype=np.float32),
        production=np.asarray(production, dtype=np.float32),
        radius=np.ones(n, dtype=np.float32),
        is_rotating=np.zeros(n, dtype=np.float32),
        is_comet=np.zeros(n, dtype=np.float32),
        distance_matrix=np.asarray(dist, dtype=np.float32),
        crosses_sun=np.zeros((n, n), dtype=np.float32),
        remaining_turns=int(remaining_turns),
    )


def test_nearby_target_outranks_far_target_of_equal_value():
    # Planet 0 = mine (source). Targets 1 and 2: equal production, but 1 is
    # nearer to the source -> arrival discount should make it more valuable.
    owner = [0, -1, -1]
    ships = [80.0, 10.0, 10.0]
    production = [1.0, 5.0, 5.0]
    dist = [
        [0.0, 0.1, 0.9],
        [0.1, 0.0, 0.9],
        [0.9, 0.9, 0.0],
    ]
    rate = eta_aware_capture_value(
        np.asarray(dist), np.asarray(owner), np.asarray(ships),
        np.asarray(production), player=0, remaining_turns=100)

    assert rate.shape == (3,)
    assert rate[0] == 0.0           # owned -> not a capture opportunity
    assert rate[1] > rate[2] > 0.0  # nearer equal-value target ranks higher


def test_cheaper_target_outranks_expensive_target_of_equal_value():
    # Equal production & distance, but target 2 has a much bigger garrison ->
    # higher demand -> lower cost-normalised value.
    owner = [0, -1, -1]
    ships = [200.0, 10.0, 120.0]
    production = [1.0, 5.0, 5.0]
    dist = [
        [0.0, 0.3, 0.3],
        [0.3, 0.0, 0.3],
        [0.3, 0.3, 0.0],
    ]
    rate = eta_aware_capture_value(
        np.asarray(dist), np.asarray(owner), np.asarray(ships),
        np.asarray(production), player=0, remaining_turns=100)
    assert rate[1] > rate[2] > 0.0


def test_no_own_source_gives_zero():
    # No planet owned by player 0 -> nothing reachable -> all zeros.
    owner = [-1, -1, 1]
    ships = [10.0, 10.0, 10.0]
    production = [5.0, 5.0, 5.0]
    dist = [
        [0.0, 0.2, 0.4],
        [0.2, 0.0, 0.4],
        [0.4, 0.4, 0.0],
    ]
    rate = eta_aware_capture_value(
        np.asarray(dist), np.asarray(owner), np.asarray(ships),
        np.asarray(production), player=0, remaining_turns=100)
    assert np.allclose(rate, 0.0)


def test_node_features_default_layout_unchanged():
    state = _board(
        owner=[0, -1, 1],
        ships=[50.0, 10.0, 20.0],
        production=[2.0, 5.0, 3.0],
        dist=[[0.0, 0.2, 0.5], [0.2, 0.0, 0.4], [0.5, 0.4, 0.0]],
    )
    node = build_reworked_node_features(state)
    assert node.shape == (3, REWORKED_NODE_FEATURE_DIM)


def test_value_rate_channel_is_opt_in_and_appended():
    state = _board(
        owner=[0, -1, 1],
        ships=[50.0, 10.0, 20.0],
        production=[2.0, 5.0, 3.0],
        dist=[[0.0, 0.2, 0.5], [0.2, 0.0, 0.4], [0.5, 0.4, 0.0]],
    )
    base = build_reworked_node_features(state)
    augmented = build_reworked_node_features(state, include_value_rate=True)

    # Exactly one extra channel, appended at the end; legacy channels identical.
    assert augmented.shape == (3, REWORKED_NODE_FEATURE_DIM + 1)
    assert np.array_equal(augmented[:, :REWORKED_NODE_FEATURE_DIM], base)

    # The new channel matches the standalone metric.
    expected = eta_aware_capture_value(
        state.distance_matrix, state.owner, state.ships, state.production,
        player=state.player, remaining_turns=state.remaining_turns)
    assert np.allclose(augmented[:, -1], expected)


if __name__ == "__main__":
    test_nearby_target_outranks_far_target_of_equal_value()
    test_cheaper_target_outranks_expensive_target_of_equal_value()
    test_no_own_source_gives_zero()
    test_node_features_default_layout_unchanged()
    test_value_rate_channel_is_opt_in_and_appended()
    print("ok")
