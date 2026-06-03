"""Unit tests for the FIFO inventory-accounting credit pass.

These build synthetic :class:`EpisodeLog`s (no env) so the attribution logic can
be checked turn-by-turn:

* a successful capture credits the attacking decision ``+dpp(target)``;
* losing a defended planet charges the reinforcing decision ``-dpp``;
* a small attack the native garrison absorbs alone credits nothing (the FIFO
  "pivotal" property -- the reinforcement was not needed).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reworked_credit import (  # noqa: E402
    EpisodeLog, LaunchRecord, TurnSnapshot, CreditConfig, attribute_credits,
)
from reworked_features import discounted_production_potential  # noqa: E402


def _snap(turn, owner, ships, prod, fleets):
    return TurnSnapshot(turn=turn, owner=dict(owner), ships=dict(ships),
                        production=dict(prod), fleet_ids=set(fleets))


def _dpp(prod, R):
    return float(discounted_production_potential(
        np.asarray([float(prod)], dtype=np.float64), R)[0])


def test_capture_credits_attacker_positive():
    # g0 mine, g1 enemy (prod 3). Attack 9 ships from g0 at t0; fleet 100
    # resolves at t1 and captures g1 at t2.
    log = EpisodeLog(max_turns=100, policy_players={0})
    log.snapshots = [
        _snap(0, {0: 0, 1: 1}, {0: 10, 1: 2}, {0: 0, 1: 3}, []),
        _snap(1, {0: 0, 1: 1}, {0: 1, 1: 5}, {0: 0, 1: 3}, [100]),
        _snap(2, {0: 0, 1: 0}, {0: 1, 1: 1}, {0: 0, 1: 3}, []),
    ]
    log.launches = [LaunchRecord(decision_id=0, player=0, launch_turn=0,
                                 src_id=0, tgt_id=1, ships=9, is_attack=True,
                                 fleet_id=100)]
    credit, stats = attribute_credits(log, CreditConfig())
    assert stats.n_capture == 1
    # dpp(g1) measured at the resolution turn (t1, R = 99)
    expected = _dpp(3, 99)
    assert abs(credit.get(0, 0.0) - expected) < 1e-4, (credit, expected)


def test_defense_loss_charges_reinforcer_negative():
    # g0 mine (prod 2), reinforced +5 from gS at t0 (fleet 100, lands t1->t2),
    # then a 20-ship enemy attack captures g0 at t3.
    log = EpisodeLog(max_turns=100, policy_players={0})
    log.snapshots = [
        _snap(0, {0: 0, 9: 0}, {0: 3, 9: 10}, {0: 2, 9: 0}, []),
        _snap(1, {0: 0, 9: 0}, {0: 5, 9: 5}, {0: 2, 9: 0}, [100]),
        _snap(2, {0: 0, 9: 0}, {0: 12, 9: 5}, {0: 2, 9: 0}, []),
        _snap(3, {0: 1, 9: 0}, {0: 6, 9: 5}, {0: 2, 9: 0}, []),
    ]
    log.launches = [LaunchRecord(decision_id=0, player=0, launch_turn=0,
                                 src_id=9, tgt_id=0, ships=5, is_attack=False,
                                 fleet_id=100)]
    credit, stats = attribute_credits(log, CreditConfig())
    assert stats.n_defense_loss == 1
    assert credit.get(0, 0.0) < 0.0, credit


def test_small_attack_absorbed_by_native_credits_nothing():
    # Reinforcement +5 lands, then a 4-ship attack the native garrison absorbs
    # alone -> FIFO consumes only NATIVE -> the reinforcement earns nothing.
    log = EpisodeLog(max_turns=100, policy_players={0})
    log.snapshots = [
        _snap(0, {0: 0, 9: 0}, {0: 10, 9: 10}, {0: 0, 9: 0}, []),
        _snap(1, {0: 0, 9: 0}, {0: 10, 9: 5}, {0: 0, 9: 0}, [100]),
        _snap(2, {0: 0, 9: 0}, {0: 15, 9: 5}, {0: 0, 9: 0}, []),
        _snap(3, {0: 0, 9: 0}, {0: 11, 9: 5}, {0: 0, 9: 0}, []),
    ]
    log.launches = [LaunchRecord(decision_id=0, player=0, launch_turn=0,
                                 src_id=9, tgt_id=0, ships=5, is_attack=False,
                                 fleet_id=100)]
    credit, _ = attribute_credits(log, CreditConfig())
    assert abs(credit.get(0, 0.0)) < 1e-9, credit


def test_big_attack_held_credits_reinforcer_positive():
    # g0 (prod 3) grows 10->13->(+5 reinf)21->24; an 18-ship attack eats
    # through all native ahead of the reinforcement (16) into its layer (2),
    # so the reinforcement was pivotal -> positive credit; planet holds at 6.
    log = EpisodeLog(max_turns=100, policy_players={0})
    log.snapshots = [
        _snap(0, {0: 0, 9: 0}, {0: 10, 9: 10}, {0: 3, 9: 0}, []),
        _snap(1, {0: 0, 9: 0}, {0: 13, 9: 5}, {0: 3, 9: 0}, [100]),
        _snap(2, {0: 0, 9: 0}, {0: 21, 9: 5}, {0: 3, 9: 0}, []),
        _snap(3, {0: 0, 9: 0}, {0: 6, 9: 5}, {0: 3, 9: 0}, []),
    ]
    log.launches = [LaunchRecord(decision_id=0, player=0, launch_turn=0,
                                 src_id=9, tgt_id=0, ships=5, is_attack=False,
                                 fleet_id=100)]
    credit, stats = attribute_credits(log, CreditConfig())
    assert stats.n_defense_hold == 1
    assert credit.get(0, 0.0) > 0.0, credit


if __name__ == "__main__":
    test_capture_credits_attacker_positive()
    test_defense_loss_charges_reinforcer_negative()
    test_small_attack_absorbed_by_native_credits_nothing()
    test_big_attack_held_credits_reinforcer_positive()
    print("all credit tests passed")
