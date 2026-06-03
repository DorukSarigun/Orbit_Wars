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


def test_failed_neutral_attack_penalizes_only_when_enabled():
    # g0 mine, g5 neutral (garrison 30, prod 3). A 10-ship attack lands at t1,
    # is absorbed, g5 stays neutral the whole game -> pure waste.
    def _log():
        log = EpisodeLog(max_turns=100, policy_players={0})
        log.snapshots = [
            _snap(0, {0: 0, 5: -1}, {0: 20, 5: 30}, {0: 0, 5: 3}, []),
            _snap(1, {0: 0, 5: -1}, {0: 10, 5: 30}, {0: 0, 5: 3}, [100]),
            _snap(2, {0: 0, 5: -1}, {0: 10, 5: 20}, {0: 0, 5: 3}, []),
        ]
        log.launches = [LaunchRecord(decision_id=0, player=0, launch_turn=0,
                                     src_id=0, tgt_id=5, ships=10, is_attack=True,
                                     fleet_id=100)]
        return log

    # Default config: no campaign penalties -> the failed attack costs nothing.
    credit, stats = attribute_credits(_log(), CreditConfig())
    assert abs(credit.get(0, 0.0)) < 1e-9, credit
    assert stats.n_fail_attack == 0

    # Enabled: the deciding turn is charged -dpp(target).
    credit, stats = attribute_credits(_log(), CreditConfig(fail_weight=0.5))
    assert credit.get(0, 0.0) < 0.0, credit
    assert stats.n_fail_attack == 1
    assert abs(credit[0] - (-0.5 * _dpp(3, 98))) < 1e-4, credit


def test_lost_race_penalizes_when_rival_captures():
    # g5 neutral; my 10-ship attack lands at t1 but the planet flips to enemy
    # player 1 (not a policy player) that turn -> I lost the race.
    def _log():
        log = EpisodeLog(max_turns=100, policy_players={0})
        log.snapshots = [
            _snap(0, {0: 0, 5: -1}, {0: 20, 5: 5}, {0: 0, 5: 3}, []),
            _snap(1, {0: 0, 5: -1}, {0: 10, 5: 5}, {0: 0, 5: 3}, [100]),
            _snap(2, {0: 0, 5: 1}, {0: 10, 5: 8}, {0: 0, 5: 3}, []),
        ]
        log.launches = [LaunchRecord(decision_id=0, player=0, launch_turn=0,
                                     src_id=0, tgt_id=5, ships=10, is_attack=True,
                                     fleet_id=100)]
        return log

    credit, stats = attribute_credits(_log(), CreditConfig())
    assert abs(credit.get(0, 0.0)) < 1e-9, credit
    assert stats.n_lost_race == 0

    credit, stats = attribute_credits(_log(), CreditConfig(lost_race_weight=1.0))
    assert credit.get(0, 0.0) < 0.0, credit
    assert stats.n_lost_race == 1


def test_multiwave_capture_does_not_penalize_softening_wave():
    # Staggered assault on neutral g5 (garrison 30): wave A (decision 0, 18
    # ships) lands t1 and only softens (12 left); wave B (decision 1, 18 ships)
    # lands t2 and captures.  Even with both penalties on, the softening wave
    # earns NO penalty (its campaign succeeded), only B gets +capture.
    log = EpisodeLog(max_turns=100, policy_players={0})
    log.snapshots = [
        _snap(0, {0: 0, 5: -1}, {0: 50, 5: 30}, {0: 0, 5: 3}, []),
        _snap(1, {0: 0, 5: -1}, {0: 32, 5: 30}, {0: 0, 5: 3}, [100]),
        _snap(2, {0: 0, 5: -1}, {0: 14, 5: 12}, {0: 0, 5: 3}, [200]),
        _snap(3, {0: 0, 5: 0}, {0: 14, 5: 6}, {0: 0, 5: 3}, []),
    ]
    log.launches = [
        LaunchRecord(decision_id=0, player=0, launch_turn=0, src_id=0, tgt_id=5,
                     ships=18, is_attack=True, fleet_id=100),
        LaunchRecord(decision_id=1, player=0, launch_turn=1, src_id=0, tgt_id=5,
                     ships=18, is_attack=True, fleet_id=200),
    ]
    cfg = CreditConfig(fail_weight=0.5, lost_race_weight=1.0)
    credit, stats = attribute_credits(log, cfg)
    assert stats.n_capture == 1
    assert stats.n_fail_attack == 0 and stats.n_lost_race == 0
    assert credit.get(1, 0.0) > 0.0, credit          # capturing wave rewarded
    assert abs(credit.get(0, 0.0)) < 1e-9, credit     # softening wave untouched


if __name__ == "__main__":
    test_capture_credits_attacker_positive()
    test_defense_loss_charges_reinforcer_negative()
    test_small_attack_absorbed_by_native_credits_nothing()
    test_big_attack_held_credits_reinforcer_positive()
    test_failed_neutral_attack_penalizes_only_when_enabled()
    test_lost_race_penalizes_when_rival_captures()
    test_multiwave_capture_does_not_penalize_softening_wave()
    print("all credit tests passed")
