"""Unit tests for the scripted comet-evacuation safety net.

A comet runs off its precomputed path after ``len(path) - 1 - path_index`` turns
and is destroyed together with every ship garrisoned on it.  The evacuation
helper rescues those ships by launching the remaining garrison to the nearest
friendly planet that will outlive the evacuation window.  These tests exercise
the pure helper (no env) with hand-built planets placed away from the sun.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_extraction import Planet  # noqa: E402
from reworked_allocator import Launch  # noqa: E402
from reworked_rollout import (  # noqa: E402
    _comet_remaining_life,
    _comet_evacuation_launches,
)


def _comet(pid, owner, x, y, ships, path_len, path_index):
    p = Planet(id=pid, owner=owner, x=float(x), y=float(y), radius=1.0,
               ships=int(ships), production=1)
    p.is_comet = True
    p.comet_path = [[float(x), float(y)] for _ in range(path_len)]
    p.comet_path_index = path_index
    return p


def _static(pid, owner, x, y, ships=10):
    p = Planet(id=pid, owner=owner, x=float(x), y=float(y), radius=1.0,
               ships=int(ships), production=1)
    p.is_comet = False
    return p


def test_remaining_life_math():
    # path of 5 points, currently at index 3 -> 5-1-3 = 1 turn left.
    c = _comet(7, 0, 80, 80, 20, path_len=5, path_index=3)
    assert _comet_remaining_life(c) == 1
    # Non-comets report -1 ("not an expiring comet"), so they never trigger an
    # evacuation and are always valid evacuation destinations.
    assert _comet_remaining_life(_static(2, 0, 85, 80)) == -1


def test_expiring_comet_is_evacuated_to_nearest_friendly():
    comet = _comet(7, 0, 80, 80, 20, path_len=5, path_index=3)   # 1 turn left
    near = _static(2, 0, 85, 80)                                  # closest friendly
    far = _static(3, 0, 20, 20)
    planets = [comet, near, far]
    ids = np.array([7, 2, 3])
    out = _comet_evacuation_launches(
        player=0, parsed_planets=planets, launches=[], ids=ids,
        angular_velocity=0.0, max_speed=2.0, evac_turns=2)
    assert len(out) == 1, out
    src_id, angle, ships = out[0]
    assert src_id == 7
    assert ships == 20            # full garrison rescued


def test_healthy_comet_not_evacuated():
    comet = _comet(7, 0, 80, 80, 20, path_len=50, path_index=3)  # ~46 turns left
    near = _static(2, 0, 85, 80)
    out = _comet_evacuation_launches(
        player=0, parsed_planets=[comet, near], launches=[], ids=np.array([7, 2]),
        angular_velocity=0.0, max_speed=2.0, evac_turns=2)
    assert out == [], out


def test_committed_ships_are_subtracted():
    # Allocator already launched 15 of the comet's 20 ships this turn (slot 0).
    comet = _comet(7, 0, 80, 80, 20, path_len=5, path_index=3)
    near = _static(2, 0, 85, 80)
    launches = [Launch(src=0, tgt=1, ships=15, kind="attack")]
    out = _comet_evacuation_launches(
        player=0, parsed_planets=[comet, near], launches=launches,
        ids=np.array([7, 2]), angular_velocity=0.0, max_speed=2.0, evac_turns=2)
    assert len(out) == 1, out
    assert out[0][2] == 5         # only the 5 remaining ships are rescued


def test_no_safe_destination_skips():
    # The only other friendly planet is itself an expiring comet -> no evac.
    comet = _comet(7, 0, 80, 80, 20, path_len=5, path_index=3)
    doomed = _comet(8, 0, 85, 80, 5, path_len=5, path_index=3)
    out = _comet_evacuation_launches(
        player=0, parsed_planets=[comet, doomed], launches=[], ids=np.array([7, 8]),
        angular_velocity=0.0, max_speed=2.0, evac_turns=2)
    assert out == [], out


def test_disabled_when_evac_turns_zero():
    comet = _comet(7, 0, 80, 80, 20, path_len=5, path_index=3)
    near = _static(2, 0, 85, 80)
    out = _comet_evacuation_launches(
        player=0, parsed_planets=[comet, near], launches=[], ids=np.array([7, 2]),
        angular_velocity=0.0, max_speed=2.0, evac_turns=0)
    assert out == [], out


if __name__ == "__main__":
    test_remaining_life_math()
    test_expiring_comet_is_evacuated_to_nearest_friendly()
    test_healthy_comet_not_evacuated()
    test_committed_ships_are_subtracted()
    test_no_safe_destination_skips()
    test_disabled_when_evac_turns_zero()
    print("all comet evacuation tests passed")
