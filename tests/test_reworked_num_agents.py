"""Smoke tests for 2- and 4-player reworked rollout wiring."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reworked_rollout import (  # noqa: E402
    ALLOWED_NUM_AGENTS,
    _validate_num_agents,
    _opponent_controller,
)


def test_validate_num_agents():
    assert _validate_num_agents(2) == 2
    assert _validate_num_agents(4) == 4
    try:
        _validate_num_agents(3)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_opponent_controller_4p():
    assert _opponent_controller(0, "scripted", 4) is None
    assert _opponent_controller(1, "scripted", 4) == "scripted"
    assert _opponent_controller(2, "scripted", 4) == "scripted"
    assert _opponent_controller(3, "scripted", 4) == "scripted"
    assert _opponent_controller(1, "frozen", 4) == "frozen"
    assert _opponent_controller(2, "frozen", 4) == "scripted"
    assert _opponent_controller(0, "self_play", 4) is None
    assert _opponent_controller(2, "self_play", 4) is None


def test_allowed_tuple():
    assert ALLOWED_NUM_AGENTS == (2, 4)


if __name__ == "__main__":
    test_validate_num_agents()
    test_opponent_controller_4p()
    test_allowed_tuple()
    print("all num_agents tests passed")
