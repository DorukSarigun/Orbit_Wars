"""Kaggle Orbit Wars — GA-tuned OrbitMCTSAgent."""

from __future__ import annotations

import os
import sys

# Kaggle's agent loader appends this file's directory to ``sys.path`` only while
# it ``exec``s this module, then pops it immediately. Any import deferred to
# act-time (e.g. ``genome_config`` or ``orbit_production_pressure``, which is
# imported lazily inside ``build_orbit_graph``) would otherwise raise
# ModuleNotFoundError during the validation episode and fail the submission.
# While the directory is still on the path during this exec, re-insert it at the
# front so it persists for every later (including lazy) sibling import.
for _p in list(sys.path):
    if _p and os.path.isfile(os.path.join(_p, "mcts_inference.py")):
        if _p not in sys.path[:1]:
            sys.path.insert(0, _p)
        break

from mcts_inference import act as _act


def agent(obs):
    player = int(obs.get("player", 0) if isinstance(obs, dict) else obs.player)
    turn = int(
        (obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)) or 0
    )
    return _act(obs, player, current_turn=turn)
