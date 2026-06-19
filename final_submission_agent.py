"""Kaggle-style entrypoint for the deterministic final micro agent.

This module is intentionally checkpoint-free.  It can be copied into the
submission bundle together with its dependencies when we decide to ship the
micro-planner agent.
"""

from __future__ import annotations

from final_agent import FinalMicroAgent


_AGENT: FinalMicroAgent | None = None


def get_agent() -> FinalMicroAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = FinalMicroAgent()
    return _AGENT


def agent(obs, config=None):
    step = 0
    if isinstance(obs, dict):
        step = int(obs.get("step", 0) or 0)
        player = int(obs.get("player", 0) or 0)
    else:
        player = int(getattr(obs, "player", 0) or 0)
    max_turns = int(getattr(config, "episodeSteps", 500) or 500) if config is not None else 500
    return get_agent().act(obs, player, current_turn=step, max_turns=max_turns)

