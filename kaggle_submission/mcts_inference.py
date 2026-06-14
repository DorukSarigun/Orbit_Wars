"""Orbit MCTS agent for Kaggle submission."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from genome_config import config_from_genome
from orbit_mcts import MCTSConfig
from orbit_mcts_agent import OrbitMCTSAgent
from orbit_scripted_agent import OrbitScriptedConfig

MAX_TURNS = 500
# Keep under Kaggle actTimeout (1s/turn) on slow workers.
MCTS_SIMULATIONS = int(os.environ.get("ORBIT_MCTS_SIMS", "12"))

_AGENT: Optional[OrbitMCTSAgent] = None


def _load_genome(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["genome"] if isinstance(raw, dict) and "genome" in raw else raw


def get_agent() -> OrbitMCTSAgent:
    global _AGENT
    if _AGENT is not None:
        return _AGENT

    genome_path = Path(__file__).with_name("genome.json")
    if genome_path.is_file():
        cfg = config_from_genome(_load_genome(genome_path))
    else:
        cfg = OrbitScriptedConfig()

    _AGENT = OrbitMCTSAgent(
        cfg=cfg,
        mcts_cfg=MCTSConfig(simulations=MCTS_SIMULATIONS),
    )
    return _AGENT


def act(env_obs: Any, player: int, *, current_turn: int) -> list[list]:
    return get_agent().act(
        env_obs, int(player), current_turn=int(current_turn), max_turns=MAX_TURNS,
    ) or []
