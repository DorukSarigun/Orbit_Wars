"""Orbit scripted agent with time-aware capture MCTS attack bonuses."""

from __future__ import annotations

from typing import Any, Optional

from orbit_defense import PlanetDefenseState
from orbit_graph import OrbitGraph
from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig
from orbit_timed_mcts import TimedCaptureMCTS, TimedMCTSConfig


class OrbitTimedMCTSAgent(OrbitScriptedAgent):
    """Base orbit scripted phases + timed MCTS for attack target ordering."""

    def __init__(
        self,
        cfg: OrbitScriptedConfig | None = None,
        timed_mcts_cfg: TimedMCTSConfig | None = None,
    ):
        super().__init__(cfg=cfg)
        self.timed_mcts_cfg = timed_mcts_cfg or TimedMCTSConfig()

    def _attack_priority_bonus(
        self,
        og: OrbitGraph,
        plan: Any,
        defense_states: list[PlanetDefenseState],
        *,
        omega: float,
        current_turn: int,
        max_turns: int,
    ) -> Optional[dict[int, float]]:
        del plan, defense_states, omega
        if (
            int(self.timed_mcts_cfg.simulations) <= 0
            or float(self.timed_mcts_cfg.bonus_weight) == 0.0
        ):
            return None
        mcts = TimedCaptureMCTS(
            og,
            self.timed_mcts_cfg,
            root_player=int(og.player),
            current_turn=int(current_turn),
            max_turns=int(max_turns),
        )
        bonus = mcts.search()
        return bonus or None


__all__ = ["OrbitTimedMCTSAgent", "TimedMCTSConfig"]
