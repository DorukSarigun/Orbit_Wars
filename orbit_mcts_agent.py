"""Orbit scripted agent + Monte-Carlo capture planning.

:class:`OrbitMCTSAgent` subclasses :class:`orbit_scripted_agent.OrbitScriptedAgent`
and only overrides ``_attack_priority_bonus`` using :mod:`orbit_mcts`.  All other
phases (evacuation, defense triage, allocation, support, consolidation) are
identical to the base agent.
"""

from __future__ import annotations

from typing import Any, Optional

from orbit_defense import PlanetDefenseState
from orbit_graph import OrbitGraph
from orbit_mcts import CaptureMCTS, MCTSConfig
from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig


class OrbitMCTSAgent(OrbitScriptedAgent):
    """:class:`OrbitScriptedAgent` + abstract capture MCTS for attack ordering.

    Set ``mcts_cfg.simulations = 0`` or ``bonus_weight = 0`` to recover exact
    base-agent attack ranking.
    """

    def __init__(
        self,
        cfg: OrbitScriptedConfig | None = None,
        mcts_cfg: MCTSConfig | None = None,
    ):
        super().__init__(cfg=cfg)
        self.mcts_cfg = mcts_cfg or MCTSConfig()

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
        del plan, defense_states, omega, current_turn, max_turns
        if int(self.mcts_cfg.simulations) <= 0 or float(self.mcts_cfg.bonus_weight) == 0.0:
            return None
        mcts = CaptureMCTS(og, self.mcts_cfg, root_player=int(og.player))
        bonus = mcts.search()
        return bonus or None


__all__ = ["OrbitMCTSAgent", "MCTSConfig"]
