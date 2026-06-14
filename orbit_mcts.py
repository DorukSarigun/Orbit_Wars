"""Lightweight Monte-Carlo planning for capture-target prioritisation.

Full ``env.step`` rollouts are far too slow for Orbit Wars (~seconds per turn).
Instead we run **abstract capture MCTS** on the orbit graph:

* **State** = who owns each planet (comets excluded); routes are fixed geometry.
* **Action** = capture one currently reachable non-owned planet (instant flip).
* **Rollout** = alternate self / enemy greedy-random captures for ``rollout_depth``
  plies, then score by production-share margin.

The search returns a per-target ``priority_bonus`` fed into the base agent's
attack ranking — identical integration to :class:`orbit_lookahead_agent.OrbitLookaheadAgent`,
so :class:`orbit_scripted_agent.OrbitScriptedAgent` behaviour is unchanged when
``bonus_weight=0`` or no simulations run.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from orbit_graph import OrbitGraph
from orbit_lookahead_agent import (
    _candidate_target_ids,
    _owned_ids,
    _reachable_etas,
)


@dataclass(frozen=True)
class MCTSConfig:
    """Hyperparameters for abstract capture MCTS."""

    simulations: int = 48
    """Number of MCTS iterations per ``act`` call."""
    rollout_depth: int = 4
    """Max plies in the random/greedy playout (self + enemy alternating)."""
    exploration_c: float = 1.4
    """UCT exploration constant."""
    top_k_actions: int = 6
    """Branching cap at the root (and on expansion)."""
    eta_horizon: float = 60.0
    """Max ETA for a directed leg to count as reachable."""
    bonus_weight: float = 1.0
    """Scale MCTS values before they join attack ranking."""
    min_bonus: float = 1e-6
    greedy_rollout: bool = True
    """Enemy/self rollout picks highest-production legal capture (else uniform)."""
    rollout_randomness: float = 0.15
    """Probability of a random legal capture during rollout (exploration)."""


@dataclass
class _MCTSNode:
  state_key: tuple[tuple[int, int], ...]
  player_to_move: int
  untried: list[int] = field(default_factory=list)
  children: dict[int, "_MCTSNode"] = field(default_factory=dict)
  visits: int = 0
  value_sum: float = 0.0
  parent: Optional["_MCTSNode"] = None
  action: Optional[int] = None

  def q(self) -> float:
    return self.value_sum / self.visits if self.visits > 0 else 0.0

  def uct(self, c: float) -> float:
    if self.parent is None or self.visits == 0:
      return float("inf")
    return self.q() + c * math.sqrt(math.log(self.parent.visits) / self.visits)


class CaptureMCTS:
    """Abstract capture-tree MCTS on a fixed :class:`OrbitGraph`."""

    def __init__(
        self,
        og: OrbitGraph,
        cfg: MCTSConfig,
        *,
        root_player: int,
        rng: Optional[random.Random] = None,
    ):
        self.og = og
        self.cfg = cfg
        self.root_player = int(root_player)
        self.enemy = 1 - self.root_player if self.root_player in (0, 1) else -1
        self.rng = rng or random.Random()
        self.candidates = _candidate_target_ids(og)
        self._prod = {int(p.id): int(p.production) for p in og.planets}
        self._owners = {
            int(p.id): int(p.owner)
            for p in og.planets
            if not getattr(p, "is_comet", False)
        }

    def _state_key(self, owners: dict[int, int]) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(owners.items()))

    def _owned(self, owners: dict[int, int], player: int) -> set[int]:
        return {pid for pid, o in owners.items() if o == player}

    def _legal_captures(self, owners: dict[int, int], player: int) -> list[int]:
        if player < 0:
            return []
        owned = self._owned(owners, player)
        if not owned:
            return []
        reach = _reachable_etas(
            self.og, owned, self.candidates, eta_horizon=self.cfg.eta_horizon,
        )
        return [
            pid for pid in reach
            if owners.get(pid, -1) != player
        ]

    def _rank_actions(self, owners: dict[int, int], player: int) -> list[int]:
        legal = self._legal_captures(owners, player)
        legal.sort(key=lambda pid: self._prod.get(pid, 0), reverse=True)
        k = max(1, int(self.cfg.top_k_actions))
        return legal[:k]

    def _apply(self, owners: dict[int, int], pid: int, player: int) -> dict[int, int]:
        nxt = dict(owners)
        nxt[int(pid)] = int(player)
        return nxt

    def _production_margin(self, owners: dict[int, int]) -> float:
        prods = {}
        for pid, prod in self._prod.items():
            o = owners.get(pid, -1)
            prods[o] = prods.get(o, 0) + prod
        mine = prods.get(self.root_player, 0)
        total = sum(prods.values())
        if total <= 0:
            return 0.0
        best_other = max(
            (v for o, v in prods.items() if o != self.root_player),
            default=0,
        )
        return float((mine - best_other) / total)

    def _rollout(self, owners: dict[int, int], player_to_move: int) -> float:
        state = dict(owners)
        player = int(player_to_move)
        depth = int(self.cfg.rollout_depth)
        for _ in range(depth):
            legal = self._legal_captures(state, player)
            if not legal:
                player = self.enemy if player == self.root_player else self.root_player
                legal = self._legal_captures(state, player)
                if not legal:
                    break
            if (
                self.cfg.greedy_rollout
                and self.rng.random() > float(self.cfg.rollout_randomness)
            ):
                pid = max(legal, key=lambda p: self._prod.get(p, 0))
            else:
                pid = self.rng.choice(legal)
            state = self._apply(state, pid, player)
            player = self.enemy if player == self.root_player else self.root_player
        return self._production_margin(state)

    def _expand(self, node: _MCTSNode, owners: dict[int, int]) -> _MCTSNode:
        pid = node.untried.pop()
        child_owners = self._apply(owners, pid, node.player_to_move)
        nxt_player = (
            self.enemy if node.player_to_move == self.root_player else self.root_player
        )
        child = _MCTSNode(
            state_key=self._state_key(child_owners),
            player_to_move=nxt_player,
            untried=self._rank_actions(child_owners, nxt_player),
            parent=node,
            action=int(pid),
        )
        node.children[int(pid)] = child
        return child

    def search(self) -> dict[int, float]:
        """Run MCTS; return per-target priority bonuses for the root player."""
        if int(self.cfg.simulations) <= 0 or float(self.cfg.bonus_weight) == 0.0:
            return {}

        owners0 = dict(self._owners)
        root_actions = self._rank_actions(owners0, self.root_player)
        if not root_actions:
            return {}

        root = _MCTSNode(
            state_key=self._state_key(owners0),
            player_to_move=self.root_player,
            untried=list(root_actions),
        )

        for _ in range(int(self.cfg.simulations)):
            node = root
            owners = dict(owners0)

            # --- selection ---
            while node.untried == [] and node.children:
                node = max(
                    node.children.values(),
                    key=lambda n: n.uct(self.cfg.exploration_c),
                )
                owners = self._apply(
                    owners, int(node.action), int(node.parent.player_to_move),
                )

            # --- expansion ---
            if node.untried:
                child = self._expand(node, owners)
                owners = self._apply(owners, int(child.action), int(node.player_to_move))
                node = child

            # --- simulation ---
            value = self._rollout(owners, node.player_to_move)

            # --- backprop ---
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent

        out: dict[int, float] = {}
        for pid, child in root.children.items():
            if child.visits <= 0:
                continue
            val = float(self.cfg.bonus_weight) * child.q()
            if val > float(self.cfg.min_bonus):
                out[int(pid)] = val
        return out


__all__ = ["MCTSConfig", "CaptureMCTS"]
