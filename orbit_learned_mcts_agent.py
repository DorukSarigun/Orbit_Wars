"""Timed MCTS agent whose branch policy can be learned.

This module intentionally does not modify the existing model classes or the
baseline timed MCTS agent.  It adds a narrow hook at the MCTS branch-selection
level:

    abstract MCTS state + candidate capture targets -> prior logits

The default prior is a small linear scorer over hand-built branch features.  A
future RL policy can replace it by implementing the same callable interface and
returning one logit per candidate target.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from orbit_defense import PlanetDefenseState
from orbit_graph import OrbitGraph
from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig
from orbit_timed_mcts import TimedCaptureMCTS, TimedMCTSConfig, _TimedState


BRANCH_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "target_prod_norm",
    "target_ships_norm",
    "target_is_enemy",
    "target_is_neutral",
    "arrival_norm",
    "ships_sent_norm",
    "hold_value_norm",
    "heuristic_norm",
    "turn_frac",
    "root_owned_frac",
)


@dataclass(frozen=True)
class BranchPriorRequest:
    """Input passed to a learned branch-prior provider."""

    root_player: int
    player_to_move: int
    relative_turn: int
    current_turn: int
    max_turns: int
    candidate_targets: tuple[int, ...]
    features_by_target: Mapping[int, Mapping[str, float]]


class BranchPriorProvider(Protocol):
    """Callable branch prior used by :class:`LearnedBranchTimedMCTS`."""

    def __call__(self, request: BranchPriorRequest) -> Mapping[int, float]:
        """Return unnormalised prior logits keyed by target id."""


@dataclass(frozen=True)
class LinearBranchPrior:
    """Small trainable linear branch-prior head.

    The weights are deliberately plain Python data so a later RL loop can save
    and load them without depending on any specific neural-network stack.
    """

    weights: Mapping[str, float] = field(default_factory=dict)

    @staticmethod
    def default() -> "LinearBranchPrior":
        return LinearBranchPrior({
            "target_prod_norm": 1.25,
            "target_is_enemy": 0.35,
            "target_is_neutral": 0.10,
            "arrival_norm": -0.45,
            "ships_sent_norm": -0.25,
            "hold_value_norm": 0.75,
            "heuristic_norm": 0.50,
        })

    def score_features(self, features: Mapping[str, float]) -> float:
        return float(sum(
            float(self.weights.get(name, 0.0)) * float(features.get(name, 0.0))
            for name in BRANCH_FEATURE_NAMES
        ))

    def __call__(self, request: BranchPriorRequest) -> Mapping[int, float]:
        return {
            int(target): self.score_features(request.features_by_target[int(target)])
            for target in request.candidate_targets
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearBranchPrior":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = data["weights"] if isinstance(data, dict) and "weights" in data else data
        return cls({str(k): float(v) for k, v in dict(weights).items()})

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"weights": dict(self.weights)}, indent=2, sort_keys=True),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class LearnedMCTSConfig:
    """Configuration for learned-prior timed MCTS."""

    timed_mcts: TimedMCTSConfig = field(default_factory=TimedMCTSConfig)
    prior_weight: float = 1.0
    """How strongly prior logits affect branch ordering and PUCT."""
    heuristic_logit_weight: float = 0.002
    """Small stabilising logit from the built-in MCTS capture heuristic."""
    prior_temperature: float = 1.0
    """Softmax temperature for PUCT priors."""
    min_prior: float = 1e-4
    """Floor so every legal branch remains explorable."""


@dataclass
class _LearnedNode:
    state: _TimedState
    untried: list[int]
    priors: dict[int, float]
    children: dict[int, "_LearnedNode"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    parent: Optional["_LearnedNode"] = None
    action: Optional[int] = None

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class LearnedBranchTimedMCTS(TimedCaptureMCTS):
    """Timed capture MCTS with learned branch priors and PUCT selection."""

    def __init__(
        self,
        og: OrbitGraph,
        cfg: LearnedMCTSConfig,
        *,
        root_player: int,
        current_turn: int = 0,
        max_turns: int = 500,
        prior_provider: Optional[BranchPriorProvider] = None,
        rng: Optional[random.Random] = None,
    ):
        self.learned_cfg = cfg
        self.prior_provider = prior_provider or LinearBranchPrior.default()
        super().__init__(
            og,
            cfg.timed_mcts,
            root_player=root_player,
            current_turn=current_turn,
            max_turns=max_turns,
            rng=rng,
        )

    def _branch_features(
        self,
        state: _TimedState,
        target: int,
        heuristic: float,
    ) -> dict[str, float]:
        action = self._plan_capture(state, int(state.player_to_move), int(target))
        if action is None:
            return {name: 0.0 for name in BRANCH_FEATURE_NAMES}

        owners = state.owner_map()
        ships = state.ship_map()
        owner = int(owners.get(int(target), -1))
        total_owned = sum(1 for o in owners.values() if int(o) == self.root_player)
        total_planets = max(1, len(owners))
        hold_value = self._capture_income(int(target), int(action.arrival_turn))

        return {
            "bias": 1.0,
            "target_prod_norm": float(self._prod.get(int(target), 0)) / 10.0,
            "target_ships_norm": float(ships.get(int(target), 0)) / 100.0,
            "target_is_enemy": 1.0 if owner >= 0 and owner != int(state.player_to_move) else 0.0,
            "target_is_neutral": 1.0 if owner < 0 else 0.0,
            "arrival_norm": float(action.arrival_turn - int(state.turn)) / max(1.0, float(self.horizon_turns)),
            "ships_sent_norm": float(action.ships) / 100.0,
            "hold_value_norm": float(hold_value) / 1000.0,
            "heuristic_norm": float(heuristic) / 1000.0,
            "turn_frac": float(self.current_turn + int(state.turn)) / max(1.0, float(self.max_turns)),
            "root_owned_frac": float(total_owned) / float(total_planets),
        }

    def _rank_action_objects(self, state: _TimedState) -> list[tuple[int, float, dict[str, float]]]:
        player = int(state.player_to_move)
        owners = state.owner_map()
        rows: list[tuple[int, float, dict[str, float]]] = []
        for target in self.candidates:
            if owners.get(int(target), -1) == player:
                continue
            action = self._plan_capture(state, player, int(target))
            if action is None:
                continue
            features = self._branch_features(state, int(target), float(action.heuristic))
            rows.append((int(target), float(action.heuristic), features))
        return rows

    def _prior_logits(
        self,
        state: _TimedState,
        rows: list[tuple[int, float, dict[str, float]]],
    ) -> dict[int, float]:
        if not rows:
            return {}
        target_ids = tuple(int(t) for t, _h, _f in rows)
        features_by_target = {int(t): f for t, _h, f in rows}
        request = BranchPriorRequest(
            root_player=int(self.root_player),
            player_to_move=int(state.player_to_move),
            relative_turn=int(state.turn),
            current_turn=int(self.current_turn),
            max_turns=int(self.max_turns),
            candidate_targets=target_ids,
            features_by_target=features_by_target,
        )
        logits = dict(self.prior_provider(request))
        out: dict[int, float] = {}
        for target, heuristic, _features in rows:
            out[int(target)] = (
                float(self.learned_cfg.prior_weight) * float(logits.get(int(target), 0.0))
                + float(self.learned_cfg.heuristic_logit_weight) * float(heuristic)
            )
        return out

    def _softmax_priors(self, logits: Mapping[int, float]) -> dict[int, float]:
        if not logits:
            return {}
        temp = max(1e-6, float(self.learned_cfg.prior_temperature))
        scaled = {int(k): float(v) / temp for k, v in logits.items()}
        max_logit = max(scaled.values())
        exp_vals = {
            int(k): math.exp(max(-60.0, min(60.0, float(v) - max_logit)))
            for k, v in scaled.items()
        }
        total = sum(exp_vals.values())
        if total <= 0.0:
            return {int(k): 1.0 / len(exp_vals) for k in exp_vals}
        floor = max(0.0, float(self.learned_cfg.min_prior))
        priors = {int(k): max(floor, float(v) / total) for k, v in exp_vals.items()}
        norm = sum(priors.values())
        return {int(k): float(v) / norm for k, v in priors.items()}

    def _rank_actions(self, state: _TimedState) -> list[int]:
        rows = self._rank_action_objects(state)
        logits = self._prior_logits(state, rows)
        rows.sort(
            key=lambda row: (
                float(logits.get(int(row[0]), 0.0)),
                float(row[1]),
            ),
            reverse=True,
        )
        k = max(1, int(self.cfg.top_k_actions))
        return [int(target) for target, _heuristic, _features in rows[:k]]

    def _node_for_state(
        self,
        state: _TimedState,
        *,
        parent: Optional[_LearnedNode] = None,
        action: Optional[int] = None,
    ) -> _LearnedNode:
        rows = self._rank_action_objects(state)
        logits = self._prior_logits(state, rows)
        ranked = self._rank_actions(state)
        priors = self._softmax_priors({target: logits.get(target, 0.0) for target in ranked})
        return _LearnedNode(
            state=state,
            untried=list(ranked),
            priors=priors,
            parent=parent,
            action=action,
        )

    def _expand_learned(self, node: _LearnedNode) -> _LearnedNode:
        target = int(node.untried.pop(0))
        child_state = self._apply_action(node.state, target)
        child = self._node_for_state(child_state, parent=node, action=target)
        node.children[target] = child
        return child

    def _puct(self, parent: _LearnedNode, child: _LearnedNode) -> float:
        if child.visits <= 0:
            q = 0.0
        else:
            q = child.q()
        prior = float(parent.priors.get(int(child.action), float(self.learned_cfg.min_prior)))
        explore = (
            float(self.cfg.exploration_c)
            * prior
            * math.sqrt(max(1.0, float(parent.visits)))
            / (1.0 + float(child.visits))
        )
        return float(q + explore)

    def search(self) -> dict[int, float]:
        """Run learned-prior timed MCTS; return root-target metric improvements."""
        if int(self.cfg.simulations) <= 0 or float(self.cfg.bonus_weight) == 0.0:
            return {}

        root = self._node_for_state(self._initial_state)
        if not root.untried:
            return {}

        for _ in range(int(self.cfg.simulations)):
            node = root
            while node.untried == [] and node.children:
                node = max(
                    node.children.values(),
                    key=lambda child: self._puct(node, child),
                )
            if node.untried:
                node = self._expand_learned(node)

            value = self._rollout(node.state)
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent

        out: dict[int, float] = {}
        for target, child in root.children.items():
            if child.visits <= 0:
                continue
            improvement = child.q() - float(self._baseline)
            val = float(self.cfg.bonus_weight) * improvement
            if val > float(self.cfg.min_bonus):
                out[int(target)] = float(val)
        return out


class OrbitLearnedMCTSAgent(OrbitScriptedAgent):
    """Timed MCTS + LP logistics with a trainable branch-prior hook."""

    def __init__(
        self,
        cfg: OrbitScriptedConfig | None = None,
        learned_mcts_cfg: LearnedMCTSConfig | None = None,
        prior_provider: Optional[BranchPriorProvider] = None,
    ):
        if cfg is None:
            cfg = OrbitScriptedConfig(
                attack_allocator="lp",
                reinforcement_allocator="lp",
            )
        super().__init__(cfg=cfg)
        self.learned_mcts_cfg = learned_mcts_cfg or LearnedMCTSConfig()
        self.prior_provider = prior_provider or LinearBranchPrior.default()

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
        timed_cfg = self.learned_mcts_cfg.timed_mcts
        if int(timed_cfg.simulations) <= 0 or float(timed_cfg.bonus_weight) == 0.0:
            return None
        mcts = LearnedBranchTimedMCTS(
            og,
            self.learned_mcts_cfg,
            root_player=int(og.player),
            current_turn=int(current_turn),
            max_turns=int(max_turns),
            prior_provider=self.prior_provider,
        )
        bonus = mcts.search()
        return bonus or None


__all__ = [
    "BRANCH_FEATURE_NAMES",
    "BranchPriorProvider",
    "BranchPriorRequest",
    "LearnedBranchTimedMCTS",
    "LearnedMCTSConfig",
    "LinearBranchPrior",
    "OrbitLearnedMCTSAgent",
]
