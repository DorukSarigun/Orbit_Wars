"""Scripted agent that maximizes graph structural metrics.

The agent evaluates candidate fleet launches by estimating how much the
empire-wide graph objective would improve after a successful capture.
Hard safety constraints (defensive reserve, neutral in-flight skip) are
applied before any metric gain is considered.

Priority order (lexicographic):
  1. Safety: do not launch if source defense shortfall would occur.
  2. Empire delta: weighted sum of node-metric improvements on self-owned
     planets (same 10 channels and weights as GNN node features / reward).
  3. Optional tie-breakers: local target delta, flow_opportunity, ETA, ships.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from feature_extraction import Planet, parse_fleets, parse_planets
from navigation import compute_launch, predict_all_fleet_targets
from obs_builder import build_observation

from scripted_graph_scorer import (
    DEFAULT_GRAPH_CHANNEL_WEIGHTS,
    ScriptedGraphScores,
    build_scripted_graph_scores,
    empire_node_objective,
    local_target_objective,
)
from scripted_strategy import ScriptedConfig, compute_src_budgets, _defense_at_arrival


def _env_omega(env_obs) -> float:
    if isinstance(env_obs, dict):
        return float(env_obs.get("angular_velocity", 0.0) or 0.0)
    return float(getattr(env_obs, "angular_velocity", 0.0) or 0.0)


def _neutral_in_flight_ships(
    target_id: int,
    *,
    fleets,
    fleet_preds: dict,
    player: int,
) -> int:
    total = 0
    for fleet in fleets:
        pred = fleet_preds.get(int(fleet.id))
        if pred is None or int(pred["target_id"]) != int(target_id):
            continue
        if int(fleet.owner) == int(player):
            total += int(fleet.ships)
    return total


def _slot_for_planet_id(obs, planet_id: int) -> int:
    for slot, pid in enumerate(obs.planet_ids):
        if int(pid) == int(planet_id):
            return int(slot)
    return -1


def _simulate_successful_capture_env(
    env_obs,
    *,
    src: Planet,
    tgt: Planet,
    ships_sent: int,
    eta: float,
    player: int,
) -> dict:
    """Return a shallow-cloned env obs after a successful capture launch."""
    out = copy.deepcopy(env_obs) if isinstance(env_obs, dict) else dict(env_obs)
    rows = [list(row) for row in out["planets"]]
    defense = _defense_at_arrival(tgt, eta, player)
    remaining_at_tgt = max(1, int(ships_sent) - int(defense))
    for row in rows:
        pid = int(row[0])
        if pid == int(src.id):
            row[5] = max(0, int(row[5]) - int(ships_sent))
        elif pid == int(tgt.id):
            row[1] = int(player)
            row[5] = int(remaining_at_tgt)
    out["planets"] = rows
    return out


@dataclass(frozen=True)
class GraphMaximizingConfig:
    """Tunable objective for :class:`GraphMaximizingAgent`."""

    # Hard safety / candidate filters.
    defense_window: int = 15
    max_launches: int = 8
    target_neutrals: bool = True
    target_enemies: bool = True
    target_comets: bool = False
    top_target_candidates: int = 12
    top_source_candidates: int = 8

    # Priority 2 uses ``channel_weights`` (same as training reward / GNN metrics).
    channel_weights: Sequence[float] = DEFAULT_GRAPH_CHANNEL_WEIGHTS

    # Optional tie-breakers after empire delta (usually 0 when aligned with RL).
    local_target_weight: float = 0.0
    flow_opportunity_weight: float = 0.0

    # Priority 5 tie-breakers.
    eta_penalty: float = 0.05
    ships_penalty: float = 0.01

    # Observation build knobs (match training defaults).
    p_max: int = 45
    f_max: int = 64


@dataclass
class _LaunchCandidate:
    score_tuple: tuple[float, ...]
    src_id: int
    tgt_id: int
    ships_sent: int
    angle: float


class GraphMaximizingAgent:
    """Scripted policy that greedily maximizes graph metric potential."""

    def __init__(
        self,
        cfg: GraphMaximizingConfig | None = None,
        *,
        reward_cfg: Any = None,
    ):
        self.cfg = cfg or GraphMaximizingConfig()
        self.reward_cfg = reward_cfg

    def act(
        self,
        env_obs,
        player: int,
        *,
        current_turn: int = 0,
        max_turns: int = 500,
    ) -> list[list]:
        cfg = self.cfg
        omega = _env_omega(env_obs)
        planets = parse_planets(env_obs)
        fleets = parse_fleets(env_obs)

        my_planets = [p for p in planets if p.owner == player and not p.is_comet]
        if not my_planets:
            return []

        obs = build_observation(
            env_obs,
            player=player,
            angular_velocity=omega,
            turn=current_turn,
            p_max=cfg.p_max,
            f_max=cfg.f_max,
            max_turns=max_turns,
            predict_targets=True,
        )
        pre_scores = build_scripted_graph_scores(
            obs, reward_cfg=self.reward_cfg, turn=int(current_turn),
        )
        if pre_scores is None:
            return []

        pre_empire = empire_node_objective(pre_scores, obs, cfg.channel_weights)
        planet_by_id = {int(p.id): p for p in planets}
        my_planet_ids = {p.id for p in my_planets}

        fleet_preds = predict_all_fleet_targets(fleets, planets, omega) if fleets else {}
        my_to_my, enemy_to_my = _group_incoming_fleets(
            fleets, planets, omega, my_planet_ids, player,
        )
        budgets = compute_src_budgets(
            my_planets,
            my_to_my,
            enemy_to_my,
            cfg=ScriptedConfig(defense_window=cfg.defense_window),
        )

        targets = _collect_targets(planets, player=player, cfg=cfg)
        if not targets:
            return []

        ranked_targets = sorted(
            targets,
            key=lambda p: (
                pre_scores.target_score(
                    _slot_for_planet_id(obs, p.id),
                    cfg.channel_weights,
                ),
                float(p.production),
                -int(p.ships),
            ),
            reverse=True,
        )[: max(1, int(cfg.top_target_candidates))]

        ranked_sources = sorted(
            my_planets,
            key=lambda p: (
                pre_scores.source_score(
                    _slot_for_planet_id(obs, p.id),
                    cfg.channel_weights,
                ),
                int(budgets.get(p.id, 0)),
                int(p.production),
            ),
            reverse=True,
        )[: max(1, int(cfg.top_source_candidates))]

        candidates: list[_LaunchCandidate] = []
        for src in ranked_sources:
            budget = int(budgets.get(src.id, 0))
            if budget <= 0:
                continue
            src_slot = _slot_for_planet_id(obs, src.id)
            if src_slot < 0:
                continue

            for tgt in ranked_targets:
                if tgt.id == src.id:
                    continue
                tgt_slot = _slot_for_planet_id(obs, tgt.id)
                if tgt_slot < 0:
                    continue

                defense = _defense_at_arrival(tgt, 0.0, player)
                needed = int(defense) + 1
                if needed > budget:
                    continue

                if tgt.owner == -1:
                    in_flight = _neutral_in_flight_ships(
                        tgt.id, fleets=fleets, fleet_preds=fleet_preds, player=player,
                    )
                    if in_flight >= needed:
                        continue

                ship_options = _ship_options(needed, budget)
                for ships_sent in ship_options:
                    launch = compute_launch(
                        src, tgt, int(ships_sent), planets, omega,
                    )
                    if launch is None:
                        continue
                    eta = float(launch["eta"])
                    defense_eta = _defense_at_arrival(tgt, eta, player)
                    if int(ships_sent) <= int(defense_eta):
                        continue

                    post_env = _simulate_successful_capture_env(
                        env_obs,
                        src=src,
                        tgt=tgt,
                        ships_sent=int(ships_sent),
                        eta=eta,
                        player=player,
                    )
                    post_obs = build_observation(
                        post_env,
                        player=player,
                        angular_velocity=omega,
                        turn=current_turn,
                        p_max=cfg.p_max,
                        f_max=cfg.f_max,
                        max_turns=max_turns,
                        predict_targets=True,
                    )
                    post_scores = build_scripted_graph_scores(
                        post_obs,
                        reward_cfg=self.reward_cfg,
                        turn=int(current_turn),
                    )
                    if post_scores is None:
                        continue

                    post_empire = empire_node_objective(
                        post_scores, post_obs, cfg.channel_weights,
                    )
                    empire_delta = post_empire - pre_empire
                    local_pre = local_target_objective(
                        pre_scores, tgt_slot, cfg.channel_weights,
                    )
                    local_post = local_target_objective(
                        post_scores, tgt_slot, cfg.channel_weights,
                    )
                    local_delta = local_post - local_pre
                    flow = pre_scores.pair_score(src_slot, tgt_slot)

                    score_tuple = (
                        empire_delta,
                        cfg.local_target_weight * local_delta,
                        cfg.flow_opportunity_weight * flow,
                        -cfg.eta_penalty * eta,
                        -cfg.ships_penalty * float(ships_sent),
                    )
                    candidates.append(_LaunchCandidate(
                        score_tuple=score_tuple,
                        src_id=int(src.id),
                        tgt_id=int(tgt.id),
                        ships_sent=int(ships_sent),
                        angle=float(launch["angle"]),
                    ))

        if not candidates:
            return []

        candidates.sort(key=lambda c: c.score_tuple, reverse=True)

        actions: list[list] = []
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        for cand in candidates:
            if len(actions) >= cfg.max_launches:
                break
            if cand.src_id in used_sources or cand.tgt_id in used_targets:
                continue
            actions.append([
                int(cand.src_id),
                float(cand.angle),
                int(cand.ships_sent),
            ])
            used_sources.add(cand.src_id)
            used_targets.add(cand.tgt_id)
        return actions


def _ship_options(needed: int, budget: int) -> list[int]:
    opts = {int(needed)}
    if budget > needed:
        opts.add(int(budget))
        mid = (needed + budget) // 2
        opts.add(int(mid))
    return sorted(opts)


def _collect_targets(planets: list[Planet], *, player: int, cfg: GraphMaximizingConfig):
    out: list[Planet] = []
    for p in planets:
        if p.owner == player:
            continue
        if p.is_comet and not cfg.target_comets:
            continue
        if p.owner == -1 and not cfg.target_neutrals:
            continue
        if p.is_enemy(player) and not cfg.target_enemies:
            continue
        out.append(p)
    return out


def _group_incoming_fleets(
    fleets,
    planets: list[Planet],
    angular_velocity: float,
    my_planet_ids: set[int],
    player: int,
):
    """Minimal copy of scripted_strategy fleet grouping (avoid extra import)."""
    from scripted_strategy import _group_fleets_by_target
    return _group_fleets_by_target(
        fleets, planets, angular_velocity, my_planet_ids, player,
    )
