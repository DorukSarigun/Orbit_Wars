"""Expansion target selection for graph-based scripted agents.

Ranks neutral / enemy capture opportunities by **efficiency**:

    efficiency = production_value_until_end / fleet_cost

where

* ``production_value_until_end`` — discounted production from capture until
  game end (``reworked_features.discounted_production_potential`` on the
  post-capture hold horizon).
* ``fleet_cost`` — ships this launch must spend (minimum capture size after
  accounting for inbound friendly / enemy fleets on the target).
* ``eta`` — turns until the fleet arrives (when the planet flips).
* ``attack_window`` — graph pressure score on the target (timing quality).

The highest-efficiency feasible candidate is the default expansion pick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from feature_extraction import Planet, parse_fleets, parse_planets
from navigation import compute_launch, predict_fleet_target
from orbit_graph import FleetMission, FleetSide, OrbitGraph, build_orbit_graph
from reworked_features import DEFAULT_GAMMA, discounted_production_potential
from scripted_strategy import (
    ScriptedConfig,
    _defense_at_arrival,
    _group_fleets_by_target,
    _own_fleets_capture_target,
    compute_src_budgets,
)


@dataclass(frozen=True)
class ExpansionCandidate:
    """One scored expansion (capture) opportunity."""

    tgt_id: int
    src_id: int
    ships_sent: int
    ships_required: int
    eta: float
    capture_turn: float
    hold_turns: float
    production_value: float
    fleet_cost: float
    efficiency: float
    net_value: float
    attack_window: float
    angle: float
    defense_at_arrival: int

    @property
    def score_tuple(self) -> tuple[float, ...]:
        """Lexicographic ranking key (higher is better)."""
        return (
            self.efficiency,
            self.net_value,
            self.attack_window,
            -self.eta,
            self.production_value,
        )


@dataclass(frozen=True)
class ExpansionConfig:
    """Tunable expansion planner knobs."""

    target_neutrals: bool = True
    target_enemies: bool = True
    target_comets: bool = False
    eta_horizon: float = 60.0
    gamma: float = DEFAULT_GAMMA
    defense_window: int = 15
    max_candidates: int = 24


def production_value_until_end(
    production: int,
    *,
    eta: float,
    current_turn: int,
    max_turns: int,
    gamma: float = DEFAULT_GAMMA,
) -> tuple[float, float]:
    """Discounted production from capture until game end.

    Returns ``(production_value, hold_turns)`` where ``hold_turns`` is the
    number of turns we expect to own the planet after arrival.
    """
    remaining = max(0, int(max_turns) - int(current_turn))
    hold = max(0.0, float(remaining) - float(eta))
    value = float(
        discounted_production_potential(
            np.array([float(production)], dtype=np.float32),
            np.array([hold], dtype=np.float32),
            gamma=float(gamma),
        )[0]
    )
    return value, hold


def estimated_defense_at_arrival(
    tgt: Planet,
    eta: float,
    player: int,
    og: OrbitGraph,
) -> int:
    """Garrison growth plus inbound fleets arriving by ``eta``."""
    base = int(_defense_at_arrival(tgt, eta, player))
    enemy_inbound = 0
    friendly_inbound = 0
    for rec in og.fleets_to(int(tgt.id)):
        if float(rec.eta) > float(eta) + 1e-6:
            continue
        if rec.side == FleetSide.ENEMY:
            enemy_inbound += int(rec.fleet.ships)
        elif rec.side == FleetSide.SELF:
            friendly_inbound += int(rec.fleet.ships)
    return max(0, base + enemy_inbound - friendly_inbound)


def _collect_expansion_targets(
    planets: list[Planet],
    *,
    player: int,
    cfg: ExpansionConfig,
) -> list[Planet]:
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


def _ship_send_options(required: int, budget: int) -> list[int]:
    """Discrete ship counts to evaluate (minimum-first, then budget)."""
    req = int(required)
    cap = int(budget)
    if cap < req:
        return []
    opts = {req}
    if cap > req:
        opts.add(cap)
        mid = (req + cap) // 2
        opts.add(int(mid))
    return sorted(opts)


def score_expansion_launch(
    src: Planet,
    tgt: Planet,
    ships_sent: int,
    *,
    eta: float,
    angle: float,
    og: OrbitGraph,
    player: int,
    current_turn: int,
    max_turns: int,
    cfg: ExpansionConfig | None = None,
) -> Optional[ExpansionCandidate]:
    """Score one concrete launch if it captures ``tgt``."""
    cfg = cfg or ExpansionConfig()
    if int(ships_sent) <= 0 or tgt.owner == player:
        return None

    defense = estimated_defense_at_arrival(tgt, eta, player, og)
    if int(ships_sent) <= defense:
        return None

    prod_value, hold = production_value_until_end(
        int(tgt.production),
        eta=eta,
        current_turn=current_turn,
        max_turns=max_turns,
        gamma=cfg.gamma,
    )
    if prod_value <= 0.0 or hold <= 0.0:
        return None

    fleet_cost = float(ships_sent)
    efficiency = prod_value / fleet_cost
    net_value = prod_value - fleet_cost
    window = og.attack_window_score(int(tgt.id))

    return ExpansionCandidate(
        tgt_id=int(tgt.id),
        src_id=int(src.id),
        ships_sent=int(ships_sent),
        ships_required=int(defense) + 1,
        eta=float(eta),
        capture_turn=float(current_turn) + float(eta),
        hold_turns=float(hold),
        production_value=float(prod_value),
        fleet_cost=fleet_cost,
        efficiency=float(efficiency),
        net_value=float(net_value),
        attack_window=float(window),
        angle=float(angle),
        defense_at_arrival=int(defense),
    )


def enumerate_expansion_candidates(
    og: OrbitGraph,
    env_obs: Any,
    player: int,
    *,
    current_turn: int = 0,
    max_turns: int = 500,
    cfg: ExpansionConfig | None = None,
) -> list[ExpansionCandidate]:
    """Enumerate and score all feasible expansion launches this turn."""
    cfg = cfg or ExpansionConfig()
    omega = float(
        env_obs.get("angular_velocity", 0.0)
        if isinstance(env_obs, dict)
        else getattr(env_obs, "angular_velocity", 0.0)
    )
    planets = og.planets
    fleets = parse_fleets(env_obs)
    my_planets = [p for p in planets if p.owner == player and not p.is_comet]
    if not my_planets:
        return []

    my_ids = {p.id for p in my_planets}
    my_to_my, enemy_to_my = _group_fleets_by_target(
        fleets, planets, omega, my_ids, player,
    )
    budgets = compute_src_budgets(
        my_planets,
        my_to_my,
        enemy_to_my,
        cfg=ScriptedConfig(defense_window=cfg.defense_window),
    )

    fleets_by_tgt: dict[int, list[tuple[int, float, int]]] = {}
    for f in fleets:
        pred = predict_fleet_target(f, planets, omega)
        if pred is None:
            continue
        tid = int(pred["target_id"])
        fleets_by_tgt.setdefault(tid, []).append(
            (int(f.ships), float(pred["eta"]), int(f.owner)),
        )

    targets = _collect_expansion_targets(planets, player=player, cfg=cfg)
    candidates: list[ExpansionCandidate] = []

    for tgt in targets:
        if _own_fleets_capture_target(tgt, fleets_by_tgt, player):
            continue

        best_for_tgt: Optional[ExpansionCandidate] = None
        for src in my_planets:
            budget = int(budgets.get(src.id, 0))
            if budget <= 0:
                continue

            # Probe with budget to estimate ETA, then evaluate ship options.
            probe = compute_launch(src, tgt, budget, planets, omega)
            if probe is None:
                continue
            eta_probe = float(probe["eta"])
            if eta_probe > cfg.eta_horizon:
                continue

            defense = estimated_defense_at_arrival(tgt, eta_probe, player, og)
            required = int(defense) + 1
            for ships in _ship_send_options(required, budget):
                launch = compute_launch(src, tgt, ships, planets, omega)
                if launch is None:
                    continue
                eta = float(launch["eta"])
                if eta > cfg.eta_horizon:
                    continue
                cand = score_expansion_launch(
                    src,
                    tgt,
                    ships,
                    eta=eta,
                    angle=float(launch["angle"]),
                    og=og,
                    player=player,
                    current_turn=current_turn,
                    max_turns=max_turns,
                    cfg=cfg,
                )
                if cand is None:
                    continue
                if best_for_tgt is None or cand.score_tuple > best_for_tgt.score_tuple:
                    best_for_tgt = cand

        if best_for_tgt is not None:
            candidates.append(best_for_tgt)

    candidates.sort(key=lambda c: c.score_tuple, reverse=True)
    if cfg.max_candidates > 0:
        candidates = candidates[: int(cfg.max_candidates)]
    return candidates


def pick_best_expansion(
    env_obs: Any,
    player: int,
    *,
    current_turn: int = 0,
    max_turns: int = 500,
    cfg: ExpansionConfig | None = None,
    og: OrbitGraph | None = None,
) -> Optional[ExpansionCandidate]:
    """Return the highest-efficiency expansion opportunity, if any."""
    graph = og or build_orbit_graph(env_obs, player)
    ranked = enumerate_expansion_candidates(
        graph,
        env_obs,
        player,
        current_turn=current_turn,
        max_turns=max_turns,
        cfg=cfg,
    )
    return ranked[0] if ranked else None
