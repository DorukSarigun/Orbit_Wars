"""Capture-efficiency scripted opponent.

A purely algorithmic, model-free policy.  Each turn it scans every
(my_source, target, ship_bin) candidate and picks up to ``max_launches``
launches that maximize:

    capture_efficiency = production_gain - ships_lost_in_combat

where

    production_gain     = tgt.production * max(0, max_turns - (turn + eta))
    ships_lost_in_combat = defense_at_arrival   (only when the launch captures)

Defensive reserve: each source planet keeps back enough ships to survive
the earliest predicted enemy fleet arriving in ``defense_window`` turns,
accounting for the source planet's own production buffer between now and
the threat arrival as well as any incoming friendly reinforcements.

Public API
----------

``ScriptedConfig``                 -- tunable hyperparameters
``CaptureEfficiencyAgent``         -- ``act(env_obs, player, ...)``
``score_capture_candidate``        -- per-candidate scoring helper
``compute_src_budgets``            -- per-source ship send budget after reserve
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from feature_extraction import Planet, Fleet, parse_planets, parse_fleets
from navigation import compute_launch, predict_fleet_target
from obs_builder import DEFAULT_SHIP_BINS


@dataclass(frozen=True)
class ScriptedConfig:
    """Hyperparameters for ``CaptureEfficiencyAgent``."""

    defense_window: int = 15
    """Only consider enemy fleets arriving within this many turns when
    computing the defensive reserve at each source planet."""

    threat_cluster_buffer: int = 2
    """Aggregate enemy threats arriving within ``e_first + buffer`` turns
    of the earliest threat into a single defensive demand.  Captures the
    case where multiple fleets arrive nearly simultaneously."""

    max_launches: int = 8
    """Per-turn launch cap (mirrors the model's MAX_LAUNCHES_PER_PLAYER_TURN)."""

    target_neutrals: bool = True
    target_enemies: bool = True
    target_comets: bool = False
    """Comet planets are skipped by default since their orbits make them
    unreliable production assets."""

    ship_bins: tuple = DEFAULT_SHIP_BINS
    """Same discrete ship fractions as the model's action space."""


@dataclass
class LaunchCandidate:
    score: float
    eta: float
    src_id: int
    tgt_id: int
    bin_idx: int
    ships_sent: int
    angle: float
    defense_at_arrival: int
    production_gain: float


def _defense_at_arrival(tgt: Planet, eta: float, player: int) -> int:
    """Estimated ship count defending ``tgt`` when our fleet arrives.

    Neutral planets are static (no production) so their defense equals
    their current ship count.  Enemy planets accrue ``production * eta``
    ships in the meantime.  Owning the target trivially has zero defense.
    """
    if tgt.owner == player:
        return 0
    if tgt.owner == -1:
        return int(tgt.ships)
    return int(tgt.ships + tgt.production * eta)


def score_capture_candidate(
    src: Planet,
    tgt: Planet,
    ships_sent: int,
    eta: float,
    *,
    current_turn: int,
    max_turns: int,
    player: int,
) -> tuple[float, int, float]:
    """Return ``(capture_efficiency, defense_at_arrival, production_gain)``.

    ``capture_efficiency`` is ``-inf`` when the launch cannot capture the
    target (insufficient ships).  Reinforcement of own planets is also
    treated as ``-inf`` since v1 only optimizes captures.
    """
    if tgt.owner == player:
        return float("-inf"), 0, 0.0

    defense = _defense_at_arrival(tgt, eta, player)
    if ships_sent <= defense:
        return float("-inf"), defense, 0.0

    remaining = max(0, max_turns - current_turn - eta)
    production_gain = float(tgt.production) * float(remaining)
    score = production_gain - float(defense)
    return float(score), int(defense), float(production_gain)


def compute_src_budgets(
    my_planets: list[Planet],
    my_fleets_to_my: dict[int, list[tuple[int, float]]],
    enemy_fleets_to_my: dict[int, list[tuple[int, float]]],
    *,
    cfg: ScriptedConfig,
) -> dict[int, int]:
    """Per-source ship send budget after holding back defensive reserve.

    Reserve formula at source ``p`` (if any threats within window):

        e_first        = min(eta of incoming enemies <= window)
        production_buf = p.production * e_first
        enemy_at_first = sum of enemies arriving within e_first + buffer
        help_at_first  = sum of own reinforcements arriving by e_first
        reserve        = max(0, enemy_at_first - help_at_first - production_buf)
        budget[p]      = max(0, p.ships - reserve)
    """
    budgets: dict[int, int] = {}
    for p in my_planets:
        threats = [
            (s, eta) for s, eta in enemy_fleets_to_my.get(p.id, [])
            if eta <= cfg.defense_window
        ]
        if not threats:
            budgets[p.id] = int(p.ships)
            continue
        e_first = min(eta for _, eta in threats)
        cluster_cutoff = e_first + cfg.threat_cluster_buffer
        enemy_at_first = sum(s for s, eta in threats if eta <= cluster_cutoff)
        helps = my_fleets_to_my.get(p.id, [])
        help_at_first = sum(s for s, eta in helps if eta <= e_first)
        production_buf = float(p.production) * float(e_first)
        reserve = max(0.0, float(enemy_at_first) - float(help_at_first) - production_buf)
        budgets[p.id] = max(0, int(p.ships - math.ceil(reserve)))
    return budgets


def _group_fleets_by_target(
    fleets: list[Fleet],
    planets: list[Planet],
    angular_velocity: float,
    my_planet_ids: set[int],
    player: int,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[tuple[int, float]]]]:
    """Split incoming fleets into (my_to_my, enemy_to_my) maps keyed by
    target planet id; values are lists of (ships, eta).
    Only fleets whose predicted target is one of my planets are kept.
    """
    my_to_my: dict[int, list[tuple[int, float]]] = {}
    enemy_to_my: dict[int, list[tuple[int, float]]] = {}
    for f in fleets:
        pred = predict_fleet_target(f, planets, angular_velocity)
        if pred is None:
            continue
        tgt_id = int(pred["target_id"])
        if tgt_id not in my_planet_ids:
            continue
        row = (int(f.ships), float(pred["eta"]))
        if f.owner == player:
            my_to_my.setdefault(tgt_id, []).append(row)
        else:
            enemy_to_my.setdefault(tgt_id, []).append(row)
    return my_to_my, enemy_to_my


class CaptureEfficiencyAgent:
    """Stateless scripted opponent.  ``act`` returns a list of env actions."""

    def __init__(self, cfg: ScriptedConfig | None = None):
        self.cfg = cfg or ScriptedConfig()

    def act(
        self,
        env_obs,
        player: int,
        *,
        current_turn: int = 0,
        max_turns: int = 500,
    ) -> list[list]:
        """Return env action list ``[[from_id, angle, ships], ...]``.

        Args:
            env_obs: kaggle env observation (dict-like) for ``player``.
                Must contain ``planets``, ``fleets``, ``initial_planets``,
                ``angular_velocity``, ``comet_planet_ids``, ``comets`` --
                same fields used by ``parse_planets`` / ``parse_fleets``.
            player: agent index (0..3).
            current_turn: zero-indexed env step (used for time horizon).
            max_turns: total game length (used for production_gain).
        """
        cfg = self.cfg
        if isinstance(env_obs, dict):
            omega = env_obs.get("angular_velocity", 0.0) or 0.0
        else:
            omega = getattr(env_obs, "angular_velocity", 0.0) or 0.0

        planets = parse_planets(env_obs)
        fleets = parse_fleets(env_obs)

        my_planets = [p for p in planets if p.owner == player and not p.is_comet]
        if not my_planets:
            return []
        my_planet_ids = {p.id for p in my_planets}

        my_to_my, enemy_to_my = _group_fleets_by_target(
            fleets, planets, omega, my_planet_ids, player,
        )
        budgets = compute_src_budgets(
            my_planets, my_to_my, enemy_to_my, cfg=cfg,
        )

        targets: list[Planet] = []
        for p in planets:
            if p.owner == player:
                continue
            if p.is_comet and not cfg.target_comets:
                continue
            if p.owner == -1 and not cfg.target_neutrals:
                continue
            if p.is_enemy(player) and not cfg.target_enemies:
                continue
            targets.append(p)
        if not targets:
            return []

        candidates: list[LaunchCandidate] = []
        for src in my_planets:
            if budgets.get(src.id, 0) <= 0:
                continue
            for tgt in targets:
                if tgt.id == src.id:
                    continue
                for bin_idx, frac in enumerate(cfg.ship_bins):
                    ships_sent = max(1, int(src.ships * float(frac)))
                    if ships_sent > budgets[src.id]:
                        continue
                    launch = compute_launch(src, tgt, ships_sent, planets, omega)
                    if launch is None:
                        continue
                    eta = float(launch["eta"])
                    score, defense, prod_gain = score_capture_candidate(
                        src, tgt, ships_sent, eta,
                        current_turn=current_turn,
                        max_turns=max_turns,
                        player=player,
                    )
                    if not math.isfinite(score) or score <= 0:
                        continue
                    candidates.append(LaunchCandidate(
                        score=score,
                        eta=eta,
                        src_id=src.id,
                        tgt_id=tgt.id,
                        bin_idx=bin_idx,
                        ships_sent=ships_sent,
                        angle=float(launch["angle"]),
                        defense_at_arrival=defense,
                        production_gain=prod_gain,
                    ))

        candidates.sort(key=lambda c: (-c.score, c.eta, c.tgt_id, c.src_id))

        actions: list[list] = []
        used_pairs: set[tuple[int, int]] = set()
        remaining_budget = dict(budgets)
        for c in candidates:
            if len(actions) >= cfg.max_launches:
                break
            if (c.src_id, c.tgt_id) in used_pairs:
                continue
            if remaining_budget.get(c.src_id, 0) < c.ships_sent:
                continue
            actions.append([int(c.src_id), float(c.angle), int(c.ships_sent)])
            used_pairs.add((c.src_id, c.tgt_id))
            remaining_budget[c.src_id] -= c.ships_sent
        return actions
