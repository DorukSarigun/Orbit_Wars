"""Graph-metric scripted agent: evacuate, defend, attack, support, consolidate.

Turn pipeline::

    1. :func:`orbit_graph.build_orbit_graph`
    2. :func:`orbit_defense.compute_owned_defense` + launch budgets
    3. :func:`orbit_attack_targets.plan_attack_targets`
    4. **act** phases (global cap ``max_launches``):

       0. **Comet evacuation** — garrisons on owned comets about to expire are
          relocated to the nearest safe friendly planet (they would otherwise
          be destroyed with the comet).
       1. **Defense triage** — CRITICAL worlds get multi-source reinforcement,
          but only when *savable*: help must land **before** the earliest
          threat and fully cover the gap.  Unsavable worlds are abandoned;
          if truly doomed their garrison is evacuated instead of fed to the
          enemy.
       2. **Attack** — :func:`orbit_attack_allocate.allocate_greedy_captures`
          (all-or-nothing waves, enemy weak spots first).  ``support_reserve_slots``
          launch slots are held back when any planet needs support.
       3. **Support** — backline ships to high enemy-influence frontline
          planets (partial reinforcement allowed, ETA-aware).
       4. **Consolidation** — idle backline surplus is pushed toward the
          owned planet closest to the enemy (rally point) so force is never
          stranded in the rear.

Public API::

    ``OrbitScriptedConfig``  — tunable hyperparameters
    ``OrbitScriptedAgent``   — ``act(env_obs, player, ...)``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from navigation import compute_launch
from orbit_attack_allocate import allocate_greedy_captures
from orbit_attack_targets import (
    attack_config_from_env,
    plan_attack_targets,
)
from orbit_defense import (
    DEFAULT_DEFENSE_HORIZON,
    DEFAULT_POTENTIAL_WEIGHT,
    DefenseStatus,
    PlanetDefenseState,
    compute_owned_defense,
)
from orbit_graph import OrbitGraph, build_orbit_graph


@dataclass(frozen=True)
class OrbitScriptedConfig:
    """Hyperparameters for :class:`OrbitScriptedAgent`."""

    max_launches: int = 8
    horizon_cap: float = DEFAULT_DEFENSE_HORIZON
    potential_weight: float = DEFAULT_POTENTIAL_WEIGHT
    strained_threshold: float = 0.0
    critical_threshold: float = -10.0

    # --- Phase 0: comet evacuation ---
    comet_evac_turns: int = 2
    """Evacuate owned comets whose remaining life is at most this many turns
    (0 disables)."""

    # --- Phase 1: defense triage ---
    max_rescue_launches: int = 3
    """Per-critical-planet cap on simultaneous rescue launches."""
    evacuate_doomed: bool = True
    """When a critical world cannot be saved and inbound enemies overwhelm it,
    relocate its garrison instead of letting the enemy destroy it."""

    # --- Phase 2: attack ---
    include_neutrals: bool = True
    include_enemies: bool = True
    include_comets: bool = False
    eta_horizon: float = 60.0
    denial_weight: float = 1.0
    influence_weight: float = 1.0
    """Weight for ship-mass influence + attack-window terms on capture targets."""
    production_pressure_weight: float = 1.0
    """Weight for production diffusion potential on neutral capture targets."""
    enemy_weakness_weight: float = 1.0
    """Scale enemy weak-spot score (fragile + exposed planets first)."""
    enemy_empire_support_weight: float = 1.0
    """Penalty for enemy reinforcement / diffusion reach on the target."""
    prioritize_enemy_weakspots: bool = True
    """Rank enemy weak spots before neutral expansion targets."""
    capture_margin: float = 0.1
    """Overkill padding on the required capture ship count (estimate buffer)."""
    max_wave_launches: int = 4
    """Cap on simultaneous launches funding a single capture wave."""
    allow_partial_enemy_waves: bool = False
    """Permit underfunded pressure waves on enemy planets (never neutrals)."""

    # --- Phase 3: backline support ---
    support_reserve_slots: int = 2
    """Launch slots held back from the attack phase while any owned planet
    has a positive support need."""
    support_threat_threshold: float = 8.0
    """Minimum realized + weighted potential threat to request support."""
    support_min_source_surplus: float = 5.0
    """Backline source must have at least this defense surplus."""
    support_max_source_threat: float = 4.0
    """Sources with higher local enemy influence are not support donors."""
    support_cushion: float = 3.0
    """Extra ships above liabilities we try to station on threatened worlds."""
    support_ship_fraction: float = 0.35
    """When surplus is thin but influence is high, cover this fraction of it."""
    max_support_launches_per_target: int = 2
    """Per-receiver cap on simultaneous support launches."""

    # --- Phase 4: consolidation ---
    consolidation_min_surplus: float = 10.0
    """Idle backline planets need at least this surplus to donate forward."""
    consolidation_fraction: float = 0.5
    """Fraction of the remaining launch budget pushed to the rally planet."""
    consolidation_min_send: int = 5
    """Skip consolidation dribbles below this ship count."""


def _env_omega(env_obs: Any) -> float:
    if isinstance(env_obs, dict):
        return float(env_obs.get("angular_velocity", 0.0) or 0.0)
    return float(getattr(env_obs, "angular_velocity", 0.0) or 0.0)


def enemy_influence(state: PlanetDefenseState, *, potential_weight: float) -> float:
    """Combined realized + potential enemy pressure on an owned planet."""
    return (
        float(state.realized_enemy_ships)
        + float(potential_weight) * float(state.potential_enemy_ships_eq)
    )


def support_need(state: PlanetDefenseState, cfg: OrbitScriptedConfig) -> int:
    """Ships we would like to add at ``state`` via friendly reinforcement."""
    gap = max(0.0, float(state.liabilities) - float(state.assets))
    need = float(math.ceil(gap)) if gap > 0 else 0.0

    infl = enemy_influence(state, potential_weight=cfg.potential_weight)
    if infl < cfg.support_threat_threshold:
        return max(0, int(math.ceil(need)))

    # Military balance (exclude production buffer): backline support when
    # garrison + inbound help cannot comfortably absorb enemy influence.
    military_assets = float(state.garrison) + float(state.friendly_inbound_ships)
    military_liabilities = float(state.realized_enemy_ships) + (
        float(cfg.potential_weight) * float(state.potential_enemy_ships_eq)
    )
    mil_gap = military_liabilities + float(cfg.support_cushion) - military_assets
    if mil_gap > 0.0:
        need = max(need, mil_gap)

    if state.status != DefenseStatus.SECURE:
        bolster = max(0.0, float(state.liabilities) + float(cfg.support_cushion) - float(state.assets))
        need = max(need, bolster)

    return max(0, int(math.ceil(need)))


def _backline_score(state: PlanetDefenseState, cfg: OrbitScriptedConfig) -> float:
    threat = enemy_influence(state, potential_weight=cfg.potential_weight)
    if float(state.defense_surplus) < float(cfg.support_min_source_surplus):
        return -math.inf
    if threat > float(cfg.support_max_source_threat):
        return -math.inf
    return float(state.defense_surplus) - 0.25 * threat


@dataclass(frozen=True)
class _ReinforcementPick:
    src_id: int
    tgt_id: int
    angle: float
    ships: int
    eta: float
    score: float


def _plan_reinforcement_wave(
    og: OrbitGraph,
    tgt_id: int,
    need: int,
    defense_states: list[PlanetDefenseState],
    remaining: dict[int, int],
    *,
    cfg: OrbitScriptedConfig,
    omega: float,
    eta_limit: float,
    relaxed_source: bool,
    max_launches: int,
    all_or_nothing: bool,
) -> list[_ReinforcementPick]:
    """Multi-source reinforcement toward ``tgt_id``.

    Every launch must land **strictly before** ``eta_limit`` (the earliest
    incoming threat) — ships arriving after the planet falls are wasted.
    With ``all_or_nothing`` the whole wave is dropped when the combined help
    cannot cover ``need`` (unsavable planet: do not feed the enemy).
    """
    if need <= 0 or max_launches <= 0:
        return []
    tgt = og.planet_by_id.get(int(tgt_id))
    if tgt is None:
        return []
    tid = int(tgt_id)

    candidates: list[tuple[float, PlanetDefenseState]] = []
    for state in defense_states:
        sid = int(state.planet_id)
        if sid == tid:
            continue
        budget = int(remaining.get(sid, 0))
        if budget <= 0:
            continue
        if relaxed_source:
            if float(state.defense_surplus) <= 0.0:
                continue
            score = float(state.defense_surplus)
        else:
            score = _backline_score(state, cfg)
            if not math.isfinite(score):
                continue
        if not og.G.has_edge(sid, tid):
            continue
        if not og.G.edges[sid, tid].get("reachable", False):
            continue
        candidates.append((score, state))
    candidates.sort(key=lambda t: t[0], reverse=True)

    picks: list[_ReinforcementPick] = []
    committed = 0
    for score, state in candidates:
        if len(picks) >= int(max_launches) or committed >= int(need):
            break
        sid = int(state.planet_id)
        budget = int(remaining.get(sid, 0))
        ships = min(budget, int(need) - committed, int(state.garrison))
        if ships <= 0:
            continue
        launch = compute_launch(og.planet_by_id[sid], tgt, ships, og.planets, omega)
        if launch is None:
            continue
        eta = float(launch["eta"])
        if eta >= float(eta_limit):
            continue
        picks.append(_ReinforcementPick(
            src_id=sid,
            tgt_id=tid,
            angle=float(launch["angle"]),
            ships=int(launch["ships"]),
            eta=eta,
            score=float(score),
        ))
        committed += int(launch["ships"])

    if all_or_nothing and committed < int(need):
        return []
    return picks


def _apply_pick(pick: _ReinforcementPick, remaining: dict[int, int]) -> list:
    remaining[pick.src_id] = int(remaining.get(pick.src_id, 0)) - int(pick.ships)
    return [pick.src_id, pick.angle, pick.ships]


# ---------------------------------------------------------------------------
# Comet / doomed-planet evacuation
# ---------------------------------------------------------------------------
def _comet_remaining_life(p: Any) -> int:
    """Turns an owned comet stays on the board (``-1`` if not a known comet)."""
    if not getattr(p, "is_comet", False):
        return -1
    path = getattr(p, "comet_path", None)
    if not path:
        return -1
    return (len(path) - 1) - int(getattr(p, "comet_path_index", -1))


def _nearest_safe_friendly(comet: Any, friendly: list, evac_turns: int):
    """Closest friendly planet that will not itself expire within the window."""
    best = None
    best_d = math.inf
    for p in friendly:
        if int(p.id) == int(comet.id):
            continue
        rem = _comet_remaining_life(p)
        if 0 <= rem <= int(evac_turns):
            continue
        d = (float(p.x) - float(comet.x)) ** 2 + (float(p.y) - float(comet.y)) ** 2
        if d < best_d:
            best_d = d
            best = p
    return best


def _comet_evacuation_actions(
    og: OrbitGraph,
    *,
    omega: float,
    evac_turns: int,
) -> tuple[list[list], set[int]]:
    """Rescue garrisons stranded on owned comets about to expire."""
    if evac_turns <= 0:
        return [], set()
    out: list[list] = []
    evacuated: set[int] = set()
    friendly = [p for p in og.planets if int(p.owner) == int(og.player)]
    for c in og.planets:
        if int(c.owner) != int(og.player):
            continue
        rem = _comet_remaining_life(c)
        if rem < 0 or rem > int(evac_turns):
            continue
        garrison = int(round(float(c.ships)))
        if garrison <= 0:
            continue
        dest = _nearest_safe_friendly(c, friendly, evac_turns)
        if dest is None:
            continue
        res = compute_launch(c, dest, garrison, og.planets, omega)
        if res is None or int(res["ships"]) <= 0:
            continue
        out.append([int(c.id), float(res["angle"]), int(res["ships"])])
        evacuated.add(int(c.id))
    return out, evacuated


def _is_doomed(state: PlanetDefenseState) -> bool:
    """Inbound enemies overwhelm everything the planet can muster locally."""
    if state.min_incoming_eta is None:
        return False
    local = (
        float(state.garrison)
        + float(state.production_buffer)
        + float(state.friendly_inbound_ships)
    )
    return float(state.realized_enemy_ships) > local


def _evacuate_garrison(
    og: OrbitGraph,
    state: PlanetDefenseState,
    defense_states: list[PlanetDefenseState],
    *,
    omega: float,
) -> Optional[list]:
    """Relocate a doomed planet's garrison to the nearest viable own planet."""
    src = og.planet_by_id.get(int(state.planet_id))
    if src is None or int(src.ships) <= 0:
        return None
    by_id = {int(s.planet_id): s for s in defense_states}
    best = None
    best_d = math.inf
    for p in og.planets:
        pid = int(p.id)
        if p.owner != og.player or pid == int(src.id) or p.is_comet:
            continue
        st = by_id.get(pid)
        if st is not None and st.status == DefenseStatus.CRITICAL:
            continue
        if not og.G.has_edge(int(src.id), pid):
            continue
        if not og.G.edges[int(src.id), pid].get("reachable", False):
            continue
        d = (float(p.x) - float(src.x)) ** 2 + (float(p.y) - float(src.y)) ** 2
        if d < best_d:
            best_d = d
            best = p
    if best is None:
        return None
    launch = compute_launch(src, best, int(src.ships), og.planets, omega)
    if launch is None or int(launch["ships"]) <= 0:
        return None
    return [int(src.id), float(launch["angle"]), int(launch["ships"])]


class OrbitScriptedAgent:
    """Evacuate → defend → attack → support → consolidate (orbit graph)."""

    def __init__(self, cfg: OrbitScriptedConfig | None = None):
        self.cfg = cfg or OrbitScriptedConfig()

    def act(
        self,
        env_obs: Any,
        player: int,
        *,
        current_turn: int = 0,
        max_turns: int = 500,
    ) -> list[list]:
        """Return env actions ``[[from_id, angle, ships], ...]``."""
        cfg = self.cfg
        og = build_orbit_graph(env_obs, player=player)
        if not any(p.owner == player and not p.is_comet for p in og.planets):
            return []

        attack_cfg = attack_config_from_env(
            env_obs,
            current_turn=int(current_turn),
            max_turns=int(max_turns),
            eta_horizon=float(cfg.eta_horizon),
            denial_weight=float(cfg.denial_weight),
            include_neutrals=cfg.include_neutrals,
            include_enemies=cfg.include_enemies,
            include_comets=cfg.include_comets,
        )
        omega = float(attack_cfg.angular_velocity)
        defense_states = compute_owned_defense(
            og,
            horizon_cap=cfg.horizon_cap,
            potential_weight=cfg.potential_weight,
        )
        plan = plan_attack_targets(
            og,
            cfg=attack_cfg,
            horizon_cap=cfg.horizon_cap,
            potential_weight=cfg.potential_weight,
        )
        remaining = dict(plan.launch_budgets)
        used_sources: set[int] = set()
        actions: list[list] = []

        # ------------------------------------------------------------------
        # Phase 0 — comet evacuation (garrison dies with the comet otherwise).
        # ------------------------------------------------------------------
        evac_actions, evacuated = _comet_evacuation_actions(
            og, omega=omega, evac_turns=int(cfg.comet_evac_turns),
        )
        for action in evac_actions:
            if len(actions) >= cfg.max_launches:
                break
            actions.append(action)
            remaining[int(action[0])] = 0

        # ------------------------------------------------------------------
        # Phase 1 — defense triage: save what is savable, abandon the rest.
        # ------------------------------------------------------------------
        critical = [
            s for s in defense_states
            if s.status == DefenseStatus.CRITICAL
            and int(s.planet_id) not in evacuated
        ]
        critical.sort(key=lambda s: s.defense_surplus)
        for state in critical:
            slots = int(cfg.max_launches) - len(actions)
            if slots <= 0:
                break
            need = max(1, int(math.ceil(-float(state.defense_surplus))))
            eta_limit = (
                float(state.min_incoming_eta)
                if state.min_incoming_eta is not None
                else math.inf
            )
            picks = _plan_reinforcement_wave(
                og,
                state.planet_id,
                need,
                defense_states,
                remaining,
                cfg=cfg,
                omega=omega,
                eta_limit=eta_limit,
                relaxed_source=True,
                max_launches=min(slots, int(cfg.max_rescue_launches)),
                all_or_nothing=True,
            )
            if picks:
                for pick in picks:
                    actions.append(_apply_pick(pick, remaining))
                continue
            # Unsavable.  If truly doomed, deny the enemy the kill.
            if cfg.evacuate_doomed and _is_doomed(state):
                evac = _evacuate_garrison(og, state, defense_states, omega=omega)
                if evac is not None:
                    actions.append(evac)
                    remaining[int(state.planet_id)] = 0

        # ------------------------------------------------------------------
        # Phase 2 — attacks, holding back slots for support when needed.
        # ------------------------------------------------------------------
        support_receivers = [s for s in defense_states if support_need(s, cfg) > 0]
        support_receivers.sort(
            key=lambda s: (
                enemy_influence(s, potential_weight=cfg.potential_weight),
                -float(s.defense_surplus),
            ),
            reverse=True,
        )
        reserve = min(int(cfg.support_reserve_slots), len(support_receivers))
        attack_slots = max(0, int(cfg.max_launches) - len(actions) - reserve)
        if attack_slots > 0:
            actions.extend(
                allocate_greedy_captures(
                    og,
                    plan,
                    remaining,
                    used_sources,
                    attack_cfg=attack_cfg,
                    influence_weight=float(cfg.influence_weight),
                    production_pressure_weight=float(cfg.production_pressure_weight),
                    denial_weight=float(cfg.denial_weight),
                    weakness_weight=float(cfg.enemy_weakness_weight),
                    empire_support_weight=float(cfg.enemy_empire_support_weight),
                    prioritize_enemy_weakspots=bool(cfg.prioritize_enemy_weakspots),
                    eta_horizon=float(cfg.eta_horizon),
                    max_launches=attack_slots,
                    capture_margin=float(cfg.capture_margin),
                    max_wave_launches=int(cfg.max_wave_launches),
                    allow_partial_enemy_waves=bool(cfg.allow_partial_enemy_waves),
                )
            )

        # ------------------------------------------------------------------
        # Phase 3 — backline support toward high enemy-influence frontline.
        # ------------------------------------------------------------------
        for state in support_receivers:
            slots = int(cfg.max_launches) - len(actions)
            if slots <= 0:
                break
            if int(state.planet_id) in evacuated:
                continue
            need = support_need(state, cfg)
            eta_limit = (
                float(state.min_incoming_eta)
                if state.min_incoming_eta is not None
                else math.inf
            )
            picks = _plan_reinforcement_wave(
                og,
                state.planet_id,
                need,
                defense_states,
                remaining,
                cfg=cfg,
                omega=omega,
                eta_limit=eta_limit,
                relaxed_source=False,
                max_launches=min(slots, int(cfg.max_support_launches_per_target)),
                all_or_nothing=False,
            )
            for pick in picks:
                actions.append(_apply_pick(pick, remaining))

        # ------------------------------------------------------------------
        # Phase 4 — consolidation: push idle backline surplus to the front.
        # ------------------------------------------------------------------
        slots = int(cfg.max_launches) - len(actions)
        if slots > 0:
            rally = None
            rally_infl = 0.0
            for s in defense_states:
                if s.status == DefenseStatus.CRITICAL:
                    continue
                if int(s.planet_id) in evacuated:
                    continue
                infl = enemy_influence(s, potential_weight=cfg.potential_weight)
                if infl > rally_infl:
                    rally_infl = infl
                    rally = s
            if rally is not None and rally_infl > 0.0:
                rid = int(rally.planet_id)
                donors = [
                    s for s in defense_states
                    if int(s.planet_id) != rid
                    and s.status == DefenseStatus.SECURE
                    and float(s.defense_surplus) >= float(cfg.consolidation_min_surplus)
                    and enemy_influence(s, potential_weight=cfg.potential_weight) < rally_infl
                ]
                donors.sort(key=lambda s: -float(s.defense_surplus))
                tgt = og.planet_by_id.get(rid)
                for s in donors:
                    if slots <= 0 or tgt is None:
                        break
                    sid = int(s.planet_id)
                    budget = int(remaining.get(sid, 0))
                    ships = int(math.floor(float(budget) * float(cfg.consolidation_fraction)))
                    if ships < int(cfg.consolidation_min_send):
                        continue
                    if not og.G.has_edge(sid, rid):
                        continue
                    if not og.G.edges[sid, rid].get("reachable", False):
                        continue
                    launch = compute_launch(
                        og.planet_by_id[sid], tgt, ships, og.planets, omega,
                    )
                    if launch is None:
                        continue
                    actions.append([sid, float(launch["angle"]), int(launch["ships"])])
                    remaining[sid] = budget - int(launch["ships"])
                    slots -= 1

        return actions


__all__ = [
    "OrbitScriptedAgent",
    "OrbitScriptedConfig",
    "enemy_influence",
    "support_need",
]
