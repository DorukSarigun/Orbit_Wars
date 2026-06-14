"""Production-potential capture tree search.

This module is intentionally standalone: it does not change the existing
agents, allocator, reward code, or training loop.  It is a first-principles
planner for the strategic question:

    "Which planets should I capture, in which order, to maximise production
    potential until the end of the game?"

The tree is event-based:

* each node stores per-planet owner/ship state,
* a launch subtracts ships from the source,
* ``compute_launch`` validates the physical route and provides ETA/path data,
* time advances to fleet arrival while owned planets produce,
* combat is resolved at arrival,
* capture value is credited only after a successful arrival.

This is still a macro planner.  It starts with single-source captures; exact
multi-source synchronisation is deliberately left for the later logistics /
simplex layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional

from feature_extraction import Planet, parse_planets
from navigation import compute_launch


@dataclass(frozen=True)
class ProductionTreeConfig:
    """Knobs for the capture-order search."""

    max_turns: int = 500
    max_depth: int = 4
    beam_width: int = 32
    eta_horizon: float = 80.0
    capture_margin: int = 1
    include_neutrals: bool = True
    include_enemies: bool = True
    include_comets: bool = False
    gamma: float = 1.0
    """Production discount. 1.0 means raw production * remaining turns."""


@dataclass(frozen=True)
class PlanetSimState:
    """Mutable-in-the-tree economic state for one planet."""

    owner: int
    ships: float
    production: int


@dataclass(frozen=True)
class FlightPath:
    """Physical route selected by ``compute_launch`` for one planned fleet."""

    src_id: int
    tgt_id: int
    ships: int
    launch_turn: float
    arrival_turn: float
    eta: float
    angle: float
    speed: float
    start_x: float
    start_y: float
    intercept_x: float
    intercept_y: float


@dataclass(frozen=True)
class CaptureOption:
    """One possible next capture from a tree node."""

    target_id: int
    source_id: int
    owner_before: int
    production: int
    eta: float
    wait_turns: float
    launch_turn: float
    capture_turn: float
    ships_required: int
    source_ships_before: float
    source_ships_after_launch: float
    target_ships_before_combat: float
    target_ships_after_capture: float
    production_value: float
    flight: FlightPath

    @property
    def kind(self) -> str:
        if self.owner_before == -1:
            return "neutral"
        return "enemy"

    @property
    def surplus_after_capture(self) -> float:
        """Backward-compatible alias for target garrison after capture."""
        return self.target_ships_after_capture


@dataclass(frozen=True)
class CaptureStep:
    """A committed edge in the selected capture sequence."""

    target_id: int
    source_id: int
    kind: str
    production: int
    launch_turn: float
    capture_turn: float
    eta: float
    wait_turns: float
    ships_required: int
    source_ships_before: float
    source_ships_after_launch: float
    target_ships_before_combat: float
    target_ships_after_capture: float
    production_value: float
    flight: FlightPath


@dataclass(frozen=True)
class CaptureTreeNode:
    """Hypothetical state in the production capture tree."""

    owned_ids: frozenset[int]
    turn: float
    ship_stock: float
    production_rate: float
    captured_value: float
    planet_states: dict[int, PlanetSimState] = field(default_factory=dict)
    in_flight: tuple[FlightPath, ...] = field(default_factory=tuple)
    steps: tuple[CaptureStep, ...] = field(default_factory=tuple)

    @property
    def depth(self) -> int:
        return len(self.steps)


def production_value_until_end(
    production: int,
    *,
    capture_turn: float,
    max_turns: int,
    gamma: float = 1.0,
) -> float:
    """Production potential earned by owning a planet after capture."""
    hold_turns = max(0.0, float(max_turns) - float(capture_turn))
    prod = float(max(0, int(production)))
    if hold_turns <= 0.0 or prod <= 0.0:
        return 0.0
    if abs(float(gamma) - 1.0) < 1e-9:
        return prod * hold_turns
    n = int(math.floor(hold_turns))
    if n <= 0:
        return 0.0
    return prod * (1.0 - float(gamma) ** n) / (1.0 - float(gamma))


def _obs_get(obs: Any, key: str, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _current_turn(obs: Any, explicit: Optional[int]) -> int:
    if explicit is not None:
        return int(explicit)
    return int(_obs_get(obs, "step", 0) or 0)


def _target_candidates(
    planets: Iterable[Planet],
    *,
    player: int,
    cfg: ProductionTreeConfig,
) -> list[Planet]:
    out: list[Planet] = []
    for p in planets:
        if int(p.owner) == int(player):
            continue
        if p.is_comet and not cfg.include_comets:
            continue
        if p.owner == -1 and not cfg.include_neutrals:
            continue
        if p.is_enemy(player) and not cfg.include_enemies:
            continue
        out.append(p)
    return out


def _initial_planet_states(planets: Iterable[Planet]) -> dict[int, PlanetSimState]:
    return {
        int(p.id): PlanetSimState(
            owner=int(p.owner),
            ships=float(max(0, int(p.ships))),
            production=int(p.production),
        )
        for p in planets
    }


def _advance_states(
    states: dict[int, PlanetSimState],
    dt: float,
) -> dict[int, PlanetSimState]:
    """Advance economic state by ``dt`` turns."""
    dt = max(0.0, float(dt))
    out: dict[int, PlanetSimState] = {}
    for pid, st in states.items():
        ships = float(st.ships)
        if int(st.owner) != -1:
            ships += float(st.production) * dt
        out[int(pid)] = PlanetSimState(
            owner=int(st.owner),
            ships=max(0.0, ships),
            production=int(st.production),
        )
    return out


def _owned_ids(states: dict[int, PlanetSimState], player: int) -> frozenset[int]:
    return frozenset(int(pid) for pid, st in states.items() if int(st.owner) == int(player))


def _ship_stock(
    states: dict[int, PlanetSimState],
    player: int,
    in_flight: tuple[FlightPath, ...] = (),
) -> float:
    planet_ships = sum(float(st.ships) for st in states.values() if int(st.owner) == int(player))
    fleet_ships = sum(float(f.ships) for f in in_flight)
    return float(planet_ships + fleet_ships)


def _production_rate(states: dict[int, PlanetSimState], player: int) -> float:
    return float(sum(float(st.production) for st in states.values() if int(st.owner) == int(player)))


def _materialize_planets(
    base_planets: list[Planet],
    states: dict[int, PlanetSimState],
    *,
    elapsed_from_root: float,
    angular_velocity: float,
) -> list[Planet]:
    """Build Planet objects at a future planning time for navigation."""
    out: list[Planet] = []
    for base in base_planets:
        st = states[int(base.id)]
        x, y = base.position_at(float(elapsed_from_root), angular_velocity=angular_velocity)
        dx = float(x) - 50.0
        dy = float(y) - 50.0
        comet_idx = base.comet_path_index
        if base.is_comet and comet_idx >= 0:
            comet_idx = comet_idx + float(elapsed_from_root)  # type: ignore[assignment]
        out.append(
            replace(
                base,
                owner=int(st.owner),
                ships=max(0, int(math.floor(float(st.ships)))),
                production=int(st.production),
                x=float(x),
                y=float(y),
                r_sun=math.hypot(dx, dy),
                theta=math.atan2(dy, dx),
                comet_path_index=comet_idx,
            )
        )
    return out


def _by_id(planets: Iterable[Planet]) -> dict[int, Planet]:
    return {int(p.id): p for p in planets}


def _wait_for_source_ships(src_state: PlanetSimState, ships_required: int) -> Optional[float]:
    if float(src_state.ships) >= float(ships_required):
        return 0.0
    if int(src_state.owner) == -1 or int(src_state.production) <= 0:
        return None
    return max(
        0.0,
        (float(ships_required) - float(src_state.ships)) / float(src_state.production),
    )


def _flight_path_from_launch(
    src: Planet,
    tgt: Planet,
    *,
    source_id: int,
    target_id: int,
    ships: int,
    launch_turn: float,
    launch: dict,
) -> FlightPath:
    angle = float(launch["angle"])
    start_x = float(src.x) + math.cos(angle) * (float(src.radius) + 0.1)
    start_y = float(src.y) + math.sin(angle) * (float(src.radius) + 0.1)
    eta = float(launch["eta"])
    return FlightPath(
        src_id=int(source_id),
        tgt_id=int(target_id),
        ships=int(ships),
        launch_turn=float(launch_turn),
        arrival_turn=float(launch_turn) + eta,
        eta=eta,
        angle=angle,
        speed=float(launch.get("speed", 0.0)),
        start_x=start_x,
        start_y=start_y,
        intercept_x=float(launch["intercept_x"]),
        intercept_y=float(launch["intercept_y"]),
    )


def estimate_capture_option(
    node: CaptureTreeNode,
    *,
    target: Planet,
    planets_by_id: dict[int, Planet],
    planets: list[Planet],
    angular_velocity: float,
    cfg: ProductionTreeConfig,
    player: Optional[int] = None,
    root_turn: float = 0.0,
) -> Optional[CaptureOption]:
    """Estimate one single-source capture from ``node``.

    The estimate is event-aware: source ships are checked per planet, all
    planets produce during wait/flight, and the target is evaluated at arrival.
    """
    if player is None:
        # Backward-compatible inference: all node-owned ids are assumed ours.
        source_ids = sorted(node.owned_ids)
        player_id = None
    else:
        source_ids = sorted(_owned_ids(node.planet_states, int(player)))
        player_id = int(player)

    target_id = int(target.id)
    target_state_now = node.planet_states.get(target_id)
    if target_state_now is None:
        return None
    if player_id is not None and int(target_state_now.owner) == player_id:
        return None

    best: Optional[CaptureOption] = None
    for sid in source_ids:
        src_state_now = node.planet_states.get(int(sid))
        if src_state_now is None:
            continue
        if player_id is not None and int(src_state_now.owner) != player_id:
            continue

        ships_required = max(1, int(math.floor(float(target_state_now.ships))) + int(cfg.capture_margin))
        opt: Optional[CaptureOption] = None

        for _ in range(10):
            wait = _wait_for_source_ships(src_state_now, ships_required)
            if wait is None:
                opt = None
                break

            launch_turn = float(node.turn) + float(wait)
            states_at_launch = _advance_states(node.planet_states, wait)
            source_at_launch = states_at_launch[int(sid)]
            if float(source_at_launch.ships) + 1e-6 < float(ships_required):
                opt = None
                break

            nav_planets = _materialize_planets(
                planets,
                states_at_launch,
                elapsed_from_root=float(launch_turn) - float(root_turn),
                angular_velocity=angular_velocity,
            )
            nav_by_id = _by_id(nav_planets)
            src = nav_by_id.get(int(sid))
            tgt = nav_by_id.get(target_id)
            if src is None or tgt is None:
                opt = None
                break

            launch = compute_launch(
                src,
                tgt,
                int(ships_required),
                nav_planets,
                angular_velocity,
            )
            if launch is None:
                opt = None
                break

            eta = float(launch["eta"])
            if eta > float(cfg.eta_horizon):
                opt = None
                break

            states_after_launch = dict(states_at_launch)
            states_after_launch[int(sid)] = PlanetSimState(
                owner=int(source_at_launch.owner),
                ships=max(0.0, float(source_at_launch.ships) - float(ships_required)),
                production=int(source_at_launch.production),
            )
            states_at_arrival = _advance_states(states_after_launch, eta)
            target_at_arrival = states_at_arrival[target_id]
            target_ships = float(target_at_arrival.ships)
            needed_next = int(math.floor(target_ships)) + int(cfg.capture_margin)
            needed_next = max(1, needed_next)
            capture_turn = launch_turn + eta

            if capture_turn >= float(cfg.max_turns):
                opt = None
                break

            if needed_next != int(ships_required):
                ships_required = needed_next
                continue

            surplus = float(ships_required) - target_ships
            if surplus <= 0.0:
                ships_required += 1
                continue

            value = production_value_until_end(
                int(target.production),
                capture_turn=capture_turn,
                max_turns=int(cfg.max_turns),
                gamma=float(cfg.gamma),
            )
            if value <= 0.0:
                opt = None
                break

            flight = _flight_path_from_launch(
                src,
                tgt,
                source_id=int(sid),
                target_id=target_id,
                ships=int(ships_required),
                launch_turn=launch_turn,
                launch=launch,
            )
            opt = CaptureOption(
                target_id=target_id,
                source_id=int(sid),
                owner_before=int(target_state_now.owner),
                production=int(target.production),
                eta=eta,
                wait_turns=float(wait),
                launch_turn=float(launch_turn),
                capture_turn=float(capture_turn),
                ships_required=int(ships_required),
                source_ships_before=float(source_at_launch.ships),
                source_ships_after_launch=max(
                    0.0,
                    float(source_at_launch.ships) - float(ships_required),
                ),
                target_ships_before_combat=target_ships,
                target_ships_after_capture=surplus,
                production_value=float(value),
                flight=flight,
            )
            break

        if opt is None:
            continue
        key = (
            opt.production_value,
            -opt.capture_turn,
            -opt.ships_required,
            opt.target_ships_after_capture,
        )
        if best is None:
            best = opt
            continue
        best_key = (
            best.production_value,
            -best.capture_turn,
            -best.ships_required,
            best.target_ships_after_capture,
        )
        if key > best_key:
            best = opt
    return best


def _apply_capture(
    node: CaptureTreeNode,
    opt: CaptureOption,
    *,
    player: int,
) -> CaptureTreeNode:
    """Apply one planned launch through arrival combat."""
    states = _advance_states(node.planet_states, opt.wait_turns)
    src_state = states[int(opt.source_id)]
    states[int(opt.source_id)] = PlanetSimState(
        owner=int(src_state.owner),
        ships=max(0.0, float(src_state.ships) - float(opt.ships_required)),
        production=int(src_state.production),
    )
    in_flight = node.in_flight + (opt.flight,)
    states = _advance_states(states, opt.eta)
    tgt_state = states[int(opt.target_id)]
    target_ships = float(tgt_state.ships)
    surplus = max(0.0, float(opt.ships_required) - target_ships)
    states[int(opt.target_id)] = PlanetSimState(
        owner=int(player),
        ships=surplus,
        production=int(tgt_state.production),
    )
    step = CaptureStep(
        target_id=int(opt.target_id),
        source_id=int(opt.source_id),
        kind=opt.kind,
        production=int(opt.production),
        launch_turn=float(opt.launch_turn),
        capture_turn=float(opt.capture_turn),
        eta=float(opt.eta),
        wait_turns=float(opt.wait_turns),
        ships_required=int(opt.ships_required),
        source_ships_before=float(opt.source_ships_before),
        source_ships_after_launch=float(opt.source_ships_after_launch),
        target_ships_before_combat=float(target_ships),
        target_ships_after_capture=float(surplus),
        production_value=float(opt.production_value),
        flight=opt.flight,
    )
    return CaptureTreeNode(
        owned_ids=_owned_ids(states, player),
        turn=float(opt.capture_turn),
        ship_stock=_ship_stock(states, player),
        production_rate=_production_rate(states, player),
        captured_value=float(node.captured_value) + float(opt.production_value),
        planet_states=states,
        in_flight=tuple(f for f in in_flight if f.arrival_turn > opt.capture_turn + 1e-6),
        steps=node.steps + (step,),
    )


def plan_capture_tree(
    env_obs: Any,
    player: int,
    *,
    current_turn: Optional[int] = None,
    cfg: ProductionTreeConfig | None = None,
) -> CaptureTreeNode:
    """Return the best capture sequence by production potential.

    The search is beam search over target sequences.  It is deterministic and
    cheap enough to run as a planning primitive before full MCTS.
    """
    cfg = cfg or ProductionTreeConfig()
    turn = _current_turn(env_obs, current_turn)
    angular_velocity = float(_obs_get(env_obs, "angular_velocity", 0.0) or 0.0)
    planets = parse_planets(env_obs)
    states = _initial_planet_states(planets)
    owned_ids = _owned_ids(states, player)
    root = CaptureTreeNode(
        owned_ids=owned_ids,
        turn=float(turn),
        ship_stock=_ship_stock(states, player),
        production_rate=_production_rate(states, player),
        captured_value=0.0,
        planet_states=states,
    )
    if not owned_ids:
        return root

    targets = _target_candidates(planets, player=player, cfg=cfg)
    if not targets:
        return root

    planets_by_id = {int(p.id): p for p in planets}
    frontier = [root]
    best = root
    for _depth in range(int(cfg.max_depth)):
        expanded: list[CaptureTreeNode] = []
        for node in frontier:
            for target in targets:
                state = node.planet_states.get(int(target.id))
                if state is None or int(state.owner) == int(player):
                    continue
                opt = estimate_capture_option(
                    node,
                    target=target,
                    planets_by_id=planets_by_id,
                    planets=planets,
                    angular_velocity=angular_velocity,
                    cfg=cfg,
                    player=player,
                    root_turn=float(turn),
                )
                if opt is None:
                    continue
                expanded.append(_apply_capture(node, opt, player=player))
        if not expanded:
            break
        expanded.sort(
            key=lambda n: (
                n.captured_value,
                n.production_rate,
                n.ship_stock,
                -n.turn,
                -len(n.steps),
            ),
            reverse=True,
        )
        frontier = expanded[: max(1, int(cfg.beam_width))]
        if frontier[0].captured_value > best.captured_value:
            best = frontier[0]
    return best


def format_capture_plan(node: CaptureTreeNode) -> str:
    """Compact human-readable capture sequence."""
    if not node.steps:
        return "no feasible captures"
    parts = []
    for i, step in enumerate(node.steps, start=1):
        parts.append(
            f"{i}. P{step.target_id}({step.kind}, prod={step.production}) "
            f"via P{step.source_id}: launch={step.launch_turn:.1f}, "
            f"arrive={step.capture_turn:.1f}, ships={step.ships_required}, "
            f"value={step.production_value:.1f}"
        )
    parts.append(
        f"total_value={node.captured_value:.1f}, "
        f"ship_stock={node.ship_stock:.1f}, production_rate={node.production_rate:.1f}"
    )
    return "\n".join(parts)


__all__ = [
    "CaptureOption",
    "CaptureStep",
    "CaptureTreeNode",
    "FlightPath",
    "PlanetSimState",
    "ProductionTreeConfig",
    "estimate_capture_option",
    "format_capture_plan",
    "plan_capture_tree",
    "production_value_until_end",
]
