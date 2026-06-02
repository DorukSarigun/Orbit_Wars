"""Scripted ship allocator for the hierarchical reworked policy.

The RL policy (:class:`reworked_model.ReworkedHierarchicalPolicy`) decides
*which* planets to attack / defend and *in what priority order* (a
Plackett-Luce ranking).  This module is the **logistics** layer: given those
target demands and the current board, it routes ships from the agent's own
planets to fulfil the demands, using the same ETA / ``fleet_speed`` physics as
:mod:`reworked_features`.

Two solvers, one interface
--------------------------
:func:`allocate_launches` exposes a single interface with a ``solver`` switch:

* ``"greedy"`` -- the original **greedy-by-priority** fill: each demand, in the
  RL priority order, draws from its nearest (lowest-ETA) sources first.  Simple,
  dependency-free, and near-optimal *for a fixed priority order*, but it commits
  sources myopically and can starve a lower-priority target of the only source
  it could have used.

* ``"mcf"`` (default) -- a **min-cost-flow** assignment solved as a
  transportation LP (:func:`scipy.optimize.linprog`, ``method="highs"``).  It
  honours the RL priority order *lexicographically* (a high-priority target is
  always filled before any lower one) **and** picks the globally cheapest
  source->target assignment, so it can route a high-priority target via a
  slightly farther source to free a nearer source for a lower-priority target --
  the global improvement greedy cannot see.  Falls back to ``"greedy"`` if SciPy
  is unavailable.

Both solvers share identical demand / surplus sizing and the same
``attack_require_full`` semantics, so they are drop-in comparable.

Self-contained: depends only on :mod:`reworked_features` (for ``fleet_speed``
and the shared defaults) plus SciPy for the MCF path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    from .reworked_features import (
        fleet_speed,
        DEFAULT_BOARD_SIZE,
        DEFAULT_MAX_SHIP_SPEED,
    )
except ImportError:  # pragma: no cover - flat layout / direct execution
    from reworked_features import (  # type: ignore
        fleet_speed,
        DEFAULT_BOARD_SIZE,
        DEFAULT_MAX_SHIP_SPEED,
    )


# Priority-weight scale for the MCF objective.  This ``priority_weight`` is an
# LP objective coefficient -- the *value* of funding a target -- NOT an RL
# reward signal.  ``W_PRIORITY`` must dominate the largest possible transport
# cost (ETA in turns, at most a few hundred) so that a single unit of a
# higher-priority target always outweighs any ETA saving on a lower-priority one
# -> strict lexicographic priority.  ``W_BASE`` keeps every
# (priority_weight - cost) strictly positive so all feasible flow is profitable.
_W_PRIORITY = 1.0e6
_W_BASE = 1.0e6


@dataclass(frozen=True)
class Launch:
    """A single fleet dispatch produced by the allocator."""

    src: int      # source planet slot (agent-owned)
    tgt: int      # target planet slot
    ships: int    # ship count to send
    kind: str     # "attack" or "defend" (the demand this launch serves)


def _eta_matrix_from_surplus(
    distance_matrix: np.ndarray,
    surplus: np.ndarray,
    *,
    board_size: float,
    max_speed: float,
) -> np.ndarray:
    """ETA (turns) per ordered pair, using each source's surplus as fleet size.

    ``eta[h, T] = raw_distance(h, T) / fleet_speed(surplus_h)``.  Sources with
    no surplus get ``+inf`` ETA (they cannot launch).  This resolves the
    circular "speed depends on fleet size" dependency by assuming a source
    would send (up to) its whole surplus.
    """
    raw = np.asarray(distance_matrix, dtype=np.float64) * float(board_size)
    speed = np.asarray(fleet_speed(surplus, max_speed=max_speed), dtype=np.float64)  # (n,)
    speed_col = speed[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = raw / speed_col
    eta[speed_col.repeat(raw.shape[1], axis=1) <= 0.0] = np.inf
    return eta


def _compute_surplus(
    owner: np.ndarray,
    ships: np.ndarray,
    *,
    player: int,
    commit_fraction: Optional[np.ndarray],
    source_reserve_frac: float,
    n: int,
) -> "tuple[np.ndarray, np.ndarray]":
    """Per-planet launchable surplus (RL commit fraction or reserve fallback)."""
    is_mine = owner == int(player)
    if commit_fraction is not None:
        frac = np.clip(np.asarray(commit_fraction, dtype=np.float64).reshape(n), 0.0, 1.0)
        surplus = np.where(is_mine, np.floor(frac * ships), 0.0).astype(np.int64)
    else:
        reserve = np.ceil(np.clip(source_reserve_frac, 0.0, 1.0) * ships)
        surplus = np.where(is_mine, np.maximum(0.0, ships - reserve), 0.0).astype(np.int64)
    return is_mine, surplus


def _target_demand(
    T: int,
    is_attack: bool,
    *,
    ships: np.ndarray,
    production: np.ndarray,
    eta: np.ndarray,
    is_mine: np.ndarray,
    surplus: np.ndarray,
    incoming_enemy: Optional[np.ndarray],
    n: int,
    capture_margin: float,
    min_send: int,
) -> "Optional[tuple[float, float]]":
    """Demand (ships) + sizing ETA for one target, or ``None`` if infeasible.

    Sizing ETA is the lowest ETA among candidate sources (mirrors the greedy
    sizing), used because the defender keeps producing while the fleet flies.
    """
    cand_etas = [eta[h, T] for h in range(n)
                 if h != T and is_mine[h] and surplus[h] >= min_send]
    cand_etas = [e for e in cand_etas if math.isfinite(e)]
    if not cand_etas:
        return None
    eta_est = float(min(cand_etas))

    if is_attack:
        need = float(ships[T]) + float(production[T]) * eta_est + float(capture_margin)
    else:
        if incoming_enemy is None:
            return None
        inc = float(np.asarray(incoming_enemy).reshape(n)[T])
        need = inc - float(ships[T]) - float(production[T]) * eta_est
        need = max(0.0, need)
        if need <= 0.0:
            return None
    return need, eta_est


def _collect_active_targets(
    priority_order: Sequence[int],
    attack_set: set,
    defend_set: set,
    *,
    ships: np.ndarray,
    production: np.ndarray,
    eta: np.ndarray,
    is_mine: np.ndarray,
    surplus: np.ndarray,
    incoming_enemy: Optional[np.ndarray],
    n: int,
    capture_margin: float,
    min_send: int,
) -> list:
    """Feasible targets in priority order with their demand + rank.

    Each entry: ``{T, is_attack, need, need_cap, pos}`` where ``pos`` is the
    rank in ``priority_order`` (0 = highest priority) and ``need_cap`` is the
    integer demand ceiling (so the demand polytope has integral RHS).
    """
    active: list = []
    seen: set = set()
    for pos, raw_t in enumerate(priority_order):
        T = int(raw_t)
        if T in seen:
            continue
        seen.add(T)
        is_attack = T in attack_set
        if not is_attack and T not in defend_set:
            continue
        dem = _target_demand(
            T, is_attack,
            ships=ships, production=production, eta=eta,
            is_mine=is_mine, surplus=surplus, incoming_enemy=incoming_enemy,
            n=n, capture_margin=capture_margin, min_send=min_send,
        )
        if dem is None:
            continue
        need, _eta_est = dem
        need_cap = int(math.ceil(need - 1e-9))
        if need_cap < max(1, int(min_send)):
            continue
        active.append({"T": T, "is_attack": is_attack, "need": need,
                       "need_cap": need_cap, "pos": pos})
    return active


# ---------------------------------------------------------------------------
# Greedy solver (original behaviour)
# ---------------------------------------------------------------------------
def _allocate_greedy(
    distance_matrix: np.ndarray,
    ships: np.ndarray,
    production: np.ndarray,
    *,
    n: int,
    is_mine: np.ndarray,
    surplus: np.ndarray,
    priority_order: Sequence[int],
    attack_set: set,
    defend_set: set,
    incoming_enemy: Optional[np.ndarray],
    board_size: float,
    max_speed: float,
    capture_margin: float,
    attack_require_full: bool,
    min_send: int,
    cost_matrix: Optional[np.ndarray] = None,
) -> list[Launch]:
    surplus = surplus.copy()
    launches: list[Launch] = []

    for raw_t in priority_order:
        T = int(raw_t)
        if T not in attack_set and T not in defend_set:
            continue
        is_attack = T in attack_set

        eta = _eta_matrix_from_surplus(
            distance_matrix, surplus, board_size=board_size, max_speed=max_speed
        )
        cand = [h for h in range(n) if h != T and is_mine[h] and surplus[h] >= min_send]
        if not cand:
            continue
        # Demand sizing uses the *physical* nearest-source ETA (production
        # accrues over real travel time); the gather order additionally honours
        # the routing-preference cost_matrix (e.g. avoid stripping a threatened
        # source) as a tie-breaker.
        eta_est = float(min(eta[h, T] for h in cand))
        if not math.isfinite(eta_est):
            continue

        if cost_matrix is not None:
            cand.sort(key=lambda h, _T=T: float(eta[h, _T]) + float(cost_matrix[h, _T]))
        else:
            cand.sort(key=lambda h, _T=T: float(eta[h, _T]))

        if is_attack:
            need = ships[T] + production[T] * eta_est + float(capture_margin)
        else:
            if incoming_enemy is None:
                continue
            inc = float(np.asarray(incoming_enemy).reshape(n)[T])
            need = inc - ships[T] - production[T] * eta_est
            need = max(0.0, need)
            if need <= 0.0:
                continue

        plan: list[tuple[int, int]] = []
        total = 0
        for h in cand:
            if total >= need:
                break
            want = max(min_send, int(math.ceil(need - total)))
            send = int(min(surplus[h], want))
            if send < min_send:
                continue
            plan.append((h, send))
            total += send

        if total <= 0:
            continue
        if is_attack and attack_require_full and total < need:
            continue

        kind = "attack" if is_attack else "defend"
        for h, send in plan:
            launches.append(Launch(src=h, tgt=T, ships=int(send), kind=kind))
            surplus[h] -= send

    return launches


# ---------------------------------------------------------------------------
# Min-cost-flow solver
# ---------------------------------------------------------------------------
def _solve_transport_lp(
    active: list,
    *,
    n: int,
    is_mine: np.ndarray,
    surplus: np.ndarray,
    eta: np.ndarray,
    n_priority: int,
    min_send: int,
    cost_matrix: Optional[np.ndarray] = None,
) -> "dict[tuple[int, int], int]":
    """One min-cost-flow assignment over the given active targets.

    Models a transportation problem:
        minimise   Σ_e (eta[h,T] + cost_matrix[h,T] - priority_weight[T]) · x_e
        s.t.       Σ_T x[h,T] <= surplus[h]   (supply)
                   Σ_h x[h,T] <= need_cap[T]  (demand ceiling)
                   0 <= x_e <= min(surplus[h], need_cap[T])
    where ``priority_weight[T] = W_BASE + W_PRIORITY·(n_priority - pos[T])`` is
    the LP value of funding target ``T`` (an objective coefficient, not an RL
    reward) and the optional ``cost_matrix[h,T]`` is an additive routing
    preference (e.g. penalise stripping a threatened source, discount interior
    bases).  Because the constraint matrix is the bipartite incidence matrix
    (totally unimodular) and all bounds / RHS are integral, the LP optimum is
    integral; we round to guard against floating point.
    """
    from scipy.optimize import linprog

    edges: list[tuple[int, int]] = []  # (source planet h, active-target index j)
    for j, a in enumerate(active):
        T = a["T"]
        for h in range(n):
            if h == T or not is_mine[h] or surplus[h] < min_send:
                continue
            if not math.isfinite(eta[h, T]):
                continue
            edges.append((h, j))
    if not edges:
        return {}

    E = len(edges)
    c = np.zeros(E, dtype=np.float64)
    ub = np.zeros(E, dtype=np.float64)
    for e, (h, j) in enumerate(edges):
        a = active[j]
        T = a["T"]
        priority_weight = _W_BASE + _W_PRIORITY * (n_priority - a["pos"])
        extra = float(cost_matrix[h, T]) if cost_matrix is not None else 0.0
        c[e] = float(eta[h, T]) + extra - priority_weight
        ub[e] = float(min(int(surplus[h]), a["need_cap"]))

    src_ids = sorted({h for h, _ in edges})
    rows: list[np.ndarray] = []
    b: list[float] = []
    # Supply rows.
    for h in src_ids:
        row = np.zeros(E)
        for e, (hh, _) in enumerate(edges):
            if hh == h:
                row[e] = 1.0
        rows.append(row)
        b.append(float(surplus[h]))
    # Demand rows.
    for j, a in enumerate(active):
        row = np.zeros(E)
        for e, (_, jj) in enumerate(edges):
            if jj == j:
                row[e] = 1.0
        rows.append(row)
        b.append(float(a["need_cap"]))

    A_ub = np.asarray(rows, dtype=np.float64)
    b_ub = np.asarray(b, dtype=np.float64)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, u) for u in ub],
                  method="highs")
    if not res.success or res.x is None:
        return {}

    flow: dict[tuple[int, int], int] = {}
    for e, (h, j) in enumerate(edges):
        s = int(round(float(res.x[e])))
        if s > 0:
            flow[(h, active[j]["T"])] = s
    return flow


def _allocate_mcf(
    distance_matrix: np.ndarray,
    ships: np.ndarray,
    production: np.ndarray,
    *,
    n: int,
    is_mine: np.ndarray,
    surplus: np.ndarray,
    priority_order: Sequence[int],
    attack_set: set,
    defend_set: set,
    incoming_enemy: Optional[np.ndarray],
    board_size: float,
    max_speed: float,
    capture_margin: float,
    attack_require_full: bool,
    min_send: int,
    cost_matrix: Optional[np.ndarray] = None,
) -> list[Launch]:
    eta = _eta_matrix_from_surplus(
        distance_matrix, surplus, board_size=board_size, max_speed=max_speed
    )
    active = _collect_active_targets(
        priority_order, attack_set, defend_set,
        ships=ships, production=production, eta=eta,
        is_mine=is_mine, surplus=surplus, incoming_enemy=incoming_enemy,
        n=n, capture_margin=capture_margin, min_send=min_send,
    )
    if not active:
        return []

    n_priority = len(priority_order)
    dropped: set = set()

    # Outer loop: solve, then drop any attack that cannot be *fully* funded
    # (a partial attack only wastes ships) and re-solve.  Dropping frees supply
    # for the remaining targets, so re-solving can complete a previously
    # starved lower-priority attack.  The active attack set shrinks every
    # iteration -> terminates.
    flow: dict[tuple[int, int], int] = {}
    act: list = active
    while True:
        act = [a for a in active if a["T"] not in dropped]
        if not act:
            return []
        flow = _solve_transport_lp(
            act, n=n, is_mine=is_mine, surplus=surplus, eta=eta,
            n_priority=n_priority, min_send=min_send, cost_matrix=cost_matrix,
        )
        # Drop sub-threshold launches (would violate min_send).
        flow = {e: s for e, s in flow.items() if s >= min_send}

        received: dict[int, int] = {a["T"]: 0 for a in act}
        for (_h, T), s in flow.items():
            received[T] = received.get(T, 0) + s

        if attack_require_full:
            newly = [a["T"] for a in act
                     if a["is_attack"] and received.get(a["T"], 0) + 1e-6 < a["need"]]
            if newly:
                dropped.update(newly)
                continue
        break

    # Emit launches grouped by priority order, sources sorted by ETA.
    launches: list[Launch] = []
    for a in act:
        T = a["T"]
        kind = "attack" if a["is_attack"] else "defend"
        srcs = [(h, s) for (h, TT), s in flow.items() if TT == T and s > 0]
        srcs.sort(key=lambda hs: eta[hs[0], T])
        for h, s in srcs:
            launches.append(Launch(src=h, tgt=T, ships=int(s), kind=kind))
    return launches


def allocate_launches(
    distance_matrix: np.ndarray,
    owner: np.ndarray,
    ships: np.ndarray,
    production: np.ndarray,
    *,
    player: int,
    priority_order: Sequence[int],
    attack_targets: Iterable[int],
    defend_targets: Iterable[int],
    incoming_enemy: Optional[np.ndarray] = None,
    commit_fraction: Optional[np.ndarray] = None,
    board_size: float = DEFAULT_BOARD_SIZE,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    capture_margin: float = 1.0,
    source_reserve_frac: float = 0.0,
    attack_require_full: bool = True,
    min_send: int = 1,
    cost_matrix: Optional[np.ndarray] = None,
    solver: str = "mcf",
) -> list[Launch]:
    r"""Resolve RL target demands into concrete fleet launches.

    Parameters
    ----------
    distance_matrix:
        ``(n, n)`` normalised pairwise distances (e.g. ``edge_bias[..., 0]``);
        ``board_size`` converts to raw units so ETA is in turns.
    owner, ships, production:
        ``(n,)`` board state.  ``owner == player`` marks the agent's planets
        (the only valid sources); ``production`` is per-turn ship growth.
    player:
        The agent's id.
    priority_order:
        Target planet slots in the **RL-chosen funding order** (the
        Plackett-Luce ranking).  Higher-priority targets are funded first when
        ships are scarce.  Entries must be a subset of
        ``attack_targets ∪ defend_targets``.
    attack_targets, defend_targets:
        The selected attack (enemy / neutral) and defend (own) planet slots.
    incoming_enemy:
        Optional ``(n,)`` estimate of enemy ships expected to hit each planet
        (e.g. :func:`reworked_features.incoming_enemy_estimate`).  Required for
        defend demands; if ``None`` defend targets are skipped.
    capture_margin:
        Extra ships required beyond break-even for an attack to commit.
    commit_fraction:
        Optional ``(n,)`` per-planet fraction (the RL ``commit_head`` output):
        the allocator may draw at most ``commit_fraction[h] * ships[h]`` from
        own planet ``h``.  When given it **overrides** ``source_reserve_frac``.
    source_reserve_frac:
        Fallback (used only when ``commit_fraction`` is ``None``): fraction of
        each source's garrison held back.
    attack_require_full:
        If ``True`` (default) an attack only launches when enough ships can be
        gathered to actually capture; partial attacks are skipped.  Defends
        always allow partial reinforcement.
    min_send:
        Minimum ships per individual launch.
    cost_matrix:
        Optional ``(n, n)`` additive **routing preference** in ETA (turn) units:
        the effective transport cost of source ``h -> target T`` becomes
        ``ETA(h, T) + cost_matrix[h, T]``.  Use it to bias *which* source feeds
        a target without changing *whether* a target is funded (priority still
        dominates) -- e.g. penalise stripping a high-``potential_threat`` source
        or discount deep-interior (``friendly_compactness``) bases.  Build one
        with :func:`reworked_pipeline.build_routing_cost_matrix`.  Demand sizing
        always uses pure physical ETA, so this never distorts capture margins.
    solver:
        ``"mcf"`` (default) for the global min-cost-flow assignment, or
        ``"greedy"`` for the priority-greedy fill.  ``"mcf"`` silently falls
        back to ``"greedy"`` if SciPy is not importable.

    Returns
    -------
    list[Launch]
        Launches to execute this turn, grouped in priority order.

    Notes
    -----
    * **Demand sizing.**  Attack need ``= garrison(T) + production(T)·ETA +
      margin``.  Defend need ``= incoming_enemy(T) − garrison(T) −
      production(T)·ETA``, clamped at 0.  ``ETA`` is the lowest ETA among
      candidate sources.  Both solvers size demand identically.
    * **Priority semantics.**  Both solvers honour ``priority_order``
      lexicographically (a higher-priority target is fully funded before any
      lower one); ``"mcf"`` additionally optimises *which* source feeds *which*
      target globally, so it can route a high-priority target via a farther
      source to free a nearer one for a lower-priority target.
    * **Multi-source timing caveat.**  Gathered fleets from several sources
      arrive at slightly different times; neither solver models that staggering
      (a known approximation).  A time-expanded MCF could refine it later.
    """
    n = int(np.asarray(owner).shape[0])
    owner = np.asarray(owner).reshape(n)
    ships = np.asarray(ships, dtype=np.float64).reshape(n)
    production = np.asarray(production, dtype=np.float64).reshape(n)
    attack_set = {int(t) for t in attack_targets}
    defend_set = {int(t) for t in defend_targets}

    is_mine, surplus = _compute_surplus(
        owner, ships, player=player, commit_fraction=commit_fraction,
        source_reserve_frac=source_reserve_frac, n=n,
    )

    use_mcf = solver == "mcf"
    if use_mcf:
        try:
            import scipy.optimize  # noqa: F401
        except Exception:  # pragma: no cover - SciPy missing -> degrade
            use_mcf = False

    common = dict(
        n=n, is_mine=is_mine, surplus=surplus,
        priority_order=priority_order, attack_set=attack_set, defend_set=defend_set,
        incoming_enemy=incoming_enemy, board_size=board_size, max_speed=max_speed,
        capture_margin=capture_margin, attack_require_full=attack_require_full,
        min_send=min_send, cost_matrix=cost_matrix,
    )

    if use_mcf:
        return _allocate_mcf(distance_matrix, ships, production, **common)
    return _allocate_greedy(distance_matrix, ships, production, **common)


__all__ = ["Launch", "allocate_launches"]
