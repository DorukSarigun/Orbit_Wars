"""Reworked self-play rollout + PPO for the hierarchical policy.

This is a clean rewrite of the training loop around the redesigned stack:

* features   -> :mod:`reworked_pipeline` (``pack_reworked_batch``)
* policy     -> :class:`reworked_model.ReworkedHierarchicalPolicy`
              (Bernoulli attack/defend + Plackett-Luce priority + Beta commit
               + Beta defense posture)
* logistics  -> :mod:`reworked_allocator` (``allocate_launches``)

Key design choices (differ from the legacy ``rl/rollout.py``):

* **One transition per player per turn.**  The whole hierarchical decision is a
  single RL action; the scripted allocator is a *deterministic* function of it
  (no extra log-prob), so the natural credit unit is the turn, not the launch.
* **Summed log-prob.**  The stored action log-prob is the sum of every sampled
  component (see :meth:`ReworkedHierarchicalPolicy.act`); PPO re-evaluates it
  with :meth:`ReworkedHierarchicalPolicy.evaluate_actions`.
* **Fixed padded width ``P_MAX``.**  Every transition stores features / masks /
  action padded to ``P_MAX`` so the PPO minibatch is a trivial stack.
* **Simple, pluggable reward.**  Default = potential-based shaping on the
  owned-production differential + a terminal win/loss bonus.  Swap via the
  ``reward_fn`` argument.

Env contract is unchanged: ``env.step([launches_p0, launches_p1, ...])`` where
each launch is ``[from_planet_id, angle, ships]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:  # package layout
    from .obs_builder import build_observation
    from .feature_extraction import parse_planets, parse_fleets, BOARD_SIZE, DEFAULT_MAX_SHIP_SPEED
    from .navigation import compute_launch
    from .reworked_pipeline import pack_reworked_batch, build_allocator_inputs
    from .reworked_allocator import allocate_launches, Launch
    from .reworked_model import ReworkedHierarchicalPolicy, ReworkedLaunchPolicy, ReworkedEdgePolicy
    from .reworked_features import (
        discounted_production_potential, DEFAULT_GAMMA, DEFAULT_NORM,
        EDGE_IDX_ENEMY_FLUX,
    )
    from .reworked_credit import (
        EpisodeLog, LaunchRecord, TurnSnapshot, CreditConfig, attribute_credits,
    )
    from .reworked_remastered_reward import (
        RemasteredRewardConfig,
        CampaignLedger,
        attribute_remastered_rewards,
        apply_remastered_credits,
        observe_enemy_kills,
        summarize_remastered_episode,
        planet_ownership_events,
    )
except ImportError:  # pragma: no cover - flat layout / direct execution
    from obs_builder import build_observation  # type: ignore
    from feature_extraction import parse_planets, parse_fleets, BOARD_SIZE, DEFAULT_MAX_SHIP_SPEED  # type: ignore
    from navigation import compute_launch  # type: ignore
    from reworked_pipeline import pack_reworked_batch, build_allocator_inputs  # type: ignore
    from reworked_allocator import allocate_launches, Launch  # type: ignore
    from reworked_model import ReworkedHierarchicalPolicy, ReworkedLaunchPolicy, ReworkedEdgePolicy  # type: ignore
    from reworked_features import (  # type: ignore
        discounted_production_potential, DEFAULT_GAMMA, DEFAULT_NORM,
        EDGE_IDX_ENEMY_FLUX,
    )
    from reworked_credit import (  # type: ignore
        EpisodeLog, LaunchRecord, TurnSnapshot, CreditConfig, attribute_credits,
    )
    from reworked_remastered_reward import (  # type: ignore
        RemasteredRewardConfig,
        CampaignLedger,
        attribute_remastered_rewards,
        apply_remastered_credits,
        observe_enemy_kills,
        summarize_remastered_episode,
        planet_ownership_events,
    )


P_MAX = 44   # fixed padded planet width (env worst case)
F_MAX = 64


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def _target_masks(state, p_max: int, player: int) -> "tuple[np.ndarray, np.ndarray]":
    """Per-planet attack / defend candidate masks padded to ``p_max``.

    * attack candidates = real planets **not** owned by ``player`` (enemy or
      neutral -- the latter being expansion);
    * defend candidates = real planets owned by ``player``.
    """
    n = state.n
    owner = np.asarray(state.owner).reshape(n)
    attack = np.zeros(p_max, dtype=bool)
    defend = np.zeros(p_max, dtype=bool)
    attack[:n] = owner != int(player)
    defend[:n] = owner == int(player)
    return attack, defend


# Minimum ships per launch for the v8 source-centric policy (slivers below this
# are dropped after flooring -- mirrors the allocator's ``min_send`` default).
MIN_SEND = 1


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute circular difference between two headings (radians)."""
    import math
    d = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


def _source_dest_masks(state, p_max: int, player: int,
                       min_send: int = MIN_SEND) -> "tuple[np.ndarray, np.ndarray]":
    """Masks for the v8 launch policy, padded to ``p_max``.

    * ``source_mask (p_max,)`` -- own planets whose garrison can fund a launch
      (``ships >= min_send``); these are the rows the agent draws from.
    * ``dest_mask (p_max, p_max)`` -- for each source row ``h``, the valid
      target columns ``t`` (every real planet except ``h`` itself); non-source
      rows are all ``False``.  This is the support of each per-source Dirichlet.
    """
    n = state.n
    owner = np.asarray(state.owner).reshape(n)
    ships = np.asarray(state.ships, dtype=np.float64).reshape(n)
    source = np.zeros(p_max, dtype=bool)
    source[:n] = (owner == int(player)) & (ships >= float(min_send))
    dest = np.zeros((p_max, p_max), dtype=bool)
    if n > 0:
        block = np.ones((n, n), dtype=bool)
        np.fill_diagonal(block, False)        # cannot send to self
        block[~source[:n], :] = False         # only source rows have targets
        dest[:n, :n] = block
    return source, dest


# Behavior-cloning target smoothing: the teacher's executed split is a sparse
# (often near one-hot) vector; mixing in a little uniform mass over the valid
# targets keeps the target reachable by a finite Dirichlet, so the BC log-prob
# gradient stays informative instead of saturating at the clamp boundary.
_BC_SPLIT_SMOOTH = 0.05
_BC_COMMIT_EPS = 0.02


def teacher_launches_to_v8_targets(
    state,
    launches: list,
    dest_mask: np.ndarray,
    *,
    p_max: int = P_MAX,
    smooth: float = _BC_SPLIT_SMOOTH,
    commit_eps: float = _BC_COMMIT_EPS,
) -> "tuple[np.ndarray, np.ndarray]":
    """Re-express the teacher's executed launches in the v8 action space.

    Returns ``(commit_target (p_max,), split_target (p_max, p_max))`` where for
    each own source ``h``:

    * ``commit_target[h] = clip(sum_t ships_{h,t} / garrison_h)`` -- the fraction
      of the garrison the teacher actually deployed;
    * ``split_target[h, :] = (1 - smooth) * (ships_{h,:} / total_h) + smooth * U``
      over the valid target columns (``U`` uniform).  Sources the teacher did not
      launch from get ``commit_eps`` and a uniform split (a benign target).
    """
    n = state.n
    ships = np.asarray(state.ships, dtype=np.float64).reshape(n)
    commit_t = np.zeros(p_max, dtype=np.float32)
    split_t = np.zeros((p_max, p_max), dtype=np.float32)
    if n == 0:
        return commit_t, split_t
    valid = np.asarray(dest_mask, dtype=bool)[:n, :n]
    sent = np.zeros((n, n), dtype=np.float64)
    for L in launches:
        if 0 <= L.src < n and 0 <= L.tgt < n and L.src != L.tgt:
            sent[L.src, L.tgt] += max(0.0, float(L.ships))
    for h in range(n):
        vrow = valid[h]
        K = int(vrow.sum())
        if K == 0:
            continue
        uni = vrow.astype(np.float64) / float(K)
        total = float(sent[h].sum())
        g = float(ships[h])
        if total > 0.0 and g > 0.0:
            commit_t[h] = float(np.clip(total / g, commit_eps, 1.0 - commit_eps))
            raw = np.where(vrow, sent[h], 0.0)
            rs = float(raw.sum())
            raw = raw / rs if rs > 0.0 else uni
            s = (1.0 - smooth) * raw + smooth * uni
        else:
            commit_t[h] = commit_eps
            s = uni
        split_t[h, :n] = s.astype(np.float32)
    return commit_t, split_t


def launch_action_to_env(
    state,
    obs,
    player: int,
    action_row: dict,
    parsed_planets: list,
    angular_velocity: float,
    *,
    min_send: int = MIN_SEND,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    comet_evac_turns: int = 2,
) -> "tuple[list, list]":
    """Turn one sampled v8 launch action into concrete env launches.

    ``action_row`` holds ``commit (P,)`` (per-source Beta fraction) and
    ``split_w (P, P)`` (per-source Dirichlet weights over targets).  For each own
    source ``h`` the deployable budget is ``commit_h * garrison_h`` and the ships
    sent to target ``t`` are ``floor(budget * split_{h,t})`` (dropping anything
    below ``min_send``).  The split is a simplex so the total per source never
    exceeds the garrison -- feasibility is automatic, no allocator required.

    Returns ``(env_action, launches)`` mirroring :func:`action_to_env_launches`.
    """
    n = state.n
    owner = np.asarray(state.owner).reshape(n)
    ships_arr = np.asarray(state.ships, dtype=np.float64).reshape(n)
    commit = np.asarray(action_row["commit"], dtype=np.float64).reshape(-1)[:n]
    split = np.asarray(action_row["split_w"], dtype=np.float64).reshape(P_MAX, P_MAX)[:n, :n]

    launches: list = []
    for h in range(n):
        if owner[h] != int(player):
            continue
        budget = float(commit[h]) * float(ships_arr[h])
        if budget < float(min_send):
            continue
        for t in range(n):
            if t == h:
                continue
            ships_ht = int(np.floor(budget * float(split[h, t])))
            if ships_ht < int(min_send):
                continue
            kind = "attack" if owner[t] != int(player) else "defend"
            launches.append(Launch(src=h, tgt=t, ships=ships_ht, kind=kind))

    planet_by_id = {int(p.id): p for p in parsed_planets}
    ids = np.asarray(obs.planet_ids).reshape(-1)
    env_action: list = []
    realized: list = []   # launches that actually became env fleets (with heading)
    for L in launches:
        src_id = int(ids[L.src]) if 0 <= L.src < len(ids) else -1
        tgt_id = int(ids[L.tgt]) if 0 <= L.tgt < len(ids) else -1
        src = planet_by_id.get(src_id)
        tgt = planet_by_id.get(tgt_id)
        if src is None or tgt is None:
            continue
        res = compute_launch(src, tgt, int(L.ships), parsed_planets, angular_velocity,
                             max_speed=max_speed)
        if res is None:
            continue
        angle = float(res["angle"])
        env_action.append([src.id, angle, int(res["ships"])])
        realized.append(replace(L, ships=int(res["ships"]), angle=angle))

    env_action.extend(_comet_evacuation_launches(
        player, parsed_planets, launches, ids,
        angular_velocity, max_speed, comet_evac_turns))
    return env_action, realized


def _edge_valid_mask(state, p_max: int, player: int,
                     min_send: int = MIN_SEND) -> np.ndarray:
    """Valid directed edges for the v9 edge policy, padded to ``p_max``.

    ``mask[h, t]`` is ``True`` when ``h`` is owned by ``player`` with a garrison
    that can fund a launch (``ships >= min_send``) and ``t`` is any other real
    planet.  Both attack (``t`` not owned) and defend (``t`` owned) edges are
    included -- the directedness keeps defence unambiguous.  A target ``t`` is an
    active "campaign" iff some source can reach it (``mask[:, t].any()``).
    """
    n = state.n
    owner = np.asarray(state.owner).reshape(n)
    ships = np.asarray(state.ships, dtype=np.float64).reshape(n)
    src_ok = (owner == int(player)) & (ships >= float(min_send))
    mask = np.zeros((p_max, p_max), dtype=bool)
    if n > 0:
        block = np.ones((n, n), dtype=bool)
        np.fill_diagonal(block, False)
        block[~src_ok, :] = False
        mask[:n, :n] = block
    return mask


def _edge_target_valid_mask(
    gate: np.ndarray,
    edge_mask: np.ndarray,
    *,
    launches: "list | None" = None,
    p_max: int = P_MAX,
) -> np.ndarray:
    """Per-target PPO slots for one edge-policy turn, padded to ``p_max``.

    ``True`` at target ``t`` when some valid edge ``h -> t`` has ``gate[h,t] > 0.5``
    or a launch actually departed toward ``t``.  Geometrically reachable targets
    with gate off and no ships sent are excluded (unlike ``edge_mask.any(axis=0)``).

    ``gate`` / ``edge_mask`` may be unpadded ``(n, n)`` (e.g. notebook demos) or
    already padded to ``(p_max, p_max)`` as in rollout buffers.
    """
    def _to_pmax(mat, *, dtype):
        a = np.asarray(mat, dtype=dtype)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError(f"expected square (n, n), got shape {a.shape}")
        side = int(a.shape[0])
        if side == p_max:
            return a, side
        out = np.zeros((p_max, p_max), dtype=dtype)
        m = min(side, p_max)
        out[:m, :m] = a[:m, :m]
        return out, side

    g, n_g = _to_pmax(gate, dtype=np.float64)
    em, n_em = _to_pmax(edge_mask, dtype=bool)
    n = min(n_g, n_em)
    tv = ((g[:n, :n] > 0.5) & em[:n, :n]).any(axis=0)
    if launches:
        for L in launches:
            tgt = int(L.tgt)
            if 0 <= tgt < n:
                tv[tgt] = True
    out = np.zeros(p_max, dtype=bool)
    if n > 0:
        out[:n] = tv
    return out


def edge_action_to_env(
    state,
    obs,
    player: int,
    action_row: dict,
    parsed_planets: list,
    angular_velocity: float,
    *,
    min_send: int = MIN_SEND,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    comet_evac_turns: int = 2,
) -> "tuple[list, list]":
    """Turn one sampled v9 edge action into concrete env launches.

    ``action_row`` holds ``gate (P, P)`` (per-edge send 0/1) and ``frac (P, P)``
    (per-edge fraction of the source garrison).  For each own source ``h`` the
    gated edges request ``floor(frac_{h,t} * garrison_h)`` ships; if a source's
    total request exceeds its garrison the requests are scaled down to fit
    (feasibility -- there is no hard simplex, so over-extension is possible and
    is taught via the source-loss credit term rather than forbidden here).

    Returns ``(env_action, launches)`` mirroring :func:`launch_action_to_env`.
    """
    n = state.n
    owner = np.asarray(state.owner).reshape(n)
    ships_arr = np.asarray(state.ships, dtype=np.float64).reshape(n)
    gate = np.asarray(action_row["gate"], dtype=np.float64).reshape(P_MAX, P_MAX)[:n, :n]
    frac = np.asarray(action_row["frac"], dtype=np.float64).reshape(P_MAX, P_MAX)[:n, :n]

    launches: list = []
    for h in range(n):
        if owner[h] != int(player):
            continue
        g = float(ships_arr[h])
        if g < float(min_send):
            continue
        req = []  # (t, ships)
        for t in range(n):
            if t == h or gate[h, t] <= 0.5:
                continue
            ships_ht = int(np.floor(g * float(frac[h, t])))
            if ships_ht > 0:
                req.append([t, ships_ht])
        total = sum(s for _, s in req)
        if total > g and total > 0:           # feasibility: scale to fit garrison
            scale = g / float(total)
            req = [[t, int(np.floor(s * scale))] for t, s in req]
        for t, ships_ht in req:
            if ships_ht < int(min_send):
                continue
            kind = "attack" if owner[t] != int(player) else "defend"
            launches.append(Launch(src=h, tgt=t, ships=int(ships_ht), kind=kind))

    planet_by_id = {int(p.id): p for p in parsed_planets}
    ids = np.asarray(obs.planet_ids).reshape(-1)
    env_action: list = []
    realized: list = []
    for L in launches:
        src_id = int(ids[L.src]) if 0 <= L.src < len(ids) else -1
        tgt_id = int(ids[L.tgt]) if 0 <= L.tgt < len(ids) else -1
        src = planet_by_id.get(src_id)
        tgt = planet_by_id.get(tgt_id)
        if src is None or tgt is None:
            continue
        res = compute_launch(src, tgt, int(L.ships), parsed_planets, angular_velocity,
                             max_speed=max_speed)
        if res is None:
            continue
        angle = float(res["angle"])
        env_action.append([src.id, angle, int(res["ships"])])
        realized.append(replace(L, ships=int(res["ships"]), angle=angle))

    env_action.extend(_comet_evacuation_launches(
        player, parsed_planets, launches, ids,
        angular_velocity, max_speed, comet_evac_turns))
    return env_action, realized


# ---------------------------------------------------------------------------
# Action -> concrete env launches
# ---------------------------------------------------------------------------
def _comet_remaining_life(p) -> int:
    """Turns an owned comet stays on the board before it expires (and takes its
    garrison with it).  ``-1`` if ``p`` is not a comet with a known path."""
    if not getattr(p, "is_comet", False):
        return -1
    path = getattr(p, "comet_path", None)
    if not path:
        return -1
    return (len(path) - 1) - int(getattr(p, "comet_path_index", -1))


def _nearest_safe_friendly(comet, friendly, evac_turns: int):
    """Closest friendly planet to ``comet`` that will not itself expire within
    ``evac_turns`` (so we never evacuate onto another doomed comet)."""
    best = None
    best_d = float("inf")
    for p in friendly:
        if int(p.id) == int(comet.id):
            continue
        rem = _comet_remaining_life(p)
        if rem >= 0 and rem <= evac_turns:
            continue  # destination comet is also about to die
        d = (float(p.x) - float(comet.x)) ** 2 + (float(p.y) - float(comet.y)) ** 2
        if d < best_d:
            best_d = d
            best = p
    return best


def _comet_evacuation_launches(player, parsed_planets, launches, ids,
                               angular_velocity, max_speed, evac_turns):
    """Extra launches that rescue ships stranded on an owned comet about to
    expire.  Ships already committed off the comet this turn (by the allocator)
    are subtracted; the rest are sent to the nearest safe friendly planet.

    These are *scripted* logistics moves (like the allocator output) and are
    deliberately NOT added to the RL launch log -- they carry no policy
    decision credit; the credit pass already absorbs the comet's disappearance.
    """
    if evac_turns <= 0:
        return []
    committed: dict = {}
    for L in launches:
        if 0 <= L.src < len(ids):
            sid = int(ids[L.src])
            committed[sid] = committed.get(sid, 0) + int(L.ships)
    friendly = [p for p in parsed_planets if int(p.owner) == int(player)]
    out: list = []
    for c in parsed_planets:
        if int(c.owner) != int(player):
            continue
        rem = _comet_remaining_life(c)
        if rem < 0 or rem > evac_turns:
            continue
        garrison = int(round(float(c.ships))) - committed.get(int(c.id), 0)
        if garrison <= 0:
            continue
        dest = _nearest_safe_friendly(c, friendly, evac_turns)
        if dest is None:
            continue
        res = compute_launch(c, dest, garrison, parsed_planets, angular_velocity,
                             max_speed=max_speed)
        if res is None or int(res["ships"]) <= 0:
            continue
        out.append([c.id, float(res["angle"]), int(res["ships"])])
    return out


def action_to_env_launches(
    obs,
    player: int,
    action_row: dict,
    parsed_planets: list,
    angular_velocity: float,
    *,
    routing_cost: bool = True,
    capture_margin: float = 1.0,
    attack_require_full: bool = True,
    max_speed: float = DEFAULT_MAX_SHIP_SPEED,
    comet_evac_turns: int = 2,
) -> "tuple[list, list]":
    """Turn one sampled hierarchical action into env launches.

    ``action_row`` holds single-sample NumPy arrays (``attack_sel`` / ``defend_sel``
    ``(P,)``, ``order`` ``(P,)``, ``commit_frac`` ``(P,)``, ``potential_weight`` /
    ``horizon`` scalars).  Returns ``(env_action, launches)`` where ``env_action``
    is the ``[[id, angle, ships], ...]`` list for ``env.step`` and ``launches``
    is the raw :class:`reworked_allocator.Launch` list (for logging / trackers).
    """
    n = int(np.asarray(obs.planet_mask).sum())
    attack_sel = np.asarray(action_row["attack_sel"]).reshape(-1)
    defend_sel = np.asarray(action_row["defend_sel"]).reshape(-1)
    order = np.asarray(action_row["order"]).reshape(-1)
    commit_frac = np.asarray(action_row["commit_frac"], dtype=np.float64).reshape(-1)[:n]

    attack_targets = [t for t in range(n) if attack_sel[t] > 0.5]
    defend_targets = [t for t in range(n) if defend_sel[t] > 0.5]
    sel = set(attack_targets) | set(defend_targets)
    priority_order = [int(t) for t in order if 0 <= int(t) < n and int(t) in sel]

    # The RL action may pick no targets, but we still run comet evacuation below
    # (stranded ships must be rescued regardless of what the policy ordered).
    launches: list = []
    if priority_order:
        inputs = build_allocator_inputs(
            obs, player,
            potential_weight=float(action_row["potential_weight"]),
            horizon=float(action_row["horizon"]),
            routing_cost=routing_cost,
            max_speed=max_speed,
        )
        launches = allocate_launches(
            **inputs,
            player=player,
            priority_order=priority_order,
            attack_targets=attack_targets,
            defend_targets=defend_targets,
            commit_fraction=commit_frac,
            capture_margin=capture_margin,
            attack_require_full=attack_require_full,
        )

    planet_by_id = {int(p.id): p for p in parsed_planets}
    ids = np.asarray(obs.planet_ids).reshape(-1)
    env_action: list = []
    realized: list = []   # launches that actually became env fleets (with heading)
    for L in launches:
        src_id = int(ids[L.src]) if 0 <= L.src < len(ids) else -1
        tgt_id = int(ids[L.tgt]) if 0 <= L.tgt < len(ids) else -1
        src = planet_by_id.get(src_id)
        tgt = planet_by_id.get(tgt_id)
        if src is None or tgt is None:
            continue
        res = compute_launch(src, tgt, int(L.ships), parsed_planets, angular_velocity,
                             max_speed=max_speed)
        if res is None:
            continue
        angle = float(res["angle"])
        env_action.append([src.id, angle, int(res["ships"])])
        realized.append(replace(L, ships=int(res["ships"]), angle=angle))

    # Scripted safety net: evacuate ships off owned comets about to expire.
    env_action.extend(_comet_evacuation_launches(
        player, parsed_planets, launches, ids,
        angular_velocity, max_speed, comet_evac_turns))
    return env_action, realized


# ---------------------------------------------------------------------------
# Reward: potential-based shaping (Ng et al. 1999) on an economic potential
# ---------------------------------------------------------------------------
# The ONLY ground-truth objective is the terminal win/loss (+/-1).  Everything
# else is *potential-based* shaping, which is provably policy-invariant (it
# never changes the optimal policy) and hack-proof (it telescopes, so any cycle
# of moves nets zero shaping reward -> no ping-pong exploit).
#
# Division of labour:
#   * the POTENTIAL Phi(s) decides *what* a state is worth -- pluggable below;
#   * the SHAPING FORMULA  F = gamma * Phi(s') - Phi(s)  is fixed and principled
#     (it is what makes the shaping reward-neutral);
#   * GAE / gamma then propagate each turn's reward back to the decision -- the
#     shaping only shortens that propagation distance by placing the reward next
#     to the event that earned it.


@dataclass
class RewardConfig:
    """Reward weights for the potential-based shaping scheme."""
    gamma: float = 0.99                 # RL discount used INSIDE the shaping term
    gamma_eco: float = DEFAULT_GAMMA    # economic horizon discount in the dpp potential
    potential_norm: float = DEFAULT_NORM  # common currency scale (keeps Phi ~O(1))
    ship_weight: float = 1.0            # value of one ship-in-hand vs. one future ship
    count_inflight: bool = True         # count own in-flight fleets as ship stock
    shaping_weight: float = 1.0
    terminal_weight: float = 1.0
    enable_shaping: bool = True         # ablation switch (shaping on/off)
    # Behavioural (non-telescoping) cost terms -- these intentionally shift the
    # optimal policy toward not wasting ships (the telescoping potential alone
    # weights ship loss at ~1/potential_norm, effectively negligible).
    loss_norm: float = 50.0             # ship-count scale for the cost terms (~O(1))
    loss_weight: float = 1.0            # penalty per OWN ship destroyed (combat + sun)
    launch_weight: float = 0.1          # small penalty per ship committed to a launch
    # Decision-indexed delayed credit (FIFO inventory accounting, see
    # :mod:`reworked_credit`).  When on, the production/ownership value of a
    # planet is credited to the *decision* whose ships fought for it (capture /
    # defence), so it is removed from the telescoping potential below to avoid
    # double counting; the ship-stock term stays.
    enable_battle_credit: bool = False
    credit_capture_weight: float = 1.0  # scale on +dpp(g) for a successful capture
    credit_defense_weight: float = 1.0  # scale on +/-dpp(g) for a resolved defence
    # Decision-indexed penalties for attacks that never paid off (campaign
    # ledger; robust to multi-source / staggered arrivals).  Default 0.0 keeps
    # the credit pass identical to capture/defence-only behaviour.
    credit_lost_race_weight: float = 0.0  # target captured by a rival mid-assault
    credit_fail_weight: float = 0.0       # target never captured by episode end
    # Over-extension: charge decisions that recently stripped a planet that then
    # fell (the departed ships escaped the defence-loss penalty).  Default 0.0.
    credit_source_loss_weight: float = 0.0
    credit_source_loss_window: int = 10
    # Attack-only ship-turn penalty for time spent in flight (see
    # :mod:`reworked_credit` ``airborne_weight``).  Default 0.0 = off.
    airborne_weight: float = 0.0
    airborne_norm: float = 50.0


def _ship_stock(planets, fleets, *, count_inflight: bool) -> dict:
    """Total ships per owner = garrisons (+ owned in-flight fleets).

    Counting in-flight fleets is what makes a *launch* leave the potential
    unchanged: ships only move between buckets, they are not destroyed.  Only
    real combat losses shrink an owner's stock.
    """
    stock: dict = {}
    for pl in planets:
        if pl.owner != -1:
            stock[pl.owner] = stock.get(pl.owner, 0.0) + float(pl.ships)
    if count_inflight:
        for fl in fleets:
            if fl.owner != -1:
                stock[fl.owner] = stock.get(fl.owner, 0.0) + float(fl.ships)
    return stock


def _own_ship_stock(env_obs, player: int, *, count_inflight: bool = True) -> float:
    """Total ships the player currently controls (garrisons + own in-flight)."""
    planets = parse_planets(env_obs)
    fleets = parse_fleets(env_obs) if count_inflight else []
    stock = _ship_stock(planets, fleets, count_inflight=count_inflight)
    return float(stock.get(player, 0.0))


def _own_production(env_obs, player: int) -> float:
    """Per-turn ship growth from the player's currently-owned planets."""
    return float(sum(p.production for p in parse_planets(env_obs) if p.owner == player))


def _build_snapshot(env_obs, turn: int) -> "TurnSnapshot":
    """Ground-truth board snapshot for the offline credit pass.

    Planets / fleets are global (shared across all player observations), so a
    single canonical observation is enough.
    """
    planets = parse_planets(env_obs)
    fleets = parse_fleets(env_obs)
    return TurnSnapshot(
        turn=int(turn),
        owner={int(p.id): int(p.owner) for p in planets},
        ships={int(p.id): float(p.ships) for p in planets},
        production={int(p.id): float(p.production) for p in planets},
        fleet_ids={int(f.id) for f in fleets},
    )


def economic_potential(env_obs, player: int, remaining_turns: int, cfg: RewardConfig) -> float:
    r"""Default potential ``Phi(s)`` = owned **total economic asset** differential.

    Two asset classes, measured in one common currency (ships):

    * **Production** -- ``dpp_raw = prod * H(R)`` is literally "how many ships
      this planet will yield over the remaining horizon", i.e. *future* ships.
    * **Ships in hand** -- garrisons plus owned in-flight fleets, valued at
      ``ship_weight`` per ship (``~1`` => a ship now ~ a future ship).

    .. math::

        \Phi(s) = \frac{1}{\text{norm}}\Big[
            \sum_{g\in\text{mine}} \big(\mathrm{dpp\_raw}(g) + w\,\text{ships}\big)
          - \sum_{g\in\text{enemy}} (\dots) \Big]

    Because ships now enter ``Phi``, spending/losing ships to capture a planet
    automatically subtracts from the shaped reward: a capture only pays off when
    the production gained exceeds the ships lost.  Still antisymmetric in
    self-play, so the shaped game stays zero-sum and policy-invariant.
    """
    planets = parse_planets(env_obs)
    if not planets:
        return 0.0
    fleets = parse_fleets(env_obs) if cfg.count_inflight else []

    prod = np.array([float(p.production) for p in planets], dtype=np.float64)
    owner = np.array([int(p.owner) for p in planets], dtype=np.int64)
    # When decision-indexed battle credit is active the production/ownership
    # value is attributed per-decision in reworked_credit, so drop it from the
    # telescoping potential to avoid double counting (ship stock stays).
    ships_only = cfg.enable_battle_credit
    if ships_only:
        dpp_raw = np.zeros_like(prod)
    else:
        # dpp_raw (future ships) on the same currency as ships, then /norm at the end.
        dpp_raw = np.asarray(
            discounted_production_potential(prod, int(remaining_turns),
                                            gamma=cfg.gamma_eco, norm=1.0),
            dtype=np.float64,
        )
    stock = _ship_stock(planets, fleets, count_inflight=cfg.count_inflight)

    def side_value(is_mine: bool) -> float:
        mask = (owner == int(player)) if is_mine else ((owner != int(player)) & (owner != -1))
        prod_val = float(dpp_raw[mask].sum())
        ship_val = sum(v for k, v in stock.items()
                       if (k == player) == is_mine and k != -1)
        return prod_val + cfg.ship_weight * float(ship_val)

    return (side_value(True) - side_value(False)) / max(cfg.potential_norm, 1e-6)


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------
@dataclass
class HierTransition:
    player: int
    node_feats: np.ndarray       # (P, 15)
    edge_feats: np.ndarray       # (P, P, 4)
    global_feats: np.ndarray     # (14,)
    planet_mask: np.ndarray      # (P,)
    attack_mask: np.ndarray      # (P,)
    defend_mask: np.ndarray      # (P,)
    # action components (padded to P / fixed)
    attack_sel: np.ndarray       # (P,)
    defend_sel: np.ndarray       # (P,)
    order: np.ndarray            # (P,) long
    commit_frac: np.ndarray      # (P,)
    posture_u: np.ndarray        # (2,)
    log_prob: float = 0.0
    value: float = 0.0
    reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    ret: float = 0.0


@dataclass
class LaunchTransition:
    """Transition for the v8 source-centric launch policy (allocator-free)."""
    player: int
    node_feats: np.ndarray       # (P, node_dim)
    edge_feats: np.ndarray       # (P, P, 4)
    global_feats: np.ndarray     # (14,)
    planet_mask: np.ndarray      # (P,)
    source_mask: np.ndarray      # (P,) bool: own planets that can send
    dest_mask: np.ndarray        # (P, P) bool: valid (source, target) pairs
    # action components
    commit: np.ndarray           # (P,)   per-source Beta fraction
    split_w: np.ndarray          # (P, P) per-source Dirichlet weights
    log_prob: float = 0.0
    value: float = 0.0
    reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    ret: float = 0.0


@dataclass
class EdgeTransition:
    """Transition for the v9 directed-edge policy with per-TARGET credit.

    One transition still covers a whole ``(player, turn)`` (a single GNN forward
    yields every per-edge log-prob), but the learning signal is decomposed per
    target campaign: ``old_target_logp`` / ``adv_target`` / ``credit_target`` are
    ``(P,)`` vectors indexed by target planet, and only entries where
    ``target_valid`` marks gated / launched targets that participate in the loss.
    ``reward`` is the turn-level
    total (shaping + losses + terminal + summed battle credit) used for the
    value/GAE backbone; ``credit_target`` holds only the per-target battle credit
    that becomes ``adv_target`` on valid targets (see ``compute_gae``).
    """
    player: int
    node_feats: np.ndarray       # (P, node_dim)
    edge_feats: np.ndarray       # (P, P, 4)
    global_feats: np.ndarray     # (14,)
    planet_mask: np.ndarray      # (P,)
    edge_mask: np.ndarray        # (P, P) bool: valid directed edges
    # action components
    gate: np.ndarray             # (P, P) per-edge send 0/1
    frac: np.ndarray             # (P, P) per-edge garrison fraction
    # per-target bookkeeping
    old_target_logp: np.ndarray  # (P,) per-target log-prob at sampling time
    target_valid: np.ndarray     # (P,) bool: gate open or launch toward target t
    value: float = 0.0
    reward: float = 0.0          # turn-level total reward (GAE backbone)
    done: bool = False
    advantage: float = 0.0       # turn-level GAE advantage (diagnostics)
    ret: float = 0.0
    credit_target: Optional[np.ndarray] = None  # (P,) per-target battle credit
    adv_target: Optional[np.ndarray] = None     # (P,) per-target advantage


@dataclass
class BCSample:
    """A teacher-forced demonstration for behavior cloning (v8 launch only).

    Stores the board the teacher acted on plus the teacher's executed allocation
    re-expressed in the v8 action space (per-source commit fraction + Dirichlet
    target split).  Consumed by a supervised ``-log pi(a_teacher | s)`` loss --
    NOT the PPO policy gradient -- so the policy is pulled toward the expert's
    concentrated, capturing launches without the off-policy bias of routing
    expert actions through PPO's reward/ratio.  Kept in a separate store so it
    never contaminates GAE / value targets (which must stay on-policy).
    """
    node_feats: np.ndarray       # (P, node_dim)
    edge_feats: np.ndarray       # (P, P, 4)
    global_feats: np.ndarray     # (14,)
    planet_mask: np.ndarray      # (P,)
    source_mask: np.ndarray      # (P,) bool
    dest_mask: np.ndarray        # (P, P) bool
    commit_target: np.ndarray    # (P,)   teacher commit fraction per source
    split_target: np.ndarray     # (P, P) teacher target split (simplex / row)


class ReworkedBuffer:
    """Flat transition store with per-player GAE (+ a side store for BC demos)."""

    def __init__(self):
        self.data: list[HierTransition] = []
        # Teacher-forced demonstrations for behavior cloning (kept OUT of the
        # PPO/GAE path; see :class:`BCSample`).
        self.bc_data: list[BCSample] = []
        # Edge-policy per-target decisions: decision_id -> (record_idx, t_local).
        # Lets battle credit (which is keyed by decision_id) route to a single
        # target campaign's slot inside its turn transition.
        self.edge_decisions: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return len(self.data)

    def add(self, tr: HierTransition) -> int:
        self.data.append(tr)
        return len(self.data) - 1

    def add_edge_decision(self, record_idx: int, t_local: int) -> int:
        """Register a per-target campaign decision; returns its ``decision_id``."""
        self.edge_decisions.append((int(record_idx), int(t_local)))
        return len(self.edge_decisions) - 1

    def add_bc(self, sample: BCSample) -> int:
        self.bc_data.append(sample)
        return len(self.bc_data) - 1

    def stack_bc(self, device=None) -> dict:
        """Stack BC demonstrations into evaluate_actions-ready tensors."""
        def st(attr, dtype):
            return torch.as_tensor(
                np.stack([getattr(s, attr) for s in self.bc_data]), dtype=dtype)
        batch = {
            "node_feats": st("node_feats", torch.float32),
            "edge_feats": st("edge_feats", torch.float32),
            "global_feats": st("global_feats", torch.float32),
            "planet_mask": st("planet_mask", torch.bool),
            "source_mask": st("source_mask", torch.bool),
            "dest_mask": st("dest_mask", torch.bool),
            "commit": st("commit_target", torch.float32),
            "split_w": st("split_target", torch.float32),
        }
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    def add_reward(self, idx: int, r: float) -> None:
        self.data[idx].reward += float(r)

    def compute_gae(self, *, gamma: float = 0.99, lam: float = 0.95) -> None:
        players = sorted({tr.player for tr in self.data})
        for player in players:
            idx = [i for i, tr in enumerate(self.data) if tr.player == player]
            last_gae = 0.0
            next_value = 0.0
            for i in reversed(idx):
                tr = self.data[i]
                mask = 0.0 if tr.done else 1.0
                delta = tr.reward + gamma * next_value * mask - tr.value
                last_gae = delta + gamma * lam * mask * last_gae
                tr.advantage = last_gae
                tr.ret = last_gae + tr.value
                next_value = tr.value
                # Edge policy: per-target policy advantage = campaign credit only.
                # Turn-level ``tr.advantage`` still feeds the value head (GAE on
                # ``ret``); policy updates use ``adv_target[k] = credit_k`` on
                # valid targets so temporal/bootstrap signal is not diluted
                # across every geometrically reachable slot.
                if isinstance(tr, EdgeTransition):
                    tv = np.asarray(tr.target_valid, dtype=bool)
                    ct = (np.zeros_like(tv, dtype=np.float32)
                          if tr.credit_target is None
                          else np.asarray(tr.credit_target, dtype=np.float32))
                    tr.adv_target = np.where(tv, ct, 0.0).astype(np.float32)

    @property
    def is_launch(self) -> bool:
        """True when the buffer holds v8 launch transitions (vs hierarchical)."""
        return bool(self.data) and isinstance(self.data[0], LaunchTransition)

    @property
    def is_edge(self) -> bool:
        """True when the buffer holds v9 directed-edge transitions."""
        return bool(self.data) and isinstance(self.data[0], EdgeTransition)

    def stack(self, device=None) -> dict:
        def st(attr, dtype):
            return torch.as_tensor(np.stack([getattr(tr, attr) for tr in self.data]), dtype=dtype)

        common = {
            "node_feats": st("node_feats", torch.float32),
            "edge_feats": st("edge_feats", torch.float32),
            "global_feats": st("global_feats", torch.float32),
            "planet_mask": st("planet_mask", torch.bool),
            "old_value": torch.tensor([tr.value for tr in self.data], dtype=torch.float32),
            "ret": torch.tensor([tr.ret for tr in self.data], dtype=torch.float32),
        }
        if self.is_edge:
            batch = common
            batch.update({
                "edge_mask": st("edge_mask", torch.bool),
                "gate": st("gate", torch.float32),
                "frac": st("frac", torch.float32),
                "old_target_logp": st("old_target_logp", torch.float32),
                "target_valid": st("target_valid", torch.bool),
                "adv_target": st("adv_target", torch.float32),
            })
            if device is not None:
                batch = {k: v.to(device) for k, v in batch.items()}
            return batch

        batch = common
        batch.update({
            "old_log_prob": torch.tensor([tr.log_prob for tr in self.data], dtype=torch.float32),
            "advantage": torch.tensor([tr.advantage for tr in self.data], dtype=torch.float32),
        })
        if self.is_launch:
            batch.update({
                "source_mask": st("source_mask", torch.bool),
                "dest_mask": st("dest_mask", torch.bool),
                "commit": st("commit", torch.float32),
                "split_w": st("split_w", torch.float32),
            })
        else:
            batch.update({
                "attack_mask": st("attack_mask", torch.bool),
                "defend_mask": st("defend_mask", torch.bool),
                "attack_sel": st("attack_sel", torch.float32),
                "defend_sel": st("defend_sel", torch.float32),
                "order": st("order", torch.long),
                "commit_frac": st("commit_frac", torch.float32),
                "posture_u": st("posture_u", torch.float32),
            })
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch


@dataclass
class EpisodeStats:
    length: int
    rewards: dict = field(default_factory=dict)
    winner: Optional[int] = None
    num_agents: int = 2
    n_launches: int = 0
    # Decision-indexed credit diagnostics (0 when battle credit is off).
    credit_total: float = 0.0
    n_capture: int = 0
    n_defense: int = 0
    # Per-episode diagnostics (planet flips, campaigns); written to episodes.jsonl.
    episode_log: Optional[dict] = None


# Supported ``num_agents`` values for :func:`collect_episode` / training.
ALLOWED_NUM_AGENTS: tuple[int, ...] = (2, 4)


def _validate_num_agents(num_agents: int) -> int:
    n = int(num_agents)
    if n not in ALLOWED_NUM_AGENTS:
        raise ValueError(f"num_agents must be one of {ALLOWED_NUM_AGENTS}, got {n}")
    return n


def _opponent_controller(player: int, opponent: str, num_agents: int) -> Optional[str]:
    """How player ``player`` is controlled when not using the live training policy.

    Returns ``None`` for the learning policy (self-play on all seats, or P0 in
    bot modes).  Otherwise ``"scripted"`` or ``"frozen"``.
    """
    if opponent == "self_play":
        return None
    if player == 0:
        return None
    if opponent == "scripted":
        return "scripted"
    if opponent == "frozen":
        # P1 = frozen checkpoint; any extra seats (4p) use the scripted bot.
        return "frozen" if player == 1 else "scripted"
    return None


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def collect_episode(
    policy: ReworkedHierarchicalPolicy,
    buffer: ReworkedBuffer,
    *,
    max_steps: int = 300,
    num_agents: int = 2,
    reward_cfg: Optional[RewardConfig] = None,
    potential_fn: Optional[Callable[[object, int, int, "RewardConfig"], float]] = None,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    routing_cost: bool = True,
    opponent: str = "self_play",
    scripted_strategy: str = "production",
    frozen_policy: Optional[ReworkedHierarchicalPolicy] = None,
    frozen_deterministic: bool = True,
    include_value_rate: bool = False,
    policy_kind: str = "hier",
    teacher_forcing: float = 0.0,
    min_send: int = MIN_SEND,
    record_bc: bool = False,
    remastered_cfg: Optional[RemasteredRewardConfig] = None,
) -> EpisodeStats:
    """Run one episode (2 or 4 players), one transition per *training* player / turn.

    ``policy_kind``:
        * ``"hier"`` -- legacy hierarchical policy + min-cost-flow allocator.
        * ``"launch"`` -- v8 source-centric launch policy (allocator-free): each
          own planet samples a commit fraction + Dirichlet target split and
          launches directly.  The learning players use this head; scripted
          opponents are unaffected (they still drive the allocator).

    Reward = potential-based shaping ``gamma * Phi(s') - Phi(s)`` on the
    pluggable potential ``potential_fn`` (default :func:`economic_potential`),
    plus the terminal win/loss bonus.  ``Phi`` of a terminal state is taken as 0
    (no future), keeping the shaping telescoping and policy-invariant.

    ``num_agents``:
        ``2`` or ``4`` (Orbit Wars env contract).  Ownership features already
        encode self + 3 enemy slots, so the reworked stack needs no arch change.

    ``opponent``:
        * ``"self_play"`` -- every seat runs the live ``policy``; all transitions
          stored (2p or 4p self-play).
        * ``"scripted"`` -- P0 = live policy (stored); P1..P{n-1} = scripted bot.
        * ``"frozen"`` -- P0 = live policy; P1 = ``frozen_policy``; in 4p, P2–P3
          scripted (same ``scripted_strategy``).

    ``teacher_forcing``:
        Per-decision probability in ``[0, 1]`` that a *learning* player executes
        the scripted bot's move (``scripted_strategy``) this turn instead of
        sampling its own.  Teacher-forced turns are **not** stored as training
        transitions, so PPO only ever trains on actions the policy actually took;
        the expert move merely steers the trajectory into strong states the agent
        reaches too rarely on its own, manufacturing the win/advantage contrast it
        needs to escape the diffuse "spray" local optimum.  ``0.0`` disables it.

    ``min_send`` (v8 launch policy only):
        Minimum ships per launch.  Per-source/target slivers below this are
        dropped after flooring, so raising it (e.g. ``3``–``5``) forces the
        Dirichlet split to concentrate mass on a few targets to get *any* launch
        through -- a direct env-level pressure against the "spray" failure mode.
        Also prunes tiny-garrison planets from the source mask.
    """
    from kaggle_environments import make
    num_agents = _validate_num_agents(num_agents)
    if opponent not in ("self_play", "scripted", "frozen"):
        raise ValueError(f"unknown opponent={opponent!r}")
    if opponent == "frozen" and frozen_policy is None:
        raise ValueError("opponent='frozen' requires frozen_policy")

    if not (0.0 <= float(teacher_forcing) <= 1.0):
        raise ValueError(f"teacher_forcing must be in [0, 1], got {teacher_forcing!r}")
    if opponent in ("scripted", "frozen") or teacher_forcing > 0.0:
        try:
            from .reworked_scripted import scripted_action_row
        except ImportError:  # pragma: no cover - flat layout
            from reworked_scripted import scripted_action_row  # type: ignore

    reward_cfg = reward_cfg or RewardConfig()
    potential_fn = potential_fn or economic_potential
    device = device or torch.device("cpu")
    # Dedicated RNG for teacher-forcing coin flips so the injection is
    # reproducible per-episode (and independent of torch/global RNG state).
    tf_rng = np.random.default_rng(seed) if teacher_forcing > 0.0 else None
    players = list(range(num_agents))
    non_training = {p for p in players if _opponent_controller(p, opponent, num_agents) is not None}
    # The frozen opponent may be a DIFFERENT policy family than the learner
    # (e.g. a v8 launch policy training against a frozen v7 hierarchical+allocator
    # opponent), so dispatch on the frozen policy's own kind, not ``policy_kind``.
    frozen_is_launch = isinstance(frozen_policy, ReworkedLaunchPolicy)
    frozen_is_edge = isinstance(frozen_policy, ReworkedEdgePolicy)

    env = make("orbit_wars", debug=False)
    if seed is not None:
        # kaggle_environments only honours ``randomSeed`` at ``make`` time; the
        # board is generated lazily on the first ``step`` from the *global* RNG,
        # which ``make`` itself advances by a variable amount.  Re-seeding the
        # global RNG right before reset is what actually makes the episode (and
        # thus the offline credit attribution) reproducible.
        import random as _random
        env.configuration["randomSeed"] = int(seed)
        _random.seed(int(seed))
    env.reset(num_agents=num_agents)
    state = env.step([[] for _ in players])

    def obs_of(p):
        return state[p]["observation"]

    ep_rewards = {p: 0.0 for p in players}
    n_launches = 0

    # Remastered (edge v10): multi-turn attack campaigns + episode-end net reward.
    use_remastered = remastered_cfg is not None and policy_kind == "edge"

    # Legacy decision-indexed delayed credit (disabled when remastered is on).
    use_credit = bool(reward_cfg.enable_battle_credit) and not use_remastered
    policy_players = set(players) - non_training
    need_ep_log = use_credit or use_remastered
    ep_log = EpisodeLog(max_turns=max_steps, policy_players=policy_players) if need_ep_log else None
    campaign_ledger = CampaignLedger() if use_remastered else None

    def _finalize_remastered(final_turn: int) -> tuple:
        if not use_remastered or ep_log is None or campaign_ledger is None:
            return 0.0, 0, 0, None
        ep_log.add_snapshot(_build_snapshot(obs_of(0), final_turn))
        slot_credits, rstats = attribute_remastered_rewards(
            ep_log, campaign_ledger, remastered_cfg)
        apply_remastered_credits(buffer, slot_credits)
        learner = min(policy_players) if policy_players else 0
        detail = summarize_remastered_episode(
            ep_log, campaign_ledger, rstats, remastered_cfg, player=learner)
        return rstats.total, rstats.n_capture, rstats.n_defense_loss, detail

    def _finalize_credit(final_turn: int) -> tuple:
        """Capture the terminal snapshot and write FIFO battle credit onto the
        buffer transitions (decision_id == buffer index) before GAE runs.

        Returns ``(credit_total, n_capture, n_defense)`` for diagnostics.
        """
        if not use_credit:
            return 0.0, 0, 0, None
        ep_log.add_snapshot(_build_snapshot(obs_of(0), final_turn))
        ccfg = CreditConfig(
            gamma_eco=reward_cfg.gamma_eco, norm=reward_cfg.potential_norm,
            capture_weight=reward_cfg.credit_capture_weight,
            defense_weight=reward_cfg.credit_defense_weight,
            lost_race_weight=reward_cfg.credit_lost_race_weight,
            fail_weight=reward_cfg.credit_fail_weight,
            source_loss_weight=reward_cfg.credit_source_loss_weight,
            source_loss_window=reward_cfg.credit_source_loss_window,
            airborne_weight=reward_cfg.airborne_weight,
            airborne_norm=reward_cfg.airborne_norm,
        )
        credits, cstats = attribute_credits(ep_log, ccfg)
        if buffer.is_edge:
            # decision_id -> (turn record, target slot): add to the turn-level
            # reward (GAE backbone) AND the per-target credit slot (GAE splits
            # it back out into the per-campaign advantage).
            for decision_id, c in credits.items():
                if 0 <= decision_id < len(buffer.edge_decisions):
                    rec_idx, t_local = buffer.edge_decisions[decision_id]
                    val = reward_cfg.shaping_weight * float(c)
                    tr = buffer.data[rec_idx]
                    tr.reward += val
                    if tr.credit_target is not None and 0 <= t_local < len(tr.credit_target):
                        tr.credit_target[t_local] += val
        else:
            for decision_id, c in credits.items():
                if 0 <= decision_id < len(buffer):
                    buffer.add_reward(decision_id, reward_cfg.shaping_weight * float(c))
        learner = min(policy_players) if policy_players else 0
        caps, losses = planet_ownership_events(ep_log, learner)
        detail = {
            "planet_captures": caps,
            "planet_losses": losses,
            "n_planet_captures": len(caps),
            "n_planet_losses": len(losses),
        }
        return cstats.total, cstats.n_capture, cstats.n_defense_hold + cstats.n_defense_loss, detail

    for t in range(max_steps):
        env_actions = [[] for _ in players]
        pending: dict[int, int] = {}   # player -> buffer index
        phi_before: dict[int, float] = {}
        stock_before: dict[int, float] = {}   # own ship stock before the step
        prod_before: dict[int, float] = {}    # own production this turn
        launched_ships: dict[int, int] = {}   # ships committed to launches this turn
        remaining_before = max(0, max_steps - t)
        remaining_after = max(0, max_steps - (t + 1))

        # Credit bookkeeping: snapshot the (global) board the agents act on now
        # and remember the next fleet id so post-step fleets can be matched.
        turn_launches: dict[int, list] = {}   # player -> [LaunchRecord, ...]
        nfid_before = 0
        if need_ep_log:
            ep_log.add_snapshot(_build_snapshot(obs_of(0), t))
            nfid_before = int(obs_of(0).get("next_fleet_id", 0) or 0)
            # planet_id -> padded slot (for passive defense-loss routing).
            ids0 = np.asarray(obs_of(0).get("planet_ids", [])).reshape(-1)
            for slot_i, pid in enumerate(ids0):
                if int(pid) >= 0:
                    ep_log.planet_slot[int(pid)] = int(slot_i)

        for p in players:
            env_obs = obs_of(p)
            av = env_obs.get("angular_velocity", 0.0) or 0.0
            obs = build_observation(env_obs, player=p, angular_velocity=av,
                                    p_max=P_MAX, f_max=F_MAX, turn=t, max_turns=max_steps)
            parsed = parse_planets(env_obs)

            ctrl = _opponent_controller(p, opponent, num_agents)
            if ctrl is not None:
                batch = pack_reworked_batch(
                    [(obs, p)], device=device, p_max=P_MAX,
                    include_value_rate=include_value_rate)
                if ctrl == "scripted":
                    action_row = scripted_action_row(
                        scripted_strategy, batch["states"][0], p, p_max=P_MAX)
                    env_action, _ = action_to_env_launches(
                        obs, p, action_row, parsed, av, routing_cost=routing_cost)
                elif frozen_is_launch:
                    st0 = batch["states"][0]
                    source_np, dest_np = _source_dest_masks(st0, P_MAX, p, min_send=min_send)
                    with torch.no_grad():
                        frozen_policy.eval()
                        act, _, _ = frozen_policy.act(
                            node_feats=batch["node_feats"],
                            edge_feats=batch["edge_feats"],
                            global_feats=batch["global_feats"],
                            planet_mask=batch["planet_mask"],
                            source_mask=torch.from_numpy(source_np)[None].to(device),
                            dest_mask=torch.from_numpy(dest_np)[None].to(device),
                            deterministic=frozen_deterministic,
                        )
                    action_row = {
                        "commit": act["commit"][0].cpu().numpy(),
                        "split_w": act["split_w"][0].cpu().numpy(),
                    }
                    env_action, _ = launch_action_to_env(
                        st0, obs, p, action_row, parsed, av, min_send=min_send)
                elif frozen_is_edge:
                    st0 = batch["states"][0]
                    edge_np = _edge_valid_mask(st0, P_MAX, p, min_send=min_send)
                    with torch.no_grad():
                        frozen_policy.eval()
                        act, _, _ = frozen_policy.act(
                            node_feats=batch["node_feats"],
                            edge_feats=batch["edge_feats"],
                            global_feats=batch["global_feats"],
                            planet_mask=batch["planet_mask"],
                            edge_mask=torch.from_numpy(edge_np)[None].to(device),
                            deterministic=frozen_deterministic,
                        )
                    action_row = {
                        "gate": act["gate"][0].cpu().numpy(),
                        "frac": act["frac"][0].cpu().numpy(),
                    }
                    env_action, _ = edge_action_to_env(
                        st0, obs, p, action_row, parsed, av, min_send=min_send)
                else:
                    attack_np, defend_np = _target_masks(batch["states"][0], P_MAX, p)
                    with torch.no_grad():
                        frozen_policy.eval()
                        act, _, _ = frozen_policy.act(
                            node_feats=batch["node_feats"],
                            edge_feats=batch["edge_feats"],
                            global_feats=batch["global_feats"],
                            planet_mask=batch["planet_mask"],
                            attack_mask=torch.from_numpy(attack_np)[None].to(device),
                            defend_mask=torch.from_numpy(defend_np)[None].to(device),
                            deterministic=frozen_deterministic,
                        )
                    action_row = {
                        "attack_sel": act["attack_sel"][0].cpu().numpy(),
                        "defend_sel": act["defend_sel"][0].cpu().numpy(),
                        "order": act["order"][0].cpu().numpy(),
                        "commit_frac": act["commit_frac"][0].cpu().numpy(),
                        "potential_weight": float(act["potential_weight"][0]),
                        "horizon": float(act["horizon"][0]),
                    }
                    env_action, _ = action_to_env_launches(
                        obs, p, action_row, parsed, av, routing_cost=routing_cost)
                env_actions[p] = env_action
                n_launches += len(env_action)
                continue

            phi_before[p] = potential_fn(env_obs, p, remaining_before, reward_cfg)
            stock_before[p] = _own_ship_stock(env_obs, p, count_inflight=reward_cfg.count_inflight)
            prod_before[p] = _own_production(env_obs, p)
            if use_remastered and campaign_ledger is not None:
                campaign_ledger.record_stock(p, t, stock_before[p])

            batch = pack_reworked_batch(
                [(obs, p)], device=device, p_max=P_MAX,
                include_value_rate=include_value_rate)
            st0 = batch["states"][0]

            # Teacher forcing: with prob `teacher_forcing`, this learning player
            # plays the scripted bot's move instead of its own.  The turn is NOT
            # stored (no buffer.add, no launch records), so PPO never trains on
            # the expert action -- it only benefits indirectly via the stronger
            # states the expert move steers the trajectory into.
            if tf_rng is not None and float(tf_rng.random()) < teacher_forcing:
                action_row = scripted_action_row(
                    scripted_strategy, st0, p, p_max=P_MAX)
                env_action, t_launches = action_to_env_launches(
                    obs, p, action_row, parsed, av, routing_cost=routing_cost)
                env_actions[p] = env_action
                n_launches += len(env_action)
                # Behavior-cloning demo: record the teacher's executed move in
                # the v8 action space (launch learner only) for the supervised
                # BC loss in ppo_update.  Not a PPO transition.
                if record_bc and policy_kind == "launch":
                    source_np, dest_np = _source_dest_masks(
                        st0, P_MAX, p, min_send=min_send)
                    commit_t, split_t = teacher_launches_to_v8_targets(
                        st0, t_launches, dest_np, p_max=P_MAX)
                    buffer.add_bc(BCSample(
                        node_feats=batch["node_feats"][0].cpu().numpy(),
                        edge_feats=batch["edge_feats"][0].cpu().numpy(),
                        global_feats=batch["global_feats"][0].cpu().numpy(),
                        planet_mask=batch["planet_mask"][0].cpu().numpy(),
                        source_mask=source_np,
                        dest_mask=dest_np,
                        commit_target=commit_t,
                        split_target=split_t,
                    ))
                continue

            if policy_kind == "edge":
                edge_np = _edge_valid_mask(st0, P_MAX, p, min_send=min_send)
                action, target_logp, value = policy.act(
                    node_feats=batch["node_feats"], edge_feats=batch["edge_feats"],
                    global_feats=batch["global_feats"], planet_mask=batch["planet_mask"],
                    edge_mask=torch.from_numpy(edge_np)[None].to(device),
                )
                gate_np = action["gate"][0].cpu().numpy()
                frac_np = action["frac"][0].cpu().numpy()
                env_action, launches = edge_action_to_env(
                    st0, obs, p, {"gate": gate_np, "frac": frac_np},
                    parsed, av, min_send=min_send)
                env_actions[p] = env_action
                n_launches += len(env_action)
                launched_ships[p] = int(sum(L.ships for L in launches))

                tr = EdgeTransition(
                    player=p,
                    node_feats=batch["node_feats"][0].cpu().numpy(),
                    edge_feats=batch["edge_feats"][0].cpu().numpy(),
                    global_feats=batch["global_feats"][0].cpu().numpy(),
                    planet_mask=batch["planet_mask"][0].cpu().numpy(),
                    edge_mask=edge_np,
                    gate=gate_np,
                    frac=frac_np,
                    old_target_logp=target_logp[0].cpu().numpy(),
                    target_valid=_edge_target_valid_mask(
                        gate_np, edge_np, launches=launches),
                    value=float(value[0]),
                    credit_target=np.zeros(P_MAX, dtype=np.float32),
                )
                pending[p] = buffer.add(tr)
                if need_ep_log:
                    ep_log.add_decision(p, t, pending[p])
            elif policy_kind == "launch":
                source_np, dest_np = _source_dest_masks(st0, P_MAX, p, min_send=min_send)
                action, log_prob, value = policy.act(
                    node_feats=batch["node_feats"], edge_feats=batch["edge_feats"],
                    global_feats=batch["global_feats"], planet_mask=batch["planet_mask"],
                    source_mask=torch.from_numpy(source_np)[None].to(device),
                    dest_mask=torch.from_numpy(dest_np)[None].to(device),
                )
                action_row = {
                    "commit": action["commit"][0].cpu().numpy(),
                    "split_w": action["split_w"][0].cpu().numpy(),
                }
                env_action, launches = launch_action_to_env(
                    st0, obs, p, action_row, parsed, av, min_send=min_send)
                env_actions[p] = env_action
                n_launches += len(env_action)
                launched_ships[p] = int(sum(L.ships for L in launches))

                tr = LaunchTransition(
                    player=p,
                    node_feats=batch["node_feats"][0].cpu().numpy(),
                    edge_feats=batch["edge_feats"][0].cpu().numpy(),
                    global_feats=batch["global_feats"][0].cpu().numpy(),
                    planet_mask=batch["planet_mask"][0].cpu().numpy(),
                    source_mask=source_np,
                    dest_mask=dest_np,
                    commit=action_row["commit"],
                    split_w=action_row["split_w"],
                    log_prob=float(log_prob[0]),
                    value=float(value[0]),
                )
                pending[p] = buffer.add(tr)
            else:
                attack_np, defend_np = _target_masks(st0, P_MAX, p)
                attack_mask = torch.from_numpy(attack_np)[None].to(device)
                defend_mask = torch.from_numpy(defend_np)[None].to(device)

                action, log_prob, value = policy.act(
                    node_feats=batch["node_feats"], edge_feats=batch["edge_feats"],
                    global_feats=batch["global_feats"], planet_mask=batch["planet_mask"],
                    attack_mask=attack_mask, defend_mask=defend_mask,
                )
                action_row = {
                    "attack_sel": action["attack_sel"][0].cpu().numpy(),
                    "defend_sel": action["defend_sel"][0].cpu().numpy(),
                    "order": action["order"][0].cpu().numpy(),
                    "commit_frac": action["commit_frac"][0].cpu().numpy(),
                    "potential_weight": float(action["potential_weight"][0]),
                    "horizon": float(action["horizon"][0]),
                }
                env_action, launches = action_to_env_launches(
                    obs, p, action_row, parsed, av, routing_cost=routing_cost)
                env_actions[p] = env_action
                n_launches += len(env_action)
                launched_ships[p] = int(sum(L.ships for L in launches))

                tr = HierTransition(
                    player=p,
                    node_feats=batch["node_feats"][0].cpu().numpy(),
                    edge_feats=batch["edge_feats"][0].cpu().numpy(),
                    global_feats=batch["global_feats"][0].cpu().numpy(),
                    planet_mask=batch["planet_mask"][0].cpu().numpy(),
                    attack_mask=attack_np,
                    defend_mask=defend_np,
                    attack_sel=action_row["attack_sel"],
                    defend_sel=action_row["defend_sel"],
                    order=action_row["order"].astype(np.int64),
                    commit_frac=action_row["commit_frac"],
                    posture_u=action["posture_u"][0].cpu().numpy(),
                    log_prob=float(log_prob[0]),
                    value=float(value[0]),
                )
                pending[p] = buffer.add(tr)

            # Build per-fleet launch records.  For hier/launch the decision unit
            # is the whole turn (decision_id == buffer index).  For the edge
            # policy the decision unit is the TARGET campaign: every launch
            # landing on the same target ``t`` shares one decision_id (mapped to
            # that target's slot inside this turn's transition), so capture /
            # defence / fail / source-loss credit attaches per campaign.
            if need_ep_log and launches:
                ids = np.asarray(obs.planet_ids).reshape(-1)
                owner_by_id = {int(pl.id): int(pl.owner) for pl in parsed}
                edge_kind = (policy_kind == "edge")
                dec_for_target: dict[int, int] = {}
                # Per-source enemy-inbound flux (what the agent sees in edge
                # features): column-sum of the enemy_flux channel = enemy ships
                # currently heading *into* each source slot.  Used to stamp
                # whether a launch stripped a planet that was already under attack.
                enemy_in_col = None
                if edge_kind and use_remastered:
                    ef = batch["edge_feats"][0].detach().cpu().numpy()
                    enemy_in_col = ef[:, :, EDGE_IDX_ENEMY_FLUX].sum(axis=0)
                recs = []
                for L in launches:
                    if not (0 <= L.src < len(ids) and 0 <= L.tgt < len(ids)):
                        continue
                    src_id = int(ids[L.src]); tgt_id = int(ids[L.tgt])
                    is_attack = owner_by_id.get(tgt_id, -1) != p
                    t_local_rec = int(L.tgt) if edge_kind else -1
                    src_under_threat = bool(
                        enemy_in_col is not None
                        and 0 <= int(L.src) < enemy_in_col.shape[0]
                        and float(enemy_in_col[int(L.src)]) > 1e-6)
                    campaign_id = -1
                    if edge_kind and use_remastered and campaign_ledger is not None:
                        t_local = int(L.tgt)
                        st = stock_before.get(p, _own_ship_stock(
                            env_obs, p, count_inflight=reward_cfg.count_inflight))
                        if is_attack:
                            tgt_owner = owner_by_id.get(tgt_id, -1)
                            tgt_was_enemy = tgt_owner not in (-1, p)
                            campaign_id = campaign_ledger.note_attack(
                                player=p, tgt_id=tgt_id, t_decision=t,
                                record_idx=pending[p], t_local=t_local,
                                ships=int(L.ships), stock=st,
                                tgt_was_enemy=tgt_was_enemy)
                        else:
                            campaign_id = campaign_ledger.note_defense(
                                player=p, tgt_id=tgt_id, t_decision=t,
                                record_idx=pending[p], t_local=t_local,
                                ships=int(L.ships), stock=st)
                        did = campaign_id
                    elif edge_kind:
                        t_local = int(L.tgt)
                        did = dec_for_target.get(t_local)
                        if did is None:
                            did = buffer.add_edge_decision(pending[p], t_local)
                            dec_for_target[t_local] = did
                    else:
                        did = pending[p]
                    recs.append(LaunchRecord(
                        decision_id=did, player=p, launch_turn=t,
                        src_id=src_id, tgt_id=tgt_id, ships=int(L.ships),
                        is_attack=is_attack,
                        angle=float(getattr(L, "angle", 0.0)),
                        campaign_id=campaign_id,
                        t_local=t_local_rec,
                        src_under_threat=src_under_threat,
                    ))
                turn_launches[p] = recs

        prev_obs_credit = obs_of(0) if use_remastered else None
        state = env.step(list(env_actions))
        done = bool(getattr(env, "done", False)) or any(
            getattr(state[p], "status", "ACTIVE") == "DONE" for p in players)

        if use_remastered and prev_obs_credit is not None and campaign_ledger is not None:
            av_kill = float(obs_of(0).get("angular_velocity", 0) or 0.0)
            for p in policy_players:
                kill_credits, _ = observe_enemy_kills(
                    campaign_ledger, ep_log, prev_obs_credit, obs_of(0),
                    turn=t, player=p, omega=av_kill, cfg=remastered_cfg)
                apply_remastered_credits(buffer, kill_credits)

        # Match the env fleets just created (id >= nfid_before) to this turn's
        # launch records by (owner, source, ships); compute_launch preserves the
        # ship count exactly, so the match is unambiguous up to identical orders.
        if need_ep_log and turn_launches:
            new_fleets = sorted(
                (f for f in parse_fleets(obs_of(0)) if int(f.id) >= nfid_before),
                key=lambda f: int(f.id),
            )
            for p, recs in turn_launches.items():
                pool = list(recs)
                for f in new_fleets:
                    if int(f.owner) != p:
                        continue
                    # Candidates share (source, ship count); when several do
                    # (one source -> several targets, same size), the launch
                    # heading uniquely identifies which target this fleet is for,
                    # so pick the unmatched record whose angle is closest.
                    best_k = -1
                    best_da = None
                    for k, rec in enumerate(pool):
                        if rec.fleet_id >= 0:
                            continue
                        if rec.src_id == int(f.from_planet_id) and rec.ships == int(f.ships):
                            da = _angle_diff(rec.angle, float(f.angle))
                            if best_da is None or da < best_da:
                                best_da = da
                                best_k = k
                    if best_k >= 0:
                        rec = pool[best_k]
                        rec.fleet_id = int(f.id)
                        ep_log.add_launch(rec)
                        pool.pop(best_k)

        for p in pending:  # only policy players have stored transitions
            r = 0.0
            if use_remastered:
                # Campaign rewards are applied at episode end; only terminal here.
                if done and remastered_cfg is not None:
                    rp = getattr(state[p], "reward", None)
                    term = float(rp) if rp is not None else 0.0
                    r += remastered_cfg.terminal_weight * term
            else:
                if reward_cfg.enable_shaping:
                    # Phi(terminal) := 0 (no future) -> shaping stays telescoping.
                    phi_after = 0.0 if done else potential_fn(obs_of(p), p, remaining_after, reward_cfg)
                    r += reward_cfg.shaping_weight * (reward_cfg.gamma * phi_after - phi_before[p])

                # Behavioural cost terms (non-telescoping): penalise ships actually
                # DESTROYED this turn (combat losses, sun crashes, failed attacks)
                # and a small per-ship launch cost to discourage gratuitous spray.
                norm = max(reward_cfg.loss_norm, 1e-6)
                if reward_cfg.loss_weight != 0.0:
                    stock_after = (0.0 if done else
                                   _own_ship_stock(obs_of(p), p, count_inflight=reward_cfg.count_inflight))
                    if not done:
                        destroyed = max(0.0, stock_before[p] + prod_before[p] - stock_after)
                        r -= reward_cfg.loss_weight * destroyed / norm
                if reward_cfg.launch_weight != 0.0:
                    r -= reward_cfg.launch_weight * float(launched_ships.get(p, 0)) / norm

                if done:
                    rp = getattr(state[p], "reward", None)
                    term = float(rp) if rp is not None else 0.0
                    r += reward_cfg.terminal_weight * term
            buffer.data[pending[p]].reward += r
            buffer.data[pending[p]].done = done
            ep_rewards[p] += r

        if done:
            rewards = {}
            for p in players:
                rp = getattr(state[p], "reward", None)
                rewards[p] = float(rp) if rp is not None else 0.0
            winner = max(rewards, key=rewards.get) if rewards else None
            finalize = _finalize_remastered if use_remastered else _finalize_credit
            ctot, ncap, ndef, detail = finalize(t + 1)
            return EpisodeStats(length=t + 1, rewards=ep_rewards, winner=winner,
                                num_agents=num_agents, n_launches=n_launches,
                                credit_total=ctot, n_capture=ncap, n_defense=ndef,
                                episode_log=detail)

    finalize = _finalize_remastered if use_remastered else _finalize_credit
    ctot, ncap, ndef, detail = finalize(max_steps)
    return EpisodeStats(length=max_steps, rewards=ep_rewards, winner=None,
                        num_agents=num_agents, n_launches=n_launches,
                        credit_total=ctot, n_capture=ncap, n_defense=ndef,
                        episode_log=detail)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_adv: bool = True
    # Per-rollout KL early stop: if an epoch's mean approx_kl exceeds this, the
    # remaining inner epochs are skipped (stabilises the Beta/Categorical heads).
    # Set to None to disable.
    target_kl: Optional[float] = 0.05
    # Behavior-cloning coefficient (v8 launch only): weight of the supervised
    # ``-log pi(a_teacher | s)`` loss on teacher-forced demonstrations.  0 = off.
    bc_coef: float = 0.0


@dataclass
class PPOStats:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    epochs_run: int = 0
    bc_loss: float = 0.0
    bc_samples: int = 0


def ppo_update(
    policy: ReworkedHierarchicalPolicy,
    buffer: ReworkedBuffer,
    optimizer: torch.optim.Optimizer,
    cfg: Optional[PPOConfig] = None,
    *,
    device: Optional[torch.device] = None,
) -> PPOStats:
    """One PPO update over the buffered hierarchical transitions."""
    cfg = cfg or PPOConfig()
    device = device or torch.device("cpu")
    if len(buffer) == 0:
        return PPOStats()

    data = buffer.stack(device=device)
    N = data["node_feats"].shape[0]
    is_launch = buffer.is_launch
    is_edge = buffer.is_edge
    if is_edge:
        # Per-target advantages normalised over the valid (campaign) entries.
        adv_t = data["adv_target"]
        tv_all = data["target_valid"]
        if cfg.normalize_adv and bool(tv_all.any()):
            v = adv_t[tv_all]
            if v.numel() > 1:
                adv_t = (adv_t - v.mean()) / (v.std() + 1e-8)
        adv = None
    else:
        adv = data["advantage"]
        if cfg.normalize_adv and N > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = PPOStats()
    n_batches = 0
    for _ in range(cfg.epochs):
        epoch_kl = 0.0
        epoch_batches = 0
        perm = torch.randperm(N, device=device)
        for start in range(0, N, cfg.minibatch_size):
            mb = perm[start:start + cfg.minibatch_size]
            if is_edge:
                ev = policy.evaluate_actions(
                    node_feats=data["node_feats"][mb], edge_feats=data["edge_feats"][mb],
                    global_feats=data["global_feats"][mb], planet_mask=data["planet_mask"][mb],
                    edge_mask=data["edge_mask"][mb],
                    action={"gate": data["gate"][mb], "frac": data["frac"][mb]},
                )
                tvm = data["target_valid"][mb].to(ev["target_logp"].dtype)  # (b, P)
                denom = tvm.sum().clamp_min(1.0)
                new_tlp = ev["target_logp"]
                old_tlp = data["old_target_logp"][mb]
                logp_diff = torch.clamp(new_tlp - old_tlp, -20.0, 20.0)
                ratio = torch.exp(logp_diff)
                a = adv_t[mb]
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * a
                policy_loss = -(torch.min(surr1, surr2) * tvm).sum() / denom
                value_loss = F.mse_loss(ev["value"], data["ret"][mb])
                entropy = (ev["entropy_target"] * tvm).sum() / denom
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                if not torch.isfinite(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                if not torch.isfinite(
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                ):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                with torch.no_grad():
                    batch_kl = float(((old_tlp - new_tlp) * tvm).sum() / denom)
                    stats.policy_loss += float(policy_loss)
                    stats.value_loss += float(value_loss)
                    stats.entropy += float(entropy)
                    stats.approx_kl += batch_kl
                epoch_kl += batch_kl
                epoch_batches += 1
                n_batches += 1
                continue
            if is_launch:
                ev = policy.evaluate_actions(
                    node_feats=data["node_feats"][mb], edge_feats=data["edge_feats"][mb],
                    global_feats=data["global_feats"][mb], planet_mask=data["planet_mask"][mb],
                    source_mask=data["source_mask"][mb], dest_mask=data["dest_mask"][mb],
                    action={"commit": data["commit"][mb], "split_w": data["split_w"][mb]},
                )
            else:
                action = {
                    "attack_sel": data["attack_sel"][mb],
                    "defend_sel": data["defend_sel"][mb],
                    "order": data["order"][mb],
                    "commit_frac": data["commit_frac"][mb],
                    "posture_u": data["posture_u"][mb],
                }
                ev = policy.evaluate_actions(
                    node_feats=data["node_feats"][mb], edge_feats=data["edge_feats"][mb],
                    global_feats=data["global_feats"][mb], planet_mask=data["planet_mask"][mb],
                    attack_mask=data["attack_mask"][mb], defend_mask=data["defend_mask"][mb],
                    action=action,
                )
            new_logp = ev["log_prob"]
            old_logp = data["old_log_prob"][mb]
            # Clamp log-prob drift so exp(delta) cannot overflow (launch Dirichlet
            # can spike on rare minibatches after a large policy shift).
            logp_diff = torch.clamp(new_logp - old_logp, -20.0, 20.0)
            ratio = torch.exp(logp_diff)
            a = adv[mb]
            surr1 = ratio * a
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * a
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(ev["value"], data["ret"][mb])
            entropy = ev["entropy"].mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            if not torch.isfinite(loss):
                continue  # skip poisoned minibatch (e.g. weights already NaN)

            optimizer.zero_grad()
            loss.backward()
            if not torch.isfinite(
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            ):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

            with torch.no_grad():
                batch_kl = float((data["old_log_prob"][mb] - new_logp).mean())
                stats.policy_loss += float(policy_loss)
                stats.value_loss += float(value_loss)
                stats.entropy += float(entropy)
                stats.approx_kl += batch_kl
            epoch_kl += batch_kl
            epoch_batches += 1
            n_batches += 1

        stats.epochs_run += 1
        if cfg.target_kl is not None and epoch_batches:
            if (epoch_kl / epoch_batches) > cfg.target_kl:
                break  # KL budget exceeded -> stop early to avoid policy collapse

    if n_batches:
        stats.policy_loss /= n_batches
        stats.value_loss /= n_batches
        stats.entropy /= n_batches
        stats.approx_kl /= n_batches

    # --- Behavior cloning on teacher-forced demonstrations (launch only) -----
    # A supervised maximum-likelihood pull toward the expert's executed launches:
    # maximise pi(a_teacher | s) so the policy directly imitates the teacher's
    # concentrated, capturing allocation.  Kept separate from PPO so it never
    # biases the on-policy ratio / advantages.
    bc_n = len(getattr(buffer, "bc_data", []))
    if is_launch and cfg.bc_coef > 0.0 and bc_n > 0:
        bc = buffer.stack_bc(device=device)
        Nbc = bc["node_feats"].shape[0]
        bc_loss_acc = 0.0
        bc_batches = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(Nbc, device=device)
            for start in range(0, Nbc, cfg.minibatch_size):
                mb = perm[start:start + cfg.minibatch_size]
                ev = policy.evaluate_actions(
                    node_feats=bc["node_feats"][mb], edge_feats=bc["edge_feats"][mb],
                    global_feats=bc["global_feats"][mb], planet_mask=bc["planet_mask"][mb],
                    source_mask=bc["source_mask"][mb], dest_mask=bc["dest_mask"][mb],
                    action={"commit": bc["commit"][mb], "split_w": bc["split_w"][mb]},
                )
                bc_loss = -ev["log_prob"].mean()
                loss = cfg.bc_coef * bc_loss
                if not torch.isfinite(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                if not torch.isfinite(
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                ):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                bc_loss_acc += float(bc_loss.detach())
                bc_batches += 1
        stats.bc_loss = bc_loss_acc / max(1, bc_batches)
        stats.bc_samples = bc_n
    return stats


__all__ = [
    "P_MAX", "F_MAX", "MIN_SEND",
    "ALLOWED_NUM_AGENTS",
    "action_to_env_launches", "launch_action_to_env", "edge_action_to_env",
    "teacher_launches_to_v8_targets",
    "RewardConfig", "economic_potential",
    "HierTransition", "LaunchTransition", "EdgeTransition", "BCSample",
    "ReworkedBuffer", "EpisodeStats",
    "collect_episode", "PPOConfig", "PPOStats", "ppo_update",
]
