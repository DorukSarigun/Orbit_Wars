"""Decision-indexed delayed credit via FIFO inventory accounting.

Motivation
----------
A launch made at turn ``t`` only pays off when its fleet *lands* at ``t + eta``
and resolves a battle (captures a planet, reinforces a defence, or is wasted).
Potential-based shaping on the economic differential does propagate that value
back through GAE, but the signal is smeared across the whole flight and -- for
**defence** -- never fires at all, because successfully holding a planet leaves
the potential unchanged (you didn't *gain* anything, you avoided a loss).

This module closes that gap with a single, ground-truth, **post-episode**
attribution pass.  Because training starts only once the game is over, we have
the *exact* per-turn board (owner / ships / production) and the *exact* set of
in-flight fleet ids -- nothing is estimated.  We replay that log and route the
realised value of every battle back to the **decision** (one stored transition
per player per turn) whose ships actually fought in it.

Crediting rule (the "distribute a planet's battles to its decisions" idea the
user asked for, made causal):

* Each planet carries a **FIFO stack of ship layers**, every layer tagged with
  the ``decision_id`` that produced it (or ``NATIVE`` for the initial garrison
  and per-turn production -- those belong to no decision and are never
  credited).
* **Attack / expansion.**  When my fleets capture an enemy/neutral planet
  ``g`` (ownership flips to me), the attack decisions that supplied the
  surviving ships share ``+dpp(g)`` by ship count, and ``g``'s new garrison is
  re-tagged with those decisions.  A failed attack pays no positive credit
  (its ship loss is handled by the separate behavioural loss penalty).
* **Defence.**  When an enemy attack actually *lands* on my planet ``g``, the
  ships it neutralises are consumed **FIFO** from ``g``'s stack:
    - planet *held*  -> the consumed decision layers share ``+dpp(g)``
      (they "saved" the planet's economy);
    - planet *lost*  -> the consumed decision layers share ``-dpp(g)``
      (they were committed to a defence that failed -- exactly the negative
      the user wanted for a defence order that couldn't hold).
  FIFO means the oldest ships (initial garrison + early production, i.e.
  ``NATIVE``) absorb first, so a reinforcement decision is only credited when
  the attack was big enough to actually need it -- a built-in *pivotal* test.
* **No attack ever came** -> the standing reinforcement earns nothing here
  (pure insurance); its opportunity cost is left to the launch / loss penalties.

The pass is **policy-agnostic** and writes its result straight onto the buffer
transitions *before* GAE, so PPO/GAE then discount it from the decision turn
forward exactly as with any other reward.

Double-counting
---------------
``+dpp(g)`` / ``-dpp(g)`` is the *production* (ownership) value of a planet.
The economic potential ``Phi`` also moves that value on ownership flips, so the
two must not both be active.  When battle credit is enabled the rollout drops
the production/ownership term from ``Phi`` (``potential_mode="ships_only"``);
the ship-stock term stays, so ship losses remain telescoped as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:  # package layout
    from .reworked_features import discounted_production_potential, DEFAULT_GAMMA, DEFAULT_NORM
except ImportError:  # pragma: no cover - flat layout
    from reworked_features import discounted_production_potential, DEFAULT_GAMMA, DEFAULT_NORM  # type: ignore


NATIVE: int = -1   # decision tag for un-credited ships (initial garrison + production)


# ---------------------------------------------------------------------------
# Episode log dataclasses (populated by the rollout)
# ---------------------------------------------------------------------------
@dataclass
class LaunchRecord:
    """One fleet launched by a *policy* (stored) player, tied to its decision.

    ``decision_id`` is the buffer index of the single stored transition for
    that player on that turn -- every fleet from the same turn shares it (one
    RL action per turn), so crediting collapses cleanly onto one transition.
    """
    decision_id: int
    player: int
    launch_turn: int
    src_id: int
    tgt_id: int
    ships: int
    is_attack: bool          # target not owned by the player at launch time
    fleet_id: int = -1       # env fleet id, filled after the step
    resolve_turn: int = -1   # turn the fleet left the board (filled offline)


@dataclass
class TurnSnapshot:
    """Ground-truth board the agents acted on at the start of turn ``turn``."""
    turn: int
    owner: dict          # planet_id -> owner (int, -1 neutral)
    ships: dict          # planet_id -> ships (float)
    production: dict      # planet_id -> production (float)
    fleet_ids: set        # set[int] of in-flight fleet ids present


@dataclass
class EpisodeLog:
    """Everything the offline credit pass needs; filled during the rollout."""
    max_turns: int
    snapshots: list = field(default_factory=list)        # list[TurnSnapshot], one per turn (+1 terminal)
    launches: list = field(default_factory=list)         # list[LaunchRecord]
    policy_players: set = field(default_factory=set)      # players whose transitions are stored

    def add_snapshot(self, snap: TurnSnapshot) -> None:
        self.snapshots.append(snap)

    def add_launch(self, rec: LaunchRecord) -> None:
        self.launches.append(rec)


# ---------------------------------------------------------------------------
# FIFO layer helpers
# ---------------------------------------------------------------------------
def _layer_sum(layers: list) -> float:
    return float(sum(s for _tag, s in layers))


def _consume_fifo(layers: list, amount: float) -> dict:
    """Remove ``amount`` ships from the front of ``layers`` (oldest first).

    Returns ``{tag: ships_consumed}`` for the layers that were drawn down.
    Mutates ``layers`` in place (depleted layers are dropped, a partially
    consumed layer is shrunk).
    """
    consumed: dict = {}
    remaining = float(amount)
    while remaining > 1e-9 and layers:
        tag, ships = layers[0]
        take = min(float(ships), remaining)
        consumed[tag] = consumed.get(tag, 0.0) + take
        remaining -= take
        if take >= float(ships) - 1e-9:
            layers.pop(0)
        else:
            layers[0] = [tag, float(ships) - take]
    return consumed


def _push_layer(layers: list, tag: int, ships: float) -> None:
    if ships > 1e-9:
        layers.append([int(tag), float(ships)])


def _distribute(credit: dict, consumed: dict, value: float) -> None:
    """Split ``value`` over the *decision* tags in ``consumed`` by ship share.

    ``NATIVE`` ships absorb their share silently (no decision to credit), which
    is what makes FIFO consumption a pivotal test: only ships that came from a
    real decision earn (or lose) credit.
    """
    decision_ships = {t: s for t, s in consumed.items() if t != NATIVE}
    total = sum(decision_ships.values())
    if total <= 1e-9:
        return
    for tag, ships in decision_ships.items():
        credit[tag] = credit.get(tag, 0.0) + value * (ships / total)


# ---------------------------------------------------------------------------
# Offline attribution pass
# ---------------------------------------------------------------------------
@dataclass
class CreditConfig:
    gamma_eco: float = DEFAULT_GAMMA
    norm: float = DEFAULT_NORM
    capture_weight: float = 1.0    # scale on +dpp(g) for a successful capture
    defense_weight: float = 1.0    # scale on +/-dpp(g) for a resolved defence
    # Decision-indexed penalties for attacks that never paid off.  Both default
    # to 0.0 so the pass is bit-identical to the capture/defence-only behaviour
    # unless explicitly enabled.  They use the *campaign ledger* (all of a
    # player's attack arrivals against a target accumulate until the planet's
    # ownership resolves), which is robust to multi-source / staggered arrivals:
    #   * ``lost_race_weight`` -- the target flips to a *different* player while
    #     my fleets were committed to it (I spent ships and the rival took it).
    #   * ``fail_weight``      -- the episode ends with the target never captured
    #     by me (ships thrown at a planet I never took -- pure waste).
    # A campaign that *succeeds* (planet flips to me) clears my pending ledger
    # with no penalty, so softening waves of a winning multi-wave assault are
    # never charged.
    lost_race_weight: float = 0.0
    fail_weight: float = 0.0
    # Optional extra discount of the realised value back toward the launch turn
    # (1.0 = none; the decision transition is the cause, so default is no extra
    # discount -- GAE already discounts from that transition forward).
    event_discount: float = 1.0


@dataclass
class CreditStats:
    total: float = 0.0
    n_capture: int = 0
    n_defense_hold: int = 0
    n_defense_loss: int = 0
    n_lost_race: int = 0
    n_fail_attack: int = 0
    capture_credit: float = 0.0
    defense_credit: float = 0.0
    lost_race_credit: float = 0.0
    fail_credit: float = 0.0


def attribute_credits(log: EpisodeLog, cfg: Optional[CreditConfig] = None
                      ) -> "tuple[dict, CreditStats]":
    """Replay the episode log and return ``{decision_id: credit}`` + stats.

    The walk processes each transition ``snap[t] -> snap[t+1]`` in env order
    (launch -> production -> combat) using observed deltas for magnitudes, so
    it never re-derives combat geometry and stays exact w.r.t. the real game.
    """
    cfg = cfg or CreditConfig()
    credit: dict = {}
    stats = CreditStats()

    snaps = log.snapshots
    if len(snaps) < 2:
        return credit, stats

    # --- index launches -----------------------------------------------------
    # by source-turn (for departures) and by resolve-turn+target (arrivals).
    launches_by_turn: dict = {}
    for rec in log.launches:
        launches_by_turn.setdefault(rec.launch_turn, []).append(rec)

    # Resolve turn = first turn the fleet id is gone.  A fleet launched at turn
    # t first appears in snap[t+1]; it resolves at the turn its id disappears.
    fleet_to_launch: dict = {r.fleet_id: r for r in log.launches if r.fleet_id >= 0}
    for i in range(len(snaps) - 1):
        gone = snaps[i].fleet_ids - snaps[i + 1].fleet_ids
        for fid in gone:
            rec = fleet_to_launch.get(fid)
            if rec is not None and rec.resolve_turn < 0:
                rec.resolve_turn = snaps[i].turn  # resolved during this step
    arrivals_by_turn: dict = {}
    for rec in log.launches:
        if rec.resolve_turn >= 0:
            arrivals_by_turn.setdefault(rec.resolve_turn, []).append(rec)

    # --- per-planet FIFO state ---------------------------------------------
    layers: dict = {}     # planet_id -> list[[tag, ships]]
    owner: dict = {}      # planet_id -> tracked owner
    # Campaign ledger: attack LaunchRecords that have arrived on a target not
    # yet owned by their launcher and whose outcome is still unresolved.  Keyed
    # by planet id; resolved (and cleared) on the next ownership flip, swept for
    # the never-captured case at episode end.  Only used when the lost-race /
    # fail penalties are enabled, so the default pass is unchanged.
    pending_attacks: dict = {}   # planet_id -> list[LaunchRecord]
    track_campaign = bool(cfg.lost_race_weight or cfg.fail_weight)

    s0 = snaps[0]
    for pid, o in s0.owner.items():
        owner[pid] = int(o)
        layers[pid] = [[NATIVE, float(s0.ships.get(pid, 0.0))]]

    def dpp_of(pid: int, prod: float, turn: int) -> float:
        R = max(0, log.max_turns - int(turn))
        return float(discounted_production_potential(
            np.asarray([float(prod)], dtype=np.float64), R,
            gamma=cfg.gamma_eco, norm=cfg.norm)[0])

    for i in range(len(snaps) - 1):
        cur, nxt = snaps[i], snaps[i + 1]
        t = cur.turn

        # ensure tracked state knows every planet present now
        for pid, o in cur.owner.items():
            if pid not in layers:
                owner[pid] = int(o)
                layers[pid] = [[NATIVE, float(cur.ships.get(pid, 0.0))]]

        # (0) departures: launches made THIS turn drain their source garrison
        for rec in launches_by_turn.get(t, []):
            if rec.src_id in layers and owner.get(rec.src_id) == rec.player:
                _consume_fifo(layers[rec.src_id], float(rec.ships))

        # (1) production: every owned planet grows by its production (NATIVE)
        for pid in list(layers.keys()):
            if owner.get(pid, -1) != -1 and pid in cur.production:
                _push_layer(layers[pid], NATIVE, float(cur.production.get(pid, 0.0)))

        # (2) combat: reconcile each planet's tracked stack with the observed
        # post-state, crediting decisions whose ships fought.
        arrivals = arrivals_by_turn.get(t, [])
        arrivals_by_tgt: dict = {}
        for rec in arrivals:
            arrivals_by_tgt.setdefault(rec.tgt_id, []).append(rec)

        touched = set(arrivals_by_tgt.keys())
        # any planet whose owner or ship count changed also needs reconciling
        for pid in layers:
            if pid not in nxt.owner:
                continue
            if int(nxt.owner[pid]) != owner.get(pid, -2) or \
               abs(float(nxt.ships.get(pid, 0.0)) - _layer_sum(layers[pid])) > 1e-6:
                touched.add(pid)

        for pid in touched:
            if pid not in nxt.owner:        # planet vanished (comet expiry)
                layers.pop(pid, None)
                owner.pop(pid, None)
                continue
            o_post = int(nxt.owner[pid])
            s_post = float(nxt.ships.get(pid, 0.0))
            o_cur = owner.get(pid, -1)
            mine_arrivals = arrivals_by_tgt.get(pid, [])

            # Campaign ledger: log my attack waves landing on a not-yet-mine
            # target.  Done before resolution so a same-turn capture/lost-race
            # sees this turn's waves too.
            if track_campaign:
                for rec in mine_arrivals:
                    if (rec.is_attack and rec.player in log.policy_players
                            and o_cur != rec.player):
                        pending_attacks.setdefault(pid, []).append(rec)

            # --- planet currently mine (a policy player's) -----------------
            if o_cur in log.policy_players:
                # push my reinforcement arrivals as decision-tagged layers
                for rec in mine_arrivals:
                    if rec.player == o_cur and not rec.is_attack:
                        _push_layer(layers[pid], rec.decision_id, float(rec.ships))
                M = _layer_sum(layers[pid])
                prod = float(cur.production.get(pid, nxt.production.get(pid, 0.0)))
                if o_post == o_cur:
                    # planet held; observed loss = ships the attack neutralised
                    diff = M - s_post
                    if diff > 1e-9:
                        consumed = _consume_fifo(layers[pid], diff)
                        val = cfg.defense_weight * dpp_of(pid, prod, t)
                        _distribute(credit, consumed, +val)
                        if any(tag != NATIVE for tag in consumed):
                            stats.n_defense_hold += 1
                            stats.defense_credit += val
                    elif diff < -1e-9:
                        _push_layer(layers[pid], NATIVE, -diff)  # unmatched gain
                else:
                    # planet LOST: whole defending stack neutralised
                    consumed = _consume_fifo(layers[pid], M)
                    val = cfg.defense_weight * dpp_of(pid, prod, t)
                    _distribute(credit, consumed, -val)
                    if any(tag != NATIVE for tag in consumed):
                        stats.n_defense_loss += 1
                        stats.defense_credit -= val
                    owner[pid] = o_post
                    layers[pid] = [[NATIVE, s_post]]

            # --- planet currently NOT mine ---------------------------------
            else:
                attackers = [r for r in mine_arrivals
                             if r.is_attack and r.player == o_post and o_post in log.policy_players]
                if o_post in log.policy_players and o_post != o_cur and attackers:
                    # capture success -> credit attackers by ship share
                    total = float(sum(r.ships for r in attackers))
                    prod = float(cur.production.get(pid, nxt.production.get(pid, 0.0)))
                    val = cfg.capture_weight * dpp_of(pid, prod, t)
                    consumed = {r.decision_id: float(r.ships) for r in attackers}
                    _distribute(credit, consumed, +val)
                    stats.n_capture += 1
                    stats.capture_credit += val
                    # new garrison tagged to the attacking decisions by share
                    owner[pid] = o_post
                    layers[pid] = ([[attackers[0].decision_id, s_post]] if total <= 1e-9
                                   else [[r.decision_id, s_post * (r.ships / total)]
                                         for r in attackers])
                else:
                    # enemy/neutral dynamics we don't credit: just resync
                    owner[pid] = o_post
                    layers[pid] = [[NATIVE, s_post]]

            # Resolve the campaign ledger on an ownership flip: the new owner's
            # waves won (cleared, no penalty -- positive credit, if any, is
            # handled by the capture branch above); every *other* player's
            # committed waves lost the race for this planet.
            if track_campaign and o_post != o_cur:
                pend = pending_attacks.pop(pid, None)
                if pend and cfg.lost_race_weight:
                    losers: dict = {}
                    for r in pend:
                        if r.player != o_post:
                            losers[r.decision_id] = losers.get(r.decision_id, 0.0) + float(r.ships)
                    if losers:
                        prod = float(cur.production.get(pid, nxt.production.get(pid, 0.0)))
                        val = cfg.lost_race_weight * dpp_of(pid, prod, t)
                        _distribute(credit, losers, -val)
                        stats.n_lost_race += 1
                        stats.lost_race_credit -= val

        # keep untouched planets aligned (production already applied)
        for pid in list(layers.keys()):
            if pid in touched or pid not in nxt.owner:
                continue
            o_post = int(nxt.owner[pid])
            if o_post != owner.get(pid, -1):
                owner[pid] = o_post
                layers[pid] = [[NATIVE, float(nxt.ships.get(pid, 0.0))]]

    # Episode end: attack waves still pending never captured their target ->
    # pure waste, charged to their decisions by ship share.
    if track_campaign and cfg.fail_weight and pending_attacks:
        last = snaps[-1]
        for pid, pend in pending_attacks.items():
            by_dec: dict = {}
            for r in pend:
                by_dec[r.decision_id] = by_dec.get(r.decision_id, 0.0) + float(r.ships)
            if not by_dec:
                continue
            prod = float(last.production.get(pid, 0.0))
            val = cfg.fail_weight * dpp_of(pid, prod, last.turn)
            _distribute(credit, by_dec, -val)
            stats.n_fail_attack += 1
            stats.fail_credit -= val

    # apply optional event discount and finalise
    if cfg.event_discount != 1.0:
        credit = {k: v * cfg.event_discount for k, v in credit.items()}
    stats.total = float(sum(credit.values()))
    return credit, stats


__all__ = [
    "NATIVE", "LaunchRecord", "TurnSnapshot", "EpisodeLog",
    "CreditConfig", "CreditStats", "attribute_credits",
]
