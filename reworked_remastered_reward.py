"""Remastered campaign reward for the edge policy.

**Attack campaign** — launches at a not-yet-owned target until capture / lost / fail.
**Defense campaign** — reinforcement launches at an owned target until the planet is
lost or the episode ends.

Attack capture (always positive when ``V > 0``)::

    R = V * max(ε, 1 - w_casualty·(K/S_end) - w_delay·Δt/max_turns)

Defense loss (planet flips away from us)::

    R = -V * (w_defense_loss + w_casualty·(K/S_end) + w_delay·Δt/max_turns)

``V = dpp(planet)``.  ``K`` = ships destroyed in the campaign.  ``S_end`` = own
stock at ``t_resolve``.  Rewards split across participating turns by launch share.
Attack launches whose fleets **arrive after** capture (``resolve_turn > t_capture``)
are reclassified as **defense** reinforcements — they do not share capture credit.
If a planet is lost without any reinforcement launch, the production penalty still
applies to the **decision turn** that last owned the planet (``TurnDecision`` ref).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from .reworked_credit import EpisodeLog, LaunchRecord, TurnSnapshot
    from .reworked_features import discounted_production_potential, DEFAULT_GAMMA, DEFAULT_NORM
except ImportError:  # pragma: no cover
    from reworked_credit import EpisodeLog, LaunchRecord, TurnSnapshot  # type: ignore
    from reworked_features import discounted_production_potential, DEFAULT_GAMMA, DEFAULT_NORM  # type: ignore


ROLE_ATTACK = "attack"
ROLE_DEFENSE = "defense"
OUTCOME_VOID = "void"  # episode ended before campaign resolved — no credit


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RemasteredRewardConfig:
    gamma_eco: float = DEFAULT_GAMMA
    norm: float = DEFAULT_NORM
    w_casualty: float = 0.35
    w_delay: float = 0.10
    w_fail: float = 0.25
    w_defense_loss: float = 1.0   # scale on -dpp when a owned planet is lost
    w_enemy_kill: float = 1.0     # +scale * clip(kills/enemy_stock, 0, 1) per fleet destroyed
    count_inflight: bool = True   # enemy stock = garrisons + in-flight (mirrors own S_end)
    min_mult: float = 0.05
    terminal_weight: float = 1.0
    # Absolute wasted-ship penalty on failed/lost attacks.  Unlike ``w_fail``
    # (which divides K by the player's *total* stock and so vanishes when the
    # agent hoards), this charges a fixed cost per ship thrown away, pricing the
    # tiny-spray behaviour regardless of empire size.
    w_waste: float = 0.0
    waste_norm: float = 50.0      # ships per "1.0" unit of waste
    # Source over-extension: when an owned planet that was *under threat* at
    # launch time is stripped by outgoing fleets and then falls within
    # ``source_strip_window`` turns, charge ``-w_source_strip * dpp(g)`` (scaled
    # by the stripped share) back onto the launching edges.  0 disables.
    w_source_strip: float = 0.0
    source_strip_window: int = 10


@dataclass
class CampaignContrib:
    record_idx: int
    t_local: int
    ships: float
    launch_turn: int = 0


@dataclass
class PlanetCampaign:
    campaign_id: int
    role: str                 # attack | defense
    player: int
    tgt_id: int
    t_decision: int
    K: float = 0.0
    E: float = 0.0                # enemy ships destroyed while defending this target
    contrib: list[CampaignContrib] = field(default_factory=list)
    stock_by_turn: dict = field(default_factory=dict)
    S_end: float = 0.0
    outcome: str = "active"
    t_resolve: int = -1
    prod_at_close: float = 0.0
    tgt_was_enemy: bool = False   # attack: target was enemy-owned at campaign open (not neutral)


@dataclass
class PassiveDefenseLoss:
    """Planet lost with no reinforcement campaign — blame the last decision turn."""
    player: int
    tgt_id: int
    record_idx: int
    t_local: int
    t_resolve: int
    K: float
    prod: float


@dataclass
class RemasteredStats:
    total: float = 0.0
    n_capture: int = 0
    n_fail: int = 0
    n_lost: int = 0
    n_defense_loss: int = 0
    capture_reward: float = 0.0
    fail_reward: float = 0.0
    defense_reward: float = 0.0
    n_enemy_kill: int = 0
    enemy_kill_reward: float = 0.0
    n_source_strip: int = 0
    source_strip_reward: float = 0.0


class CampaignLedger:
    """Multi-turn attack and defense campaigns."""

    def __init__(self) -> None:
        self._next_id = 0
        self._active: dict[tuple[int, int, str], PlanetCampaign] = {}
        self.closed: list[PlanetCampaign] = []
        self.passive_losses: list[PassiveDefenseLoss] = []
        # Over-extension penalties: (record_idx, t_local, reward<0) charged to the
        # launching edge that stripped a threatened source which then fell.
        self.source_strips: list[tuple[int, int, float]] = []

    def _camp_key(self, player: int, tgt_id: int, role: str) -> tuple[int, int, str]:
        return (int(player), int(tgt_id), str(role))

    def _open(
        self, *, role: str, player: int, tgt_id: int, t_decision: int,
        record_idx: int, t_local: int, ships: float, stock: float,
        tgt_was_enemy: bool = False,
    ) -> int:
        key = self._camp_key(player, tgt_id, role)
        camp = self._active.get(key)
        if camp is None:
            camp = PlanetCampaign(
                campaign_id=self._next_id,
                role=str(role),
                player=int(player),
                tgt_id=int(tgt_id),
                t_decision=int(t_decision),
                tgt_was_enemy=bool(tgt_was_enemy),
            )
            self._next_id += 1
            self._active[key] = camp
        camp.contrib.append(CampaignContrib(
            record_idx=int(record_idx), t_local=int(t_local), ships=float(ships),
            launch_turn=int(t_decision)))
        camp.stock_by_turn[int(t_decision)] = max(float(stock), 1.0)
        return camp.campaign_id

    def note_attack(self, *, player: int, tgt_id: int, t_decision: int,
                    record_idx: int, t_local: int, ships: int, stock: float,
                    tgt_was_enemy: bool = False) -> int:
        return self._open(role=ROLE_ATTACK, player=player, tgt_id=tgt_id,
                          t_decision=t_decision, record_idx=record_idx,
                          t_local=t_local, ships=float(ships), stock=stock,
                          tgt_was_enemy=tgt_was_enemy)

    def note_defense(self, *, player: int, tgt_id: int, t_decision: int,
                     record_idx: int, t_local: int, ships: int, stock: float) -> int:
        return self._open(role=ROLE_DEFENSE, player=player, tgt_id=tgt_id,
                          t_decision=t_decision, record_idx=record_idx,
                          t_local=t_local, ships=float(ships), stock=stock)

    def record_stock(self, player: int, turn: int, stock: float) -> None:
        t, s = int(turn), max(float(stock), 1.0)
        for camp in self._active.values():
            if camp.player == int(player):
                camp.stock_by_turn[t] = s

    def get_active(self, player: int, tgt_id: int, role: str) -> Optional[PlanetCampaign]:
        return self._active.get(self._camp_key(player, tgt_id, role))

    def close(self, player: int, tgt_id: int, role: str, *, outcome: str,
              t_resolve: int, prod: float) -> Optional[PlanetCampaign]:
        key = self._camp_key(player, tgt_id, role)
        camp = self._active.pop(key, None)
        if camp is None:
            return None
        camp.outcome = str(outcome)
        camp.t_resolve = int(t_resolve)
        camp.prod_at_close = float(prod)
        camp.S_end = _stock_at_resolve(camp)
        self.closed.append(camp)
        return camp

    def close_remaining(self, *, t_end: int, prod_lookup: dict[int, float],
                        owner_lookup: dict[int, int]) -> None:
        """Close unresolved campaigns at episode end.

        * **Attack** with wasted ships (``K > 0``) → ``fail`` (chip spam penalised).
        * **Attack** with no committed loss → ``void`` (opened but never fought).
        * **Defense** still held → ``void`` (no loss event; zero credit by design).
        """
        for key in list(self._active.keys()):
            camp = self._active.pop(key)
            camp.t_resolve = int(t_end)
            camp.prod_at_close = float(prod_lookup.get(camp.tgt_id, 0.0))
            camp.S_end = _stock_at_resolve(camp)
            if camp.role == ROLE_ATTACK and camp.K > 1e-9:
                camp.outcome = "fail"
            else:
                camp.outcome = OUTCOME_VOID
            self.closed.append(camp)


# ---------------------------------------------------------------------------
# Reward formulas
# ---------------------------------------------------------------------------
def _player_garrison_stock(snapshot: TurnSnapshot, player: int,
                           *, count_inflight: bool = True) -> float:
    """``player``'s total ship stock at one snapshot.

    Garrisons on owned planets plus (when ``count_inflight``) the player's
    in-flight fleets, so the passive-defense ``S_end`` matches the active-campaign
    stock convention (``_stock_at_resolve`` is recorded with the same flag).
    """
    total = 0.0
    for pid, owner in snapshot.owner.items():
        if int(owner) == int(player):
            total += float(snapshot.ships.get(int(pid), 0.0))
    if count_inflight:
        total += float(getattr(snapshot, "fleet_stock", {}).get(int(player), 0.0))
    return max(total, 1.0)


def _stock_at_resolve(camp: PlanetCampaign) -> float:
    t = int(camp.t_resolve)
    if t in camp.stock_by_turn:
        return max(float(camp.stock_by_turn[t]), 1.0)
    prior = [tt for tt in camp.stock_by_turn if tt <= t]
    if prior:
        return max(float(camp.stock_by_turn[max(prior)]), 1.0)
    if camp.stock_by_turn:
        return max(float(max(camp.stock_by_turn.values())), 1.0)
    return 1.0


def dpp_value(prod: float, remaining_turns: int, cfg: RemasteredRewardConfig) -> float:
    R = max(0, int(remaining_turns))
    return float(discounted_production_potential(
        np.asarray([float(prod)], dtype=np.float64), R,
        gamma=cfg.gamma_eco, norm=cfg.norm)[0])


def _fractions(K: float, S_end: float, delta_steps: int, max_turns: int
               ) -> tuple[float, float]:
    f_loss = float(np.clip(K / max(S_end, 1.0), 0.0, 1.0))
    f_delay = float(np.clip(delta_steps / max(int(max_turns), 1), 0.0, 1.0))
    return f_loss, f_delay


def capture_reward(V: float, K: float, S_end: float, delta_steps: int,
                   max_turns: int, cfg: RemasteredRewardConfig) -> float:
    V = max(float(V), 0.0)
    if V <= 0.0:
        return 0.0
    f_loss, f_delay = _fractions(K, S_end, delta_steps, max_turns)
    mult = max(1.0 - cfg.w_casualty * f_loss - cfg.w_delay * f_delay, cfg.min_mult)
    return V * mult


def fail_reward(V: float, K: float, S_end: float, cfg: RemasteredRewardConfig) -> float:
    V = max(float(V), 0.0)
    f_loss, _ = _fractions(K, S_end, 0, 1)
    # Value-relative term (cheap when hoarding) + absolute wasted-ship term
    # (fixed cost per ship lost, so spamming tiny fleets is no longer free).
    waste = max(float(K), 0.0) / max(float(cfg.waste_norm), 1.0)
    return -cfg.w_fail * V * f_loss - cfg.w_waste * waste


def _ship_stock_by_owner(planets, fleets, *, count_inflight: bool) -> dict[int, float]:
    """Total ships per owner = garrisons (+ in-flight fleets when enabled)."""
    stock: dict[int, float] = {}
    for pl in planets:
        if pl.owner != -1:
            stock[pl.owner] = stock.get(pl.owner, 0.0) + float(pl.ships)
    if count_inflight:
        for fl in fleets:
            if fl.owner != -1:
                stock[fl.owner] = stock.get(fl.owner, 0.0) + float(fl.ships)
    return stock


def _enemy_stock(enemy_owner: int, planets, fleets, *, count_inflight: bool) -> float:
    stock = _ship_stock_by_owner(planets, fleets, count_inflight=count_inflight)
    return max(float(stock.get(int(enemy_owner), 0.0)), 1.0)


def enemy_kill_reward(kills: float, enemy_stock: float,
                      cfg: RemasteredRewardConfig) -> float:
    """Positive credit for destroying enemy ships, scaled by enemy empire size.

    Mirrors own casualty scaling ``K / S_end``: the reward is proportional to
    the fraction of the enemy's total fleet eliminated (clipped to [0, 1]).
    """
    frac = float(np.clip(
        max(float(kills), 0.0) / max(float(enemy_stock), 1.0), 0.0, 1.0))
    return cfg.w_enemy_kill * frac


def _distribute_kill(
    contrib: list[CampaignContrib],
    kill_turn: int,
    enemy_ships: float,
    enemy_stock: float,
    cfg: RemasteredRewardConfig,
) -> dict[tuple[int, int], float]:
    """Split enemy-kill reward among campaign contributors by launch share."""
    eligible = [c for c in contrib if int(c.launch_turn) <= int(kill_turn)]
    total = sum(c.ships for c in eligible)
    if total <= 1e-9:
        return {}
    base = enemy_kill_reward(enemy_ships, enemy_stock, cfg)
    out: dict[tuple[int, int], float] = {}
    for c in eligible:
        key = (c.record_idx, c.t_local)
        out[key] = out.get(key, 0.0) + base * (c.ships / total)
    return out


def observe_enemy_kills(
    ledger: CampaignLedger,
    log: EpisodeLog,
    prev_obs,
    curr_obs,
    *,
    turn: int,
    player: int,
    omega: float,
    cfg: RemasteredRewardConfig,
) -> tuple[dict[tuple[int, int], float], float]:
    """Detect enemy fleets destroyed at planets during active campaigns.

    Called once per env step (``prev_obs`` -> ``curr_obs``).

    * **Defense** — predicted target is still owned by ``player`` after combat.
      Credit the active defense campaign (or the last decision slot if none).
    * **Attack** — active assault on an enemy-owned target (not neutral).  Credit
      when the target is still enemy-owned (chip) or was just captured by
      ``player``.  Neutral targets are excluded.
    """
    try:
        from .feature_extraction import parse_planets, parse_fleets
        from .navigation import predict_all_fleet_targets
    except ImportError:  # pragma: no cover
        from feature_extraction import parse_planets, parse_fleets  # type: ignore
        from navigation import predict_all_fleet_targets  # type: ignore

    if int(player) not in log.policy_players:
        return {}, 0.0

    prev_planets_list = parse_planets(prev_obs)
    curr_planets_list = parse_planets(curr_obs)
    prev_planets = {p.id: p for p in prev_planets_list}
    curr_planets = {p.id: p for p in curr_planets_list}
    prev_fleets_list = parse_fleets(prev_obs)
    curr_fleets_list = parse_fleets(curr_obs)
    prev_fleets = {f.id: f for f in prev_fleets_list}
    curr_fleets = {f.id: f for f in curr_fleets_list}

    prev_preds = (
        predict_all_fleet_targets(prev_fleets_list, prev_planets_list, omega)
        if prev_fleets_list else {}
    )
    enemy_stocks = _ship_stock_by_owner(
        prev_planets_list, prev_fleets_list, count_inflight=cfg.count_inflight)

    slot_credits: dict[tuple[int, int], float] = {}
    total_r = 0.0
    kill_turn = int(turn)

    for f_id, prev_fl in prev_fleets.items():
        if prev_fl.owner == int(player) or prev_fl.owner == -1:
            continue
        if f_id in curr_fleets:
            continue
        pred = prev_preds.get(f_id)
        if pred is None:
            continue
        target_id = int(pred["target_id"])
        curr_planet = curr_planets.get(target_id)
        if curr_planet is None:
            continue

        kills = float(prev_fl.ships)
        if kills <= 0.0:
            continue

        enemy_stock = max(float(enemy_stocks.get(int(prev_fl.owner), 0.0)), 1.0)
        part: dict[tuple[int, int], float] = {}
        acamp = ledger.get_active(int(player), target_id, ROLE_ATTACK)
        if acamp is not None and acamp.tgt_was_enemy and curr_planet.owner != -1:
            acamp.E += kills
            part = _distribute_kill(
                acamp.contrib, kill_turn, kills, enemy_stock, cfg)
        elif curr_planet.owner == int(player):
            dcamp = ledger.get_active(int(player), target_id, ROLE_DEFENSE)
            if dcamp is not None:
                dcamp.E += kills
                part = _distribute_kill(
                    dcamp.contrib, kill_turn, kills, enemy_stock, cfg)
            else:
                slot = int(log.planet_slot.get(target_id, -1))
                rec_idx = log.record_idx_at(int(player), kill_turn)
                if rec_idx < 0:
                    rec_idx = log.record_idx_at(int(player), max(0, kill_turn - 1))
                if rec_idx >= 0 and slot >= 0:
                    part = {(rec_idx, slot): enemy_kill_reward(
                        kills, enemy_stock, cfg)}
        elif curr_planet.owner not in (-1, int(player)):
            slot = int(log.planet_slot.get(target_id, -1))
            rec_idx = log.record_idx_at(int(player), kill_turn)
            if rec_idx < 0:
                rec_idx = log.record_idx_at(int(player), max(0, kill_turn - 1))
            if rec_idx >= 0 and slot >= 0:
                part = {(rec_idx, slot): enemy_kill_reward(
                    kills, enemy_stock, cfg)}

        for key, val in part.items():
            slot_credits[key] = slot_credits.get(key, 0.0) + val
            total_r += val

    return slot_credits, total_r


def defense_loss_reward(V: float, K: float, S_end: float, delta_steps: int,
                        max_turns: int, cfg: RemasteredRewardConfig) -> float:
    """Lost production potential + optional casualty / delay extras (always <= 0)."""
    V = max(float(V), 0.0)
    if V <= 0.0:
        return 0.0
    f_loss, f_delay = _fractions(K, S_end, delta_steps, max_turns)
    penalty = cfg.w_defense_loss + cfg.w_casualty * f_loss + cfg.w_delay * f_delay
    return -V * penalty


# ---------------------------------------------------------------------------
# Offline attribution
# ---------------------------------------------------------------------------
def _fill_resolve_turns(log: EpisodeLog) -> dict[int, LaunchRecord]:
    fleet_to_launch = {r.fleet_id: r for r in log.launches if r.fleet_id >= 0}
    snaps = log.snapshots
    for i in range(len(snaps) - 1):
        gone = snaps[i].fleet_ids - snaps[i + 1].fleet_ids
        for fid in gone:
            rec = fleet_to_launch.get(fid)
            if rec is not None and rec.resolve_turn < 0:
                rec.resolve_turn = int(snaps[i + 1].turn)
    return fleet_to_launch


def _arrivals_by_turn(log: EpisodeLog) -> dict[int, list[LaunchRecord]]:
    _fill_resolve_turns(log)
    out: dict[int, list[LaunchRecord]] = {}
    for rec in log.launches:
        if rec.resolve_turn >= 0:
            out.setdefault(int(rec.resolve_turn), []).append(rec)
    return out


def _close_at_episode_end(log: EpisodeLog, ledger: CampaignLedger) -> None:
    if not log.snapshots:
        return
    last = log.snapshots[-1]
    prod_lookup = {int(k): float(v) for k, v in last.production.items()}
    owner_lookup = {int(k): int(v) for k, v in last.owner.items()}
    ledger.close_remaining(
        t_end=int(last.turn), prod_lookup=prod_lookup, owner_lookup=owner_lookup)


def _source_strip_loss(
    log: EpisodeLog, cfg: RemasteredRewardConfig, *,
    player: int, pid: int, t_step: int, prod: float, garrison_lost: float,
) -> list[tuple[int, int, float]]:
    """Over-extension penalty for stripping a *threatened* source that then fell.

    Returns ``(record_idx, t_local, reward<0)`` entries for each launch that left
    ``pid`` while it was under attack, within ``source_strip_window`` turns before
    the loss.  The total ``-w_source_strip * dpp(pid)`` is split between the
    stripped ships and the garrison that stayed (shared blame), then the strip
    share is distributed across the launching edges by ship count.
    """
    if cfg.w_source_strip <= 0.0:
        return []
    strips = [
        r for r in log.launches
        if int(r.src_id) == int(pid) and int(r.player) == int(player)
        and bool(getattr(r, "src_under_threat", False))
        and 0 <= (int(t_step) - int(r.launch_turn)) <= int(cfg.source_strip_window)
    ]
    S = float(sum(float(r.ships) for r in strips))
    if S <= 1e-9:
        return []
    R_rem = max(0, log.max_turns - int(t_step))
    V = dpp_value(prod, R_rem, cfg)
    if V <= 0.0:
        return []
    denom = S + max(0.0, float(garrison_lost))
    total_penalty = -cfg.w_source_strip * V * (S / max(denom, 1e-9))
    out: list[tuple[int, int, float]] = []
    for r in strips:
        rec_idx = log.record_idx_at(int(player), int(r.launch_turn))
        t_local = int(getattr(r, "t_local", -1))
        if rec_idx < 0 or t_local < 0:
            continue
        out.append((rec_idx, t_local, total_penalty * (float(r.ships) / S)))
    return out


def _accumulate_and_close(log: EpisodeLog, ledger: CampaignLedger,
                          cfg: RemasteredRewardConfig) -> None:
    snaps = log.snapshots
    if len(snaps) < 2:
        _close_at_episode_end(log, ledger)
        return

    arrivals = _arrivals_by_turn(log)
    fleet_to_launch = {r.fleet_id: r for r in log.launches if r.fleet_id >= 0}

    for i in range(len(snaps) - 1):
        cur, nxt = snaps[i], snaps[i + 1]
        t_step = int(nxt.turn)
        t_decision = int(cur.turn)

        gone = cur.fleet_ids - nxt.fleet_ids
        arrivals_t = arrivals.get(t_step, [])
        arrivals_by_tgt: dict[int, list[LaunchRecord]] = {}
        for rec in arrivals_t:
            arrivals_by_tgt.setdefault(int(rec.tgt_id), []).append(rec)

        flipped: list[tuple[int, int, int]] = []
        for pid in set(cur.owner) | set(nxt.owner):
            if pid not in nxt.owner:
                continue
            o_before = int(cur.owner.get(pid, -1))
            o_after = int(nxt.owner.get(pid, -1))
            if o_before != o_after:
                flipped.append((pid, o_before, o_after))

        for pid, o_before, o_after in flipped:
            prod = float(nxt.production.get(pid, cur.production.get(pid, 0.0)))

            # --- Attack capture ---
            if o_after in log.policy_players:
                attackers = [r for r in arrivals_by_tgt.get(pid, [])
                             if r.is_attack and r.player == o_after]
                acamp = ledger.get_active(o_after, pid, ROLE_ATTACK)
                if acamp is not None and attackers:
                    s_post = float(nxt.ships.get(pid, 0.0))
                    total = float(sum(r.ships for r in attackers))
                    for r in attackers:
                        if total <= 1e-9:
                            acamp.K += float(r.ships)
                        else:
                            survived = s_post * (float(r.ships) / total)
                            acamp.K += max(0.0, float(r.ships) - survived)
                    closed_atk = ledger.close(
                        o_after, pid, ROLE_ATTACK, outcome="capture",
                        t_resolve=t_step, prod=prod)
                    if closed_atk is not None:
                        _spin_off_post_capture_reinforcements(
                            log, ledger, closed_atk, t_step)

            # --- Defense loss: planet was ours, now isn't ---
            if o_before in log.policy_players and o_after != o_before:
                player = int(o_before)
                dcamp = ledger.get_active(player, pid, ROLE_DEFENSE)
                garrison_lost = max(0.0, float(cur.ships.get(pid, 0.0)))
                # Over-extension: blame launches that stripped this planet while
                # it was under attack (charged to the launching edges).
                ledger.source_strips.extend(_source_strip_loss(
                    log, cfg, player=player, pid=int(pid), t_step=t_step,
                    prod=prod, garrison_lost=garrison_lost))
                if dcamp is not None:
                    dcamp.K += garrison_lost
                    for r in arrivals_by_tgt.get(pid, []):
                        if not r.is_attack and r.player == player:
                            dcamp.K += float(r.ships)
                    ledger.close(player, pid, ROLE_DEFENSE, outcome="defense_loss",
                                 t_resolve=t_step, prod=prod)
                else:
                    # No reinforcement campaign — still penalise lost dpp on the
                    # last decision turn that owned the planet.
                    slot = int(log.planet_slot.get(pid, -1))
                    rec_idx = log.record_idx_at(player, t_decision)
                    if rec_idx >= 0 and slot >= 0:
                        K = garrison_lost
                        for r in arrivals_by_tgt.get(pid, []):
                            if not r.is_attack and r.player == player:
                                K += float(r.ships)
                        ledger.passive_losses.append(PassiveDefenseLoss(
                            player=player, tgt_id=int(pid),
                            record_idx=rec_idx, t_local=slot,
                            t_resolve=t_step, K=K, prod=prod))

            # --- Attack lost race ---
            for player in log.policy_players:
                if o_after == int(player):
                    continue
                acamp = ledger.get_active(int(player), pid, ROLE_ATTACK)
                if acamp is None:
                    continue
                for r in arrivals_by_tgt.get(pid, []):
                    if r.is_attack and r.player == int(player):
                        acamp.K += float(r.ships)
                ledger.close(int(player), pid, ROLE_ATTACK, outcome="lost",
                             t_resolve=t_step, prod=prod)

        # Failed attack chips (planet still not ours).
        for fid in gone:
            rec = fleet_to_launch.get(fid)
            if rec is None or rec.player not in log.policy_players:
                continue
            if rec.resolve_turn != t_step:
                continue
            role = ROLE_ATTACK if rec.is_attack else ROLE_DEFENSE
            camp = ledger.get_active(rec.player, rec.tgt_id, role)
            if camp is None:
                continue
            o_after = int(nxt.owner.get(rec.tgt_id, -1))
            if role == ROLE_ATTACK:
                if o_after == int(rec.player):
                    continue
                if not any(p == rec.tgt_id and oa == int(rec.player)
                           for p, _, oa in flipped):
                    camp.K += float(rec.ships)
            else:
                if o_after == int(rec.player):
                    camp.K += float(rec.ships)

    _close_at_episode_end(log, ledger)


def _campaign_net_reward(camp: PlanetCampaign, max_turns: int,
                         cfg: RemasteredRewardConfig) -> float:
    R_rem = max(0, max_turns - camp.t_resolve)
    V = dpp_value(camp.prod_at_close, R_rem, cfg)
    delta = max(0, camp.t_resolve - camp.t_decision)
    if camp.role == ROLE_ATTACK:
        if camp.outcome == "capture":
            return capture_reward(V, camp.K, camp.S_end, delta, max_turns, cfg)
        if camp.outcome in ("lost", "fail"):
            # Price the waste by the value the agent was *pursuing* when it
            # committed (horizon from ``t_decision``), not the value left at
            # resolve: an attack that resolves at the episode end would otherwise
            # have ``R_rem -> 0 -> V -> 0`` and escape the value-relative penalty.
            R_dec = max(0, max_turns - camp.t_decision)
            V_dec = dpp_value(camp.prod_at_close, R_dec, cfg)
            return fail_reward(V_dec, camp.K, camp.S_end, cfg)
        if camp.outcome == OUTCOME_VOID:
            return 0.0
    elif camp.role == ROLE_DEFENSE:
        if camp.outcome == "defense_loss":
            return defense_loss_reward(V, camp.K, camp.S_end, delta, max_turns, cfg)
        if camp.outcome == OUTCOME_VOID:
            return 0.0
    return 0.0


def _distribute(camp: PlanetCampaign, R: float,
                slot_credits: dict[tuple[int, int], float]) -> None:
    by_slot: dict[tuple[int, int], float] = {}
    for c in camp.contrib:
        key = (c.record_idx, c.t_local)
        by_slot[key] = by_slot.get(key, 0.0) + c.ships
    total = sum(by_slot.values())
    if total <= 1e-9:
        return
    for key, ships in by_slot.items():
        slot_credits[key] = slot_credits.get(key, 0.0) + R * (ships / total)


def _launch_record_idx(
    r: LaunchRecord, camp: PlanetCampaign, log: EpisodeLog,
) -> int:
    """Buffer index for the decision turn that issued ``r``."""
    rec_idx = log.record_idx_at(int(r.player), int(r.launch_turn))
    if rec_idx >= 0:
        return rec_idx
    t_local = int(getattr(r, "t_local", -1))
    for c in camp.contrib:
        if (int(c.launch_turn) == int(r.launch_turn)
                and int(c.t_local) == t_local
                and abs(float(c.ships) - float(r.ships)) < 1e-6):
            return int(c.record_idx)
    return -1


def _capture_eligible_launches(
    camp: PlanetCampaign, log: EpisodeLog,
) -> list[LaunchRecord]:
    """Launches that arrived on or before capture — eligible for capture credit."""
    t_cap = int(camp.t_resolve)
    return [
        r for r in log.launches
        if int(r.campaign_id) == int(camp.campaign_id)
        and r.is_attack
        and int(r.player) == int(camp.player)
        and int(r.tgt_id) == int(camp.tgt_id)
        and int(r.resolve_turn) >= 0
        and int(r.resolve_turn) <= t_cap
    ]


def _spin_off_post_capture_reinforcements(
    log: EpisodeLog,
    ledger: CampaignLedger,
    camp: PlanetCampaign,
    t_resolve: int,
) -> None:
    """Attack launches landing after capture → defense campaign (no capture share)."""
    t_cap = int(t_resolve)
    for r in log.launches:
        if int(r.campaign_id) != int(camp.campaign_id):
            continue
        if not r.is_attack:
            continue
        if int(r.player) != int(camp.player) or int(r.tgt_id) != int(camp.tgt_id):
            continue
        if int(r.resolve_turn) < 0 or int(r.resolve_turn) <= t_cap:
            continue
        r.is_attack = False
        rec_idx = _launch_record_idx(r, camp, log)
        t_local = int(getattr(r, "t_local", -1))
        if rec_idx < 0 or t_local < 0:
            continue
        st = float(camp.stock_by_turn.get(
            int(r.launch_turn), max(float(camp.S_end), 1.0)))
        ledger.note_defense(
            player=int(r.player), tgt_id=int(r.tgt_id),
            t_decision=int(r.launch_turn), record_idx=rec_idx,
            t_local=t_local, ships=int(r.ships), stock=st,
        )


def _distribute_capture(
    camp: PlanetCampaign,
    R: float,
    log: EpisodeLog,
    slot_credits: dict[tuple[int, int], float],
) -> None:
    """Split capture credit only across launches that arrived by ``t_resolve``."""
    eligible = _capture_eligible_launches(camp, log)
    if not eligible:
        _distribute(camp, R, slot_credits)
        return
    by_slot: dict[tuple[int, int], float] = {}
    for r in eligible:
        rec_idx = _launch_record_idx(r, camp, log)
        t_local = int(getattr(r, "t_local", -1))
        if rec_idx < 0 or t_local < 0:
            continue
        key = (rec_idx, t_local)
        by_slot[key] = by_slot.get(key, 0.0) + float(r.ships)
    total = sum(by_slot.values())
    if total <= 1e-9:
        _distribute(camp, R, slot_credits)
        return
    for key, ships in by_slot.items():
        slot_credits[key] = slot_credits.get(key, 0.0) + R * (ships / total)


def attribute_remastered_rewards(
    log: EpisodeLog,
    ledger: CampaignLedger,
    cfg: RemasteredRewardConfig,
) -> tuple[dict[tuple[int, int], float], RemasteredStats]:
    _accumulate_and_close(log, ledger, cfg)

    slot_credits: dict[tuple[int, int], float] = {}
    stats = RemasteredStats()

    for camp in ledger.closed:
        R = _campaign_net_reward(camp, log.max_turns, cfg)
        stats.total += R
        if camp.role == ROLE_DEFENSE and camp.outcome == "defense_loss":
            stats.n_defense_loss += 1
            stats.defense_reward += R
        elif camp.outcome == "capture":
            stats.n_capture += 1
            stats.capture_reward += R
        elif camp.outcome == "lost":
            stats.n_lost += 1
            stats.fail_reward += R
        elif camp.outcome == "fail":
            stats.n_fail += 1
            stats.fail_reward += R
        if abs(R) > 1e-9:
            if camp.role == ROLE_ATTACK and camp.outcome == "capture":
                _distribute_capture(camp, R, log, slot_credits)
            else:
                _distribute(camp, R, slot_credits)

    for pl in ledger.passive_losses:
        R_rem = max(0, log.max_turns - pl.t_resolve)
        V = dpp_value(pl.prod, R_rem, cfg)
        delta = 0
        S_end = 1.0
        for snap in reversed(log.snapshots):
            if int(snap.turn) <= int(pl.t_resolve):
                S_end = _player_garrison_stock(
                    snap, pl.player, count_inflight=cfg.count_inflight)
                break
        R = defense_loss_reward(V, pl.K, S_end, delta, log.max_turns, cfg)
        stats.total += R
        stats.n_defense_loss += 1
        stats.defense_reward += R
        key = (pl.record_idx, pl.t_local)
        slot_credits[key] = slot_credits.get(key, 0.0) + R

    for rec_idx, t_local, R in ledger.source_strips:
        if abs(R) <= 1e-9:
            continue
        stats.total += R
        stats.n_source_strip += 1
        stats.source_strip_reward += R
        key = (int(rec_idx), int(t_local))
        slot_credits[key] = slot_credits.get(key, 0.0) + R

    return slot_credits, stats


# Back-compat alias
attribute_remastered_attack_rewards = attribute_remastered_rewards


def planet_ownership_events(
    log: EpisodeLog, player: int,
) -> tuple[list[dict], list[dict]]:
    """Replay board snapshots and list planets captured / lost by ``player``.

    Ground-truth from ``TurnSnapshot.owner`` diffs (independent of the reward pass).
    """
    captures: list[dict] = []
    losses: list[dict] = []
    snaps = log.snapshots
    for i in range(len(snaps) - 1):
        cur, nxt = snaps[i], snaps[i + 1]
        t = int(nxt.turn)
        pids = set(cur.owner) | set(nxt.owner)
        for pid in pids:
            o0 = int(cur.owner.get(pid, -1))
            o1 = int(nxt.owner.get(pid, -1))
            if o0 == o1:
                continue
            prod = float(nxt.production.get(pid, cur.production.get(pid, 0.0)))
            ships = float(nxt.ships.get(pid, cur.ships.get(pid, 0.0)))
            if o1 == int(player) and o0 != int(player):
                captures.append({
                    "planet_id": int(pid), "turn": t, "from_owner": o0,
                    "production": prod, "ships": ships,
                })
            elif o0 == int(player) and o1 != int(player):
                losses.append({
                    "planet_id": int(pid), "turn": t, "to_owner": o1,
                    "production": prod, "ships": ships,
                })
    return captures, losses


def summarize_remastered_episode(
    log: EpisodeLog,
    ledger: CampaignLedger,
    stats: RemasteredStats,
    cfg: RemasteredRewardConfig,
    *,
    player: int = 0,
) -> dict:
    """Compact per-episode record for ``episodes.jsonl`` logging."""
    captures, losses = planet_ownership_events(log, int(player))
    campaigns: list[dict] = []
    for camp in ledger.closed:
        campaigns.append({
            "campaign_id": int(camp.campaign_id),
            "role": str(camp.role),
            "tgt_id": int(camp.tgt_id),
            "outcome": str(camp.outcome),
            "tgt_was_enemy": bool(camp.tgt_was_enemy),
            "t_decision": int(camp.t_decision),
            "t_resolve": int(camp.t_resolve),
            "K": float(camp.K),
            "enemy_kills": float(camp.E),
            "reward": float(_campaign_net_reward(camp, log.max_turns, cfg)),
        })
    for pl in ledger.passive_losses:
        R_rem = max(0, log.max_turns - pl.t_resolve)
        V = dpp_value(pl.prod, R_rem, cfg)
        S_end = 1.0
        for snap in reversed(log.snapshots):
            if int(snap.turn) <= int(pl.t_resolve):
                S_end = _player_garrison_stock(
                    snap, pl.player, count_inflight=cfg.count_inflight)
                break
        campaigns.append({
            "campaign_id": -1,
            "role": ROLE_DEFENSE,
            "tgt_id": int(pl.tgt_id),
            "outcome": "passive_defense_loss",
            "tgt_was_enemy": False,
            "t_decision": int(pl.t_local),
            "t_resolve": int(pl.t_resolve),
            "K": float(pl.K),
            "enemy_kills": 0.0,
            "reward": float(defense_loss_reward(V, pl.K, S_end, 0, log.max_turns, cfg)),
        })
    return {
        "planet_captures": captures,
        "planet_losses": losses,
        "n_planet_captures": len(captures),
        "n_planet_losses": len(losses),
        "campaigns": campaigns,
        "reward_stats": {
            "total": float(stats.total),
            "n_capture": int(stats.n_capture),
            "n_fail": int(stats.n_fail),
            "n_lost": int(stats.n_lost),
            "n_defense_loss": int(stats.n_defense_loss),
            "capture_reward": float(stats.capture_reward),
            "fail_reward": float(stats.fail_reward),
            "defense_reward": float(stats.defense_reward),
            "n_source_strip": int(stats.n_source_strip),
            "source_strip_reward": float(stats.source_strip_reward),
        },
    }


def apply_remastered_credits(buffer, slot_credits: dict[tuple[int, int], float]) -> None:
    for (rec_idx, t_local), val in slot_credits.items():
        if not (0 <= rec_idx < len(buffer.data)):
            continue
        tr = buffer.data[rec_idx]
        v = float(val)
        tr.reward += v
        if getattr(tr, "credit_target", None) is not None and 0 <= t_local < len(tr.credit_target):
            tr.credit_target[t_local] += v
            # Passive defense / kill fallback / source-strip credits land on turns
            # where the agent did not gate that target; still train on them.
            if abs(v) > 1e-9 and getattr(tr, "target_valid", None) is not None:
                tr.target_valid[t_local] = True
