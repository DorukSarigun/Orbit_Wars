from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from feature_extraction import parse_planets


@dataclass
class ProductionCaptureLaunch:
    buffer_idx: int
    launch_turn: int
    player: int
    target_planet_id: int
    ships_sent: int
    est_arrival_turn: float
    launch_target_owner: int
    launch_target_ships: int = 0
    my_total_ships_at_launch: int = 0


@dataclass
class ProductionCaptureOutcome:
    ships_sent: int
    total_ships_sent: int
    total_ship_investment: float
    planet_production: float
    capture_turn: int
    max_turns: int
    captured_from_enemy: bool
    num_opponents: int
    my_total_ships_at_launch: int = 0


@dataclass
class ProductionNeutralGiftOutcome:
    damage_done: float
    launch_target_ships: int
    planet_production: float
    capture_turn: int
    max_turns: int


@dataclass
class ProductionDefenseMissLaunch:
    buffer_idx: int
    launch_turn: int
    player: int
    threatened_planet_id: int
    ships_sent: int
    est_defense_arrival_turn: float
    # Number of distinct threatened planets this same launch was registered
    # against at launch time. Used to normalize the per-miss penalty so a
    # single launch is not penalized N times if it "could have" defended N
    # different planets.
    num_threatened_at_launch: int = 1


@dataclass
class ProductionPlanetLossOutcome:
    ships_sent: int
    total_ships_sent: int
    planet_production: float
    loss_turn: int
    max_turns: int
    num_opponents: int
    # How many distinct threatened planets the launch was originally
    # associated with. The penalty is divided by this so the worst-case
    # sum (all N planets lost) equals one full loss penalty.
    num_threatened_at_launch: int = 1


class ProductionRewardTracker:
    """Tracks launch -> capture / defense-miss credit assignment.

    Capture credit is shared across ALL launches that arrived at the
    target within a `softening_window` of turns before the capture. This
    means a fleet that "softens" the defenders (arrives, kills garrison,
    but doesn't capture) still gets proportional credit if a follow-up
    fleet completes the capture shortly after. Without this, a softening
    fleet would be punished (waste penalty) but receive no reward.
    """

    def __init__(self, *, max_turns: int = 500, num_agents: int = 2,
                 softening_window: int = 30):
        self.max_turns = int(max_turns)
        self.num_opponents = max(1, int(num_agents) - 1)
        self.softening_window = max(0, int(softening_window))
        self._capture_pending: dict[tuple[int, int], list[ProductionCaptureLaunch]] = {}
        self._loss_pending: dict[tuple[int, int], list[ProductionDefenseMissLaunch]] = {}
        self._resolved_captures: dict[int, deque] = {}
        self._resolved_losses: dict[int, deque] = {}
        self._resolved_neutral_gifts: dict[int, deque] = {}

    def register_capture_launch(self, *, buffer_idx: int, launch_turn: int, player: int,
                                target_planet_id: int, ships_sent: int,
                                est_arrival_turn: float, launch_target_owner: int,
                                launch_target_ships: int = 0,
                                my_total_ships_at_launch: int = 0) -> None:
        if ships_sent <= 0 or launch_target_owner == player:
            return
        key = (int(player), int(target_planet_id))
        self._capture_pending.setdefault(key, []).append(ProductionCaptureLaunch(
            buffer_idx=int(buffer_idx),
            launch_turn=int(launch_turn),
            player=int(player),
            target_planet_id=int(target_planet_id),
            ships_sent=int(ships_sent),
            est_arrival_turn=float(est_arrival_turn),
            launch_target_owner=int(launch_target_owner),
            launch_target_ships=int(launch_target_ships),
            my_total_ships_at_launch=int(my_total_ships_at_launch),
        ))

    def register_defense_miss_candidate(self, *, buffer_idx: int, launch_turn: int,
                                        player: int, threatened_planet_id: int,
                                        ships_sent: int,
                                        est_defense_arrival_turn: float,
                                        num_threatened_at_launch: int = 1) -> None:
        if ships_sent <= 0:
            return
        key = (int(player), int(threatened_planet_id))
        self._loss_pending.setdefault(key, []).append(ProductionDefenseMissLaunch(
            buffer_idx=int(buffer_idx),
            launch_turn=int(launch_turn),
            player=int(player),
            threatened_planet_id=int(threatened_planet_id),
            ships_sent=int(ships_sent),
            est_defense_arrival_turn=float(est_defense_arrival_turn),
            num_threatened_at_launch=max(1, int(num_threatened_at_launch)),
        ))

    def observe_turn(self, turn: int, prev_obs, curr_obs, player: int) -> None:
        prev_planets = {p.id: p for p in parse_planets(prev_obs) if not p.is_comet}
        curr_planets = {p.id: p for p in parse_planets(curr_obs) if not p.is_comet}
        for planet_id, curr in curr_planets.items():
            prev = prev_planets.get(planet_id)
            prev_owner = prev.owner if prev is not None else -1
            self._resolve_neutral_gift_if_enemy_captured(
                turn,
                player,
                planet_id,
                prev_owner,
                curr.owner,
                float(curr.production),
            )
            if prev_owner == player or curr.owner != player:
                continue
            key = (int(player), int(planet_id))
            launches = [
                fl for fl in self._capture_pending.pop(key, [])
                if fl.est_arrival_turn <= turn + 1e-6
                and fl.launch_turn <= turn
                and (turn - fl.est_arrival_turn) <= self.softening_window
            ]
            total_ships = sum(max(0, fl.ships_sent) for fl in launches)
            if total_ships <= 0:
                continue
            captured_from_enemy = prev_owner not in (-1, player)
            observed_target_ships = max(0, int(prev.ships)) if prev is not None else 0
            launch_target_ships = max(max(0, int(fl.launch_target_ships)) for fl in launches)
            target_ship_investment = max(observed_target_ships, launch_target_ships)
            total_ship_investment = min(float(total_ships), float(target_ship_investment))
            for fl in launches:
                outcome = ProductionCaptureOutcome(
                    ships_sent=fl.ships_sent,
                    total_ships_sent=total_ships,
                    total_ship_investment=total_ship_investment,
                    planet_production=float(curr.production),
                    capture_turn=int(turn),
                    max_turns=self.max_turns,
                    captured_from_enemy=bool(captured_from_enemy),
                    num_opponents=self.num_opponents,
                    my_total_ships_at_launch=fl.my_total_ships_at_launch,
                )
                self._resolved_captures.setdefault(player, deque()).append((fl.buffer_idx, outcome))
        self._expire_arrived_non_captures(turn, player, curr_planets)
        for planet_id, prev in prev_planets.items():
            if prev.owner != player:
                continue
            curr = curr_planets.get(planet_id)
            if curr is not None and curr.owner == player:
                continue
            key = (int(player), int(planet_id))
            candidates = [
                fl for fl in self._loss_pending.pop(key, [])
                if fl.est_defense_arrival_turn <= turn + 1e-6 and fl.launch_turn <= turn
            ]
            total_ships = sum(max(0, fl.ships_sent) for fl in candidates)
            if total_ships <= 0:
                continue
            for fl in candidates:
                outcome = ProductionPlanetLossOutcome(
                    ships_sent=fl.ships_sent,
                    total_ships_sent=total_ships,
                    planet_production=float(prev.production),
                    loss_turn=int(turn),
                    max_turns=self.max_turns,
                    num_opponents=self.num_opponents,
                    num_threatened_at_launch=fl.num_threatened_at_launch,
                )
                self._resolved_losses.setdefault(player, deque()).append((fl.buffer_idx, outcome))

    def _resolve_neutral_gift_if_enemy_captured(self, turn: int, player: int,
                                                planet_id: int, prev_owner: int,
                                                curr_owner: int,
                                                planet_production: float) -> None:
        if prev_owner != -1 or curr_owner in (-1, player):
            return
        key = (int(player), int(planet_id))
        pending = self._capture_pending.get(key, [])
        if not pending:
            return
        eligible = [
            fl for fl in pending
            if fl.launch_target_owner == -1
            and fl.est_arrival_turn <= turn + 1e-6
            and fl.launch_turn <= turn
        ]
        if not eligible:
            return
        initial_ships = max(max(0, fl.launch_target_ships) for fl in eligible)
        if initial_ships <= 0:
            return
        raw_damage = [
            min(float(max(0, fl.ships_sent)), float(initial_ships))
            for fl in eligible
        ]
        raw_total = sum(raw_damage)
        if raw_total <= 0.0:
            return
        damage_scale = min(1.0, float(initial_ships) / raw_total)
        eligible_keys = {id(fl) for fl in eligible}
        remaining = [fl for fl in pending if id(fl) not in eligible_keys]
        if remaining:
            self._capture_pending[key] = remaining
        else:
            self._capture_pending.pop(key, None)
        for fl, damage in zip(eligible, raw_damage):
            outcome = ProductionNeutralGiftOutcome(
                damage_done=float(damage) * damage_scale,
                launch_target_ships=int(initial_ships),
                planet_production=float(planet_production),
                capture_turn=int(turn),
                max_turns=self.max_turns,
            )
            self._resolved_neutral_gifts.setdefault(player, deque()).append((fl.buffer_idx, outcome))

    def _expire_arrived_non_captures(self, turn: int, player: int, curr_planets: dict) -> None:
        """Keep arrived-but-not-captured launches for `softening_window`
        turns so they can share capture credit if a follow-up fleet
        completes the capture shortly after. A launch is dropped only
        when either:
          - The planet has already been captured this turn (handled by
            the capture-resolution path that pops the key), or
          - The launch arrived more than `softening_window` turns ago
            without a capture occurring in that window.
        """
        for key in list(self._capture_pending):
            key_player, planet_id = key
            if key_player != player:
                continue
            curr = curr_planets.get(planet_id)
            if curr is not None and curr.owner == player:
                # Planet flipped to us through something we're not crediting
                # here (e.g. terminal state or other tracker); drop stale.
                self._capture_pending.pop(key, None)
                continue
            launches = [
                fl for fl in self._capture_pending[key]
                if (fl.est_arrival_turn > turn + 1e-6)                              # not yet arrived
                or (curr is not None and curr.owner == -1 and fl.launch_target_owner == -1)
                or ((turn - fl.est_arrival_turn) <= self.softening_window)    # within softening window
            ]
            if launches:
                self._capture_pending[key] = launches
            else:
                self._capture_pending.pop(key, None)

    def pop_capture_resolved(self, player: int):
        q = self._resolved_captures.get(player)
        if q:
            while q:
                yield q.popleft()

    def pop_loss_resolved(self, player: int):
        q = self._resolved_losses.get(player)
        if q:
            while q:
                yield q.popleft()

    def pop_neutral_gift_resolved(self, player: int):
        q = self._resolved_neutral_gifts.get(player)
        if q:
            while q:
                yield q.popleft()


def opponent_count_multiplier(num_opponents: int) -> float:
    total_players = max(2, int(num_opponents) + 1)
    return (total_players + 1.0) / total_players


def remaining_steps(max_turns: int, turn: int) -> int:
    return max(0, int(max_turns) - int(turn))


def production_reward_value(
    production: float,
    *,
    production_reward_beta: float = 0.0,
    production_reward_reference: float = 3.0,
) -> float:
    value = max(0.0, float(production))
    beta = max(0.0, float(production_reward_beta))
    if value <= 0.0 or beta <= 0.0:
        return value
    reference = max(1e-6, float(production_reward_reference))
    return value * ((value / reference) ** beta)


def compute_production_capture_reward(
    outcome: ProductionCaptureOutcome,
    *,
    reward_scale: float = 0.001,
    production_reward_beta: float = 0.0,
    production_reward_reference: float = 3.0,
) -> float:
    if outcome.total_ships_sent <= 0:
        return 0.0
    max_turns = max(1, int(outcome.max_turns))
    time_factor = remaining_steps(max_turns, outcome.capture_turn)
    reward = production_reward_value(
        outcome.planet_production,
        production_reward_beta=production_reward_beta,
        production_reward_reference=production_reward_reference,
    ) * time_factor
    if outcome.captured_from_enemy:
        reward *= opponent_count_multiplier(outcome.num_opponents)
    reward -= max(0.0, float(outcome.total_ship_investment))
    return reward * max(0.0, outcome.ships_sent / outcome.total_ships_sent) * float(reward_scale)


def compute_production_neutral_gift_penalty(
    outcome: ProductionNeutralGiftOutcome,
    *,
    reward_scale: float = 0.001,
    production_reward_beta: float = 0.0,
    production_reward_reference: float = 3.0,
) -> float:
    if outcome.launch_target_ships <= 0:
        return 0.0
    max_turns = max(1, int(outcome.max_turns))
    time_factor = remaining_steps(max_turns, outcome.capture_turn)
    damage_fraction = max(
        0.0,
        min(1.0, float(outcome.damage_done) / float(outcome.launch_target_ships)),
    )
    production_value = production_reward_value(
        outcome.planet_production,
        production_reward_beta=production_reward_beta,
        production_reward_reference=production_reward_reference,
    )
    return -production_value * time_factor * damage_fraction * float(reward_scale)


def compute_production_planet_loss_penalty(
    outcome: ProductionPlanetLossOutcome,
    *,
    reward_scale: float = 0.001,
    production_reward_beta: float = 0.0,
    production_reward_reference: float = 3.0,
) -> float:
    """Per-launch planet-loss penalty.

    A single launch may be registered as a defense-miss candidate for
    several threatened planets at launch time. To avoid compounding the
    penalty (so that losing N planets gives N× the penalty on one
    launch), we divide the per-miss penalty by `num_threatened_at_launch`.
    This means:
      - Worst case (all N threatened planets lost): total per-launch
        penalty ≈ 1× (one planet's production-weighted loss).
      - Common case (only some lost): proportional fraction.
    """
    if outcome.total_ships_sent <= 0:
        return 0.0
    max_turns = max(1, int(outcome.max_turns))
    time_factor = remaining_steps(max_turns, outcome.loss_turn)
    penalty = -production_reward_value(
        outcome.planet_production,
        production_reward_beta=production_reward_beta,
        production_reward_reference=production_reward_reference,
    ) * time_factor
    penalty *= opponent_count_multiplier(outcome.num_opponents)
    ship_share = max(0.0, outcome.ships_sent / outcome.total_ships_sent)
    threat_normalizer = 1.0 / max(1, int(outcome.num_threatened_at_launch))
    return penalty * ship_share * threat_normalizer * float(reward_scale)
