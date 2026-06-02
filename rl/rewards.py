"""Reward shaping for OrbitWars self-play (PPO + DQN).

Env emits a sparse terminal reward (+1 winner / -1 everyone else). For a
zero-sum game that's clean but slow to learn from. We layer dense
shaping on top.

# Capture-value shaping (primary signal)

A captured planet's economic value is roughly

    expected_future_ships = production * remaining_turns

so the *true* reward of capturing planet `p` at turn `t` should scale as

    capture_value(p, t) = planet_value(p) * (max_turns - t) / max_turns

- `planet_value(p)` defaults to `p.production` (gameplay-correct), but
  can switch to `p.radius` (logarithmically-related proxy).
- The (max_turns - t) / max_turns factor decays linearly so capturing
  early is worth more than capturing late.
- Losing a planet to the opponent is the negative of the capture value.
- Stealing from the opponent (vs. capturing neutral) is a 2-unit swing
  in the (mine - opp) total, so the formulation below naturally double
  counts it.

# Auxiliary signals

- `ship_diff_weight`: small bonus for ship-count advantage growing.
- `step_cost`: tiny per-step penalty (encourage decisive play).
- `planet_diff_weight`: legacy un-weighted planet count delta;
  default 0 since `capture_value_weight` supersedes it.

All shaping is potential-shaped-like in expectation (telescopes ~0 in
the long run for a balanced game), so it does not break the optimal
policy under PPO/DQN; it just shortens credit-assignment chains.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from feature_extraction import parse_planets, parse_fleets


@dataclass
class RewardConfig:
    reward_system: str = "current"
    # Terminal reward scale (env emits +/- 1; we can amplify if we want).
    terminal_scale: float = 1.0

    # ---- Capture-value shaping (the primary dense signal) ----
    # Two-part decomposition with independent weights:
    #   gain   = +capture_value_weight * time_factor * d_mine_total
    #   denial = -denial_value_weight  * time_factor * d_opp_total
    # When denial == capture they collapse to the symmetric (mine - opp)
    # swing form (zero-sum-balanced). Setting denial > capture pushes the
    # agent to prefer stealing/disrupting the opponent over neutral grabs.
    # NOTE: capture credit is also assigned per-fleet at arrival via
    # FleetTracker (see rl/fleet_tracker.py). These step-level weights are kept
    # smaller than fleet_capture_weight so they provide state-change feedback
    # without dominating launch-specific delayed credit.
    #
    # ⚠️ LEGACY (DEPRECATED for graph-native pipeline):
    # `capture_value_weight` and `denial_value_weight` are **production-delta**
    # step-shaping channels from the original (pre-graph) reward stack. They
    # are retained for backwards compatibility with the `current` / `aggressive`
    # / `production_simple` presets and for ablations. In the modern graph-
    # based stack they are explicitly zeroed.  Dense signal for ``graph_native``
    # comes from arrival credit (``enable_arrival_credit``) and terminal reward.
    # Keep these at 0.0 for any new training runs that consume the graph
    # encoder; they exist for legacy reproduction only.
    capture_value_weight: float = 0.01    # gain side: my production growth (LEGACY)
    denial_value_weight: float = 0.01     # denial side: opp's production loss (LEGACY)
    # 'production' (gameplay-true: ships/turn) or 'radius' (log-proxy)
    capture_value_kind: str = "production"
    # Total game length used to normalize the time-decay factor. Default
    # matches kaggle_environments orbit_wars (~500 steps episode).
    max_turns: int = 500
    # Floor for the time-decay factor. 0.0 lets the late-game capture
    # reward go to zero; 0.1 keeps a baseline so terminal captures still
    # register a small signal.
    time_decay_floor: float = 0.0
    # Power applied to the *normalized* remaining-time fraction
    # ((max_turns - turn) / max_turns) before the (max_turns) scale is
    # restored. curvature=1.0 matches the legacy linear decay; values
    # > 1.0 amplify the EARLY/LATE reward ratio, biasing the policy
    # toward early captures (and symmetrically penalizing early losses).
    # Affects both `current`/`aggressive` step-shaping and
    # `production_simple` capture/loss/defense-window credit.
    capture_time_curvature: float = 1.0

    # ---- Auxiliary fast-feedback signals ----
    ship_diff_weight: float = 0.0005    # per ship advantage delta (normalized by 100)
    ship_loss_weight: float = 0.001     # direct penalty per my ship lost
    fleet_airborne_weight: float = 0.00002
    fleet_capture_weight: float = 0.05
    fleet_kill_weight: float = 0.001
    fleet_waste_weight: float = 0.0015
    fleet_overcommit_weight: float = 0.0
    enemy_capture_radius_weight: float = 0.02
    planet_loss_weight: float = 0.05
    planet_loss_credit_horizon: int = 30
    # Defense-Window credit assignment for planet loss (Tasarım 2). Used
    # only by the production_simple reward system. The penalty is split:
    #   depart_share to launches FROM the lost planet (took ships away)
    #   hold_share   to passive holds (NO-OP / kept ships at planet)
    # If only one side has records in window, it absorbs the full share.
    planet_defense_horizon: int = 30
    planet_defense_depart_share: float = 0.7
    planet_defense_hold_share: float = 0.3
    # Sorun 1 (DEPRECATED): post-capture reinforcement reward. The legacy
    # mechanism unconditionally rewarded ships_sent toward a freshly
    # captured planet, even when the planet was sitting idle with no
    # threat. PPO learned to pile ships onto safe planets ("blue team
    # passing fleets back and forth" exploit). Default weight is now 0
    # so the reward never fires; use `defense_kills_weight` (below) for
    # a threat-aware replacement.
    production_reinforcement_weight: float = 0.0
    production_reinforcement_decay_window: int = 30
    production_upstream_credit_weight: float = 0.2
    production_upstream_credit_window: int = 50
    # Defense kills reward (threat-aware replacement for the legacy
    # reinforcement bonus). Tags my planets that have at least one enemy
    # fleet en route as 'under attack', then distributes the enemy ships
    # killed during the resulting combat to my launches that targeted the
    # planet AND arrived before/during the combat. Reward magnitude per
    # defender = enemy_killed * (ships_sent / total_eligible_ships)
    # * defense_kills_weight * reward_scale.
    defense_kills_weight: float = 0.001
    # Sorun 2: capture-then-lose clawback. When a captured planet is
    # lost within `clawback_window` turns of capture, a fraction of the
    # original capture reward is returned (negative reward) to the
    # original capturing transitions. Magnitude is decayed linearly with
    # the gap between capture and loss.
    production_capture_clawback_window: int = 30
    production_capture_clawback_fraction: float = 0.5
    # Continuous Beta ship head: the policy can request more ships than the
    # source planet has. The env clamps to `available`, but we still
    # penalize the over-allocation so the agent learns to track its own
    # garrison. Penalty per turn = -ship_overflow_weight * overflow_ships
    # * reward_scale (i.e. per-ship overshoot, same scale as defense_deficit).
    ship_overflow_weight: float = 1.0
    defense_hold_weight: float = 0.001
    defense_hold_horizon: int = 15
    defense_arrival_window: float = 1.5
    delayed_reward_discount: float = 1.0
    production_reward_beta: float = 0.0
    production_reward_reference: float = 3.0
    invalid_action_penalty: float = 0.01
    reward_scale: float = 1.0
    draw_terminal_reward: float = 0.0
    step_cost: float = 0.0              # tiny per-step alive cost (typically 0 or negative)
    # Deprecated compatibility knobs. Step-level shaping is no longer spread
    # backward over recent actions; rollout assigns it only to the transitions
    # created on that env step. Fleet capture/kill credit is still assigned
    # directly to the original launching transition by FleetTracker.
    eligibility_trace_horizon: int = 0
    eligibility_trace_decay: float = 0.0
    noop_trace_weight: float = 0.0
    # If True, the eligibility trace assigns MORE credit to OLDER actions in
    # the trace window and LESS to the most recent ones. This is the right
    # default for delayed-effect domains like Orbit Wars where a fleet
    # launched ~30 turns ago is what's currently producing the
    # production-diff signal, while a launch made this turn has not yet
    # arrived. Weight at lookback k (0 = current, L-1 = oldest in window):
    #   forward  : decay^k          -> recent gets max
    #   reverse  : decay^(L-1-k)    -> oldest gets max
    eligibility_trace_reverse: bool = False

    # ---- Legacy / backward-compatible ----
    # Un-weighted planet count differential (superseded by capture-value).
    # Kept at 0 by default; set non-zero to re-enable the old behavior.
    planet_diff_weight: float = 0.0

    # ---- Held-production metric (GNN node feature + arrival-credit channel) ----
    # Per-owned-planet discounted future production:
    #   prod * (1 + x + x^2 + ... + x^{R-1}),  R = remaining turns,
    #   x = per-turn hold survival (optionally from vulnerability).
    held_production_norm: float = 500.0
    hold_survival_base: float = 0.99
    hold_survival_from_vulnerability: bool = True
    hold_survival_min: float = 0.5
    hold_survival_max: float = 0.99

    # ---- Flux-augmented node metrics (observation + arrival credit) ----
    # When enabled, four additional per-node channels are appended to
    # ``compute_planet_metric_vectors`` (friendly/enemy in/out flux).
    enable_flux_metrics: bool = False
    flux_mode: str = "exp_decay"      # "raw" | "linear" | "exp_decay"
    flux_tau: float = 10.0            # decay timescale for exp_decay (turns)
    flux_eta_norm: float = 50.0       # cutoff for linear mode (turns)
    flux_ship_norm: float = 50.0      # divisor for output magnitude

    # ---- Arrival-credit channel weights (independent of any step shaping) ----
    # Per-channel weights for counterfactual credit at fleet arrival.
    # Order mirrors ``model_graph_pyg._graph_node_metrics`` (base 10 + flux 4).
    # ``influence_balance`` and ``strategic_potential`` are redundant linear
    # combos — keep at 0 unless ablating.
    arrival_credit_friendly_influence_weight: float = 0.0
    arrival_credit_enemy_influence_weight: float = 0.0
    arrival_credit_influence_balance_weight: float = 0.0
    arrival_credit_friendly_support_weight: float = 0.0
    arrival_credit_supportability_weight: float = 0.0
    arrival_credit_vulnerability_weight: float = 0.0
    arrival_credit_production_potential_weight: float = 0.0
    arrival_credit_friendly_production_potential_weight: float = 0.0
    arrival_credit_strategic_potential_weight: float = 0.0
    arrival_credit_held_discounted_production_weight: float = 0.0
    arrival_credit_flux_friendly_in_weight: float = 0.0
    arrival_credit_flux_friendly_out_weight: float = 0.0
    arrival_credit_flux_enemy_in_weight: float = 0.0
    arrival_credit_flux_enemy_out_weight: float = 0.0

    # ---- Arrival-credit eligibility trace (ETA-weighted backward credit) ----
    #
    # On launch: snapshot per-planet metric vector.  On fleet arrival:
    #
    #     credit = w(eta) * arrival_credit_weight * reward_scale *
    #              Σ_c arrival_credit_*_weight[c] · Δmetric[c]
    #
    # Default off; opt-in via ``enable_arrival_credit=True``.
    enable_arrival_credit: bool = False
    arrival_credit_mode: str = "exp_decay"   # "raw" | "linear" | "exp_decay"
    arrival_credit_tau: float = 10.0         # decay timescale (turns)
    arrival_credit_eta_norm: float = 50.0    # cutoff for "linear" mode
    arrival_credit_weight: float = 1.0       # global scale on the credit
    # If True, arrival credit compares actual metrics at arrival against
    # a counterfactual branch where the fleet did not arrive and stayed at
    # the source.  If removing the arrival makes a self-owned target go
    # negative, the counterfactual target is marked non-self so friendly
    # production/influence metrics are disabled naturally.
    arrival_credit_counterfactual: bool = False
    # "target" | "target_source" | "full_graph".  Full-graph sums the
    # weighted metric delta across all planets, so diffusion effects from
    # target/source ownership changes are captured.
    arrival_credit_counterfactual_scope: str = "full_graph"
    arrival_credit_counterfactual_return_to_source: bool = True
    # Optional short horizon after arrival.  When >0, both actual and
    # counterfactual branches resolve the soonest enemy arrival at the
    # affected self-owned planets if its ETA is within this cap.
    arrival_credit_counterfactual_horizon: float = 0.0
    # If True, the snapshot also captures the *source* planet vector and
    # the credit is the sum of (target delta) + (-source delta), which
    # rewards the agent both for what the fleet achieves at tgt and for
    # the cost it pays at src (e.g. source becoming more vulnerable).
    # Default False to keep the signal focused on tgt impact.
    arrival_credit_include_source: bool = False

    # In counterfactual ``full_graph`` scope, which planets contribute to
    # the (curr - cf) sum:
    #   "self"  -> only planets currently owned by the launching player
    #              (the natural "what happened to MY empire" reading)
    #   "all"   -> every real planet (legacy behaviour; double-counts
    #              enemy-side effects that the player can't directly use)
    arrival_credit_owner_filter: str = "self"


REWARD_SYSTEMS: tuple[str, ...] = (
    "current", "aggressive", "production_simple", "production_diff", "production_only",
    "graph_native",
)


def build_reward_config(reward_system: str = "current", **overrides) -> RewardConfig:
    if reward_system == "current":
        cfg = RewardConfig(reward_system="current")
    elif reward_system == "production_diff":
        # Minimal "production differential + eligibility trace" system.
        # Per-step reward = (sum_my_production - sum_opp_production / num_opp)
        #                   * reward_scale
        # That single scalar is then distributed back to launch transitions
        # over the last `eligibility_trace_horizon` turns with exponential
        # decay (`eligibility_trace_decay`). NO-OP credit is scaled by
        # `noop_trace_weight`. ALL legacy tracker channels (capture, defense
        # kills, planet defense, fleet combat, airborne, overflow,
        # defense_deficit, ...) are disabled here so the reward landscape
        # is exactly: dense production diff + terminal +/-1.
        cfg = RewardConfig(
            reward_system="production_diff",
            # All shaping channels off.
            capture_value_weight=0.0,
            denial_value_weight=0.0,
            ship_diff_weight=0.0,
            ship_loss_weight=0.0,
            fleet_airborne_weight=0.0,
            fleet_capture_weight=0.0,
            fleet_kill_weight=0.0,
            fleet_waste_weight=0.0,
            fleet_overcommit_weight=0.0,
            enemy_capture_radius_weight=0.0,
            planet_loss_weight=0.0,
            defense_hold_weight=0.0,
            ship_overflow_weight=0.0,
            defense_kills_weight=0.0,
            production_reinforcement_weight=0.0,
            invalid_action_penalty=0.0,
            planet_diff_weight=0.0,
            # Eligibility trace defaults. Reverse=True so older launches
            # (whose fleets are arriving NOW and producing the current
            # production-diff signal) receive the bulk of the credit.
            eligibility_trace_horizon=30,
            eligibility_trace_decay=0.9,
            eligibility_trace_reverse=True,
            noop_trace_weight=1.0,
            delayed_reward_discount=1.0,
            reward_scale=0.001,
            draw_terminal_reward=0.0,
        )
    elif reward_system == "production_only":
        cfg = RewardConfig(
            reward_system="production_only",
            capture_value_weight=0.0,
            denial_value_weight=0.0,
            ship_diff_weight=0.0,
            ship_loss_weight=0.0,
            fleet_airborne_weight=0.0,
            fleet_capture_weight=0.0,
            fleet_kill_weight=1.0,
            fleet_waste_weight=0.0,
            fleet_overcommit_weight=0.0,
            enemy_capture_radius_weight=0.0,
            planet_loss_weight=0.0,
            defense_hold_weight=0.0,
            ship_overflow_weight=0.0,
            defense_kills_weight=0.0,
            production_reinforcement_weight=0.0,
            invalid_action_penalty=0.0,
            planet_diff_weight=0.0,
            eligibility_trace_horizon=30,
            eligibility_trace_decay=0.9,
            eligibility_trace_reverse=True,
            noop_trace_weight=1.0,
            delayed_reward_discount=1.0,
            reward_scale=0.001,
            draw_terminal_reward=0.0,
        )
    elif reward_system == "graph_native":
        cfg = RewardConfig(
            reward_system="graph_native",
            capture_value_weight=0.0,
            denial_value_weight=0.0,
            ship_diff_weight=0.0,
            ship_loss_weight=0.0,
            fleet_airborne_weight=0.0,
            fleet_capture_weight=0.0,
            fleet_kill_weight=0.0,
            fleet_waste_weight=0.0,
            fleet_overcommit_weight=0.0,
            enemy_capture_radius_weight=0.0,
            planet_loss_weight=0.0,
            defense_hold_weight=0.0,
            ship_overflow_weight=0.0,
            defense_kills_weight=0.0,
            production_reinforcement_weight=0.0,
            invalid_action_penalty=0.0,
            planet_diff_weight=0.0,
            # Arrival-credit channel defaults (independent channels only).
            arrival_credit_friendly_influence_weight=1.0,
            arrival_credit_enemy_influence_weight=-1.0,
            arrival_credit_influence_balance_weight=0.0,
            arrival_credit_friendly_support_weight=1.0,
            arrival_credit_supportability_weight=1.0,
            arrival_credit_vulnerability_weight=-1.0,
            arrival_credit_production_potential_weight=1.0,
            arrival_credit_friendly_production_potential_weight=1.0,
            arrival_credit_held_discounted_production_weight=1.0,
            arrival_credit_strategic_potential_weight=0.0,
            held_production_norm=500.0,
            hold_survival_from_vulnerability=True,
            reward_scale=1.0,
            draw_terminal_reward=0.0,
            step_cost=0.0,
        )
    elif reward_system == "aggressive":
        cfg = RewardConfig(
            reward_system="aggressive",
            capture_value_weight=0.015,
            denial_value_weight=0.025,
            fleet_airborne_weight=0.000025,
            fleet_capture_weight=0.06,
            fleet_kill_weight=0.001,
            fleet_waste_weight=0.0015,
            enemy_capture_radius_weight=0.035,
            planet_loss_weight=0.075,
            defense_hold_weight=0.0015,
        )
    elif reward_system == "production_simple":
        cfg = RewardConfig(
            reward_system="production_simple",
            capture_value_weight=0.0,
            denial_value_weight=0.0,
            ship_diff_weight=0.0,
            ship_loss_weight=0.0,
            # NOTE: airborne is risk-aware in rollout and intentionally
            # uses an unnormalized base weight in production_simple.
            fleet_airborne_weight=2.0,
            fleet_capture_weight=0.0,
            fleet_kill_weight=1.0,
            fleet_waste_weight=1.0,
            fleet_overcommit_weight=0.0,
            enemy_capture_radius_weight=0.0,
            planet_loss_weight=1.0,
            defense_hold_weight=0.0,
            defense_kills_weight=0.05,
            production_upstream_credit_weight=0.2,
            production_upstream_credit_window=50,
            delayed_reward_discount=1.0,
            reward_scale=0.001,
            draw_terminal_reward=-1.0,
            planet_diff_weight=0.0,
        )
    else:
        raise ValueError(
            f"Unknown reward_system={reward_system!r}; "
            f"expected one of {', '.join(REWARD_SYSTEMS)}"
        )

    clean_overrides = {
        key: value
        for key, value in overrides.items()
        if value is not None
    }
    return replace(cfg, **clean_overrides)


# ---- helpers --------------------------------------------------------------

def _planet_value(p, kind: str) -> float:
    if kind == "radius":
        return float(p.radius)
    if kind == "production":
        return float(p.production)
    raise ValueError(f"Unknown capture_value_kind: {kind!r} (use 'production' or 'radius')")


def curved_time_factor(max_turns: int, turn: int, *,
                       curvature: float = 1.0,
                       floor_norm: float = 0.0) -> float:
    """Time-decay factor with optional super-linear curvature.

    Returns ``max_turns * max(floor_norm, ((max_turns - turn) / max_turns) ** curvature)``.

    With ``curvature=1.0`` this matches the legacy linear ``max_turns - turn``
    (modulo the ``floor_norm`` clamp). Values > 1.0 keep the t=0 peak at
    ``max_turns`` but accelerate the dropoff toward t=max_turns, so the
    EARLY-vs-LATE reward ratio grows without inflating the per-capture
    scale (PPO value head stays well-behaved).
    """
    T = max(1, int(max_turns))
    remaining_norm = max(0.0, (T - int(turn)) / T)
    decayed = remaining_norm ** max(0.0, float(curvature))
    return max(float(floor_norm), decayed) * float(T)


def _capture_value_components(prev_env_obs, curr_env_obs, player: int, *,
                               kind: str, time_factor: float) -> tuple[float, float]:
    """Time-weighted production-value deltas, split into (gain, denial).

    Returns
    -------
    gain   : float  -- time_factor * d_mine_total   (positive when I capture)
    denial : float  -- time_factor * (-d_opp_total) (positive when opp loses)

    Capturing neutral -> gain only.
    Stealing from opp -> gain AND denial (both positive).
    Losing to opp     -> negative gain AND negative denial.

    Comets are excluded (transient, production=1, ownership churn noisy).
    """
    prev_planets = {p.id: p for p in parse_planets(prev_env_obs) if not p.is_comet}
    curr_planets = {p.id: p for p in parse_planets(curr_env_obs) if not p.is_comet}
    def total_for(pmap, owner):
        if owner == "opponents":
            return sum(
                _planet_value(p, kind)
                for p in pmap.values()
                if p.owner not in (player, -1)
            )
        return sum(_planet_value(p, kind) for p in pmap.values() if p.owner == owner)

    d_mine = total_for(curr_planets, player) - total_for(prev_planets, player)
    d_opp = total_for(curr_planets, "opponents") - total_for(prev_planets, "opponents")
    return time_factor * d_mine, time_factor * (-d_opp)


def _player_score(env_obs, player: int) -> tuple[int, int]:
    """Return (n_planets_owned, total_ships_owned_garrisoned_or_in_flight)."""
    planets = parse_planets(env_obs)
    fleets = parse_fleets(env_obs)
    n_planets = sum(1 for p in planets if p.owner == player and not p.is_comet)
    ships = sum(p.ships for p in planets if p.owner == player and not p.is_comet)
    ships += sum(f.ships for f in fleets if f.owner == player)
    return n_planets, ships


def _current_turn(env_obs) -> int:
    if isinstance(env_obs, dict):
        return int(env_obs.get("step", 0) or 0)
    return int(getattr(env_obs, "step", 0) or 0)


def normalize_terminal_rewards(raw_rewards: list[float], *, draw_value: float = 0.0) -> list[float]:
    if not raw_rewards:
        return []
    best_reward = max(raw_rewards)
    winners = [i for i, r in enumerate(raw_rewards) if r == best_reward]
    if len(winners) != 1 or best_reward <= 0:
        return [float(draw_value) for _ in raw_rewards]
    winner = winners[0]
    return [1.0 if i == winner else -1.0 for i in range(len(raw_rewards))]


def compute_step_reward(
    prev_env_obs,
    curr_env_obs,
    player: int,
    *,
    terminal: bool,
    env_terminal_reward: float,
    cfg: Optional[RewardConfig] = None,
    invalid_actions: int = 0,
) -> float:
    """Compute the per-step reward for `player`.

    Args
    ----
    prev_env_obs : env observation BEFORE the step (player's view)
    curr_env_obs : env observation AFTER the step (player's view)
    player : 0 or 1
    terminal : whether the episode ended this step
    env_terminal_reward : `state[player].reward` at this step (env's
        sparse signal; non-zero only at terminal)
    cfg : RewardConfig

    Returns
    -------
    float reward in [~-1, ~+1] range typically.
    """
    cfg = cfg or RewardConfig()

    # ---- Time-decay factor (1.0 at start, -> floor at end of game) ----
    # Curvature > 1.0 amplifies the early/late reward ratio (capture
    # *and* loss are weighted by the same factor, so symmetry is
    # preserved automatically inside _capture_value_components).
    turn = _current_turn(curr_env_obs)
    T = max(1, int(cfg.max_turns))
    raw_factor = max(0.0, (T - turn) / T)
    curved = raw_factor ** max(0.0, float(cfg.capture_time_curvature))
    time_factor = max(cfg.time_decay_floor, curved)

    # ---- Primary capture-value shaping (production-weighted, time-decayed) ----
    # gain   = +production captured by me (mine grows)
    # denial = +production destroyed for opp (opp shrinks)
    # Independent weights let you tune aggression vs growth bias.
    gain, denial = _capture_value_components(
        prev_env_obs, curr_env_obs, player,
        kind=cfg.capture_value_kind, time_factor=time_factor,
    )

    # ---- Auxiliary signals (legacy planet diff + ship diff + step cost) ----
    p_prev, s_prev = _player_score(prev_env_obs, player)
    p_curr, s_curr = _player_score(curr_env_obs, player)
    prev_planets = parse_planets(prev_env_obs)
    prev_fleets = parse_fleets(prev_env_obs)
    curr_planets = parse_planets(curr_env_obs)
    curr_fleets = parse_fleets(curr_env_obs)
    op_prev = sum(1 for p in prev_planets if p.owner not in (player, -1) and not p.is_comet)
    os_prev = sum(p.ships for p in prev_planets if p.owner not in (player, -1) and not p.is_comet)
    os_prev += sum(f.ships for f in prev_fleets if f.owner not in (player, -1))
    op_curr = sum(1 for p in curr_planets if p.owner not in (player, -1) and not p.is_comet)
    os_curr = sum(p.ships for p in curr_planets if p.owner not in (player, -1) and not p.is_comet)
    os_curr += sum(f.ships for f in curr_fleets if f.owner not in (player, -1))
    d_planets = (p_curr - p_prev) - (op_curr - op_prev)
    d_ship_diff = ((s_curr - os_curr) - (s_prev - os_prev)) / 100.0
    ship_loss = max(0.0, float(s_prev - s_curr))

    shaped = (cfg.capture_value_weight * gain
              + cfg.denial_value_weight * denial
              + cfg.planet_diff_weight * d_planets
              + cfg.ship_diff_weight * d_ship_diff
              - cfg.ship_loss_weight * ship_loss
              - cfg.invalid_action_penalty * max(0, int(invalid_actions))
              + cfg.step_cost)

    # Terminal step: legacy capture/diff shaping is suppressed; terminal
    # +/-1 is added separately.
    if terminal:
        shaped = 0.0

    terminal_term = cfg.terminal_scale * float(env_terminal_reward) if terminal else 0.0
    return float(shaped + terminal_term)


def arrival_credit_channel_weights(cfg: Optional[RewardConfig] = None) -> tuple[float, ...]:
    """Per-node metric weights for arrival credit and scripted graph agents.

    Order mirrors ``model_graph_pyg._graph_node_metrics`` base 10 (no flux).
    """
    cfg = cfg or RewardConfig()
    return (
        float(cfg.arrival_credit_friendly_influence_weight),
        float(cfg.arrival_credit_enemy_influence_weight),
        float(cfg.arrival_credit_influence_balance_weight),
        float(cfg.arrival_credit_friendly_support_weight),
        float(cfg.arrival_credit_supportability_weight),
        float(cfg.arrival_credit_vulnerability_weight),
        float(cfg.arrival_credit_production_potential_weight),
        float(cfg.arrival_credit_friendly_production_potential_weight),
        float(cfg.arrival_credit_strategic_potential_weight),
        float(cfg.arrival_credit_held_discounted_production_weight),
    )


def _hold_metric_kwargs_from_cfg(
    cfg: Optional[RewardConfig],
    *,
    turn: int = 0,
    obs=None,
) -> dict[str, float | int | bool]:
    """Kwargs forwarded to ``_graph_node_metrics`` for held production."""
    if obs is not None:
        try:
            from model_graph_pyg import _hold_metric_kwargs_from_obs

            meta = getattr(obs, "meta", None) or {}
            out = _hold_metric_kwargs_from_obs(obs)
            if cfg is not None:
                out["held_production_norm"] = float(getattr(cfg, "held_production_norm", 500.0))
                out["hold_survival_base"] = float(getattr(cfg, "hold_survival_base", 0.99))
                out["hold_survival_from_vulnerability"] = bool(
                    getattr(cfg, "hold_survival_from_vulnerability", True)
                )
                out["hold_survival_min"] = float(getattr(cfg, "hold_survival_min", 0.5))
                out["hold_survival_max"] = float(getattr(cfg, "hold_survival_max", 0.99))
                if "remaining_turns" not in meta:
                    out["remaining_turns"] = max(
                        0, int(getattr(cfg, "max_turns", 500)) - int(turn)
                    )
            return out
        except Exception:
            pass
    max_turns = int(getattr(cfg, "max_turns", 500)) if cfg is not None else 500
    return {
        "remaining_turns": max(0, max_turns - int(turn)),
        "held_production_norm": float(getattr(cfg, "held_production_norm", 500.0)),
        "hold_survival_base": float(getattr(cfg, "hold_survival_base", 0.99)),
        "hold_survival_from_vulnerability": bool(
            getattr(cfg, "hold_survival_from_vulnerability", True)
        ),
        "hold_survival_min": float(getattr(cfg, "hold_survival_min", 0.5)),
        "hold_survival_max": float(getattr(cfg, "hold_survival_max", 0.99)),
    }


def compute_planet_metric_vectors(
    obs,
    *,
    cfg: Optional[RewardConfig] = None,
    turn: int = 0,
):
    """Per-planet graph metric matrix used by arrival-credit snapshots.

    Returns a ``(num_real_planets, D)`` ``numpy.ndarray`` where ``D`` is
    10 (base) or 14 (base + flux) depending on ``cfg.enable_flux_metrics``.
    Padded planet slots are *excluded* — index 0..num_real_planets-1
    corresponds to the first ``num_real_planets`` planet slots in
    ``obs.planet_mask`` (which mirrors the convention used elsewhere in
    the codebase: real planets occupy the leading slots).

    Channel order matches :func:`rl.arrival_credit.build_channel_weights`:

        [friendly_influence, enemy_influence, influence_balance,
         friendly_support, supportability, vulnerability,
         production_potential, friendly_production_potential,
         strategic_potential, held_discounted_production,
         flux_friendly_in, flux_friendly_out,
         flux_enemy_in, flux_enemy_out]

    Returns ``None`` if torch / model_graph_pyg is unavailable or there
    are no real planets, so callers can no-op gracefully.
    """
    try:
        import numpy as np
        import torch
        from model_graph_pyg import (
            _estimated_intercept_eta_matrix,
            _graph_node_metrics,
        )
    except Exception:
        return None

    flux_enabled = bool(cfg is not None and getattr(cfg, "enable_flux_metrics", False))

    with torch.no_grad():
        planet_feats = torch.as_tensor(obs.planet_feats, dtype=torch.float32)
        edge_bias = torch.as_tensor(obs.edge_bias, dtype=torch.float32)
        planet_mask = torch.as_tensor(obs.planet_mask, dtype=torch.bool)
        n = int(planet_mask.sum().item())
        if n <= 0:
            return None
        eta = _estimated_intercept_eta_matrix(planet_feats, edge_bias, n)
        flux_features = None
        if flux_enabled:
            from .flux import compute_node_flux_features
            flux_np = compute_node_flux_features(
                obs,
                p_max=int(planet_feats.shape[0]),
                mode=getattr(cfg, "flux_mode", "exp_decay"),
                tau=float(getattr(cfg, "flux_tau", 10.0)),
                eta_norm=float(getattr(cfg, "flux_eta_norm", 50.0)),
                ship_norm=float(getattr(cfg, "flux_ship_norm", 50.0)),
            )
            flux_features = torch.as_tensor(flux_np[:n], dtype=torch.float32)
        hold_kwargs = _hold_metric_kwargs_from_cfg(cfg, turn=turn, obs=obs)
        node_metrics = _graph_node_metrics(
            planet_feats, edge_bias, n, eta=eta, flux_features=flux_features, **hold_kwargs,
        )
        return node_metrics.detach().cpu().numpy().astype(np.float32, copy=False)



def sum_planet_metric_channels(
    obs,
    *,
    cfg: Optional[RewardConfig] = None,
    turn: int = 0,
) -> list[float]:
    """Sum of per-planet metric vectors (diagnostics / tests).

    Returns a list of length 10 or 14 matching ``compute_planet_metric_vectors``.
    """
    mat = compute_planet_metric_vectors(obs, cfg=cfg, turn=turn)
    if mat is None:
        flux_enabled = bool(cfg is not None and getattr(cfg, "enable_flux_metrics", False))
        return [0.0] * (14 if flux_enabled else 10)
    return mat.sum(axis=0).tolist()
