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
    capture_value_weight: float = 0.01    # gain side: my production growth
    denial_value_weight: float = 0.01     # denial side: opp's production loss
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

    # ---- Graph-native shaping (structure delta over graph metrics) ----
    # Per-channel weights applied to the *delta* of summed node metrics.
    # NOTE: ``influence_balance`` and ``strategic_potential`` are linear
    # combinations of the other three (friendly/enemy/production); leaving
    # them weighted in addition to the primaries triple-counts the same
    # signal.  Defaults are 0.0 so only independent channels contribute.
    graph_friendly_influence_weight: float = 0.0
    graph_enemy_influence_weight: float = 0.0
    graph_influence_balance_weight: float = 0.0      # redundant: friendly - enemy
    graph_friendly_support_weight: float = 0.0
    graph_supportability_weight: float = 0.0
    graph_vulnerability_weight: float = 0.0
    graph_production_potential_weight: float = 0.0
    graph_strategic_potential_weight: float = 0.0    # redundant: production + friendly - enemy
    # Edge-level shaping. Activates a single weighted channel on the sum
    # of an edge metric across all real planet pairs (default
    # ``flow_opportunity`` -- the only directed, non-anti-symmetric edge
    # metric that captures progress toward high-strategic targets).
    # Choices: ``flow_opportunity`` | ``movement_resistance`` | ``same_owner``.
    graph_edge_efficiency_weight: float = 0.0
    graph_edge_efficiency_signal: str = "flow_opportunity"

    # ---- Potential-based shaping & normalization ----
    # Discount used in the potential-based shaping formula
    #   r_t = graph_potential_discount * φ(s_{t+1}) - φ(s_t)
    # which preserves the optimal policy under any γ <= this value (Ng et
    # al. 1999).  Should match (or be slightly less than) the PPO ``gamma``
    # to keep the shaping theoretically valid.
    graph_potential_discount: float = 0.99
    # If True, per-step graph_native rewards are divided by a running
    # standard deviation (per-player) before being added to the buffer.
    # Reduces variance from heterogeneous metric scales without changing
    # the sign or relative ordering of step rewards.
    graph_reward_normalize: bool = False
    graph_reward_normalize_eps: float = 1e-6
    graph_reward_normalize_warmup: int = 64

    # If True, shaping signals also fire on the terminal step.
    shape_on_terminal: bool = False


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
            # Graph-native channel weights.  Independent channels only:
            # influence_balance and strategic_potential are linear combos
            # of friendly/enemy/production and would triple-count the
            # same signal (kept at 0 by default).
            graph_friendly_influence_weight=1.0,
            graph_enemy_influence_weight=-1.0,
            graph_influence_balance_weight=0.0,           # redundant
            graph_friendly_support_weight=1.0,
            graph_supportability_weight=1.0,
            graph_vulnerability_weight=-1.0,
            graph_production_potential_weight=1.0,
            graph_strategic_potential_weight=0.0,         # redundant
            # Edge-level signal: flow_opportunity is positive and
            # asymmetric (relu(strategic_gradient) / (eta+1)), so it
            # rewards launches that progress toward high-strategic
            # targets relative to current control.
            graph_edge_efficiency_weight=0.5,
            graph_edge_efficiency_signal="flow_opportunity",
            # Potential-based shaping (Ng et al. 1999): r = γ φ(s') - φ(s).
            graph_potential_discount=0.99,
            # Running-std normalization helps balance heterogeneous
            # metric scales (ships vs. production vs. 0-1 metrics).
            graph_reward_normalize=True,
            reward_scale=0.001,
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

    # Suppress shaping on terminal step if requested (some prefer clean
    # terminal-only credit assignment).
    if terminal and not cfg.shape_on_terminal:
        shaped = 0.0

    terminal_term = cfg.terminal_scale * float(env_terminal_reward) if terminal else 0.0
    return float(shaped + terminal_term)


# Mapping of edge-efficiency signal name -> column index in
# GRAPH_EDGE_METRIC_DIM (see model_graph_pyg._graph_edge_metric_matrix).
_EDGE_SIGNAL_INDEX = {
    "eta_norm": 0,
    "movement_resistance": 1,
    "same_owner": 2,
    "is_self_src": 3,
    "is_self_tgt": 4,
    "is_enemy_tgt": 5,
    "is_neutral_tgt": 6,
    "strategic_gradient": 7,
    "flow_opportunity": 8,
}


def compute_graph_metrics_summary(obs, *, edge_signal: str = "flow_opportunity") -> dict:
    """Lightweight scalar summary of graph metrics for reward shaping.

    Returns ``{"node_sum": np.ndarray(GRAPH_NODE_METRIC_DIM,), "edge_sum": float}``.
    Computed without constructing a PyG ``Data`` object: ETA is computed
    once and shared between node + edge metric matrices.

    Falls back to a zeroed summary if torch / model_graph_pyg is unavailable.
    """
    try:
        import torch
        from model_graph_pyg import (
            GRAPH_NODE_METRIC_DIM,
            _estimated_intercept_eta_matrix,
            _graph_edge_metric_matrix,
            _graph_node_metrics,
        )
    except Exception:
        return {"node_sum": [0.0] * 8, "edge_sum": 0.0}

    edge_idx = _EDGE_SIGNAL_INDEX.get(edge_signal, _EDGE_SIGNAL_INDEX["flow_opportunity"])

    with torch.no_grad():
        planet_feats = torch.as_tensor(obs.planet_feats, dtype=torch.float32)
        edge_bias = torch.as_tensor(obs.edge_bias, dtype=torch.float32)
        planet_mask = torch.as_tensor(obs.planet_mask, dtype=torch.bool)
        n = int(planet_mask.sum().item())
        if n <= 0:
            return {"node_sum": [0.0] * GRAPH_NODE_METRIC_DIM, "edge_sum": 0.0}
        eta = _estimated_intercept_eta_matrix(planet_feats, edge_bias, n)
        node_metrics = _graph_node_metrics(planet_feats, edge_bias, n, eta=eta)
        edge_metrics = _graph_edge_metric_matrix(
            planet_feats, edge_bias, n, eta=eta, node_metrics=node_metrics,
        )
        node_sum = node_metrics.sum(dim=0).cpu().tolist()
        edge_sum = float(edge_metrics[..., edge_idx].sum().item())
    return {"node_sum": node_sum, "edge_sum": edge_sum}


def compute_graph_native_reward(
    prev_obs_or_summary,
    curr_obs_or_summary,
    player: int,
    *,
    cfg: Optional[RewardConfig] = None,
    turn: int = 0,
) -> float:
    """Graph-native potential-based reward.

    Either argument may be an ``Observation`` (in which case a metric
    summary is computed on the fly) or a pre-computed summary dict from
    :func:`compute_graph_metrics_summary`.  Passing pre-computed summaries
    avoids a duplicate metric pass when the rollout caches the previous
    step's summary.

    Implements **potential-based shaping** (Ng, Harada & Russell 1999):

        r_t = γ · φ(s_{t+1}) − φ(s_t)

    where φ is the weighted sum of graph node + edge metric channels and
    γ is :attr:`RewardConfig.graph_potential_discount` (matching PPO
    ``gamma`` keeps the optimal policy invariant).

    Falls back to ``0.0`` if torch / model_graph_pyg is unavailable.
    """
    cfg = cfg or RewardConfig()

    def _as_summary(x):
        if isinstance(x, dict):
            return x
        return compute_graph_metrics_summary(x, edge_signal=cfg.graph_edge_efficiency_signal)

    prev = _as_summary(prev_obs_or_summary)
    curr = _as_summary(curr_obs_or_summary)

    prev_node = prev.get("node_sum") or []
    curr_node = curr.get("node_sum") or []
    if not prev_node or not curr_node or len(prev_node) != len(curr_node):
        return 0.0

    weights = (
        cfg.graph_friendly_influence_weight,
        cfg.graph_enemy_influence_weight,
        cfg.graph_influence_balance_weight,
        cfg.graph_friendly_support_weight,
        cfg.graph_supportability_weight,
        cfg.graph_vulnerability_weight,
        cfg.graph_production_potential_weight,
        cfg.graph_strategic_potential_weight,
    )
    if len(weights) != len(curr_node):
        weights = weights[: len(curr_node)] + (0.0,) * max(0, len(curr_node) - len(weights))

    discount = float(cfg.graph_potential_discount)
    # Potential-based shaping per channel: r_i = γ * curr_i - prev_i
    raw_reward = 0.0
    for w, p_i, c_i in zip(weights, prev_node, curr_node):
        if w == 0.0:
            continue
        raw_reward += float(w) * (discount * float(c_i) - float(p_i))

    # Edge-level signal (single scalar channel).
    if cfg.graph_edge_efficiency_weight != 0.0:
        prev_edge = float(prev.get("edge_sum", 0.0))
        curr_edge = float(curr.get("edge_sum", 0.0))
        raw_reward += cfg.graph_edge_efficiency_weight * (discount * curr_edge - prev_edge)

    T = max(1, int(cfg.max_turns))
    time_norm = max(0.0, (T - turn) / T)
    time_factor = max(
        cfg.time_decay_floor,
        time_norm ** max(0.0, float(cfg.capture_time_curvature)),
    )

    return float(raw_reward * time_factor * cfg.reward_scale)


class RunningMeanStd:
    """Welford-style running estimator of mean and variance for scalars.

    Used by the graph_native reward path to normalize per-step rewards
    by a running standard deviation (per-player), which keeps the
    advantage scale stable across the heterogeneous graph metric channels
    (ships ~ 0..1000 vs. supportability ~ 0..1).

    Notes
    -----
    - Only the standard deviation is consumed (the mean is *not*
      subtracted); subtracting the mean would shift the equilibrium of a
      potential-based shaping signal away from zero.
    - During the first ``warmup`` updates the std is treated as ``1.0`` so
      we don't amplify near-zero noise into huge rewards.
    """

    __slots__ = ("count", "_mean", "_m2", "warmup")

    def __init__(self, *, warmup: int = 64) -> None:
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self.warmup = max(0, int(warmup))

    def update(self, value: float) -> None:
        v = float(value)
        self.count += 1
        delta = v - self._mean
        self._mean += delta / self.count
        delta2 = v - self._mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def normalize(self, value: float, *, eps: float = 1e-6) -> float:
        """Return ``value / max(std, eps)`` once warmup is satisfied; else ``value``."""
        if self.count < self.warmup:
            return float(value)
        s = self.std
        return float(value) / max(s, float(eps))
