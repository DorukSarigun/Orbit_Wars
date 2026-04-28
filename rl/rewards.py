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

from dataclasses import dataclass
from typing import Optional

from feature_extraction import parse_planets, parse_fleets


@dataclass
class RewardConfig:
    # Terminal reward scale (env emits +/- 1; we can amplify if we want).
    terminal_scale: float = 1.0

    # ---- Capture-value shaping (the primary dense signal) ----
    # Two-part decomposition with independent weights:
    #   gain   = +capture_value_weight * time_factor * d_mine_total
    #   denial = -denial_value_weight  * time_factor * d_opp_total
    # When denial == capture they collapse to the symmetric (mine - opp)
    # swing form (zero-sum-balanced). Setting denial > capture pushes the
    # agent to prefer stealing/disrupting the opponent over neutral grabs.
    capture_value_weight: float = 0.05    # gain side: my production growth
    denial_value_weight: float = 0.05     # denial side: opp's production loss
    # 'production' (gameplay-true: ships/turn) or 'radius' (log-proxy)
    capture_value_kind: str = "production"
    # Total game length used to normalize the time-decay factor. Default
    # matches kaggle_environments orbit_wars (~500 steps episode).
    max_turns: int = 500
    # Floor for the time-decay factor. 0.0 lets the late-game capture
    # reward go to zero; 0.1 keeps a baseline so terminal captures still
    # register a small signal.
    time_decay_floor: float = 0.0

    # ---- Auxiliary fast-feedback signals ----
    ship_diff_weight: float = 0.0005    # per ship advantage delta (normalized by 100)
    step_cost: float = 0.0              # tiny per-step alive cost (typically 0 or negative)

    # ---- Legacy / backward-compatible ----
    # Un-weighted planet count differential (superseded by capture-value).
    # Kept at 0 by default; set non-zero to re-enable the old behavior.
    planet_diff_weight: float = 0.0

    # If True, shaping signals also fire on the terminal step.
    shape_on_terminal: bool = False


# ---- helpers --------------------------------------------------------------

def _planet_value(p, kind: str) -> float:
    if kind == "radius":
        return float(p.radius)
    if kind == "production":
        return float(p.production)
    raise ValueError(f"Unknown capture_value_kind: {kind!r} (use 'production' or 'radius')")


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
    opp = 1 - player

    def total_for(pmap, owner):
        return sum(_planet_value(p, kind) for p in pmap.values() if p.owner == owner)

    d_mine = total_for(curr_planets, player) - total_for(prev_planets, player)
    d_opp = total_for(curr_planets, opp) - total_for(prev_planets, opp)
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


def compute_step_reward(
    prev_env_obs,
    curr_env_obs,
    player: int,
    *,
    terminal: bool,
    env_terminal_reward: float,
    cfg: Optional[RewardConfig] = None,
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
    turn = _current_turn(curr_env_obs)
    raw_factor = max(0.0, (cfg.max_turns - turn) / max(1, cfg.max_turns))
    time_factor = max(cfg.time_decay_floor, raw_factor)

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
    opp = 1 - player
    op_prev, os_prev = _player_score(prev_env_obs, opp)
    p_curr, s_curr = _player_score(curr_env_obs, player)
    op_curr, os_curr = _player_score(curr_env_obs, opp)
    d_planets = (p_curr - p_prev) - (op_curr - op_prev)
    d_ship_diff = ((s_curr - os_curr) - (s_prev - os_prev)) / 100.0

    shaped = (cfg.capture_value_weight * gain
              + cfg.denial_value_weight * denial
              + cfg.planet_diff_weight * d_planets
              + cfg.ship_diff_weight * d_ship_diff
              + cfg.step_cost)

    # Suppress shaping on terminal step if requested (some prefer clean
    # terminal-only credit assignment).
    if terminal and not cfg.shape_on_terminal:
        shaped = 0.0

    terminal_term = cfg.terminal_scale * float(env_terminal_reward) if terminal else 0.0
    return float(shaped + terminal_term)
