"""Reward shaping for OrbitWars self-play PPO.

Env emits a sparse terminal reward (+1 winner / -1 everyone else). For a
2-player zero-sum game that's clean but slow to learn from. We add an
optional *dense shaping* signal computed from observation deltas:

    shaping_t = w_p * (Δ planets_owned)
              + w_s * (Δ ship_diff_normalized)

where the delta is taken w.r.t. the previous turn from the player's own
perspective. Shaping is potential-based-like (telescopes to ~zero on
average if we don't win), so it is unbiased asymptotically and doesn't
break optimal policies.

To preserve the unbiased property we use a *small* coefficient. Set
weights to 0 to recover pure terminal reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from feature_extraction import parse_planets, parse_fleets


@dataclass
class RewardConfig:
    # Terminal reward scale (env emits +/- 1; we can amplify if we want).
    terminal_scale: float = 1.0
    # Dense shaping weights (small, to bias exploration, not dominate).
    planet_diff_weight: float = 0.01    # per planet captured (net)
    ship_diff_weight: float = 0.0005    # per ship advantage delta (normalized by 100)
    # Per-step "alive" cost (encourages decisive play). Negative -> small penalty.
    step_cost: float = 0.0
    # If True, shaping signals only kick in for non-terminal steps.
    shape_on_terminal: bool = False


def _player_score(env_obs, player: int) -> tuple[int, int]:
    """Return (n_planets_owned, total_ships_owned_garrisoned_or_in_flight)."""
    planets = parse_planets(env_obs)
    fleets = parse_fleets(env_obs)
    n_planets = sum(1 for p in planets if p.owner == player and not p.is_comet)
    ships = sum(p.ships for p in planets if p.owner == player and not p.is_comet)
    ships += sum(f.ships for f in fleets if f.owner == player)
    return n_planets, ships


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

    # Dense shaping from before -> after.
    p_prev, s_prev = _player_score(prev_env_obs, player)
    # Opponent is the "other" player; with 2 agents this is 1 - player.
    opp = 1 - player
    op_prev, os_prev = _player_score(prev_env_obs, opp)

    p_curr, s_curr = _player_score(curr_env_obs, player)
    op_curr, os_curr = _player_score(curr_env_obs, opp)

    # Net planet swing (mine gained - opp gained), so capturing from opponent
    # counts double (intuitive: it's a 2-unit swing).
    d_planets = (p_curr - p_prev) - (op_curr - op_prev)
    # Ship differential delta, normalized.
    d_ship_diff = ((s_curr - os_curr) - (s_prev - os_prev)) / 100.0

    shaped = (cfg.planet_diff_weight * d_planets
              + cfg.ship_diff_weight * d_ship_diff
              + cfg.step_cost)

    # Suppress shaping on terminal step if requested (some prefer clean
    # terminal-only credit assignment).
    if terminal and not cfg.shape_on_terminal:
        shaped = 0.0

    terminal_term = cfg.terminal_scale * float(env_terminal_reward) if terminal else 0.0
    return float(shaped + terminal_term)
