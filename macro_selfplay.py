"""Self-play in the abstract macro model (AlphaZero data generation).

The macro MCTS searches an *abstract* turn-horizon model, so the cleanest and
most self-consistent AlphaZero formulation is to **play that same abstract
model**: at every decision we run a search for the player to move, record the
root visit distribution (policy target) and the abstract board features, play a
move, and advance the abstract game.  When the game ends we label every record
with the final root-relative outcome ``z`` (value target).

This keeps the policy/value targets and the search dynamics in one representation
(no root/leaf gap).  Real-env strength is recovered separately at evaluation
time by gating new nets in actual ``orbit_wars`` matches (see
``train_macro_alphazero.py``).

A ``MacroSample`` carries everything the trainer needs:

* ``planet_feats`` / ``global_feats`` -- value-head inputs for the root state.
* ``action_feats`` / ``policy_target`` -- prior-head inputs + visit-count target.
* ``player`` / ``z`` -- which side moved and the eventual outcome (its sign).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from orbit_graph import build_orbit_graph
from orbit_macro_mcts_agent import (
    MacroAction,
    MacroActionType,
    MacroMCTSConfig,
    MacroSearchResult,
    MacroTimedMCTS,
)
from orbit_timed_mcts import _TimedState
from macro_alphazero_net import (
    MacroAlphaZeroNet,
    abstract_state_arrays,
    action_feature_vector,
    attach_net,
)


@dataclass
class MacroSample:
    """One AlphaZero training record from a self-play decision."""

    planet_feats: np.ndarray          # (P, Fp)
    global_feats: np.ndarray          # (G,)
    action_feats: np.ndarray          # (A, Fa)
    policy_target: np.ndarray         # (A,) visit distribution
    player: int
    z: float = 0.0


@dataclass
class SelfPlayConfig:
    max_decisions: int = 60
    """Hard cap on macro decisions per self-play game."""
    temperature: float = 1.0
    """Visit-count temperature for the *policy target* and move sampling."""
    temperature_moves: int = 12
    """Sample moves with temperature for this many plies, then play greedily."""
    num_players: int = 2
    resign_owned_frac: float = 1.0
    """End the game when one player owns this fraction of non-neutral planets."""


# --------------------------------------------------------------------------- #
# Random board generation (variety for self-play).
# --------------------------------------------------------------------------- #
def random_board(
    rng: random.Random,
    *,
    n_planets: int = 7,
    num_players: int = 2,
    span: float = 100.0,
) -> dict:
    """A simple random Orbit Wars board: one home each, the rest neutral."""
    planets = []
    for pid in range(n_planets):
        x = rng.uniform(10.0, span - 10.0)
        y = rng.uniform(10.0, span - 10.0)
        prod = rng.randint(1, 9)
        if pid < num_players:
            owner = pid
            ships = rng.randint(40, 80)
        else:
            owner = -1
            ships = rng.randint(1, 12)
        planets.append([pid, owner, x, y, 1.0, ships, prod])
    return {
        "planets": planets,
        "fleets": [],
        "initial_planets": planets,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "comets": [],
    }


# --------------------------------------------------------------------------- #
# Game-end helpers.
# --------------------------------------------------------------------------- #
def _non_neutral_owner_counts(state: _TimedState) -> dict[int, int]:
    counts: dict[int, int] = {}
    for owner in state.owner_map().values():
        if int(owner) >= 0:
            counts[int(owner)] = counts.get(int(owner), 0) + 1
    return counts


def _is_terminal(state: _TimedState, cfg: SelfPlayConfig) -> bool:
    counts = _non_neutral_owner_counts(state)
    if len(counts) <= 1:
        return True
    total = sum(counts.values())
    if total <= 0:
        return True
    top = max(counts.values())
    return (top / total) >= float(cfg.resign_owned_frac)


def _sample_action(
    result: MacroSearchResult,
    policy: dict[tuple[str, int], float],
    rng: random.Random,
    *,
    greedy: bool,
) -> MacroAction:
    if greedy or not policy:
        best = result.best_action()
        return best if best is not None else MacroAction(type=MacroActionType.PASS)
    keys = list(policy.keys())
    weights = [max(0.0, policy[k]) for k in keys]
    total = sum(weights)
    if total <= 0.0:
        best = result.best_action()
        return best if best is not None else MacroAction(type=MacroActionType.PASS)
    pick = rng.choices(keys, weights=weights, k=1)[0]
    return result.actions[pick]


def _final_outcome(og, macro_cfg: MacroMCTSConfig, live: _TimedState, abs_turn: int) -> float:
    """Root(player 0)-relative abstract score at the end of self-play."""
    closer = MacroTimedMCTS(
        og, macro_cfg, root_player=0, current_turn=int(abs_turn),
        max_turns=int(macro_cfg.timed.horizon_turns) + int(abs_turn) + 1,
        init_state=live,
    )
    return float(closer._score(closer._advance_to_horizon(closer._initial_state)))


# --------------------------------------------------------------------------- #
# Self-play.
# --------------------------------------------------------------------------- #
def play_abstract_game(
    env_board: dict,
    macro_cfg: MacroMCTSConfig,
    *,
    selfplay_cfg: Optional[SelfPlayConfig] = None,
    net: Optional[MacroAlphaZeroNet] = None,
    rng: Optional[random.Random] = None,
    device=None,
) -> list[MacroSample]:
    """Play one abstract self-play game; return labelled training samples."""
    sp = selfplay_cfg or SelfPlayConfig()
    rng = rng or random.Random()

    og = build_orbit_graph(env_board, player=0)
    base = MacroTimedMCTS(og, macro_cfg, root_player=0, current_turn=0, rng=rng)
    live = base._initial_state

    samples: list[MacroSample] = []
    abs_turn = 0
    for ply in range(int(sp.max_decisions)):
        cur = int(live.player_to_move)
        mcts = MacroTimedMCTS(
            og, macro_cfg, root_player=cur, current_turn=int(abs_turn),
            max_turns=int(macro_cfg.timed.horizon_turns) + int(abs_turn) + 1,
            init_state=live, rng=rng,
        )
        if net is not None:
            attach_net(net, mcts, device=device)

        result = mcts.run()
        if not result.actions:
            break

        policy = result.visit_distribution(temperature=sp.temperature)
        if any(result.visits.values()):
            root_state = mcts._initial_state
            pf, gf = abstract_state_arrays(mcts, root_state)
            keys = list(result.actions.keys())
            action_feats = np.stack([
                action_feature_vector(mcts._macro_features(root_state, result.actions[k]))
                for k in keys
            ])
            pi = np.array([float(policy.get(k, 0.0)) for k in keys], dtype=np.float32)
            samples.append(MacroSample(
                planet_feats=pf, global_feats=gf,
                action_feats=action_feats, policy_target=pi, player=cur,
            ))

        greedy = ply >= int(sp.temperature_moves)
        action = _sample_action(result, policy, rng, greedy=greedy)
        live = mcts._apply_macro(mcts._initial_state, action)
        abs_turn += int(live.turn)
        if _is_terminal(live, sp):
            break

    z0 = _final_outcome(og, macro_cfg, live, abs_turn)
    for s in samples:
        s.z = float(z0 if s.player == 0 else -z0)
    return samples


def generate_selfplay(
    n_games: int,
    macro_cfg: MacroMCTSConfig,
    *,
    selfplay_cfg: Optional[SelfPlayConfig] = None,
    net: Optional[MacroAlphaZeroNet] = None,
    seed: int = 0,
    n_planets: int = 7,
    device=None,
) -> list[MacroSample]:
    """Generate samples from ``n_games`` random-board self-play games."""
    sp = selfplay_cfg or SelfPlayConfig()
    rng = random.Random(seed)
    out: list[MacroSample] = []
    for _ in range(int(n_games)):
        board = random_board(rng, n_planets=n_planets, num_players=sp.num_players)
        out.extend(play_abstract_game(
            board, macro_cfg, selfplay_cfg=sp, net=net, rng=rng, device=device,
        ))
    return out


__all__ = [
    "MacroSample",
    "SelfPlayConfig",
    "random_board",
    "play_abstract_game",
    "generate_selfplay",
]
