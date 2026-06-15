"""Self-play in the **real** ``orbit_wars`` environment (AlphaZero data gen).

The abstract self-play in :mod:`macro_selfplay` keeps the search and the game in
one (abstract) representation, which is fast but suffers a representation gap:
the net never sees real launch dynamics, fleet travel, sun crossings, or the
true turn-cap economy.  This module closes that gap by playing the *actual*
``kaggle_environments`` game:

* Each turn, for every live player, a :class:`MacroAlphaZeroAgent` runs its
  macro MCTS (steered by the current net) and then executes the real launches
  through the scripted LP allocator -- i.e. exactly the agent we ship.
* The macro search performed inside ``act()`` is captured (root visit
  distribution + macro state) and turned into a :class:`MacroSample` via the
  shared :func:`macro_selfplay.build_macro_sample`, so the training tensors are
  identical to the abstract path.
* When the game ends (elimination or turn cap) every record is labelled with
  the **real** game outcome ``z`` from its player's perspective.

This is *much* more expensive than abstract self-play (a full real game with a
macro search every turn), but the value/policy targets are grounded in the game
we actually win or lose.
"""

from __future__ import annotations

import random
from typing import Optional

import torch

from orbit_macro_mcts_agent import MacroMCTSConfig
from orbit_scripted_agent import OrbitScriptedConfig
from orbit_match import _alive, _own_ships
from macro_alphazero_net import MacroAlphaZeroAgent, MacroAlphaZeroNet
from macro_selfplay import MacroSample, SelfPlayConfig, build_macro_sample


def _game_outcome(final_obs, num_agents: int) -> list[float]:
    """Per-player value target ``z`` in ``[-1, 1]`` from the final board.

    A clean elimination is a hard win/loss (``+1`` / ``-1``); otherwise we use a
    bounded ship-count margin (``(mine - best_other) / total``) so a turn-capped
    game still produces a graded, learnable signal.  For two players this is
    anti-symmetric (``z_1 = -z_0``).
    """
    survivors = [p for p in range(num_agents) if _alive(final_obs, p)]
    if len(survivors) == 1:
        win = survivors[0]
        return [1.0 if p == win else -1.0 for p in range(num_agents)]
    ships = [_own_ships(final_obs, p) for p in range(num_agents)]
    total = sum(ships)
    if total <= 0.0:
        return [0.0] * num_agents
    out: list[float] = []
    for p in range(num_agents):
        best_other = max((ships[q] for q in range(num_agents) if q != p), default=0.0)
        out.append(max(-1.0, min(1.0, (ships[p] - best_other) / total)))
    return out


def _make_agent(
    net: MacroAlphaZeroNet,
    macro_cfg: MacroMCTSConfig,
    cfg: Optional[OrbitScriptedConfig],
    device: Optional[torch.device],
) -> MacroAlphaZeroAgent:
    return MacroAlphaZeroAgent(
        net,
        cfg=cfg or OrbitScriptedConfig(attack_allocator="lp", reinforcement_allocator="lp"),
        macro_cfg=macro_cfg,
        device=device,
    )


def play_real_game(
    macro_cfg: MacroMCTSConfig,
    *,
    net: MacroAlphaZeroNet,
    selfplay_cfg: Optional[SelfPlayConfig] = None,
    max_steps: int = 300,
    seed: Optional[int] = None,
    cfg: Optional[OrbitScriptedConfig] = None,
    device: Optional[torch.device] = None,
) -> list[MacroSample]:
    """Play one real ``orbit_wars`` self-play game; return labelled samples.

    Both players are driven by the *same* net (standard AlphaZero self-play); a
    separate agent instance per player keeps each side's captured search isolated.
    ``selfplay_cfg.temperature`` shapes the recorded policy target; move
    execution itself is deterministic (the scripted allocator consumes the macro
    capture bonus).
    """
    from kaggle_environments import make
    import random as _random

    sp = selfplay_cfg or SelfPlayConfig()
    num_agents = int(sp.num_players)

    env = make("orbit_wars", debug=False)
    if seed is not None:
        env.configuration["randomSeed"] = int(seed)
        _random.seed(int(seed))
    env.reset(num_agents=num_agents)
    state = env.step([[] for _ in range(num_agents)])

    agents = [_make_agent(net, macro_cfg, cfg, device) for _ in range(num_agents)]
    samples: list[MacroSample] = []
    sample_players: list[int] = []

    cap = min(int(max_steps), int(sp.max_decisions))
    t = 0
    for t in range(cap):
        env_actions = []
        for p in range(num_agents):
            env_obs = state[p]["observation"]
            if not _alive(env_obs, p):
                env_actions.append([])
                continue
            ea = agents[p].act(env_obs, p, current_turn=t, max_turns=cap)
            mcts = agents[p].last_mcts
            result = agents[p].last_result
            if mcts is not None and result is not None and result.actions:
                policy = result.visit_distribution(temperature=sp.temperature)
                sample = build_macro_sample(mcts, result, policy, player=p)
                if sample is not None:
                    samples.append(sample)
                    sample_players.append(p)
            env_actions.append(ea or [])
        state = env.step(env_actions)
        done = env.done or any(
            getattr(state[p], "status", "ACTIVE") == "DONE"
            for p in range(num_agents)
        )
        if done:
            break

    final_obs = state[0]["observation"]
    outcome = _game_outcome(final_obs, num_agents)
    for s, p in zip(samples, sample_players):
        s.z = float(outcome[int(p)])
    return samples


def generate_selfplay_real(
    n_games: int,
    macro_cfg: MacroMCTSConfig,
    *,
    net: MacroAlphaZeroNet,
    selfplay_cfg: Optional[SelfPlayConfig] = None,
    max_steps: int = 300,
    seed: int = 0,
    cfg: Optional[OrbitScriptedConfig] = None,
    device: Optional[torch.device] = None,
    log_fn=None,
    log_every: int = 1,
) -> list[MacroSample]:
    """Generate samples from ``n_games`` real-env self-play games."""
    sp = selfplay_cfg or SelfPlayConfig()
    out: list[MacroSample] = []
    total = int(n_games)
    for gi in range(total):
        out.extend(play_real_game(
            macro_cfg,
            net=net,
            selfplay_cfg=sp,
            max_steps=int(max_steps),
            seed=int(seed) + gi,
            cfg=cfg,
            device=device,
        ))
        if log_fn is not None and log_every > 0 and (
            (gi + 1) % int(log_every) == 0 or (gi + 1) == total
        ):
            log_fn(gi + 1, total, len(out))
    return out


__all__ = [
    "play_real_game",
    "generate_selfplay_real",
]
