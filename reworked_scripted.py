"""Deterministic scripted opponents for the reworked stack.

These bots are *not* neural — they emit a fixed, rule-based hierarchical
"action_row" (the same dict the RL policy produces) and then route ships through
the **same** :func:`reworked_allocator.allocate_launches` logistics layer via
:func:`reworked_rollout.action_to_env_launches`.  So a scripted opponent plays
with identical physics / allocation to the agent; only the *strategic choice*
(which targets, what priority, how much to commit) is hand-coded.

Production-maximising bot
-------------------------
``production_action_row`` implements the requested behaviour: **always go for
the most valuable (highest-production) planet to maximise production growth.**
It selects every non-owned planet as an attack candidate and every owned planet
as a defend candidate, then ranks them by production (the value at stake), so
the allocator funds the highest-production affordable target first and drops the
unaffordable ones (``attack_require_full``).  The net effect is a greedy
production-grabber that also shields its own high-production worlds.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

try:  # package layout
    from .reworked_pipeline import board_state_from_obs, BoardState
except ImportError:  # pragma: no cover - flat layout / direct execution
    from reworked_pipeline import board_state_from_obs, BoardState  # type: ignore


def production_action_row(
    board: BoardState,
    player: int,
    *,
    p_max: int,
    commit: float = 0.85,
    potential_weight: float = 0.0,
    horizon: float = 15.0,
) -> dict:
    """Greedy production-maximising ``action_row`` (see module docstring).

    * attack = all non-owned planets (neutral expansion + enemy conquest),
    * defend = all owned planets,
    * priority = production descending (highest value at stake funded first),
    * commit = ``commit`` fraction on each owned source.

    The allocator then funds the highest-production *affordable* targets first
    and drops the rest, so the bot consistently captures the best planet it can
    pay for this turn.
    """
    n = board.n
    owner = np.asarray(board.owner).reshape(n)
    prod = np.asarray(board.production, dtype=np.float64).reshape(n)

    attack = np.zeros(p_max, dtype=np.float32)
    defend = np.zeros(p_max, dtype=np.float32)
    commit_frac = np.zeros(p_max, dtype=np.float32)

    is_mine = owner == int(player)
    not_mine = ~is_mine
    attack[:n][not_mine] = 1.0
    defend[:n][is_mine] = 1.0
    commit_frac[:n][is_mine] = float(commit)

    # Priority: all real planets by production descending (value at stake),
    # padded with the remaining slots so the order is a full permutation.
    real_order = np.argsort(-prod, kind="stable")
    order = np.concatenate([real_order, np.arange(n, p_max)]).astype(np.int64)[:p_max]

    return {
        "attack_sel": attack,
        "defend_sel": defend,
        "order": order,
        "commit_frac": commit_frac,
        "potential_weight": float(potential_weight),
        "horizon": float(horizon),
    }


# Registry: name -> (board, player, *, p_max) -> action_row
SCRIPTED_STRATEGIES: dict[str, Callable[..., dict]] = {
    "production": production_action_row,
}


def scripted_action_row(strategy: str, board: BoardState, player: int, *, p_max: int) -> dict:
    """Dispatch to a named scripted strategy."""
    try:
        fn = SCRIPTED_STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(
            f"unknown scripted strategy {strategy!r}; "
            f"available: {sorted(SCRIPTED_STRATEGIES)}"
        ) from exc
    return fn(board, player, p_max=p_max)


__all__ = [
    "production_action_row",
    "scripted_action_row",
    "SCRIPTED_STRATEGIES",
    "board_state_from_obs",
]
