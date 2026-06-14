"""AlphaZero-style training loop for the macro MCTS net (abstract model).

Pipeline per iteration:

1. **Self-play** -- generate games with the current best net steering the macro
   MCTS (root Dirichlet noise + temperature for exploration).  Each decision is
   labelled with the root visit distribution (policy target) and, at game end,
   the root-relative abstract outcome (value target).
2. **Train** -- fit ``f(s) -> (prior, value)`` against
   ``policy = visit distribution`` (cross-entropy) and ``value = outcome`` (MSE),
   over a replay buffer of recent self-play.
3. **Eval-gate** -- pit the freshly trained net against the current best in fast
   abstract head-to-head games; promote it only if it clears ``gate_winrate``
   (AlphaZero's stability mechanism).

Value targets are kept in the raw :meth:`MacroTimedMCTS._score` scale so the net
value stays consistent with the heuristic rollout it is blended with at search
time; ``value_coef`` is therefore small to balance the (large) value MSE against
the policy cross-entropy.
"""

from __future__ import annotations

import argparse
import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from orbit_macro_mcts_agent import MacroMCTSConfig
from orbit_timed_mcts import TimedMCTSConfig
from macro_alphazero_net import MacroAlphaZeroNet, save_net
from macro_selfplay import (
    MacroSample,
    SelfPlayConfig,
    generate_selfplay,
    play_abstract_game,
    random_board,
)


# --------------------------------------------------------------------------- #
# Batching.
# --------------------------------------------------------------------------- #
def collate(samples: list[MacroSample], device: torch.device) -> dict:
    """Pad a list of samples into padded value + policy training tensors."""
    b = len(samples)
    p_max = max(s.planet_feats.shape[0] for s in samples)
    a_max = max(s.action_feats.shape[0] for s in samples)
    fp = samples[0].planet_feats.shape[1]
    fa = samples[0].action_feats.shape[1]
    g = samples[0].global_feats.shape[0]

    planet_feats = np.zeros((b, p_max, fp), dtype=np.float32)
    planet_mask = np.zeros((b, p_max), dtype=np.float32)
    global_feats = np.zeros((b, g), dtype=np.float32)
    action_feats = np.zeros((b, a_max, fa), dtype=np.float32)
    action_mask = np.zeros((b, a_max), dtype=np.float32)
    policy_target = np.zeros((b, a_max), dtype=np.float32)
    value_target = np.zeros((b,), dtype=np.float32)

    for i, s in enumerate(samples):
        p = s.planet_feats.shape[0]
        a = s.action_feats.shape[0]
        planet_feats[i, :p] = s.planet_feats
        planet_mask[i, :p] = 1.0
        global_feats[i] = s.global_feats
        action_feats[i, :a] = s.action_feats
        action_mask[i, :a] = 1.0
        policy_target[i, :a] = s.policy_target
        value_target[i] = s.z

    t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    return {
        "planet_feats": t(planet_feats),
        "planet_mask": t(planet_mask),
        "global_feats": t(global_feats),
        "action_feats": t(action_feats),
        "action_mask": t(action_mask),
        "policy_target": t(policy_target),
        "value_target": t(value_target),
    }


@dataclass
class TrainStats:
    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    n: int = 0


def train_epoch(
    net: MacroAlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    buffer: list[MacroSample],
    *,
    device: torch.device,
    batch_size: int,
    value_coef: float,
    grad_clip: float,
    rng: random.Random,
) -> TrainStats:
    net.train()
    order = list(range(len(buffer)))
    rng.shuffle(order)
    stats = TrainStats()
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        if not idx:
            continue
        batch = collate([buffer[i] for i in idx], device)

        value_pred = net.forward_value(
            batch["planet_feats"], batch["global_feats"], batch["planet_mask"])
        value_loss = F.mse_loss(value_pred, batch["value_target"])

        logits = net.forward_prior(batch["action_feats"], batch["action_mask"])
        log_probs = torch.log_softmax(logits, dim=1)
        log_probs = torch.nan_to_num(log_probs, neginf=0.0)
        policy_loss = -(batch["policy_target"] * log_probs).sum(dim=1).mean()

        loss = policy_loss + float(value_coef) * value_loss
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        stats.loss += float(loss.detach()) * len(idx)
        stats.policy_loss += float(policy_loss.detach()) * len(idx)
        stats.value_loss += float(value_loss.detach()) * len(idx)
        stats.n += len(idx)
    if stats.n > 0:
        stats.loss /= stats.n
        stats.policy_loss /= stats.n
        stats.value_loss /= stats.n
    return stats


# --------------------------------------------------------------------------- #
# Eval-gate: abstract head-to-head between two nets.
# --------------------------------------------------------------------------- #
def _play_match(
    board: dict,
    macro_cfg: MacroMCTSConfig,
    net_p0: Optional[MacroAlphaZeroNet],
    net_p1: Optional[MacroAlphaZeroNet],
    *,
    rng: random.Random,
    max_decisions: int,
    device,
) -> float:
    """Return the player-0-relative final abstract score for a two-net game.

    Player 0 is steered by ``net_p0``, player 1 by ``net_p1`` (``None`` = no
    net, i.e. heuristic rollout + uniform priors).  The shared abstract game is
    advanced by whichever net owns the side to move.
    """
    from orbit_graph import build_orbit_graph
    from orbit_macro_mcts_agent import MacroActionType, MacroTimedMCTS
    from macro_alphazero_net import attach_net
    from macro_selfplay import _final_outcome, _is_terminal

    og = build_orbit_graph(board, player=0)
    base = MacroTimedMCTS(og, macro_cfg, root_player=0, current_turn=0, rng=rng)
    live = base._initial_state
    sp = SelfPlayConfig(max_decisions=max_decisions)
    abs_turn = 0
    for _ in range(max_decisions):
        cur = int(live.player_to_move)
        mcts = MacroTimedMCTS(
            og, macro_cfg, root_player=cur, current_turn=int(abs_turn),
            max_turns=int(macro_cfg.timed.horizon_turns) + int(abs_turn) + 1,
            init_state=live, rng=rng,
        )
        net = net_p0 if cur == 0 else net_p1
        if net is not None:
            attach_net(net, mcts, device=device)
        result = mcts.run()
        action = result.best_action()
        if action is None:
            action = next(
                (a for a in mcts._macro_actions(mcts._initial_state)),
                None,
            )
        if action is None:
            break
        live = mcts._apply_macro(mcts._initial_state, action)
        abs_turn += int(live.turn)
        if _is_terminal(live, sp):
            break
    return _final_outcome(og, macro_cfg, live, abs_turn)


def eval_gate(
    candidate: MacroAlphaZeroNet,
    best: Optional[MacroAlphaZeroNet],
    macro_cfg: MacroMCTSConfig,
    *,
    n_games: int,
    seed: int,
    max_decisions: int,
    n_planets: int,
    device,
) -> float:
    """Win rate of ``candidate`` vs ``best`` over colour-balanced abstract games."""
    rng = random.Random(seed)
    wins = 0.0
    games = 0
    for g in range(int(n_games)):
        board = random_board(rng, n_planets=n_planets, num_players=2)
        # Candidate as player 0.
        s0 = _play_match(board, macro_cfg, candidate, best, rng=rng,
                         max_decisions=max_decisions, device=device)
        wins += 1.0 if s0 > 0 else (0.5 if abs(s0) < 1e-9 else 0.0)
        games += 1
        # Candidate as player 1 (swap colours; player-0 score negated).
        s1 = _play_match(board, macro_cfg, best, candidate, rng=rng,
                         max_decisions=max_decisions, device=device)
        wins += 1.0 if (-s1) > 0 else (0.5 if abs(s1) < 1e-9 else 0.0)
        games += 1
    return wins / max(1, games)


# --------------------------------------------------------------------------- #
# Main training loop.
# --------------------------------------------------------------------------- #
def train(
    *,
    iterations: int = 10,
    games_per_iter: int = 16,
    simulations: int = 48,
    horizon_turns: int = 12,
    n_planets: int = 7,
    d_model: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    value_coef: float = 5e-3,
    grad_clip: float = 5.0,
    batch_size: int = 128,
    epochs_per_iter: int = 4,
    buffer_games: int = 64,
    gate_winrate: float = 0.55,
    gate_games: int = 16,
    value_blend: float = 0.5,
    dirichlet_alpha: float = 0.5,
    seed: int = 0,
    device: str = "cpu",
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
) -> MacroAlphaZeroNet:
    dev = torch.device(device)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    macro_cfg = MacroMCTSConfig(
        timed=TimedMCTSConfig(
            simulations=simulations,
            horizon_turns=horizon_turns,
            source_reserve_frac=0.0,
            gamma=1.0,
        ),
        value_blend=value_blend,
        root_dirichlet_alpha=dirichlet_alpha,
    )
    selfplay_cfg = SelfPlayConfig(
        max_decisions=max(20, 4 * horizon_turns),
        temperature=1.0,
        temperature_moves=max(4, horizon_turns // 2),
    )

    net = MacroAlphaZeroNet(d_model=d_model).to(dev)
    best = copy.deepcopy(net).eval()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    # Replay buffer of recent self-play samples (approx by game count).
    approx_samples_per_game = max(1, selfplay_cfg.max_decisions)
    buffer: deque[MacroSample] = deque(maxlen=buffer_games * approx_samples_per_game)

    for it in range(int(iterations)):
        samples = generate_selfplay(
            games_per_iter, macro_cfg, selfplay_cfg=selfplay_cfg,
            net=best, seed=seed + 1000 + it, n_planets=n_planets, device=dev,
        )
        buffer.extend(samples)

        last = TrainStats()
        for _ in range(int(epochs_per_iter)):
            last = train_epoch(
                net, optimizer, list(buffer), device=dev,
                batch_size=batch_size, value_coef=value_coef,
                grad_clip=grad_clip, rng=rng,
            )

        winrate = eval_gate(
            net, best, macro_cfg, n_games=gate_games, seed=seed + 7000 + it,
            max_decisions=selfplay_cfg.max_decisions, n_planets=n_planets, device=dev,
        )
        promoted = winrate >= gate_winrate
        if promoted:
            best = copy.deepcopy(net).eval()

        if verbose:
            print(
                f"iter {it:03d} | samples={len(samples)} buf={len(buffer)} "
                f"loss={last.loss:.3f} (pol={last.policy_loss:.3f} "
                f"val={last.value_loss:.1f}) winrate={winrate:.2f} "
                f"{'PROMOTED' if promoted else 'kept'}"
            )
        if checkpoint_dir and promoted:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_net(best, os.path.join(checkpoint_dir, "best.pt"),
                     meta={"iter": it, "winrate": winrate})

    if checkpoint_dir:
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_net(best, os.path.join(checkpoint_dir, "best.pt"))
    return best


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaZero-style macro MCTS training")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--games-per-iter", type=int, default=16)
    p.add_argument("--simulations", type=int, default=48)
    p.add_argument("--horizon-turns", type=int, default=12)
    p.add_argument("--n-planets", type=int, default=7)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-coef", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs-per-iter", type=int, default=4)
    p.add_argument("--buffer-games", type=int, default=64)
    p.add_argument("--gate-winrate", type=float, default=0.55)
    p.add_argument("--gate-games", type=int, default=16)
    p.add_argument("--value-blend", type=float, default=0.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_macro_alphazero")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    train(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        simulations=args.simulations,
        horizon_turns=args.horizon_turns,
        n_planets=args.n_planets,
        d_model=args.d_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        value_coef=args.value_coef,
        batch_size=args.batch_size,
        epochs_per_iter=args.epochs_per_iter,
        buffer_games=args.buffer_games,
        gate_winrate=args.gate_winrate,
        gate_games=args.gate_games,
        value_blend=args.value_blend,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()


__all__ = ["collate", "train_epoch", "eval_gate", "train", "main"]
