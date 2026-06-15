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
import logging
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from orbit_macro_mcts_agent import MacroMCTSConfig
from orbit_timed_mcts import TimedMCTSConfig
from macro_alphazero_net import MacroAlphaZeroNet, build_net, load_net, save_net
from macro_selfplay import (
    MacroSample,
    SelfPlayConfig,
    generate_selfplay,
    play_abstract_game,
    random_board,
    sample_env_board,
)
from macro_selfplay_real import generate_selfplay_real


# --------------------------------------------------------------------------- #
# Logging (always file + stdout, flushed each line).
# --------------------------------------------------------------------------- #
class _FlushStreamHandler(logging.StreamHandler):
    """Stdout handler that flushes after every record (nohup / pipe safe)."""

    def emit(self, record) -> None:
        super().emit(record)
        self.flush()


def setup_training_logger(log_file: Path, *, append: bool = False) -> logging.Logger:
    """Configure the training logger: mirror every message to file and stdout."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # ``orbit_match`` (pulled in for real-env self-play / gate) calls
    # ``logging.disable(INFO)`` at import to mute kaggle_environments' import
    # spam; that global override would also swallow our INFO training logs, so
    # clear it here.  The kaggle import noise already happened before this runs.
    logging.disable(logging.NOTSET)

    logger = logging.getLogger("macro_alphazero.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, mode="a" if append else "w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = _FlushStreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _default_log_file(checkpoint_dir: Optional[str]) -> Path:
    if checkpoint_dir:
        return Path(checkpoint_dir) / "train.log"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / "macro_alphazero" / f"train_{stamp}.log"


# --------------------------------------------------------------------------- #
# Self-play length schedule (macro decisions per game).
# --------------------------------------------------------------------------- #
def default_min_decisions(horizon_turns: int) -> int:
    """Starting cap before the ramp (matches the old fixed formula)."""
    return max(20, 4 * int(horizon_turns))


def scheduled_max_decisions(
    iter_idx: int,
    *,
    min_decisions: int,
    max_decisions: int,
    ramp_iters: int,
) -> int:
    """Linear ramp from ``min_decisions`` at iter 0 to ``max_decisions`` at iter ``ramp_iters - 1``.

    Uses the absolute training-iteration index so resuming at iter 30 keeps the
    schedule at the 30th step of the ramp.  After the ramp completes the cap
    stays at ``max_decisions`` (hard ceiling).
    """
    lo = int(min_decisions)
    hi = int(max_decisions)
    if hi <= lo:
        return lo
    span = max(1, int(ramp_iters) - 1)
    t = min(1.0, max(0.0, float(iter_idx) / float(span)))
    return int(round(lo + t * (hi - lo)))


# --------------------------------------------------------------------------- #
# Batching.
# --------------------------------------------------------------------------- #
def collate(samples: list[MacroSample], device: torch.device) -> dict:
    """Pad a list of samples into padded value + policy training tensors."""
    b = len(samples)
    p_max = max(s.planet_feats.shape[0] for s in samples)
    a_max = max(s.action_feats.shape[0] for s in samples)
    fp = samples[0].planet_feats.shape[1]
    fe = samples[0].edge_feats.shape[2]
    fa = samples[0].action_feats.shape[1]
    g = samples[0].global_feats.shape[0]

    planet_feats = np.zeros((b, p_max, fp), dtype=np.float32)
    edge_feats = np.zeros((b, p_max, p_max, fe), dtype=np.float32)
    planet_mask = np.zeros((b, p_max), dtype=np.float32)
    global_feats = np.zeros((b, g), dtype=np.float32)
    action_feats = np.zeros((b, a_max, fa), dtype=np.float32)
    action_mask = np.zeros((b, a_max), dtype=np.float32)
    action_target_idx = np.full((b, a_max), -1, dtype=np.int64)
    policy_target = np.zeros((b, a_max), dtype=np.float32)
    value_target = np.zeros((b,), dtype=np.float32)

    for i, s in enumerate(samples):
        p = s.planet_feats.shape[0]
        a = s.action_feats.shape[0]
        planet_feats[i, :p] = s.planet_feats
        edge_feats[i, :p, :p] = s.edge_feats
        planet_mask[i, :p] = 1.0
        global_feats[i] = s.global_feats
        action_feats[i, :a] = s.action_feats
        action_mask[i, :a] = 1.0
        action_target_idx[i, :a] = s.action_target_idx
        policy_target[i, :a] = s.policy_target
        value_target[i] = s.z

    t = lambda x: torch.from_numpy(x).to(device)  # noqa: E731
    return {
        "planet_feats": t(planet_feats),
        "edge_feats": t(edge_feats),
        "planet_mask": t(planet_mask),
        "global_feats": t(global_feats),
        "action_feats": t(action_feats),
        "action_mask": t(action_mask),
        "action_target_idx": t(action_target_idx),
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

        if getattr(net, "arch", "gnn") == "mlp":
            value_pred = net.forward_value(
                batch["planet_feats"], batch["global_feats"], batch["planet_mask"])
            logits = net.forward_prior(batch["action_feats"], batch["action_mask"])
        else:
            enc = net.encode(
                batch["planet_feats"], batch["edge_feats"],
                batch["global_feats"], batch["planet_mask"])
            value_pred = net.value_from_encoding(enc)
            logits = net.prior_from_encoding(
                enc, batch["action_feats"], batch["action_target_idx"],
                batch["action_mask"])
        value_loss = F.mse_loss(value_pred, batch["value_target"])

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
    device,
) -> float:
    """Win rate of ``candidate`` vs ``best`` over colour-balanced abstract games."""
    rng = random.Random(seed)
    wins = 0.0
    games = 0
    for g in range(int(n_games)):
        board = sample_env_board(seed=int(seed) + g, num_players=2)
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


def real_eval_gate(
    candidate: MacroAlphaZeroNet,
    best: Optional[MacroAlphaZeroNet],
    macro_cfg: MacroMCTSConfig,
    *,
    n_games: int,
    seed: int,
    max_steps: int,
    device,
) -> float:
    """Win rate of ``candidate`` vs ``best`` over colour-balanced **real** games.

    Each pairing is played twice (candidate as P0 then P1) in the live
    ``orbit_wars`` env via :func:`orbit_match.play_match`; the winner is the env
    reward leader.  If ``best`` is ``None`` the opponent is the plain scripted LP
    agent (a sane baseline for the very first iteration).
    """
    from orbit_match import play_match
    from macro_alphazero_net import MacroAlphaZeroAgent
    from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig

    def _cand() -> MacroAlphaZeroAgent:
        return MacroAlphaZeroAgent(candidate, macro_cfg=macro_cfg, device=device)

    def _opp():
        if best is not None:
            return MacroAlphaZeroAgent(best, macro_cfg=macro_cfg, device=device)
        return OrbitScriptedAgent(
            cfg=OrbitScriptedConfig(attack_allocator="lp", reinforcement_allocator="lp"))

    wins = 0.0
    games = 0
    for g in range(int(n_games)):
        res0 = play_match([_cand(), _opp()], num_agents=2,
                          max_steps=int(max_steps), seed=int(seed) + g)
        wins += 1.0 if res0.winner == 0 else (0.5 if res0.winner < 0 else 0.0)
        games += 1
        res1 = play_match([_opp(), _cand()], num_agents=2,
                          max_steps=int(max_steps), seed=int(seed) + g)
        wins += 1.0 if res1.winner == 1 else (0.5 if res1.winner < 0 else 0.0)
        games += 1
    return wins / max(1, games)


# --------------------------------------------------------------------------- #
# Main training loop.
# --------------------------------------------------------------------------- #
def train(
    *,
    iterations: int = 10,
    arch: str = "gnn",
    env: str = "abstract",
    real_max_steps: int = 300,
    games_per_iter: int = 16,
    simulations: int = 48,
    horizon_turns: int = 12,
    min_decisions: Optional[int] = None,
    max_decisions: int = 500,
    max_decisions_ramp_iters: int = 200,
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
    log_file: Optional[str] = None,
    log_every: int = 8,
    resume_from: Optional[str] = None,
    start_iter: Optional[int] = None,
) -> MacroAlphaZeroNet:
    dev = torch.device(device)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    log_path = Path(log_file) if log_file else _default_log_file(checkpoint_dir)
    resuming = resume_from is not None
    log = setup_training_logger(log_path, append=resuming)

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
    min_dec = int(min_decisions) if min_decisions is not None else default_min_decisions(horizon_turns)
    max_dec = max(int(max_decisions), min_dec)
    ramp_iters = max(1, int(max_decisions_ramp_iters))
    selfplay_cfg = SelfPlayConfig(
        max_decisions=min_dec,
        temperature=1.0,
        temperature_moves=max(4, horizon_turns // 2),
    )

    if resuming:
        ckpt_path = Path(resume_from)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        raw = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
        d_model = int(raw.get("d_model", d_model))
        best = load_net(str(ckpt_path), device=dev)
        net = copy.deepcopy(best).train()
        meta = raw.get("meta") or {}
        if start_iter is None:
            start_iter = int(meta.get("iter", -1)) + 1
        log.info(
            "resumed from %s (checkpoint iter=%s) -> start_iter=%d",
            ckpt_path.resolve(), meta.get("iter", "?"), start_iter,
        )
    else:
        start_iter = 0
        net = build_net(arch, d_model=d_model).to(dev)
        best = copy.deepcopy(net).eval()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    env = str(env).lower()
    if env not in ("abstract", "real"):
        raise ValueError(f"--env must be 'abstract' or 'real', got {env!r}")

    # Replay buffer sized for the longest games we will eventually play. Real-env
    # games record up to ~2 samples per turn (both colours), capped by real_max_steps.
    if env == "real":
        approx_samples_per_game = max(1, 2 * min(max_dec, int(real_max_steps)))
    else:
        approx_samples_per_game = max(1, max_dec)
    buffer: deque[MacroSample] = deque(maxlen=buffer_games * approx_samples_per_game)

    n_params = sum(p.numel() for p in net.parameters())
    log.info("=" * 72)
    log.info("macro AlphaZero training %s", "resumed" if resuming else "started")
    log.info("log file: %s", log_path.resolve())
    log.info(
        "arch=%s env=%s start_iter=%d +%d iters games/iter=%d sims=%d horizon=%d "
        "boards=orbit_wars d_model=%d buffer_games=%d value_coef=%.4g device=%s params=%d",
        getattr(net, "arch", "gnn"), env, start_iter, iterations, games_per_iter, simulations,
        horizon_turns, d_model, buffer_games, value_coef, device, n_params,
    )
    if env == "real":
        log.info("real-env self-play: real_max_steps=%d (value target = real game outcome in [-1,1])", real_max_steps)
    log.info(
        "max_decisions ramp: %d -> %d over %d iters (iter 0 .. %d)",
        min_dec, max_dec, ramp_iters, ramp_iters - 1,
    )
    log.info("=" * 72)

    end_iter = start_iter + int(iterations)
    for it in range(start_iter, end_iter):
        t_iter = time.monotonic()
        iter_max_dec = scheduled_max_decisions(
            it, min_decisions=min_dec, max_decisions=max_dec, ramp_iters=ramp_iters,
        )
        iter_sp_cfg = replace(selfplay_cfg, max_decisions=iter_max_dec)
        iter_real_steps = min(int(real_max_steps), iter_max_dec) if env == "real" else iter_max_dec
        log.info(
            "iter %03d | self-play starting (%d games, %s cap=%d)...",
            it, games_per_iter,
            "real-env steps" if env == "real" else "max_decisions",
            iter_real_steps if env == "real" else iter_max_dec,
        )
        t_sp = time.monotonic()

        def _sp_progress(done: int, total: int, n_samples: int) -> None:
            log.info(
                "iter %03d | self-play %d/%d games | samples=%d",
                it, done, total, n_samples,
            )

        if env == "real":
            samples = generate_selfplay_real(
                games_per_iter, macro_cfg, net=best, selfplay_cfg=iter_sp_cfg,
                max_steps=iter_real_steps, seed=seed + 1000 + it, device=dev,
                log_fn=_sp_progress, log_every=max(1, int(log_every)),
            )
        else:
            samples = generate_selfplay(
                games_per_iter, macro_cfg, selfplay_cfg=iter_sp_cfg,
                net=best, seed=seed + 1000 + it, device=dev,
                log_fn=_sp_progress, log_every=int(log_every),
            )
        buffer.extend(samples)
        log.info(
            "iter %03d | self-play done in %.1fs | +%d samples buf=%d",
            it, time.monotonic() - t_sp, len(samples), len(buffer),
        )

        t_tr = time.monotonic()
        last = TrainStats()
        for ep in range(int(epochs_per_iter)):
            last = train_epoch(
                net, optimizer, list(buffer), device=dev,
                batch_size=batch_size, value_coef=value_coef,
                grad_clip=grad_clip, rng=rng,
            )
        log.info(
            "iter %03d | train %d epochs in %.1fs | loss=%.3f pol=%.3f val=%.1f",
            it, epochs_per_iter, time.monotonic() - t_tr,
            last.loss, last.policy_loss, last.value_loss,
        )

        t_ev = time.monotonic()
        log.info("iter %03d | eval-gate starting (%d games, %s)...",
                 it, gate_games, "real env" if env == "real" else "abstract")
        if env == "real":
            winrate = real_eval_gate(
                net, best, macro_cfg, n_games=gate_games, seed=seed + 7000 + it,
                max_steps=iter_real_steps, device=dev,
            )
        else:
            winrate = eval_gate(
                net, best, macro_cfg, n_games=gate_games, seed=seed + 7000 + it,
                max_decisions=iter_max_dec, device=dev,
            )
        promoted = winrate >= gate_winrate
        if promoted:
            best = copy.deepcopy(net).eval()

        log.info(
            "iter %03d | eval-gate %.1fs winrate=%.2f %s | total %.1fs",
            it, time.monotonic() - t_ev, winrate,
            "PROMOTED" if promoted else "kept",
            time.monotonic() - t_iter,
        )
        if checkpoint_dir and promoted:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt = os.path.join(checkpoint_dir, "best.pt")
            save_net(best, ckpt, meta={"iter": it, "winrate": winrate})
            log.info("iter %03d | saved checkpoint %s", it, ckpt)

    if checkpoint_dir:
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_ckpt = os.path.join(checkpoint_dir, "best.pt")
        save_net(best, final_ckpt, meta={"iter": end_iter - 1})
        log.info("training finished | final checkpoint %s", final_ckpt)
    else:
        log.info("training finished")
    return best


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AlphaZero-style macro MCTS training")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--games-per-iter", type=int, default=16)
    p.add_argument("--simulations", type=int, default=48)
    p.add_argument("--horizon-turns", type=int, default=12)
    p.add_argument(
        "--min-decisions", type=int, default=None,
        help="Self-play macro-step floor at iter 0 (default: max(20, 4*horizon-turns))",
    )
    p.add_argument(
        "--max-decisions", type=int, default=500,
        help="Hard cap on macro decisions per self-play game (ramp target)",
    )
    p.add_argument(
        "--max-decisions-ramp-iters", type=int, default=200,
        help="Linear ramp length: min at iter 0, max at iter ramp-iters-1",
    )
    p.add_argument(
        "--arch", type=str, default="gnn", choices=["gnn", "mlp"],
        help="Network architecture for fresh training (ignored when --resume-from)",
    )
    p.add_argument(
        "--env", type=str, default="abstract", choices=["abstract", "real"],
        help="Self-play / eval-gate environment: 'abstract' macro model (fast) "
             "or 'real' orbit_wars env (grounded, much slower)",
    )
    p.add_argument(
        "--real-max-steps", type=int, default=300,
        help="Max real env turns per game when --env real (also capped by the "
             "max-decisions ramp)",
    )
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--value-coef", type=float, default=None,
        help="Value-loss weight (default: 5e-3 for abstract big-scale targets, "
             "1.0 for real-env [-1,1] outcomes)",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs-per-iter", type=int, default=4)
    p.add_argument("--buffer-games", type=int, default=64)
    p.add_argument("--gate-winrate", type=float, default=0.55)
    p.add_argument("--gate-games", type=int, default=16)
    p.add_argument("--value-blend", type=float, default=0.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_macro_alphazero_gnn")
    p.add_argument(
        "--log-file", type=str, default=None,
        help="Training log path (default: <checkpoint-dir>/train.log)",
    )
    p.add_argument(
        "--log-every", type=int, default=8,
        help="Self-play progress log interval in games (0 = only start/end)",
    )
    p.add_argument(
        "--resume-from", type=str, default=None,
        help="Checkpoint to resume from (e.g. checkpoints_macro_alphazero/best.pt)",
    )
    p.add_argument(
        "--start-iter", type=int, default=None,
        help="First iter index when resuming (default: checkpoint meta iter + 1)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    value_coef = args.value_coef
    if value_coef is None:
        value_coef = 1.0 if args.env == "real" else 5e-3
    train(
        iterations=args.iterations,
        arch=args.arch,
        env=args.env,
        real_max_steps=args.real_max_steps,
        games_per_iter=args.games_per_iter,
        simulations=args.simulations,
        horizon_turns=args.horizon_turns,
        min_decisions=args.min_decisions,
        max_decisions=args.max_decisions,
        max_decisions_ramp_iters=args.max_decisions_ramp_iters,
        d_model=args.d_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        value_coef=value_coef,
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
        log_file=args.log_file,
        log_every=args.log_every,
        resume_from=args.resume_from,
        start_iter=args.start_iter,
    )


if __name__ == "__main__":
    main()


__all__ = ["collate", "train_epoch", "eval_gate", "real_eval_gate", "train", "main"]
