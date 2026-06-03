"""Action-sensitivity diagnostic for the reworked hierarchical policy.

Question we are testing (hypothesis E): *do the high-level policy's choices
actually change the return?*  If they don't, the actor has no advantage signal,
entropy stays maxed, and the agent only looks good because the scripted allocator
executes any reasonable selection competently.

Method (fully deterministic — no neural net, no sampling noise):

1. Replay a fixed seed with BOTH sides scripted for ``branch_turn`` turns to
   reach a representative mid-game state.
2. At ``branch_turn``, player 0 plays one *candidate* high-level action_row
   (the rest of the turn / future is a fixed scripted continuation for both
   sides for ``horizon`` turns).
3. Measure, for that single substituted action:
   * launches it produced (fleets + total ships) at the branch turn,
   * cumulative potential-based shaped reward for P0 over the continuation,
   * final economic-asset differential Phi at the end of the continuation.

Because the prefix and the continuation are identical across candidates, any
difference in cumulative reward is caused *only* by the one substituted action.
A tiny spread across candidates => the action barely matters => the actor cannot
learn (confirms the hypothesis).

Run::

    python reworked_diag_sensitivity.py --branch-turn 30 --horizon 25 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import random
import numpy as np

from kaggle_environments import make


def _fresh_env(seed: int):
    """A reproducible env: orbit_wars seeds the global ``random`` only at board
    generation, so we must seed *before* reset (configuration[randomSeed] set
    after make() is ignored).  Steps consume no randomness, so this makes the
    whole trajectory deterministic given the action sequence."""
    random.seed(int(seed))
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=2)
    return env

from obs_builder import build_observation
from feature_extraction import parse_planets
from reworked_pipeline import pack_reworked_batch, board_state_from_obs
from reworked_rollout import (
    action_to_env_launches, economic_potential, RewardConfig, P_MAX, F_MAX,
)
from reworked_scripted import production_action_row


# ---------------------------------------------------------------------------
# Candidate high-level action_rows, built from a BoardState (player 0).
# ---------------------------------------------------------------------------
def _empty_row(p_max: int) -> dict:
    return {
        "attack_sel": np.zeros(p_max, np.float32),
        "defend_sel": np.zeros(p_max, np.float32),
        "order": np.arange(p_max, dtype=np.int64),
        "commit_frac": np.zeros(p_max, np.float32),
        "potential_weight": 0.0,
        "horizon": 15.0,
    }


def _row_from_masks(board, player, attack_idx, defend_idx, commit, p_max):
    """Build an action_row selecting given target indices, priority by production."""
    n = board.n
    prod = np.asarray(board.production, np.float64).reshape(n)
    owner = np.asarray(board.owner).reshape(n)
    row = _empty_row(p_max)
    for t in attack_idx:
        row["attack_sel"][t] = 1.0
    for t in defend_idx:
        row["defend_sel"][t] = 1.0
    is_mine = owner == int(player)
    row["commit_frac"][:n][is_mine] = float(commit)
    sel = list(attack_idx) + list(defend_idx)
    sel_sorted = sorted(sel, key=lambda t: -prod[t])
    rest = [t for t in range(p_max) if t not in sel_sorted]
    row["order"] = np.asarray(sel_sorted + rest, dtype=np.int64)[:p_max]
    return row


def candidate_rows(board, player, p_max, rng) -> dict:
    """Named candidate action_rows spanning the commit and selection axes."""
    n = board.n
    owner = np.asarray(board.owner).reshape(n)
    prod = np.asarray(board.production, np.float64).reshape(n)
    not_mine = [t for t in range(n) if owner[t] != int(player)]
    mine = [t for t in range(n) if owner[t] == int(player)]
    order_by_prod = sorted(not_mine, key=lambda t: -prod[t])

    rows: dict[str, dict] = {}
    rows["noop"] = _empty_row(p_max)
    # Commit axis (select-all = the de-facto current behaviour at 3 commit levels)
    for c in (0.30, 0.85, 1.00):
        rows[f"all_c{int(c*100):03d}"] = production_action_row(board, player, p_max=p_max, commit=c)
    # Selection axis (fixed commit 0.85)
    rows["best1"] = _row_from_masks(board, player, order_by_prod[:1], mine, 0.85, p_max)
    rows["top3"] = _row_from_masks(board, player, order_by_prod[:3], mine, 0.85, p_max)
    rows["defend_only"] = _row_from_masks(board, player, [], mine, 0.85, p_max)
    rows["attack_only"] = _row_from_masks(board, player, not_mine, [], 0.85, p_max)
    # Random 50% attack subsets (within-axis noise baseline)
    for s in range(3):
        k = max(1, len(not_mine) // 2)
        pick = list(rng.choice(not_mine, size=k, replace=False)) if not_mine else []
        rows[f"rand{s}"] = _row_from_masks(board, player, pick, mine, 0.85, p_max)
    return rows


# ---------------------------------------------------------------------------
# Deterministic branching rollout
# ---------------------------------------------------------------------------
def _scripted_env_action(env_obs, player, turn, max_turns):
    av = env_obs.get("angular_velocity", 0.0) or 0.0
    obs = build_observation(env_obs, player=player, angular_velocity=av,
                            p_max=P_MAX, f_max=F_MAX, turn=turn, max_turns=max_turns)
    board = board_state_from_obs(obs, player)
    row = production_action_row(board, player, p_max=P_MAX)
    ea, _ = action_to_env_launches(obs, player, row, parse_planets(env_obs), av)
    return ea


def _row_env_action(env_obs, player, row, turn, max_turns):
    av = env_obs.get("angular_velocity", 0.0) or 0.0
    obs = build_observation(env_obs, player=player, angular_velocity=av,
                            p_max=P_MAX, f_max=F_MAX, turn=turn, max_turns=max_turns)
    ea, launches = action_to_env_launches(obs, player, row, parse_planets(env_obs), av)
    return ea, launches


def run_branch(seed, branch_turn, horizon, candidate_row, cfg, max_turns):
    """Replay prefix (both scripted), substitute P0's action at branch_turn for
    `candidate_row`, then continue both scripted for `horizon` turns.

    Returns (cum_shaped_reward_p0, n_fleets, total_ships, phi_final).
    """
    env = _fresh_env(seed)
    state = env.step([[], []])

    # Prefix: both scripted.
    for t in range(branch_turn):
        a0 = _scripted_env_action(state[0]["observation"], 0, t, max_turns)
        a1 = _scripted_env_action(state[1]["observation"], 1, t, max_turns)
        state = env.step([a0, a1])
        if env.done:
            return 0.0, 0, 0, 0.0

    cum_r = 0.0
    n_fleets = total_ships = 0
    for k in range(horizon):
        t = branch_turn + k
        remaining = max_turns - t
        phi_before = economic_potential(state[0]["observation"], 0, remaining, cfg)
        if k == 0:
            a0, launches = _row_env_action(state[0]["observation"], 0, candidate_row, t, max_turns)
            n_fleets = len(launches)
            total_ships = int(sum(L.ships for L in launches))
        else:
            a0 = _scripted_env_action(state[0]["observation"], 0, t, max_turns)
        a1 = _scripted_env_action(state[1]["observation"], 1, t, max_turns)
        state = env.step([a0, a1])
        phi_after = economic_potential(state[0]["observation"], 0, max_turns - (t + 1), cfg)
        cum_r += cfg.shaping_weight * (cfg.gamma * phi_after - phi_before)
        if env.done:
            break

    phi_final = economic_potential(state[0]["observation"], 0, max(1, max_turns - (branch_turn + horizon)), cfg)
    return cum_r, n_fleets, total_ships, phi_final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-turn", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--max-turns", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    cfg = RewardConfig()
    rng = np.random.default_rng(0)

    # Build candidate names once from a reference state (turn = branch_turn, seed[0]).
    env = _fresh_env(args.seeds[0])
    state = env.step([[], []])
    for t in range(args.branch_turn):
        a0 = _scripted_env_action(state[0]["observation"], 0, t, args.max_turns)
        a1 = _scripted_env_action(state[1]["observation"], 1, t, args.max_turns)
        state = env.step([a0, a1])
    av = state[0]["observation"].get("angular_velocity", 0.0) or 0.0
    obs0 = build_observation(state[0]["observation"], player=0, angular_velocity=av,
                             p_max=P_MAX, f_max=F_MAX, turn=args.branch_turn, max_turns=args.max_turns)
    board0 = board_state_from_obs(obs0, 0)
    names = list(candidate_rows(board0, 0, P_MAX, np.random.default_rng(0)).keys())

    # Per-candidate metrics averaged over seeds.
    print(f"\nbranch_turn={args.branch_turn}  horizon={args.horizon}  seeds={args.seeds}")
    print(f"{'candidate':14s} {'cum_reward':>12s} {'fleets':>7s} {'ships':>7s} {'phi_final':>10s}")
    print("-" * 56)
    summary = {}
    for name in names:
        rs, fl, sh, pf = [], [], [], []
        for seed in args.seeds:
            # rebuild candidate for THIS seed's branch state (so masks match the board)
            e = _fresh_env(seed)
            st = e.step([[], []])
            for t in range(args.branch_turn):
                a0 = _scripted_env_action(st[0]["observation"], 0, t, args.max_turns)
                a1 = _scripted_env_action(st[1]["observation"], 1, t, args.max_turns)
                st = e.step([a0, a1])
            avv = st[0]["observation"].get("angular_velocity", 0.0) or 0.0
            o0 = build_observation(st[0]["observation"], player=0, angular_velocity=avv,
                                   p_max=P_MAX, f_max=F_MAX, turn=args.branch_turn, max_turns=args.max_turns)
            b0 = board_state_from_obs(o0, 0)
            row = candidate_rows(b0, 0, P_MAX, np.random.default_rng(100 + seed))[name]
            r, nf, ts, phf = run_branch(seed, args.branch_turn, args.horizon, row, cfg, args.max_turns)
            rs.append(r); fl.append(nf); sh.append(ts); pf.append(phf)
        summary[name] = float(np.mean(rs))
        print(f"{name:14s} {np.mean(rs):12.4f} {np.mean(fl):7.1f} {np.mean(sh):7.1f} {np.mean(pf):10.4f}")

    vals = np.array(list(summary.values()))
    spread = float(vals.max() - vals.min())
    best = max(summary, key=summary.get); worst = min(summary, key=summary.get)
    print("-" * 56)
    print(f"cum_reward  mean={vals.mean():.4f}  std={vals.std():.4f}  spread(max-min)={spread:.4f}")
    print(f"best={best} ({summary[best]:+.4f})   worst={worst} ({summary[worst]:+.4f})")
    print("\nInterpretation: small spread/std vs. the noop->all gap => the high-level "
          "action barely changes the return => low learning signal for the actor.")


if __name__ == "__main__":
    main()
