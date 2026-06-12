"""Remastered campaign reward wired to the hierarchical policy (hier + MCF).

The hier policy has no per-target credit slots, so the remastered campaign
credits land on the launching turn's transition ``reward`` (turn-level), and
the episodic terminal is stored on ``terminal_reward`` and spread across the
player's turns inside the GAE backbone by ``compute_gae``.

Covers:
* ``apply_remastered_credits`` on a hier buffer (no ``credit_target``).
* terminal spread for HierTransition (and no double count for EdgeTransition).
* end-to-end ``collect_episode(policy_kind="hier", remastered_cfg=...)`` +
  ``ppo_update`` smoke run.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reworked_model import (  # noqa: E402
    ReworkedHierarchicalPolicy,
    REWORKED_NODE_FEATURE_DIM,
    REWORKED_EDGE_FEATURE_DIM,
    REWORKED_GLOBAL_FEATURE_DIM,
)
from reworked_remastered_reward import (  # noqa: E402
    RemasteredRewardConfig,
    apply_remastered_credits,
)
from reworked_rollout import (  # noqa: E402
    P_MAX,
    EdgeTransition,
    HierTransition,
    PPOConfig,
    ReworkedBuffer,
    RewardConfig,
    ppo_update,
)


# --------------------------------------------------------------------------- helpers
def _hier_tr(player=0, P=8, value=0.0):
    return HierTransition(
        player=player,
        node_feats=np.zeros((P, REWORKED_NODE_FEATURE_DIM), dtype=np.float32),
        edge_feats=np.zeros((P, P, 4), dtype=np.float32),
        global_feats=np.zeros(14, dtype=np.float32),
        planet_mask=np.ones(P, dtype=bool),
        attack_mask=np.ones(P, dtype=bool),
        defend_mask=np.ones(P, dtype=bool),
        attack_sel=np.zeros(P, dtype=np.float32),
        defend_sel=np.zeros(P, dtype=np.float32),
        order=np.zeros(P, dtype=np.int64),
        commit_frac=np.zeros(P, dtype=np.float32),
        posture_u=np.zeros(2, dtype=np.float32),
        value=value,
    )


def _edge_tr(player=0, P=8):
    return EdgeTransition(
        player=player,
        node_feats=np.zeros((P, REWORKED_NODE_FEATURE_DIM), dtype=np.float32),
        edge_feats=np.zeros((P, P, 4), dtype=np.float32),
        global_feats=np.zeros(14, dtype=np.float32),
        planet_mask=np.ones(P, dtype=bool),
        edge_mask=np.ones((P, P), dtype=bool),
        gate=np.zeros((P, P), dtype=np.float32),
        frac=np.zeros((P, P), dtype=np.float32),
        old_target_logp=np.zeros(P, dtype=np.float32),
        old_target_value=np.zeros(P, dtype=np.float32),
        target_valid=np.zeros(P, dtype=bool),
        credit_target=np.zeros(P, dtype=np.float32),
    )


# --------------------------------------------------------------------------- credits
def test_apply_remastered_credits_lands_on_hier_turn_reward():
    """HierTransition has no credit_target; campaign credit goes to tr.reward."""
    buf = ReworkedBuffer()
    buf.data.append(_hier_tr())
    buf.data.append(_hier_tr())
    apply_remastered_credits(buf, {(0, 5): 0.7, (1, 2): -0.3, (1, 6): 0.1})
    assert buf.data[0].reward == pytest.approx(0.7)
    assert buf.data[1].reward == pytest.approx(-0.3 + 0.1)
    # No per-target slot machinery is conjured up on hier transitions.
    assert getattr(buf.data[0], "credit_target", None) is None


def test_apply_remastered_credits_ignores_out_of_range_record():
    buf = ReworkedBuffer()
    buf.data.append(_hier_tr())
    apply_remastered_credits(buf, {(7, 0): 1.0})
    assert buf.data[0].reward == 0.0


# --------------------------------------------------------------------------- terminal spread
def test_hier_terminal_reward_spreads_into_gae_backbone():
    """terminal_reward on the done turn is spread over the player's turns; the
    first turn's undiscounted return therefore recovers the full terminal."""
    buf = ReworkedBuffer()
    for _ in range(3):
        buf.data.append(_hier_tr())
    buf.data[-1].done = True
    buf.data[-1].terminal_reward = 3.0
    buf.compute_gae(gamma=1.0, lam=1.0)
    # values are 0 -> ret == sum of future (spread) rewards.
    assert buf.data[0].ret == pytest.approx(3.0)
    assert sum(tr.advantage for tr in buf.data) > 0.0
    # Later turns weigh at least as much as earlier ones (discounted weights).
    assert buf.data[2].ret <= buf.data[0].ret + 1e-6


def test_hier_legacy_reward_path_unchanged_without_terminal_reward():
    """terminal_reward defaults to 0 -> compute_gae behaves exactly as before."""
    buf = ReworkedBuffer()
    for r in (0.5, -0.2, 1.0):
        tr = _hier_tr()
        tr.reward = r
        buf.data.append(tr)
    buf.data[-1].done = True
    buf.compute_gae(gamma=1.0, lam=1.0)
    assert buf.data[0].ret == pytest.approx(0.5 - 0.2 + 1.0)
    assert buf.data[2].ret == pytest.approx(1.0)


def test_edge_terminal_not_double_counted_in_backbone():
    """For EdgeTransition the terminal share feeds only the per-target return
    (existing behaviour); the turn-level GAE backbone must not also absorb it."""
    buf = ReworkedBuffer()
    tr = _edge_tr()
    tr.done = True
    tr.terminal_reward = 2.0
    buf.data.append(tr)
    buf.compute_gae(gamma=1.0, lam=1.0)
    assert buf.data[0].ret == pytest.approx(0.0)        # backbone untouched
    # The terminal share is routed onto the actionable per-target slots instead.
    assert float(np.max(buf.data[0].ret_target)) == pytest.approx(2.0)


# --------------------------------------------------------------------------- per-target path
def test_hier_act_evaluate_target_logp_consistency():
    """evaluate_actions on the sampled action must reproduce act's per-target
    log-probs, and their sum must equal the attack+defend share of log_prob."""
    torch.manual_seed(1)
    P = 6
    policy = ReworkedHierarchicalPolicy(
        node_dim=REWORKED_NODE_FEATURE_DIM, d_model=32, n_layers=1, n_heads=4)
    nf = torch.randn(1, P, REWORKED_NODE_FEATURE_DIM)
    ef = torch.randn(1, P, P, REWORKED_EDGE_FEATURE_DIM)
    gf = torch.randn(1, REWORKED_GLOBAL_FEATURE_DIM)
    pm = torch.ones(1, P, dtype=torch.bool)
    am = torch.tensor([[0, 1, 1, 1, 0, 0]], dtype=torch.bool)   # non-self
    dm = torch.tensor([[1, 0, 0, 0, 1, 1]], dtype=torch.bool)   # own

    action, log_prob, _ = policy.act(
        node_feats=nf, edge_feats=ef, global_feats=gf, planet_mask=pm,
        attack_mask=am, defend_mask=dm)
    assert action["target_logp"].shape == (1, P)
    assert action["target_value"].shape == (1, P)

    ev = policy.evaluate_actions(
        node_feats=nf, edge_feats=ef, global_feats=gf, planet_mask=pm,
        attack_mask=am, defend_mask=dm,
        action={k: action[k] for k in
                ("attack_sel", "defend_sel", "order", "commit_frac", "posture_u")})
    assert torch.allclose(ev["target_logp"], action["target_logp"], atol=1e-5)
    assert torch.allclose(ev["log_prob"], log_prob, atol=1e-4)
    # Per-target logps live only on attack/defend candidates.
    cand = am | dm
    assert torch.all(ev["target_logp"][~cand] == 0.0)


def test_hier_per_target_gae_advantage_decomposition():
    """With per-target fields populated, adv_target = credit - V_t on the
    selected slots and stays zero elsewhere (no terminal)."""
    P = 8
    buf = ReworkedBuffer()
    tr = _hier_tr(P=P)
    tr.old_target_logp = np.zeros(P, dtype=np.float32)
    tr.old_target_value = np.full(P, 0.25, dtype=np.float32)
    tv = np.zeros(P, dtype=bool)
    tv[3] = True
    tr.target_valid = tv
    tr.credit_target = np.zeros(P, dtype=np.float32)
    tr.done = True
    buf.data.append(tr)
    assert buf.has_hier_target_credit

    apply_remastered_credits(buf, {(0, 3): 1.5, (0, 6): -0.4})
    buf.compute_gae(gamma=0.99, lam=0.95)
    # slot 3: selected, credited; slot 6: passive credit flips it valid.
    assert tr.adv_target[3] == pytest.approx(1.5 - 0.25)
    assert tr.adv_target[6] == pytest.approx(-0.4 - 0.25)
    assert tr.ret_target[3] == pytest.approx(1.5)
    untouched = [k for k in range(P) if k not in (3, 6)]
    assert all(tr.adv_target[k] == 0.0 for k in untouched)
    # Turn-level backbone also absorbed the summed credit via tr.reward.
    assert tr.reward == pytest.approx(1.5 - 0.4)


# --------------------------------------------------------------------------- end-to-end
def test_collect_episode_hier_remastered_smoke():
    """Short hier episode with the remastered reward: campaigns register, the
    episode finalizes without error, and a PPO update runs on the buffer."""
    pytest.importorskip("kaggle_environments")
    from reworked_rollout import collect_episode

    torch.manual_seed(0)
    policy = ReworkedHierarchicalPolicy(
        node_dim=REWORKED_NODE_FEATURE_DIM, d_model=32, n_layers=1, n_heads=4)
    buffer = ReworkedBuffer()
    reward_cfg = RewardConfig(
        enable_shaping=False, loss_weight=0.0, launch_weight=0.0,
        enable_battle_credit=False, terminal_weight=0.0)
    remastered_cfg = RemasteredRewardConfig()

    stats = collect_episode(
        policy, buffer,
        max_steps=40, seed=123,
        opponent="scripted", scripted_strategy="production",
        policy_kind="hier",
        reward_cfg=reward_cfg,
        remastered_cfg=remastered_cfg,
    )
    assert len(buffer) > 0
    assert all(isinstance(tr, HierTransition) for tr in buffer.data)
    assert not buffer.is_edge
    # Remastered + hier populates the per-target credit fields.
    assert buffer.has_hier_target_credit
    assert buffer.data[0].old_target_logp is not None
    assert buffer.data[0].credit_target is not None
    # The remastered pass produced a diagnostics detail dict.
    assert stats.episode_log is not None
    assert "reward_stats" in stats.episode_log

    buffer.compute_gae(gamma=0.99, lam=0.95)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    ppo_stats = ppo_update(policy, buffer, optimizer,
                           PPOConfig(epochs=1, minibatch_size=8))
    assert ppo_stats.epochs_run >= 1
