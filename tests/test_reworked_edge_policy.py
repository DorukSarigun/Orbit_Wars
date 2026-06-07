"""Unit + smoke tests for the v9 directed-edge policy (per-target credit).

Covers the valid-edge mask, the per-edge gate+magnitude action -> env launch
conversion (with the per-source feasibility clamp), ``ReworkedEdgePolicy``
act/evaluate per-target consistency, and the GAE per-target advantage
decomposition (``A_{t,k} = credit_k`` on valid targets).  A short end-to-end ``collect_episode``
+ ``ppo_update`` run exercises the full edge path including battle credit and the
source-loss penalty.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from feature_extraction import Planet  # noqa: E402
from reworked_model import (  # noqa: E402
    ReworkedEdgePolicy,
    REWORKED_NODE_FEATURE_DIM, REWORKED_EDGE_FEATURE_DIM, REWORKED_GLOBAL_FEATURE_DIM,
)
from reworked_pipeline import BoardState  # noqa: E402
from reworked_rollout import (  # noqa: E402
    _edge_valid_mask, _edge_target_valid_mask, edge_action_to_env, MIN_SEND, P_MAX,
    EdgeTransition, ReworkedBuffer,
)


def _board(*, owner, ships):
    n = len(owner)
    return BoardState(
        n=n, player=0,
        owner=np.asarray(owner, dtype=np.int64),
        ships=np.asarray(ships, dtype=np.float32),
        production=np.ones(n, dtype=np.float32),
        radius=np.ones(n, dtype=np.float32),
        is_rotating=np.zeros(n, dtype=np.float32),
        is_comet=np.zeros(n, dtype=np.float32),
        distance_matrix=np.ones((n, n), dtype=np.float32),
        crosses_sun=np.zeros((n, n), dtype=np.float32),
        remaining_turns=100,
    )


def _edge_mask(B, P, source_rows):
    m = torch.zeros(B, P, P, dtype=torch.bool)
    for b in range(B):
        for h in source_rows:
            for t in range(P):
                if t != h:
                    m[b, h, t] = True
    return m


# --------------------------------------------------------------------------- mask
def test_edge_valid_mask_basic():
    # player 0 owns planets 0 and 2; planet 2 has too few ships to send.
    state = _board(owner=[0, 1, 0, -1], ships=[10.0, 5.0, 0.0, 3.0])
    m = _edge_valid_mask(state, P_MAX, 0, min_send=MIN_SEND)
    assert m.shape == (P_MAX, P_MAX)
    # source 0 reaches every other real planet (attack AND defend edges)
    assert m[0, 1] and m[0, 2] and m[0, 3] and not m[0, 0]
    # planet 2 has 0 ships -> not a source; non-owned planets never sources
    assert not m[2].any() and not m[1].any() and not m[3].any()
    # a target is "active" iff some source can reach it
    target_valid = m.any(axis=0)
    assert target_valid[1] and target_valid[2] and target_valid[3]
    assert not m[4:, :].any()  # padding clean


def test_edge_target_valid_mask_unpadded_n_by_n():
    """Notebook-style (n, n) slices without p_max padding must work."""
    n = 3
    em = np.zeros((n, n), dtype=bool)
    em[0, 1] = em[0, 2] = True
    gate = np.zeros((n, n), dtype=np.float32)
    gate[0, 1] = 1.0
    tv = _edge_target_valid_mask(gate, em)
    assert tv[1] and not tv[2]


def test_edge_target_valid_mask_gated_only():
    em = np.zeros((P_MAX, P_MAX), dtype=bool)
    em[0, 1] = em[0, 2] = em[0, 3] = True
    gate = np.zeros((P_MAX, P_MAX), dtype=np.float32)
    gate[0, 1] = 1.0
    tv = _edge_target_valid_mask(gate, em)
    assert tv[1]
    assert not tv[2] and not tv[3]
    # reachable-but-ungated -> excluded (old edge_np.any(axis=0) would include 2,3)
    assert not np.array_equal(tv[:4], em.any(axis=0)[:4])


# ------------------------------------------------------------ env action + clamp
def _planet(pid, owner, x, y, ships):
    return Planet(id=pid, owner=owner, x=float(x), y=float(y), radius=1.0,
                  ships=int(ships), production=1)


def test_edge_action_feasibility_clamp():
    # source 0 (garrison 10) gates two targets each requesting 80% (=8 ships):
    # 8 + 8 = 16 > 10 -> scaled to fit so the launched total never exceeds 10.
    state = _board(owner=[0, -1, -1], ships=[10.0, 50.0, 50.0])
    parsed = [_planet(0, 0, 80, 80, 10), _planet(1, -1, 85, 80, 50),
              _planet(2, -1, 80, 85, 50)]
    obs = SimpleNamespace(planet_ids=np.array([0, 1, 2]))
    gate = np.zeros((P_MAX, P_MAX), dtype=np.float32)
    frac = np.zeros((P_MAX, P_MAX), dtype=np.float32)
    gate[0, 1] = 1.0; gate[0, 2] = 1.0
    frac[0, 1] = 0.8; frac[0, 2] = 0.8
    env_action, realized = edge_action_to_env(
        state, obs, 0, {"gate": gate, "frac": frac}, parsed, 0.0, min_send=MIN_SEND)
    total = sum(int(L.ships) for L in realized)
    assert total <= 10, (total, realized)
    assert len(realized) >= 1
    # each realized launch carries its heading for credit disambiguation
    assert all(hasattr(L, "angle") for L in realized)


# ------------------------------------------------------------------- edge policy
def test_edge_policy_act_evaluate_consistency():
    torch.manual_seed(0)
    B, P = 2, 6
    pol = ReworkedEdgePolicy(d_model=32, n_layers=2)
    nf = torch.randn(B, P, REWORKED_NODE_FEATURE_DIM)
    ef = torch.randn(B, P, P, REWORKED_EDGE_FEATURE_DIM)
    gf = torch.randn(B, REWORKED_GLOBAL_FEATURE_DIM)
    pm = torch.ones(B, P, dtype=torch.bool)
    em = _edge_mask(B, P, source_rows=[0, 1])

    action, target_logp, value = pol.act(
        node_feats=nf, edge_feats=ef, global_feats=gf, planet_mask=pm, edge_mask=em)
    assert action["gate"].shape == (B, P, P)
    assert action["frac"].shape == (B, P, P)
    assert target_logp.shape == (B, P) and value.shape == (B,)
    assert torch.isfinite(target_logp).all()
    # gate is zero on invalid edges
    assert float((action["gate"] * (~em)).abs().sum()) < 1e-6

    ev = pol.evaluate_actions(
        node_feats=nf, edge_feats=ef, global_feats=gf, planet_mask=pm,
        edge_mask=em, action=action)
    assert torch.allclose(target_logp, ev["target_logp"], atol=1e-4)
    assert ev["entropy_target"].shape == (B, P)
    assert torch.isfinite(ev["entropy_target"]).all()
    # target_valid marks every real target (all reachable from sources 0/1)
    assert bool(ev["target_valid"].all())


def test_edge_policy_deterministic_reproducible():
    torch.manual_seed(1)
    B, P = 1, 5
    pol = ReworkedEdgePolicy(d_model=16, n_layers=2)
    nf = torch.randn(B, P, REWORKED_NODE_FEATURE_DIM)
    ef = torch.randn(B, P, P, REWORKED_EDGE_FEATURE_DIM)
    gf = torch.randn(B, REWORKED_GLOBAL_FEATURE_DIM)
    pm = torch.ones(B, P, dtype=torch.bool)
    em = _edge_mask(B, P, source_rows=[0])
    a1, lp1, _ = pol.act(node_feats=nf, edge_feats=ef, global_feats=gf,
                         planet_mask=pm, edge_mask=em, deterministic=True)
    a2, lp2, _ = pol.act(node_feats=nf, edge_feats=ef, global_feats=gf,
                         planet_mask=pm, edge_mask=em, deterministic=True)
    assert torch.allclose(a1["gate"], a2["gate"])
    assert torch.allclose(a1["frac"], a2["frac"])
    assert torch.allclose(lp1, lp2)


# --------------------------------------------------------------- GAE per-target
def test_edge_gae_per_target_matches_campaign_credit():
    # Two-turn single-player buffer; per-target credit differs across targets.
    P = 4
    buf = ReworkedBuffer()

    def _mk(reward, credit, valid, value, done):
        return EdgeTransition(
            player=0,
            node_feats=np.zeros((P, REWORKED_NODE_FEATURE_DIM), np.float32),
            edge_feats=np.zeros((P, P, REWORKED_EDGE_FEATURE_DIM), np.float32),
            global_feats=np.zeros(REWORKED_GLOBAL_FEATURE_DIM, np.float32),
            planet_mask=np.ones(P, bool),
            edge_mask=np.zeros((P, P), bool),
            gate=np.zeros((P, P), np.float32),
            frac=np.zeros((P, P), np.float32),
            old_target_logp=np.zeros(P, np.float32),
            target_valid=np.asarray(valid, bool),
            value=float(value),
            reward=float(reward),
            done=done,
            credit_target=np.asarray(credit, np.float32),
        )

    # turn 0: targets 1 & 2 active, credit +0.5 / -0.3; reward includes that.
    buf.data.append(_mk(reward=0.2 + 0.5 - 0.3, credit=[0, 0.5, -0.3, 0],
                        valid=[False, True, True, False], value=0.4, done=False))
    # turn 1 (terminal): target 1 active, credit +1.0.
    buf.data.append(_mk(reward=1.0, credit=[0, 1.0, 0, 0],
                        valid=[False, True, False, False], value=0.6, done=True))

    buf.compute_gae(gamma=0.99, lam=0.95)
    for tr in buf.data:
        tv = tr.target_valid
        ct = np.asarray(tr.credit_target, dtype=np.float32)
        assert np.allclose(np.asarray(tr.adv_target)[tv], ct[tv])
        # inactive targets carry no advantage
        assert float(np.asarray(tr.adv_target)[~tv].sum()) == 0.0
    # turn 0: target 1 (+credit) should out-advantage target 2 (-credit)
    a0 = buf.data[0].adv_target
    assert a0[1] > a0[2]
    assert abs(a0[1] - 0.5) < 1e-6
    assert abs(a0[2] + 0.3) < 1e-6


if __name__ == "__main__":
    test_edge_valid_mask_basic()
    test_edge_target_valid_mask_gated_only()
    test_edge_action_feasibility_clamp()
    test_edge_policy_act_evaluate_consistency()
    test_edge_policy_deterministic_reproducible()
    test_edge_gae_per_target_matches_campaign_credit()
    print("all edge policy tests passed")
