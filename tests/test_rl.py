"""Unit tests for the RL training stack (PPO + DQN/Dueling)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model import OrbitWarsPolicy, pack_observations
from model_q import OrbitWarsQNet
from obs_builder import build_observation
from rl.buffer import RolloutBuffer, Transition
from rl.replay import ReplayBuffer, DQNTransition
from rl.rewards import RewardConfig, compute_step_reward


# ----------------------------------------------------------------------- helpers

def _toy_env_obs(player: int = 0):
    """A minimal env-style observation dict that obs_builder accepts."""
    return {
        "planets": [
            [0,  0, 70.0, 70.0, 3.0, 100, 3],
            [1,  1, 30.0, 30.0, 2.0,  80, 2],
            [2, -1, 50.0, 90.0, 2.0,  20, 1],
        ],
        "fleets": [],
        "initial_planets": [
            [0,  0, 70.0, 70.0, 3.0, 100, 3],
            [1,  1, 30.0, 30.0, 2.0,  80, 2],
            [2, -1, 50.0, 90.0, 2.0,  20, 1],
        ],
        "comet_planet_ids": [],
        "comets": [],
        "angular_velocity": 0.05,
        "step": 0,
        "player": player,
    }


def _toy_obs(player: int = 0, p_max: int = 4, f_max: int = 4):
    return build_observation(_toy_env_obs(player), player=player,
                             angular_velocity=0.05, p_max=p_max, f_max=f_max)


# ------------------------------------------------------------- evaluate_actions

class TestEvaluateActions:
    def test_log_prob_matches_act(self):
        """evaluate_actions() must reproduce the log_prob that act() returned."""
        torch.manual_seed(0)
        net = OrbitWarsPolicy(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        net.eval()
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        with torch.no_grad():
            out = net(**batch)
            action = net.act(out, deterministic=False)

        ev = net.evaluate_actions(out, action["src"], action["tgt"], action["ship_bin"])
        assert torch.allclose(ev["log_prob"], action["log_prob"], atol=1e-5)

    def test_entropy_is_nonnegative(self):
        torch.manual_seed(1)
        net = OrbitWarsPolicy(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        out = net(**batch)
        action = net.act(out)
        ev = net.evaluate_actions(out, action["src"], action["tgt"], action["ship_bin"])
        assert (ev["entropy"] >= 0).all()

    def test_gradient_flows(self):
        """Loss = -log_prob.sum() should produce non-zero grads."""
        torch.manual_seed(2)
        net = OrbitWarsPolicy(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        out = net(**batch)
        action = net.act(out)
        ev = net.evaluate_actions(out, action["src"], action["tgt"], action["ship_bin"])
        loss = -ev["log_prob"].sum()
        loss.backward()
        any_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in net.parameters())
        assert any_grad

    def test_noop_only_uses_src_branch(self):
        """For NO-OP actions tgt/ship branches must NOT contribute to log_prob."""
        torch.manual_seed(3)
        net = OrbitWarsPolicy(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        out = net(**batch)
        # Force a NO-OP action.
        src = torch.tensor([net.p_max], dtype=torch.long)
        tgt = torch.tensor([0], dtype=torch.long)
        ship = torch.tensor([0], dtype=torch.long)
        ev = net.evaluate_actions(out, src, tgt, ship)
        # log_prob should equal log_softmax(src_logits)[NOOP].
        from torch.distributions import Categorical
        expected = Categorical(logits=out.src_logits).log_prob(src)
        assert torch.allclose(ev["log_prob"], expected, atol=1e-6)


# ----------------------------------------------------------------- RolloutBuffer

def _dummy_transition(reward: float = 0.0, done: bool = False, p_max: int = 4,
                      f_max: int = 4) -> Transition:
    obs = _toy_obs(p_max=p_max, f_max=f_max)
    return Transition(
        planet_feats=obs.planet_feats, fleet_feats=obs.fleet_feats,
        global_feats=obs.global_feats, edge_bias=obs.edge_bias,
        planet_mask=obs.planet_mask, fleet_mask=obs.fleet_mask,
        source_mask=obs.source_mask,
        src=0, tgt=1, ship_bin=2, is_noop=False,
        log_prob=-1.5, value=0.1,
        reward=reward, done=done,
    )


class TestRolloutBuffer:
    def test_gae_terminal_advantage_equals_reward_minus_value(self):
        buf = RolloutBuffer()
        buf.add(_dummy_transition(reward=1.0, done=True))
        buf.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)
        # On terminal: A = r + gamma*0*mask - V = 1.0 - 0.1
        assert pytest.approx(buf._advantages[0], abs=1e-5) == 0.9

    def test_returns_equal_advantages_plus_values(self):
        buf = RolloutBuffer()
        for r in [0.0, 0.1, 1.0]:
            buf.add(_dummy_transition(reward=r, done=False))
        buf.data[-1].done = True
        buf.compute_gae(last_value=0.0)
        for adv, ret, t in zip(buf._advantages, buf._returns, buf.data):
            assert pytest.approx(ret, abs=1e-5) == adv + t.value

    def test_minibatches_cover_all_samples(self):
        buf = RolloutBuffer()
        n = 7
        for _ in range(n):
            buf.add(_dummy_transition(reward=0.1))
        buf.data[-1].done = True
        buf.compute_gae(last_value=0.0)
        seen = 0
        for batch in buf.iter_minibatches(batch_size=3, shuffle=False):
            seen += batch["src"].shape[0]
        assert seen == n


# ------------------------------------------------------------------ ReplayBuffer

def _dummy_dqn_transition(reward: float = 0.0, done: bool = False) -> DQNTransition:
    cur = _toy_obs(p_max=4, f_max=4)
    nxt = _toy_obs(p_max=4, f_max=4)
    return DQNTransition(
        planet_feats=cur.planet_feats, fleet_feats=cur.fleet_feats,
        global_feats=cur.global_feats, edge_bias=cur.edge_bias,
        planet_mask=cur.planet_mask, fleet_mask=cur.fleet_mask,
        source_mask=cur.source_mask,
        src=0, tgt=1, ship_bin=2, is_noop=False,
        reward=reward, done=done,
        next_planet_feats=nxt.planet_feats, next_fleet_feats=nxt.fleet_feats,
        next_global_feats=nxt.global_feats, next_edge_bias=nxt.edge_bias,
        next_planet_mask=nxt.planet_mask, next_fleet_mask=nxt.fleet_mask,
        next_source_mask=nxt.source_mask,
    )


class TestReplayBuffer:
    def test_capacity_enforced(self):
        buf = ReplayBuffer(capacity=3)
        for _ in range(5):
            buf.add(_dummy_dqn_transition())
        assert len(buf) == 3

    def test_can_sample_batch(self):
        buf = ReplayBuffer(capacity=10)
        assert not buf.can_sample(2)
        for _ in range(4):
            buf.add(_dummy_dqn_transition(reward=0.5))
        assert buf.can_sample(4)
        sample = buf.sample(2)
        assert sample["reward"].shape == (2,)
        # next_* fields must round-trip correctly.
        assert sample["next_planet_feats"].shape == sample["planet_feats"].shape


# ----------------------------------------------------------------- reward shape

class TestRewards:
    def test_pure_terminal_reward(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        cfg = RewardConfig(planet_diff_weight=0.0, ship_diff_weight=0.0)
        r = compute_step_reward(prev, curr, 0, terminal=True,
                                env_terminal_reward=+1, cfg=cfg)
        assert r == 1.0

    def test_zero_when_no_change_no_terminal(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=RewardConfig())
        # Same state -> shaping = 0.
        assert pytest.approx(r, abs=1e-6) == 0.0

    def test_planet_capture_increases_reward(self):
        prev = _toy_env_obs(player=0)
        # Player 0 captures planet 2 (was neutral).
        curr = _toy_env_obs(player=0)
        curr["planets"][2][1] = 0  # owner -1 -> 0
        cfg = RewardConfig(planet_diff_weight=0.1, ship_diff_weight=0.0)
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=cfg)
        assert r > 0


# ---------------------------------------------------------------------- Q-net

class TestQNet:
    @pytest.mark.parametrize("dueling", [False, True])
    def test_forward_shapes(self, dueling: bool):
        net = OrbitWarsQNet(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2,
                            dueling=dueling)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        out = net(**batch)
        assert out.q_src.shape == (1, 5)        # P+1
        assert out.q_tgt.shape == (1, 4, 4)     # P x P
        assert out.q_ship.shape == (1, 4, 5)    # P x n_bins
        if dueling:
            assert out.v_state is not None
            assert out.v_state.shape == (1,)
        else:
            assert out.v_state is None

    def test_dueling_centers_advantage(self):
        """For dueling, mean(A) is subtracted, so mean(Q_branch - V) ≈ 0."""
        torch.manual_seed(0)
        net = OrbitWarsQNet(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2,
                            dueling=True)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        out = net(**batch)
        # Per-source advantage should be mean-centered along last dim.
        assert torch.allclose(out.q_tgt.mean(dim=-1) - out.v_state[:, None],
                              torch.zeros_like(out.q_tgt.mean(dim=-1)),
                              atol=1e-5)

    def test_select_actions_respects_source_mask(self):
        torch.manual_seed(1)
        net = OrbitWarsQNet(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        # Only slot 0 is launchable in toy obs.
        out = net(**batch)
        action = OrbitWarsQNet.select_actions(
            out, source_mask=batch["source_mask"], planet_mask=batch["planet_mask"],
            epsilon=0.0,
        )
        # src must be either 0 (only valid planet) or NOOP (= p_max=4).
        assert action["src"][0].item() in (0, 4)

    def test_joint_q_noop_only_uses_src(self):
        torch.manual_seed(2)
        net = OrbitWarsQNet(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        out = net(**batch)
        src = torch.tensor([net.p_max], dtype=torch.long)
        tgt = torch.tensor([0], dtype=torch.long)
        ship = torch.tensor([0], dtype=torch.long)
        q = net.joint_q(out, src, tgt, ship)
        # For NOOP, joint_q == q_src[NOOP].
        assert torch.allclose(q, out.q_src[:, net.p_max], atol=1e-6)

    def test_max_joint_q_finite(self):
        torch.manual_seed(3)
        net = OrbitWarsQNet(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        out = net(**batch)
        q_max = net.max_joint_q(out, batch["source_mask"], batch["planet_mask"])
        assert torch.isfinite(q_max).all()
