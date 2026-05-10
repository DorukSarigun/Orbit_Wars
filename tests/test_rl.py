"""Unit tests for the RL training stack (PPO + DQN/Dueling)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from model import OrbitWarsPolicy, pack_observations
from model_q import OrbitWarsQNet
from obs_builder import build_observation
from rl.buffer import RolloutBuffer, Transition
from rl.defense_tracker import DefenseTracker, compute_defense_reward
from rl.fleet_tracker import (
    FleetOutcome,
    FleetTracker,
    compute_fleet_reward,
    compute_production_simple_fleet_combat_reward,
)
from rl.planet_loss_tracker import (
    PlanetLossOutcome,
    PlanetLossTracker,
    compute_planet_loss_launch_penalty,
)
from rl.planet_defense_tracker import PlanetDefenseTracker
from rl.production_reward_tracker import (
    ProductionRewardTracker,
    ProductionCaptureOutcome,
    ProductionNeutralGiftOutcome,
    ProductionReinforcementOutcome,
    ProductionCaptureClawbackOutcome,
    compute_production_capture_reward,
    compute_production_neutral_gift_penalty,
    compute_production_reinforcement_reward,
    compute_production_capture_clawback,
    production_reward_value,
)
from rl.rollout import (
    _airborne_risk_multiplier,
    _compute_airborne_cost,
    _compute_source_defense_deficit_penalty,
    _compute_source_defense_shortfall,
)
from rl.replay import ReplayBuffer, DQNTransition
from rl.rewards import REWARD_SYSTEMS, RewardConfig, build_reward_config, compute_step_reward, normalize_terminal_rewards
from rl.train import TrainConfig, _episode_num_agents, _parse_num_agent_choices
from rl.metrics import (
    WeightSnapshot, grad_norms_by_module, weight_norms_by_module,
)


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

        ev = net.evaluate_actions(out, action["src"], action["tgt"], action["ship_value"])
        assert torch.allclose(ev["log_prob"], action["log_prob"], atol=1e-5)

    def test_entropy_is_nonnegative(self):
        torch.manual_seed(1)
        net = OrbitWarsPolicy(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        out = net(**batch)
        action = net.act(out)
        ev = net.evaluate_actions(out, action["src"], action["tgt"], action["ship_value"])
        assert (ev["entropy"] >= 0).all()

    def test_gradient_flows(self):
        """Loss = -log_prob.sum() should produce non-zero grads."""
        torch.manual_seed(2)
        net = OrbitWarsPolicy(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        out = net(**batch)
        action = net.act(out)
        ev = net.evaluate_actions(out, action["src"], action["tgt"], action["ship_value"])
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
        ship_value = torch.tensor([0.5], dtype=torch.float32)
        ev = net.evaluate_actions(out, src, tgt, ship_value)
        # log_prob should equal log_softmax(src_logits)[NOOP].
        from torch.distributions import Categorical
        expected = Categorical(logits=out.src_logits).log_prob(src)
        assert torch.allclose(ev["log_prob"], expected, atol=1e-6)


# ----------------------------------------------------------------- RolloutBuffer

def _dummy_transition(reward: float = 0.0, done: bool = False, p_max: int = 4,
                      f_max: int = 4) -> Transition:
    obs = _toy_obs(p_max=p_max, f_max=f_max)
    return Transition(
        player=0,
        planet_feats=obs.planet_feats, fleet_feats=obs.fleet_feats,
        global_feats=obs.global_feats, edge_bias=obs.edge_bias,
        planet_mask=obs.planet_mask, fleet_mask=obs.fleet_mask,
        source_mask=obs.source_mask,
        src=0, tgt=1, ship_value=0.5, is_noop=False,
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

    def test_add_returns_index_and_add_reward_mutates_transition(self):
        buf = RolloutBuffer()
        idx = buf.add(_dummy_transition(reward=0.1))
        assert idx == 0
        buf.add_reward(idx, 0.4)
        assert buf.data[idx].reward == pytest.approx(0.5, abs=1e-6)

    def test_add_delayed_reward_discounts_by_event_delay(self):
        buf = RolloutBuffer()
        tr = _dummy_transition(reward=0.0)
        tr.turn = 10
        idx = buf.add(tr)
        buf.add_delayed_reward(idx, 8.0, event_turn=13, discount=0.5)
        assert buf.data[idx].reward == pytest.approx(1.0, abs=1e-6)

    def test_gae_separates_interleaved_self_play_players(self):
        buf = RolloutBuffer()
        for player, value, reward in [
            (0, 10.0, 0.0),
            (1, 100.0, 0.0),
            (0, 20.0, 1.0),
            (1, 200.0, 2.0),
        ]:
            tr = _dummy_transition(reward=0.0, done=False)
            tr.player = player
            tr.value = value
            tr.reward = reward
            buf.add(tr)
        buf.data[2].done = True
        buf.data[3].done = True

        buf.compute_gae(last_value=0.0, gamma=1.0, lam=1.0)

        assert buf._returns[0] == pytest.approx(1.0, abs=1e-6)
        assert buf._returns[1] == pytest.approx(2.0, abs=1e-6)
        assert buf._returns[2] == pytest.approx(1.0, abs=1e-6)
        assert buf._returns[3] == pytest.approx(2.0, abs=1e-6)


# ------------------------------------------------------------------ ReplayBuffer

class TestDefenseTracker:
    def test_latest_noop_gets_reward_when_enemy_attack_is_held(self):
        prev = {
            "planets": [
                [0, 0, 80.0, 50.0, 3.0, 100, 3],
                [1, 1, 20.0, 20.0, 3.0, 50, 2],
            ],
            "fleets": [
                [10, 1, 75.0, 50.0, 0.0, 1, 10],
            ],
            "initial_planets": [
                [0, 0, 80.0, 50.0, 3.0, 100, 3],
                [1, 1, 20.0, 20.0, 3.0, 50, 2],
            ],
            "comet_planet_ids": [],
            "comets": [],
            "angular_velocity": 0.0,
        }
        curr = {
            **prev,
            "planets": [
                [0, 0, 80.0, 50.0, 3.0, 93, 3],
                [1, 1, 20.0, 20.0, 3.0, 50, 2],
            ],
            "fleets": [],
        }
        tracker = DefenseTracker(horizon=15, arrival_window=2.0)
        tracker.prime(prev, 0)
        tracker.register_hold(buffer_idx=3, player=0, planet_id=0, hold_turn=5, ships_held=100)
        tracker.observe_turn(5, curr, 0)

        resolved = list(tracker.pop_resolved())
        assert len(resolved) == 1
        idx, outcome = resolved[0]
        assert idx == 3
        assert outcome.ships_absorbed == pytest.approx(10.0, abs=1e-6)
        assert compute_defense_reward(outcome, hold_weight=0.001) == pytest.approx(0.01, abs=1e-6)

    def test_no_reward_without_arriving_enemy_threat(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        curr["planets"][0][5] = 80
        tracker = DefenseTracker(horizon=15, arrival_window=2.0)
        tracker.prime(prev, 0)
        tracker.register_hold(buffer_idx=3, player=0, planet_id=0, hold_turn=5, ships_held=100)
        tracker.observe_turn(5, curr, 0)

        assert list(tracker.pop_resolved()) == []

    def test_multi_turn_hold_splits_credit_with_recency_weighting(self):
        """Multiple NO-OPs over several turns on the same threatened planet
        should all share the defense reward, with newer holds weighted
        slightly higher (recency decay=0.9^delay).
        """
        prev = {
            "planets": [
                [0, 0, 80.0, 50.0, 3.0, 100, 3],
                [1, 1, 20.0, 20.0, 3.0, 50, 2],
            ],
            "fleets": [
                [10, 1, 75.0, 50.0, 0.0, 1, 10],
            ],
            "initial_planets": [
                [0, 0, 80.0, 50.0, 3.0, 100, 3],
                [1, 1, 20.0, 20.0, 3.0, 50, 2],
            ],
            "comet_planet_ids": [],
            "comets": [],
            "angular_velocity": 0.0,
        }
        curr = {
            **prev,
            "planets": [
                [0, 0, 80.0, 50.0, 3.0, 93, 3],
                [1, 1, 20.0, 20.0, 3.0, 50, 2],
            ],
            "fleets": [],
        }
        tracker = DefenseTracker(horizon=15, arrival_window=2.0)
        tracker.prime(prev, 0)
        # Three holds across three turns, same ships_held for symmetry.
        tracker.register_hold(buffer_idx=1, player=0, planet_id=0, hold_turn=3, ships_held=100)
        tracker.register_hold(buffer_idx=2, player=0, planet_id=0, hold_turn=4, ships_held=100)
        tracker.register_hold(buffer_idx=3, player=0, planet_id=0, hold_turn=5, ships_held=100)
        tracker.observe_turn(5, curr, 0)

        resolved = list(tracker.pop_resolved())
        # All three holds must be credited.
        assert sorted(idx for idx, _ in resolved) == [1, 2, 3]
        total_absorbed = sum(out.ships_absorbed for _, out in resolved)
        # Conservation: absorbed total still equals what was actually
        # absorbed by the planet (10 ships).
        assert total_absorbed == pytest.approx(10.0, abs=1e-6)
        # Recency: newer holds get >= older holds' share.
        by_idx = {idx: out.ships_absorbed for idx, out in resolved}
        assert by_idx[3] > by_idx[2] > by_idx[1]


class TestFleetReward:
    def test_airborne_cost_scales_with_ships_and_eta(self):
        assert _compute_airborne_cost(
            ships_sent=20,
            est_distance=40.0,
            weight=0.00002,
            speed=1.0,
        ) == pytest.approx(-0.016, abs=1e-6)

    def test_capture_reward_is_positive(self):
        outcome = FleetOutcome(
            captured_planet=True,
            killed_enemy_ships=False,
            destroyed_enroute=False,
            self_reinforce=False,
            planet_production=3.0,
        )
        assert compute_fleet_reward(outcome) == pytest.approx(0.15, abs=1e-6)

    def test_enemy_capture_gets_radius_and_remaining_turn_bonus(self):
        outcome = FleetOutcome(
            captured_planet=True,
            killed_enemy_ships=False,
            destroyed_enroute=False,
            self_reinforce=False,
            captured_from_enemy=True,
            planet_production=3.0,
            planet_radius=5.0,
            arrival_turn=250,
        )
        assert compute_fleet_reward(outcome) == pytest.approx(0.20, abs=1e-6)

    def test_enemy_kill_reward_is_net_of_waste(self):
        outcome = FleetOutcome(
            captured_planet=False,
            killed_enemy_ships=True,
            destroyed_enroute=False,
            self_reinforce=False,
            enemy_ships_killed=20.0,
            own_waste=10.0,
        )
        assert compute_fleet_reward(outcome) == pytest.approx(0.005, abs=1e-6)

    def test_failed_neutral_attack_has_no_kill_reward_and_penalizes_waste(self):
        outcome = FleetOutcome(
            captured_planet=False,
            killed_enemy_ships=False,
            destroyed_enroute=False,
            self_reinforce=False,
            failed_neutral_attack=True,
            own_waste=16.0,
        )
        assert compute_fleet_reward(outcome) == pytest.approx(-0.024, abs=1e-6)

    def test_self_reinforce_has_no_waste_penalty(self):
        outcome = FleetOutcome(
            captured_planet=False,
            killed_enemy_ships=False,
            destroyed_enroute=False,
            self_reinforce=True,
            own_waste=16.0,
        )
        assert compute_fleet_reward(outcome) == pytest.approx(0.0, abs=1e-6)


class TestProductionSimpleFleetCombatReward:
    def test_failed_attack_is_empire_relative(self):
        # Same 20-ship loss; different empires get very different penalties.
        big_empire = FleetOutcome(
            captured_planet=False,
            killed_enemy_ships=False,
            destroyed_enroute=False,
            self_reinforce=False,
            failed_neutral_attack=True,
            own_waste=20.0,
            my_total_ships_at_launch=200,
        )
        small_empire = FleetOutcome(
            captured_planet=False,
            killed_enemy_ships=False,
            destroyed_enroute=False,
            self_reinforce=False,
            failed_neutral_attack=True,
            own_waste=20.0,
            my_total_ships_at_launch=50,
        )
        assert compute_production_simple_fleet_combat_reward(big_empire) == pytest.approx(0.0, abs=1e-6)
        assert compute_production_simple_fleet_combat_reward(small_empire) == pytest.approx(0.0, abs=1e-6)

    def test_enemy_capture_combat_signal_rewards_kills(self):
        outcome = FleetOutcome(
            captured_planet=True,
            killed_enemy_ships=False,
            destroyed_enroute=False,
            self_reinforce=False,
            captured_from_enemy=True,
            own_waste=20.0,
            enemy_ships_killed=20.0,
            my_total_ships_at_launch=100,
        )
        assert compute_production_simple_fleet_combat_reward(outcome) == pytest.approx(0.02, abs=1e-6)

    def test_enemy_kill_is_positive_when_empire_is_stronger(self):
        outcome = FleetOutcome(
            captured_planet=False,
            killed_enemy_ships=True,
            destroyed_enroute=False,
            self_reinforce=False,
            enemy_ships_killed=20.0,
            my_total_ships_at_launch=200,
            enemy_total_ships_at_launch=100,
        )
        assert compute_production_simple_fleet_combat_reward(outcome) == pytest.approx(0.04, abs=1e-6)

    def test_enemy_kill_is_positive_but_reduced_when_empire_is_weaker(self):
        outcome = FleetOutcome(
            captured_planet=False,
            killed_enemy_ships=True,
            destroyed_enroute=False,
            self_reinforce=False,
            enemy_ships_killed=20.0,
            my_total_ships_at_launch=100,
            enemy_total_ships_at_launch=200,
        )
        assert compute_production_simple_fleet_combat_reward(outcome) == pytest.approx(0.01, abs=1e-6)

    def test_enemy_kill_uses_base_value_when_empires_are_equal(self):
        outcome = FleetOutcome(
            captured_planet=False,
            killed_enemy_ships=True,
            destroyed_enroute=False,
            self_reinforce=False,
            enemy_ships_killed=20.0,
            my_total_ships_at_launch=100,
            enemy_total_ships_at_launch=100,
        )
        assert compute_production_simple_fleet_combat_reward(outcome) == pytest.approx(0.02, abs=1e-6)

    def test_self_reinforce_returns_zero(self):
        outcome = FleetOutcome(
            captured_planet=False,
            killed_enemy_ships=False,
            destroyed_enroute=False,
            self_reinforce=True,
            own_waste=20.0,
            my_total_ships_at_launch=50,
        )
        assert compute_production_simple_fleet_combat_reward(outcome) == 0.0

    def test_zero_empire_falls_back_to_default_denominator(self):
        # If empire ships were never recorded, fall back to the legacy /100.
        outcome = FleetOutcome(
            captured_planet=False,
            killed_enemy_ships=False,
            self_reinforce=False,
            destroyed_enroute=True,
            own_waste=20.0,
            my_total_ships_at_launch=0,
        )
        assert compute_production_simple_fleet_combat_reward(outcome) == pytest.approx(-0.0002, abs=1e-6)


class TestFleetTrackerMultiFleetAttribution:
    """Multi-fleet same-turn arrivals must split enemy_killed and own_waste
    proportionally; otherwise the combat reward is over-credited.
    """

    def _obs_with_planet(self, planet_id: int, owner: int, ships: int,
                         x: float = 50.0, y: float = 50.0, prod: float = 2.0):
        return {
            "planets": [[int(planet_id), int(owner), float(x), float(y),
                         float(prod), int(ships), 3]],
            "fleets": [],
            "initial_planets": [[int(planet_id), int(owner), float(x), float(y),
                                 float(prod), int(ships), 3]],
            "comet_planet_ids": [],
            "comets": [],
        }

    def test_single_fleet_capture_matches_legacy_semantics(self):
        """Single-fleet capture: enemy_killed = prev_ships, waste = prev_ships."""
        tracker = FleetTracker()
        tracker.register_launch(
            buffer_idx=1, launch_turn=0, player=0,
            src_planet_id=99, tgt_planet_id=7, ships_sent=50, angle=0.0,
            est_distance=0.0, launch_tgt_owner=1, launch_tgt_ships=30,
            my_total_ships_at_launch=500,
        )
        # After arrival: planet mine, 20 ships remain (50 - 30 = 20).
        env_obs = self._obs_with_planet(planet_id=7, owner=0, ships=20)
        tracker.observe_turn(turn=5, env_obs=env_obs, player=0)
        resolved = list(tracker.pop_resolved(0))
        assert len(resolved) == 1
        _, outcome = resolved[0]
        assert outcome.captured_planet
        assert outcome.enemy_ships_killed == pytest.approx(30.0, abs=1e-6)
        assert outcome.own_waste == pytest.approx(30.0, abs=1e-6)

    def test_multi_fleet_capture_splits_kills_proportionally(self):
        """Two fleets (20, 50) combine to capture a 30-defender planet.
        Total kills should sum to 30, total waste should sum to 30, and
        each fleet should get its proportional share.
        """
        tracker = FleetTracker()
        tracker.register_launch(
            buffer_idx=1, launch_turn=0, player=0,
            src_planet_id=99, tgt_planet_id=7, ships_sent=20, angle=0.0,
            est_distance=0.0, launch_tgt_owner=1, launch_tgt_ships=30,
            my_total_ships_at_launch=500,
        )
        tracker.register_launch(
            buffer_idx=2, launch_turn=0, player=0,
            src_planet_id=98, tgt_planet_id=7, ships_sent=50, angle=1.0,
            est_distance=0.0, launch_tgt_owner=1, launch_tgt_ships=30,
            my_total_ships_at_launch=500,
        )
        # After arrival: planet mine, 40 ships remain (20+50-30=40).
        env_obs = self._obs_with_planet(planet_id=7, owner=0, ships=40)
        tracker.observe_turn(turn=5, env_obs=env_obs, player=0)
        resolved = list(tracker.pop_resolved(0))
        assert len(resolved) == 2
        outcomes = {idx: out for idx, out in resolved}
        # Shares: 20/70 and 50/70.
        # Fleet 1: 30 * 20/70 ≈ 8.571
        assert outcomes[1].enemy_ships_killed == pytest.approx(30 * 20 / 70, abs=1e-6)
        assert outcomes[1].own_waste == pytest.approx(30 * 20 / 70, abs=1e-6)
        # Fleet 2: 30 * 50/70 ≈ 21.429
        assert outcomes[2].enemy_ships_killed == pytest.approx(30 * 50 / 70, abs=1e-6)
        assert outcomes[2].own_waste == pytest.approx(30 * 50 / 70, abs=1e-6)
        # Combined totals are conserved (no double-counting).
        assert (outcomes[1].enemy_ships_killed + outcomes[2].enemy_ships_killed
                == pytest.approx(30.0, abs=1e-6))
        assert (outcomes[1].own_waste + outcomes[2].own_waste
                == pytest.approx(30.0, abs=1e-6))

    def test_multi_fleet_capture_produces_scaled_kill_credit(self):
        tracker = FleetTracker()
        tracker.register_launch(
            buffer_idx=1, launch_turn=0, player=0,
            src_planet_id=99, tgt_planet_id=7, ships_sent=20, angle=0.0,
            est_distance=0.0, launch_tgt_owner=1, launch_tgt_ships=30,
            my_total_ships_at_launch=500,
        )
        tracker.register_launch(
            buffer_idx=2, launch_turn=0, player=0,
            src_planet_id=98, tgt_planet_id=7, ships_sent=50, angle=1.0,
            est_distance=0.0, launch_tgt_owner=1, launch_tgt_ships=30,
            my_total_ships_at_launch=500,
        )
        env_obs = self._obs_with_planet(planet_id=7, owner=0, ships=40)
        tracker.observe_turn(turn=5, env_obs=env_obs, player=0)
        resolved = list(tracker.pop_resolved(0))
        for _, outcome in resolved:
            r = compute_production_simple_fleet_combat_reward(outcome)
            assert r == pytest.approx(outcome.enemy_ships_killed * 0.001, abs=1e-6)

    def test_multi_fleet_failed_attack_splits_damage(self):
        """Two fleets fail against heavily defended enemy. Each fleet dies,
        but the damage dealt is split proportionally.
        """
        tracker = FleetTracker()
        tracker.register_launch(
            buffer_idx=1, launch_turn=0, player=0,
            src_planet_id=99, tgt_planet_id=7, ships_sent=20, angle=0.0,
            est_distance=0.0, launch_tgt_owner=1, launch_tgt_ships=100,
            my_total_ships_at_launch=500,
        )
        tracker.register_launch(
            buffer_idx=2, launch_turn=0, player=0,
            src_planet_id=98, tgt_planet_id=7, ships_sent=30, angle=1.0,
            est_distance=0.0, launch_tgt_owner=1, launch_tgt_ships=100,
            my_total_ships_at_launch=500,
        )
        # Planet remains enemy-owned, garrison reduced from 100 to 50
        # (20+30=50 damage dealt, planet still has 50 defenders).
        env_obs = self._obs_with_planet(planet_id=7, owner=1, ships=50)
        tracker.observe_turn(turn=5, env_obs=env_obs, player=0)
        resolved = list(tracker.pop_resolved(0))
        outcomes = {idx: out for idx, out in resolved}
        assert outcomes[1].killed_enemy_ships
        assert outcomes[2].killed_enemy_ships
        # Total damage = 50, split 20/50 and 30/50.
        assert outcomes[1].enemy_ships_killed == pytest.approx(20.0, abs=1e-6)
        assert outcomes[2].enemy_ships_killed == pytest.approx(30.0, abs=1e-6)
        # Each fleet fully dies (all ships were used).
        assert outcomes[1].own_waste == pytest.approx(20.0, abs=1e-6)
        assert outcomes[2].own_waste == pytest.approx(30.0, abs=1e-6)


class TestRiskAwareAirborneCost:
    def test_safe_launch_uses_unit_multiplier(self):
        assert _airborne_risk_multiplier(0.0) == pytest.approx(1.0, abs=1e-9)

    def test_threatened_source_increases_cost(self):
        safe = _compute_airborne_cost(
            ships_sent=20, est_distance=40.0, weight=0.0001, speed=2.0,
            risk_multiplier=1.0,
        )
        threatened = _compute_airborne_cost(
            ships_sent=20, est_distance=40.0, weight=0.0001, speed=2.0,
            risk_multiplier=_airborne_risk_multiplier(1.0),
        )
        # Threat intensity 1.0 -> 3x base cost.
        assert threatened == pytest.approx(safe * 3.0, abs=1e-9)
        assert threatened < safe < 0.0

    def test_overrun_threat_saturates(self):
        # Even if incoming/remaining = 100, the multiplier saturates because
        # threat_intensity is capped at 3.0 inside the helper.
        # Implementation: 1 + 2*min(t, 3) -> at t>=3, multiplier = 7.
        assert _airborne_risk_multiplier(3.0) == pytest.approx(7.0, abs=1e-9)
        assert _airborne_risk_multiplier(100.0) == pytest.approx(7.0, abs=1e-9)
        # Strictly increasing in the unsaturated region.
        assert _airborne_risk_multiplier(0.5) < _airborne_risk_multiplier(1.0)


class TestRolloutRewardAttribution:
    """End-to-end checks: launches must credit only the *correct* transition.

    These run a short `collect_episode` with a deterministic seed and then
    inspect the resulting buffer for invariants that would catch any
    mis-routed reward (wrong player, wrong transition, leaked airborne to
    NO-OP, etc.).
    """

    def _make_policy_and_buffer(self):
        from model_planet_only import PlanetOnlyOrbitWarsPolicy
        from rl.buffer import RolloutBuffer
        torch.manual_seed(0)
        # The env spawns up to ~36 planets (incl. comets) at default
        # settings; use a generous p_max so build_observation never has to
        # truncate.
        policy = PlanetOnlyOrbitWarsPolicy(
            p_max=64, f_max=0, d_model=32, n_heads=2, n_layers=2,
        )
        return policy, RolloutBuffer()

    def test_launch_count_per_player_lower_bounded_by_is_noop_transitions(self):
        """`stats.n_launches[p]` must equal the number of *successful*
        launches (i.e. a valid env action was generated). Buffer transitions
        with `is_noop=False` cover both successful launches AND invalid
        actions (policy wanted to launch but the decode rejected the
        target). The invariant therefore is:

            count(is_noop=False, player=p)  >=  n_launches[p]

        The slack is exactly the number of invalid actions for player p.
        """
        from rl.rollout import collect_episode
        policy, buf = self._make_policy_and_buffer()
        cfg = build_reward_config("production_simple")
        stats = collect_episode(
            policy, buf,
            max_steps=20,
            num_agents=2,
            reward_cfg=cfg,
            seed=42,
        )
        counted = [0, 0]
        for tr in buf.data:
            if not tr.is_noop:
                counted[tr.player] += 1
        assert counted[0] >= stats.n_launches[0]
        assert counted[1] >= stats.n_launches[1]
        # Sanity: at random init we expect at least some launches.
        assert sum(stats.n_launches) > 0, (
            "No launches at random init -- check action sampling / source_mask"
        )

    def test_no_op_transitions_have_zero_reward_in_production_simple(self):
        """In `production_simple`, NO-OP transitions should never receive
        airborne, capture, loss, or combat rewards. The only reward they can
        receive is the per-step share, which is 0 in non-terminal steps and
        the terminal share at episode end.
        """
        from rl.rollout import collect_episode
        policy, buf = self._make_policy_and_buffer()
        cfg = build_reward_config("production_simple")
        collect_episode(
            policy, buf,
            max_steps=20,
            num_agents=2,
            reward_cfg=cfg,
            seed=42,
        )
        # Group transitions by (player, turn) so we know which step they
        # belong to. Within each step, all NO-OP transitions of the same
        # player should have identical reward (the step-share).
        for tr in buf.data:
            if tr.is_noop and not tr.done:
                assert tr.reward == pytest.approx(0.0, abs=1e-9), (
                    f"NO-OP transition got nonzero reward in non-terminal step: "
                    f"reward={tr.reward}, player={tr.player}, turn={tr.turn}"
                )

    def test_launch_transitions_receive_airborne_cost(self):
        """Every launch transition in `production_simple` must end up with
        a non-positive immediate reward (airborne is always <= 0, and step
        shaping is zero for non-terminal steps)."""
        from rl.rollout import collect_episode
        policy, buf = self._make_policy_and_buffer()
        cfg = build_reward_config("production_simple")
        collect_episode(
            policy, buf,
            max_steps=20,
            num_agents=2,
            reward_cfg=cfg,
            seed=42,
        )
        launch_transitions = [tr for tr in buf.data if not tr.is_noop and not tr.done]
        # We need a few launches to make this meaningful.
        assert len(launch_transitions) > 5
        # In production_simple, immediate launch reward = airborne_cost (<= 0).
        # Delayed combat/capture/loss may add either sign on top, but at
        # least *some* launch transitions should be strictly negative
        # (airborne always fires for non-zero launches with travel time).
        any_negative = any(tr.reward < 0 for tr in launch_transitions)
        assert any_negative, (
            "No launch transition received an airborne penalty; "
            "either weights are zero or rewards are not being routed."
        )

    def test_player_attribution_no_cross_leak(self):
        """A launch by player p must never credit a transition of player q != p.

        We assert the simpler, equivalent invariant: every transition that
        has a non-zero reward must belong to a player that produced at
        least one launch or saw at least one terminal share.
        """
        from rl.rollout import collect_episode
        policy, buf = self._make_policy_and_buffer()
        cfg = build_reward_config("production_simple")
        stats = collect_episode(
            policy, buf,
            max_steps=20,
            num_agents=2,
            reward_cfg=cfg,
            seed=42,
        )
        # At least one player must have launched something for this test to
        # be meaningful.
        assert sum(stats.n_launches) > 0
        # All transitions must have player in {0, 1}; no leaks.
        for tr in buf.data:
            assert tr.player in (0, 1), f"Unexpected player id: {tr.player}"


class TestPlanetLossTracker:
    def test_planet_loss_penalty_splits_by_sent_ship_ratio(self):
        outcome = PlanetLossOutcome(
            ships_sent=20,
            total_ships_sent=50,
            production_lost=3.0,
        )
        assert compute_planet_loss_launch_penalty(outcome, loss_weight=0.05) == pytest.approx(-0.06, abs=1e-6)

    def test_launches_from_threatened_lost_planet_get_penalty_credit(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        curr["planets"][0][1] = 1
        curr["planets"][0][5] = 20
        tracker = PlanetLossTracker(horizon=30)
        tracker.register_launch(
            buffer_idx=3,
            launch_turn=4,
            player=0,
            planet_id=0,
            ships_sent=20,
            source_had_enemy_incoming=True,
        )
        tracker.register_launch(
            buffer_idx=7,
            launch_turn=4,
            player=0,
            planet_id=0,
            ships_sent=30,
            source_had_enemy_incoming=True,
        )
        tracker.observe_turn(5, prev, curr, 0)

        resolved = list(tracker.pop_resolved(0))
        assert [idx for idx, _ in resolved] == [3, 7]
        rewards = [
            compute_planet_loss_launch_penalty(outcome, loss_weight=0.05)
            for _, outcome in resolved
        ]
        assert rewards == pytest.approx([-0.06, -0.09], abs=1e-6)

    def test_no_planet_loss_penalty_without_incoming_at_launch(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        curr["planets"][0][1] = 1
        tracker = PlanetLossTracker(horizon=30)
        tracker.register_launch(
            buffer_idx=3,
            launch_turn=4,
            player=0,
            planet_id=0,
            ships_sent=20,
            source_had_enemy_incoming=False,
        )
        tracker.observe_turn(5, prev, curr, 0)

        assert list(tracker.pop_resolved(0)) == []


class TestPlanetDefenseTracker:
    """Defense Window credit assignment for planet loss (Tasarım 2)."""

    def _flip_p0_to_enemy(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        curr["planets"][0][1] = 1   # P0 ownership flips 0 -> 1
        curr["planets"][0][5] = 25  # leftover ships (irrelevant for tracker)
        return prev, curr

    def test_depart_only_absorbs_full_penalty_split_by_ships(self):
        """When only DEPART actions are recorded, they share the full
        (depart_share + hold_share) = 1.0 penalty proportional to ships."""
        prev, curr = self._flip_p0_to_enemy()
        tracker = PlanetDefenseTracker(
            horizon=30, depart_share=0.7, hold_share=0.3,
            max_turns=500, num_opponents=1,
        )
        # Two launches FROM planet 0 with ratio 1:3
        tracker.register_depart(buffer_idx=10, player=0,
                                src_planet_id=0, turn=4, ships_sent=10)
        tracker.register_depart(buffer_idx=11, player=0,
                                src_planet_id=0, turn=4, ships_sent=30)
        tracker.observe_turn(5, prev, curr, 0,
                             reward_scale=0.001, loss_weight=1.0)
        resolved = list(tracker.pop_resolved(0))
        # loss_value = production(3) * remaining(495) * opp_mult(3/2) * 0.001 * 1.0
        loss_value = 3.0 * 495 * 1.5 * 0.001 * 1.0
        # Full share since no holds present.
        full_share = 0.7 + 0.3
        expected = {
            10: -loss_value * full_share * (10 / 40),
            11: -loss_value * full_share * (30 / 40),
        }
        assert {idx: r for idx, r in resolved} == pytest.approx(expected, abs=1e-6)

    def test_hold_only_absorbs_full_penalty(self):
        prev, curr = self._flip_p0_to_enemy()
        tracker = PlanetDefenseTracker(
            horizon=30, depart_share=0.7, hold_share=0.3,
            max_turns=500, num_opponents=1,
        )
        tracker.register_hold(buffer_idx=20, player=0,
                              planet_id=0, turn=4, ships_held=100)
        tracker.observe_turn(5, prev, curr, 0,
                             reward_scale=0.001, loss_weight=1.0)
        resolved = list(tracker.pop_resolved(0))
        loss_value = 3.0 * 495 * 1.5 * 0.001 * 1.0
        # Full share absorbed by single hold.
        assert resolved == [(20, pytest.approx(-loss_value, abs=1e-6))]

    def test_depart_and_hold_split_per_configured_shares(self):
        prev, curr = self._flip_p0_to_enemy()
        tracker = PlanetDefenseTracker(
            horizon=30, depart_share=0.7, hold_share=0.3,
            max_turns=500, num_opponents=1,
        )
        tracker.register_depart(buffer_idx=10, player=0,
                                src_planet_id=0, turn=4, ships_sent=40)
        tracker.register_hold(buffer_idx=20, player=0,
                              planet_id=0, turn=4, ships_held=60)
        tracker.observe_turn(5, prev, curr, 0,
                             reward_scale=0.001, loss_weight=1.0)
        resolved = dict(tracker.pop_resolved(0))
        loss_value = 3.0 * 495 * 1.5 * 0.001 * 1.0
        assert resolved[10] == pytest.approx(-loss_value * 0.7, abs=1e-6)
        assert resolved[20] == pytest.approx(-loss_value * 0.3, abs=1e-6)

    def test_no_records_yields_no_penalty(self):
        prev, curr = self._flip_p0_to_enemy()
        tracker = PlanetDefenseTracker(horizon=30)
        tracker.observe_turn(5, prev, curr, 0,
                             reward_scale=0.001, loss_weight=1.0)
        assert list(tracker.pop_resolved(0)) == []

    def test_actions_outside_horizon_are_expired(self):
        prev, curr = self._flip_p0_to_enemy()
        tracker = PlanetDefenseTracker(horizon=10, max_turns=500, num_opponents=1)
        # Action way before window
        tracker.register_depart(buffer_idx=10, player=0,
                                src_planet_id=0, turn=0, ships_sent=50)
        # Action inside window
        tracker.register_depart(buffer_idx=11, player=0,
                                src_planet_id=0, turn=98, ships_sent=20)
        tracker.observe_turn(100, prev, curr, 0,
                             reward_scale=0.001, loss_weight=1.0)
        resolved = list(tracker.pop_resolved(0))
        # Only buffer_idx=11 should appear
        assert [idx for idx, _ in resolved] == [11]

    def test_planet_still_owned_emits_no_penalty(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        # Planet 0 still owned by player 0 in curr -> no loss
        tracker = PlanetDefenseTracker(horizon=30, max_turns=500, num_opponents=1)
        tracker.register_depart(buffer_idx=10, player=0,
                                src_planet_id=0, turn=4, ships_sent=20)
        tracker.observe_turn(5, prev, curr, 0,
                             reward_scale=0.001, loss_weight=1.0)
        assert list(tracker.pop_resolved(0)) == []
        # Records should still be pending (not popped) for future loss
        # detection if it occurs later in window.
        tracker.observe_turn(10, _toy_env_obs(player=0),
                             {**_toy_env_obs(player=0),
                              "planets": [
                                  [0, 1, 70.0, 70.0, 3.0, 25, 3],
                                  [1, 1, 30.0, 30.0, 2.0, 80, 2],
                                  [2, -1, 50.0, 90.0, 2.0, 20, 1],
                              ]}, 0,
                             reward_scale=0.001, loss_weight=1.0)
        # Now planet 0 lost, depart record from turn 4 (within horizon=30) gets penalty
        resolved = list(tracker.pop_resolved(0))
        assert [idx for idx, _ in resolved] == [10]


class TestProductionRewardTracker:
    def test_production_capture_reward_uses_remaining_steps_ship_share_and_investment(self):
        outcome = ProductionCaptureOutcome(
            ships_sent=20,
            total_ships_sent=50,
            total_ship_investment=50.0,
            planet_production=3.0,
            capture_turn=100,
            max_turns=500,
            captured_from_enemy=False,
            num_opponents=1,
        )
        # Scaled: (production * remaining - investment) * ship_share * 0.001
        # = (3 * 400 - 50) * (20/50) * 0.001 = 0.46
        assert compute_production_capture_reward(outcome) == pytest.approx(0.46, abs=1e-6)

    def test_capture_reward_subtracts_ship_investment_not_empire_fraction(self):
        outcome = ProductionCaptureOutcome(
            ships_sent=90,
            total_ships_sent=90,
            total_ship_investment=90.0,
            planet_production=3.0,
            capture_turn=100,
            max_turns=500,
            captured_from_enemy=False,
            num_opponents=1,
            my_total_ships_at_launch=100,
        )
        assert compute_production_capture_reward(outcome) == pytest.approx(1.11, abs=1e-6)

    def test_production_reward_beta_preserves_reference_and_boosts_high_production(self):
        assert production_reward_value(
            3.0,
            production_reward_beta=0.5,
            production_reward_reference=3.0,
        ) == pytest.approx(3.0, abs=1e-6)
        assert production_reward_value(
            5.0,
            production_reward_beta=0.5,
            production_reward_reference=3.0,
        ) == pytest.approx(5.0 * ((5.0 / 3.0) ** 0.5), abs=1e-6)
        assert production_reward_value(
            1.0,
            production_reward_beta=0.5,
            production_reward_reference=3.0,
        ) < 1.0

    def test_production_capture_reward_supports_superlinear_production_value(self):
        outcome = ProductionCaptureOutcome(
            ships_sent=50,
            total_ships_sent=50,
            total_ship_investment=50.0,
            planet_production=5.0,
            capture_turn=100,
            max_turns=500,
            captured_from_enemy=False,
            num_opponents=1,
        )
        effective_production = 5.0 * ((5.0 / 3.0) ** 0.5)
        expected = (effective_production * 400 - 50.0) * 0.001
        assert compute_production_capture_reward(
            outcome,
            production_reward_beta=0.5,
            production_reward_reference=3.0,
        ) == pytest.approx(expected, abs=1e-6)

    def test_neutral_gift_penalty_uses_damage_fraction(self):
        outcome = ProductionNeutralGiftOutcome(
            damage_done=90.0,
            launch_target_ships=100,
            planet_production=3.0,
            capture_turn=100,
            max_turns=500,
        )
        assert compute_production_neutral_gift_penalty(outcome) == pytest.approx(-1.08, abs=1e-6)

    def test_tracker_assigns_neutral_gift_penalty_when_enemy_captures_softened_planet(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        prev["planets"][2][1] = -1
        prev["planets"][2][5] = 100
        prev["planets"][2][6] = 3
        curr["planets"][2][1] = 1
        curr["planets"][2][5] = 5
        curr["planets"][2][6] = 3
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2)
        tracker.register_capture_launch(
            buffer_idx=7,
            launch_turn=10,
            player=0,
            target_planet_id=2,
            ships_sent=90,
            est_arrival_turn=80,
            launch_target_owner=-1,
            launch_target_ships=100,
        )
        tracker.observe_turn(100, prev, curr, 0)
        resolved = list(tracker.pop_neutral_gift_resolved(0))
        assert [idx for idx, _ in resolved] == [7]
        penalties = [compute_production_neutral_gift_penalty(outcome) for _, outcome in resolved]
        assert penalties == pytest.approx([-1.08], abs=1e-6)

    def test_neutral_gift_penalty_can_resolve_after_softening_window(self):
        prev = _toy_env_obs(player=0)
        mid = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        prev["planets"][2][1] = -1
        prev["planets"][2][5] = 100
        prev["planets"][2][6] = 3
        mid["planets"][2][1] = -1
        mid["planets"][2][5] = 10
        mid["planets"][2][6] = 3
        curr["planets"][2][1] = 1
        curr["planets"][2][5] = 5
        curr["planets"][2][6] = 3
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2, softening_window=10)
        tracker.register_capture_launch(
            buffer_idx=7,
            launch_turn=10,
            player=0,
            target_planet_id=2,
            ships_sent=90,
            est_arrival_turn=80,
            launch_target_owner=-1,
            launch_target_ships=100,
        )
        tracker.observe_turn(100, prev, mid, 0)
        assert list(tracker.pop_neutral_gift_resolved(0)) == []
        tracker.observe_turn(140, mid, curr, 0)
        resolved = list(tracker.pop_neutral_gift_resolved(0))
        assert [idx for idx, _ in resolved] == [7]
        penalties = [compute_production_neutral_gift_penalty(outcome) for _, outcome in resolved]
        assert penalties == pytest.approx([-0.972], abs=1e-6)

    def test_enemy_capture_multiplier_depends_on_total_player_count(self):
        outcome = ProductionCaptureOutcome(
            ships_sent=50,
            total_ships_sent=50,
            total_ship_investment=50.0,
            planet_production=3.0,
            capture_turn=100,
            max_turns=500,
            captured_from_enemy=True,
            num_opponents=2,
        )
        assert compute_production_capture_reward(outcome) == pytest.approx(1.55, abs=1e-6)

    def test_tracker_splits_capture_reward_only_across_arrived_launches(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        prev["planets"][2][1] = -1
        prev["planets"][2][6] = 3
        curr["planets"][2][1] = 0
        curr["planets"][2][6] = 3
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2)
        tracker.register_capture_launch(
            buffer_idx=1,
            launch_turn=10,
            player=0,
            target_planet_id=2,
            ships_sent=20,
            est_arrival_turn=100,
            launch_target_owner=-1,
        )
        tracker.register_capture_launch(
            buffer_idx=2,
            launch_turn=10,
            player=0,
            target_planet_id=2,
            ships_sent=30,
            est_arrival_turn=100,
            launch_target_owner=-1,
        )
        tracker.register_capture_launch(
            buffer_idx=3,
            launch_turn=10,
            player=0,
            target_planet_id=2,
            ships_sent=100,
            est_arrival_turn=101,
            launch_target_owner=-1,
        )
        tracker.observe_turn(100, prev, curr, 0)
        resolved = list(tracker.pop_capture_resolved(0))
        assert [idx for idx, _ in resolved] == [1, 2]
        rewards = [compute_production_capture_reward(outcome) for _, outcome in resolved]
        # Scaled: (3 * 400 - 20) * ship_share * 0.001 -> 1.18 * [0.4, 0.6]
        assert rewards == pytest.approx([0.472, 0.708], abs=1e-6)

    def test_neutral_capture_base_reward_still_splits_by_ship_share(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        prev["planets"][2][1] = -1
        prev["planets"][2][6] = 3
        curr["planets"][2][1] = 0
        curr["planets"][2][6] = 3
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2)
        tracker.register_capture_launch(
            buffer_idx=1,
            launch_turn=10,
            player=0,
            target_planet_id=2,
            ships_sent=20,
            est_arrival_turn=100,
            launch_target_owner=-1,
        )
        tracker.register_capture_launch(
            buffer_idx=2,
            launch_turn=90,
            player=0,
            target_planet_id=2,
            ships_sent=20,
            est_arrival_turn=100,
            launch_target_owner=-1,
        )
        tracker.observe_turn(100, prev, curr, 0)
        resolved = list(tracker.pop_capture_resolved(0))
        assert [idx for idx, _ in resolved] == [1, 2]
        rewards = [compute_production_capture_reward(outcome) for _, outcome in resolved]
        total_reward = (3.0 * (500 - 100) - 20.0) * 0.001
        assert rewards == pytest.approx([total_reward * 0.5, total_reward * 0.5], abs=1e-6)

    def test_delayed_discount_makes_recent_equal_capture_launch_more_valuable(self):
        buf = RolloutBuffer()
        old_idx = buf.add(_dummy_transition(reward=0.0))
        recent = _dummy_transition(reward=0.0)
        recent.turn = 90
        recent_idx = buf.add(recent)
        base_reward = 600.0
        buf.add_delayed_reward(old_idx, base_reward, event_turn=100, discount=0.99)
        buf.add_delayed_reward(recent_idx, base_reward, event_turn=100, discount=0.99)
        assert buf.data[recent_idx].reward > buf.data[old_idx].reward
        assert buf.data[old_idx].reward == pytest.approx(base_reward * (0.99 ** 100), abs=1e-6)
        assert buf.data[recent_idx].reward == pytest.approx(base_reward * (0.99 ** 10), abs=1e-6)

    def test_enemy_capture_still_splits_by_ship_share_not_recency(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        prev["planets"][2][1] = 1
        prev["planets"][2][6] = 3
        curr["planets"][2][1] = 0
        curr["planets"][2][6] = 3
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2)
        tracker.register_capture_launch(
            buffer_idx=1,
            launch_turn=10,
            player=0,
            target_planet_id=2,
            ships_sent=20,
            est_arrival_turn=100,
            launch_target_owner=1,
        )
        tracker.register_capture_launch(
            buffer_idx=2,
            launch_turn=90,
            player=0,
            target_planet_id=2,
            ships_sent=20,
            est_arrival_turn=100,
            launch_target_owner=1,
        )
        tracker.observe_turn(100, prev, curr, 0)
        resolved = list(tracker.pop_capture_resolved(0))
        rewards = [compute_production_capture_reward(outcome) for _, outcome in resolved]
        total_reward = (3.0 * (500 - 100) * 1.5 - 20.0) * 0.001
        assert rewards == pytest.approx([total_reward * 0.5, total_reward * 0.5], abs=1e-6)

    def test_failed_arrived_launch_outside_window_does_not_get_later_credit(self):
        """Fleet 1 arrives at turn 50 and doesn't capture. Fleet 2 captures
        at turn 100 (50 turns later). With default softening_window=30, Fleet
        1 is out-of-window so ONLY Fleet 2 gets credit.
        """
        prev = _toy_env_obs(player=0)
        mid = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        prev["planets"][2][1] = -1
        mid["planets"][2][1] = -1
        curr["planets"][2][1] = 0
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2)
        tracker.register_capture_launch(
            buffer_idx=1,
            launch_turn=10,
            player=0,
            target_planet_id=2,
            ships_sent=20,
            est_arrival_turn=50,
            launch_target_owner=-1,
        )
        tracker.observe_turn(50, prev, mid, 0)
        assert list(tracker.pop_capture_resolved(0)) == []
        tracker.register_capture_launch(
            buffer_idx=2,
            launch_turn=60,
            player=0,
            target_planet_id=2,
            ships_sent=30,
            est_arrival_turn=100,
            launch_target_owner=-1,
        )
        tracker.observe_turn(100, mid, curr, 0)
        resolved = list(tracker.pop_capture_resolved(0))
        assert [idx for idx, _ in resolved] == [2]
        rewards = [compute_production_capture_reward(outcome) for _, outcome in resolved]
        # Scaled: (production(=1 default) * 400 - 20) * 1.0 * 0.001 = 0.38
        assert rewards == pytest.approx([0.38], abs=1e-6)

    def test_softening_fleet_within_window_shares_capture_credit(self):
        """Fleet 1 arrives at turn 50 and softens the target (no capture).
        Fleet 2 captures at turn 70 (20 turns later). With default
        softening_window=30, BOTH fleets share the capture reward
        proportional to their ships_sent.
        """
        prev = _toy_env_obs(player=0)
        mid = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        prev["planets"][2][1] = -1
        prev["planets"][2][6] = 2
        mid["planets"][2][1] = -1
        mid["planets"][2][6] = 2
        curr["planets"][2][1] = 0
        curr["planets"][2][6] = 2
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2)
        # Fleet 1: softening launch, arrives turn 50
        tracker.register_capture_launch(
            buffer_idx=1,
            launch_turn=10,
            player=0,
            target_planet_id=2,
            ships_sent=20,
            est_arrival_turn=50,
            launch_target_owner=-1,
        )
        tracker.observe_turn(50, prev, mid, 0)
        assert list(tracker.pop_capture_resolved(0)) == []
        # Fleet 2: capturing launch, arrives turn 70
        tracker.register_capture_launch(
            buffer_idx=2,
            launch_turn=60,
            player=0,
            target_planet_id=2,
            ships_sent=30,
            est_arrival_turn=70,
            launch_target_owner=-1,
        )
        tracker.observe_turn(70, mid, curr, 0)
        resolved = list(tracker.pop_capture_resolved(0))
        # Both Fleet 1 and Fleet 2 are credited, proportional to ships_sent.
        # Fleet 1 share = 20/50 = 0.4, Fleet 2 share = 30/50 = 0.6.
        assert [idx for idx, _ in resolved] == [1, 2]
        rewards = [compute_production_capture_reward(outcome) for _, outcome in resolved]
        # (production(=2) * 430 - 20) * 1.0 * 0.001 = 0.84
        total = (2.0 * (500 - 70) - 20.0) * 0.001
        assert rewards == pytest.approx([total * 0.4, total * 0.6], abs=1e-6)

    def test_softening_fleet_outside_window_gets_no_credit(self):
        """Edge case: Fleet 1 arrives exactly `softening_window + 1` turns
        before capture. It should NOT be credited.
        """
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        prev["planets"][2][1] = -1
        curr["planets"][2][1] = 0
        tracker = ProductionRewardTracker(
            max_turns=500, num_agents=2, softening_window=10,
        )
        tracker.register_capture_launch(
            buffer_idx=1,
            launch_turn=0,
            player=0,
            target_planet_id=2,
            ships_sent=20,
            est_arrival_turn=50,
            launch_target_owner=-1,
        )
        tracker.register_capture_launch(
            buffer_idx=2,
            launch_turn=50,
            player=0,
            target_planet_id=2,
            ships_sent=30,
            est_arrival_turn=61,  # 11 turns after Fleet 1 arrival, capture here
            launch_target_owner=-1,
        )
        tracker.observe_turn(61, prev, curr, 0)
        resolved = list(tracker.pop_capture_resolved(0))
        # Only Fleet 2 should be in the resolved list.
        assert [idx for idx, _ in resolved] == [2]

    def test_softening_window_configurable(self):
        """Custom softening_window should be respected."""
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        prev["planets"][2][1] = -1
        curr["planets"][2][1] = 0
        tracker = ProductionRewardTracker(
            max_turns=500, num_agents=2, softening_window=5,
        )
        tracker.register_capture_launch(
            buffer_idx=1,
            launch_turn=0,
            player=0,
            target_planet_id=2,
            ships_sent=20,
            est_arrival_turn=50,
            launch_target_owner=-1,
        )
        tracker.register_capture_launch(
            buffer_idx=2,
            launch_turn=50,
            player=0,
            target_planet_id=2,
            ships_sent=30,
            est_arrival_turn=55,  # exactly 5 turns (within window)
            launch_target_owner=-1,
        )
        tracker.observe_turn(55, prev, curr, 0)
        resolved = list(tracker.pop_capture_resolved(0))
        assert [idx for idx, _ in resolved] == [1, 2]

    def test_source_defense_deficit_penalty_uses_incoming_minus_remaining(self):
        penalty = _compute_source_defense_deficit_penalty(
            enemy_incoming=60.0,
            ships_remaining_after_launch=50,
        )
        assert penalty == pytest.approx(-10.0, abs=1e-6)

    def test_source_defense_deficit_penalty_zero_when_remaining_covers_incoming(self):
        penalty = _compute_source_defense_deficit_penalty(
            enemy_incoming=60.0,
            ships_remaining_after_launch=60,
        )
        assert penalty == pytest.approx(0.0, abs=1e-6)

    def test_source_defense_deficit_penalty_respects_weight(self):
        penalty = _compute_source_defense_deficit_penalty(
            enemy_incoming=60.0,
            ships_remaining_after_launch=50,
            weight=0.5,
        )
        assert penalty == pytest.approx(-5.0, abs=1e-6)


class TestSourceDefenseShortfall:
    """Per-fleet ETA simulation: garrison evolves with production, each
    incoming enemy fleet attacks the CURRENT garrison (not the final
    total). Distant threats are auto-absorbed if production catches up
    in time; only overwhelming arrivals contribute to the shortfall."""

    def test_no_enemy_arrivals_returns_zero(self):
        assert _compute_source_defense_shortfall(
            ships_remaining_after_launch=0,
            production=2.0,
            enemy_arrivals=[],
        ) == pytest.approx(0.0, abs=1e-9)

    def test_distant_threat_covered_by_production_contributes_zero(self):
        # 3 ships remaining, production 2/turn, enemy 30 ships at ETA 80.
        # By t=80 garrison = 3 + 2*80 = 163 >> 30. Zero shortfall.
        assert _compute_source_defense_shortfall(
            ships_remaining_after_launch=3,
            production=2.0,
            enemy_arrivals=[(80.0, 30.0)],
        ) == pytest.approx(0.0, abs=1e-9)

    def test_imminent_threat_exceeding_garrison_reports_exact_shortfall(self):
        # 3 ships remaining, production 2/turn, enemy 30 ships at ETA 5.
        # At t=5: garrison = 3 + 2*5 = 13. Deficit = 30 - 13 = 17.
        assert _compute_source_defense_shortfall(
            ships_remaining_after_launch=3,
            production=2.0,
            enemy_arrivals=[(5.0, 30.0)],
        ) == pytest.approx(17.0, abs=1e-6)

    def test_user_example_two_sequential_attacks(self):
        # User's concrete example:
        #   ships_remaining=3, production=2
        #   Attack 1: 20 ships @ ETA 10
        #     garrison = 3 + 2*10 = 23, deficit = max(0, 20-23) = 0
        #     garrison after = 23 - 20 = 3
        #   Attack 2: 100 ships @ ETA 35 (25 turns later)
        #     garrison = 3 + 2*25 = 53, deficit = max(0, 100-53) = 47
        #     garrison after = 0
        # Total shortfall = 47.
        shortfall = _compute_source_defense_shortfall(
            ships_remaining_after_launch=3,
            production=2.0,
            enemy_arrivals=[(10.0, 20.0), (35.0, 100.0)],
        )
        assert shortfall == pytest.approx(47.0, abs=1e-6)

    def test_arrivals_sorted_internally_independent_of_input_order(self):
        unsorted_first = _compute_source_defense_shortfall(
            ships_remaining_after_launch=3,
            production=2.0,
            enemy_arrivals=[(35.0, 100.0), (10.0, 20.0)],
        )
        sorted_first = _compute_source_defense_shortfall(
            ships_remaining_after_launch=3,
            production=2.0,
            enemy_arrivals=[(10.0, 20.0), (35.0, 100.0)],
        )
        assert unsorted_first == pytest.approx(sorted_first, abs=1e-9)

    def test_self_reinforcement_increases_garrison_before_attack(self):
        # Enemy 50 ships @ ETA 20. Without help: garrison = 5 + 2*20 = 45,
        # shortfall = 50-45 = 5.
        # With self reinforcement 30 ships @ ETA 15: garrison at t=15 is
        # 5 + 2*15 = 35 -> +30 reinforcement = 65. At t=20 garrison is
        # 65 + 2*5 = 75 > 50 -> zero shortfall.
        without_help = _compute_source_defense_shortfall(
            ships_remaining_after_launch=5,
            production=2.0,
            enemy_arrivals=[(20.0, 50.0)],
        )
        with_help = _compute_source_defense_shortfall(
            ships_remaining_after_launch=5,
            production=2.0,
            enemy_arrivals=[(20.0, 50.0)],
            self_arrivals=[(15.0, 30.0)],
        )
        assert without_help == pytest.approx(5.0, abs=1e-6)
        assert with_help == pytest.approx(0.0, abs=1e-9)

    def test_zero_production_planet_only_uses_initial_garrison(self):
        # Comet-like: no production. 10 ships remaining, enemy 15 @ ETA 50.
        # Garrison never grows -> shortfall = 15 - 10 = 5.
        assert _compute_source_defense_shortfall(
            ships_remaining_after_launch=10,
            production=0.0,
            enemy_arrivals=[(50.0, 15.0)],
        ) == pytest.approx(5.0, abs=1e-6)

    def test_multiple_small_enemies_chain_deplete_garrison(self):
        # 10 ships remaining, production 1/turn, three small attacks at
        # ETAs 5, 10, 15 (each 8 ships).
        #   t=5 : garrison = 10 + 5 = 15, combat: 8 vs 15 -> 7 left, no deficit
        #   t=10: garrison = 7 + 5  = 12, combat: 8 vs 12 -> 4 left, no deficit
        #   t=15: garrison = 4 + 5  = 9,  combat: 8 vs 9  -> 1 left, no deficit
        # Total shortfall = 0 (every attack is absorbed).
        assert _compute_source_defense_shortfall(
            ships_remaining_after_launch=10,
            production=1.0,
            enemy_arrivals=[(5.0, 8.0), (10.0, 8.0), (15.0, 8.0)],
        ) == pytest.approx(0.0, abs=1e-9)


class TestPostCaptureReinforcement:
    """Sorun 1: a fleet launched at a non-mine target that is captured
    by an earlier-arriving friendly fleet should still receive an
    outcome-tied reward when it eventually arrives at the now-mine
    planet."""

    def _capture_neutral_p2_at_t10(self):
        """Helper: prev has P2 neutral, curr has P2 owned by player 0."""
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        prev["planets"][2][1] = -1   # P2 neutral
        prev["planets"][2][6] = 3    # P2 production
        curr["planets"][2][1] = 0    # P2 mine after capture
        curr["planets"][2][6] = 3
        return prev, curr

    def test_in_flight_launch_moves_to_reinforce_pending_at_capture(self):
        """F1 arrives at T10 and captures. F2 still in flight (arrives
        T15). After observe_turn(10), F2 must be queued in
        _reinforce_pending, NOT silently dropped."""
        prev, curr = self._capture_neutral_p2_at_t10()
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2)
        # F1: arrives at T10 (capture launch)
        tracker.register_capture_launch(
            buffer_idx=1, launch_turn=4, player=0,
            target_planet_id=2, ships_sent=30,
            est_arrival_turn=10, launch_target_owner=-1,
        )
        # F2: still in flight, arrives T15
        tracker.register_capture_launch(
            buffer_idx=2, launch_turn=4, player=0,
            target_planet_id=2, ships_sent=10,
            est_arrival_turn=15, launch_target_owner=-1,
        )
        tracker.observe_turn(10, prev, curr, 0)
        # F1 captured; F2 must still be tracked as a reinforcement.
        captures = list(tracker.pop_capture_resolved(0))
        assert [idx for idx, _ in captures] == [1]
        reinforcements = list(tracker.pop_reinforcement_resolved(0))
        assert reinforcements == []  # not yet arrived
        # Internal state: F2 is in _reinforce_pending.
        assert (0, 2) in tracker._reinforce_pending
        assert len(tracker._reinforce_pending[(0, 2)]) == 1
        assert tracker._reinforce_pending[(0, 2)][0].buffer_idx == 2

    def test_arrival_at_still_mine_planet_emits_reinforcement_reward(self):
        """F2 arrives at T15 to a still-mine P2 -> small + reward."""
        prev, curr = self._capture_neutral_p2_at_t10()
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2)
        tracker.register_capture_launch(
            buffer_idx=1, launch_turn=4, player=0,
            target_planet_id=2, ships_sent=30,
            est_arrival_turn=10, launch_target_owner=-1,
        )
        tracker.register_capture_launch(
            buffer_idx=2, launch_turn=4, player=0,
            target_planet_id=2, ships_sent=10,
            est_arrival_turn=15, launch_target_owner=-1,
        )
        tracker.observe_turn(10, prev, curr, 0)
        list(tracker.pop_capture_resolved(0))  # drain
        # Now T15: P2 still mine, F2 arrives.
        prev2 = _toy_env_obs(player=0)
        prev2["planets"][2][1] = 0
        prev2["planets"][2][6] = 3
        curr2 = _toy_env_obs(player=0)
        curr2["planets"][2][1] = 0
        curr2["planets"][2][6] = 3
        tracker.observe_turn(15, prev2, curr2, 0)
        reinforcements = list(tracker.pop_reinforcement_resolved(0))
        assert [idx for idx, _ in reinforcements] == [2]
        outcome = reinforcements[0][1]
        assert outcome.ships_sent == 10
        assert outcome.planet_production == pytest.approx(3.0)
        assert outcome.arrival_turn == 15
        assert outcome.capture_turn == 10
        # Reward: 10 * 3 * (1 - 5/30) * 0.1 * 0.001 = ~0.0025
        reward = compute_production_reinforcement_reward(
            outcome,
            reinforcement_weight=0.1,
            reward_scale=0.001,
            decay_window=30,
        )
        expected = 10.0 * 3.0 * (1.0 - 5.0 / 30.0) * 0.1 * 0.001
        assert reward == pytest.approx(expected, abs=1e-6)
        # And it must be positive.
        assert reward > 0.0

    def test_arrival_at_lost_planet_emits_no_reinforcement_reward(self):
        """F2 arrives at T15 but P2 was lost to enemy at T13. No bonus."""
        prev, curr = self._capture_neutral_p2_at_t10()
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2)
        tracker.register_capture_launch(
            buffer_idx=1, launch_turn=4, player=0,
            target_planet_id=2, ships_sent=30,
            est_arrival_turn=10, launch_target_owner=-1,
        )
        tracker.register_capture_launch(
            buffer_idx=2, launch_turn=4, player=0,
            target_planet_id=2, ships_sent=10,
            est_arrival_turn=15, launch_target_owner=-1,
        )
        tracker.observe_turn(10, prev, curr, 0)
        list(tracker.pop_capture_resolved(0))
        # Drain any clawback pending events from the capture so the loss
        # observed below doesn't mix with this test's assertions.
        list(tracker.pop_capture_clawback_resolved(0))
        # T15: P2 owned by enemy now (lost in between).
        prev2 = _toy_env_obs(player=0)
        prev2["planets"][2][1] = 1   # enemy
        prev2["planets"][2][6] = 3
        curr2 = _toy_env_obs(player=0)
        curr2["planets"][2][1] = 1
        curr2["planets"][2][6] = 3
        tracker.observe_turn(15, prev2, curr2, 0)
        reinforcements = list(tracker.pop_reinforcement_resolved(0))
        assert reinforcements == []  # no reinforcement reward when not mine

    def test_reinforcement_decay_floors_at_0_1(self):
        """Beyond decay_window the reward is 0; at the boundary it floors
        at 10% of the base magnitude."""
        outcome = ProductionReinforcementOutcome(
            ships_sent=10, planet_production=3.0,
            arrival_turn=40, capture_turn=10,  # gap=30 == window
            max_turns=500,
        )
        # gap exactly == decay_window: time_factor = max(0.1, 1 - 30/30) = 0.1
        reward = compute_production_reinforcement_reward(
            outcome,
            reinforcement_weight=0.1, reward_scale=0.001, decay_window=30,
        )
        assert reward == pytest.approx(10 * 3 * 0.1 * 0.1 * 0.001, abs=1e-9)

        # gap > decay_window: 0
        outcome.arrival_turn = 41
        reward = compute_production_reinforcement_reward(
            outcome,
            reinforcement_weight=0.1, reward_scale=0.001, decay_window=30,
        )
        assert reward == pytest.approx(0.0, abs=1e-9)

    def test_self_reinforce_launch_target_is_filtered_at_register(self):
        """A launch toward a target ALREADY mine at launch time is
        filtered by `register_capture_launch` (existing behavior). The
        reinforcement path must NOT receive such launches."""
        tracker = ProductionRewardTracker(max_turns=500, num_agents=2)
        tracker.register_capture_launch(
            buffer_idx=99, launch_turn=4, player=0,
            target_planet_id=2, ships_sent=10,
            est_arrival_turn=15, launch_target_owner=0,  # mine at launch
        )
        assert tracker._capture_pending == {}
        # Even after observe_turn, no reinforcement record appears.
        prev, curr = self._capture_neutral_p2_at_t10()
        tracker.observe_turn(10, prev, curr, 0)
        assert list(tracker.pop_reinforcement_resolved(0)) == []


class TestCaptureClawback:
    """Sorun 2: a captured planet lost within the clawback window
    triggers a partial refund of the original capture reward."""

    def _scenario_capture_then_lose(self, *, capture_turn: int = 10,
                                     loss_turn: int = 13):
        """Build prev/curr observations for capture at `capture_turn`
        and a separate prev/curr for loss at `loss_turn`."""
        # T(capture-1) -> T(capture): neutral -> mine
        prev_cap = _toy_env_obs(player=0)
        curr_cap = _toy_env_obs(player=0)
        prev_cap["planets"][2][1] = -1
        prev_cap["planets"][2][6] = 3
        curr_cap["planets"][2][1] = 0
        curr_cap["planets"][2][6] = 3
        # T(loss-1) -> T(loss): mine -> enemy
        prev_loss = _toy_env_obs(player=0)
        curr_loss = _toy_env_obs(player=0)
        prev_loss["planets"][2][1] = 0
        prev_loss["planets"][2][6] = 3
        curr_loss["planets"][2][1] = 1
        curr_loss["planets"][2][6] = 3
        return prev_cap, curr_cap, prev_loss, curr_loss

    def test_capture_then_lose_emits_negative_clawback(self):
        """Capture at T10, lose at T13 (gap=3, well inside window=30):
        original capturing transitions get a clawback (negative) reward."""
        prev_cap, curr_cap, prev_loss, curr_loss = self._scenario_capture_then_lose(
            capture_turn=10, loss_turn=13,
        )
        tracker = ProductionRewardTracker(
            max_turns=500, num_agents=2,
            capture_clawback_window=30, capture_clawback_fraction=0.5,
        )
        tracker.register_capture_launch(
            buffer_idx=1, launch_turn=4, player=0,
            target_planet_id=2, ships_sent=30,
            est_arrival_turn=10, launch_target_owner=-1,
        )
        tracker.observe_turn(10, prev_cap, curr_cap, 0)
        captures = list(tracker.pop_capture_resolved(0))
        assert [idx for idx, _ in captures] == [1]
        capture_reward = compute_production_capture_reward(
            captures[0][1], reward_scale=0.001,
        )
        assert capture_reward > 0.0

        # T13: planet lost.
        tracker.observe_turn(13, prev_loss, curr_loss, 0)
        clawbacks = list(tracker.pop_capture_clawback_resolved(0))
        assert [idx for idx, _ in clawbacks] == [1]
        clawback_outcome = clawbacks[0][1]
        clawback_reward = compute_production_capture_clawback(
            clawback_outcome, reward_scale=0.001,
        )
        # gap = 13 - 10 = 3, time_factor = max(0.1, 1 - 2/30) = 28/30
        # clawback_reward = -capture_reward * 0.5 * 28/30
        expected = -capture_reward * 0.5 * (1.0 - 2.0 / 30.0)
        assert clawback_reward == pytest.approx(expected, abs=1e-9)
        # And clawback strictly less than capture (net positive).
        assert capture_reward + clawback_reward > 0.0

    def test_loss_outside_clawback_window_emits_no_clawback(self):
        """Capture at T10, lose at T100 (gap=90, beyond window=30): no
        clawback emitted because the recent-capture record has expired."""
        prev_cap, curr_cap, prev_loss, curr_loss = self._scenario_capture_then_lose(
            capture_turn=10, loss_turn=100,
        )
        tracker = ProductionRewardTracker(
            max_turns=500, num_agents=2,
            capture_clawback_window=30, capture_clawback_fraction=0.5,
        )
        tracker.register_capture_launch(
            buffer_idx=1, launch_turn=4, player=0,
            target_planet_id=2, ships_sent=30,
            est_arrival_turn=10, launch_target_owner=-1,
        )
        tracker.observe_turn(10, prev_cap, curr_cap, 0)
        list(tracker.pop_capture_resolved(0))
        # Walk forward via observe_turn (no transitions) to expire the
        # recent-capture record.
        unchanged = _toy_env_obs(player=0)
        unchanged["planets"][2][1] = 0
        unchanged["planets"][2][6] = 3
        tracker.observe_turn(45, unchanged, unchanged, 0)  # > 10 + 30
        assert (0, 2) not in tracker._recent_captures
        # Now lose at T100 - no clawback record exists.
        tracker.observe_turn(100, prev_loss, curr_loss, 0)
        assert list(tracker.pop_capture_clawback_resolved(0)) == []

    def test_loss_without_recent_capture_emits_no_clawback(self):
        """A planet I owned all along (no recent capture record) loses;
        no clawback emitted (Defense Window handles this case)."""
        prev_cap = _toy_env_obs(player=0)
        prev_cap["planets"][2][1] = 0  # already mine
        prev_cap["planets"][2][6] = 3
        curr_cap = _toy_env_obs(player=0)
        curr_cap["planets"][2][1] = 0
        curr_cap["planets"][2][6] = 3
        prev_loss = _toy_env_obs(player=0)
        prev_loss["planets"][2][1] = 0
        prev_loss["planets"][2][6] = 3
        curr_loss = _toy_env_obs(player=0)
        curr_loss["planets"][2][1] = 1
        curr_loss["planets"][2][6] = 3
        tracker = ProductionRewardTracker(
            max_turns=500, num_agents=2,
            capture_clawback_window=30, capture_clawback_fraction=0.5,
        )
        tracker.observe_turn(10, prev_cap, curr_cap, 0)
        tracker.observe_turn(13, prev_loss, curr_loss, 0)
        assert list(tracker.pop_capture_clawback_resolved(0)) == []

    def test_clawback_disabled_when_fraction_or_window_zero(self):
        """Setting fraction=0 or window=0 disables clawback emission."""
        outcome = ProductionCaptureOutcome(
            ships_sent=30, total_ships_sent=30, total_ship_investment=0.0,
            planet_production=3.0, capture_turn=10, max_turns=500,
            captured_from_enemy=False, num_opponents=1,
        )
        clawback_outcome = ProductionCaptureClawbackOutcome(
            capture_outcome=outcome, loss_turn=13,
            clawback_window=0, clawback_fraction=0.5,
        )
        # Window=0 -> reward=0. (window is clamped to >=1, gap=3 > 1)
        reward = compute_production_capture_clawback(
            clawback_outcome, reward_scale=0.001,
        )
        assert reward == pytest.approx(0.0, abs=1e-9)


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
    def test_reward_factory_current_matches_default_config(self):
        cfg = build_reward_config("current")
        default = RewardConfig()
        assert "current" in REWARD_SYSTEMS
        assert cfg.capture_value_weight == pytest.approx(default.capture_value_weight)
        assert cfg.denial_value_weight == pytest.approx(default.denial_value_weight)
        assert cfg.fleet_capture_weight == pytest.approx(default.fleet_capture_weight)

    def test_reward_factory_aggressive_is_separate_preset(self):
        cfg = build_reward_config("aggressive")
        default = RewardConfig()
        assert "aggressive" in REWARD_SYSTEMS
        assert cfg.denial_value_weight > default.denial_value_weight
        assert cfg.enemy_capture_radius_weight > default.enemy_capture_radius_weight
        assert cfg.planet_loss_weight > default.planet_loss_weight
        assert cfg.fleet_kill_weight == pytest.approx(default.fleet_kill_weight)

    def test_reward_factory_production_simple_is_separate_preset(self):
        cfg = build_reward_config("production_simple")
        assert "production_simple" in REWARD_SYSTEMS
        assert cfg.capture_value_weight == pytest.approx(0.0)
        assert cfg.denial_value_weight == pytest.approx(0.0)
        assert cfg.fleet_kill_weight == pytest.approx(1.0)
        assert cfg.fleet_waste_weight == pytest.approx(1.0)
        # Base airborne weight is unnormalized; rollout applies a risk
        # multiplier when source is under enemy attack.
        assert cfg.fleet_airborne_weight == pytest.approx(1.0)
        assert cfg.reward_scale == pytest.approx(0.001)
        assert cfg.draw_terminal_reward == pytest.approx(-1.0)
        assert cfg.delayed_reward_discount == pytest.approx(1.0)

    def test_reward_factory_overrides_preset_values(self):
        cfg = build_reward_config(
            "aggressive",
            denial_value_weight=0.123,
            fleet_capture_weight=None,
        )
        aggressive = build_reward_config("aggressive")
        assert cfg.denial_value_weight == pytest.approx(0.123)
        assert cfg.fleet_capture_weight == pytest.approx(aggressive.fleet_capture_weight)

    def test_reward_factory_rejects_unknown_system(self):
        with pytest.raises(ValueError):
            build_reward_config("unknown")

    def test_pure_terminal_reward(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        cfg = RewardConfig(
            capture_value_weight=0.0,
            denial_value_weight=0.0,
            planet_diff_weight=0.0,
            ship_diff_weight=0.0,
        )
        r = compute_step_reward(prev, curr, 0, terminal=True,
                                env_terminal_reward=+1, cfg=cfg)
        assert r == 1.0

    def test_terminal_reward_normalization_zeroes_draws(self):
        assert normalize_terminal_rewards([1.0, 1.0]) == [0.0, 0.0]
        assert normalize_terminal_rewards([0.0, 0.0]) == [0.0, 0.0]
        assert normalize_terminal_rewards([1.0, 0.0]) == [1.0, -1.0]
        assert normalize_terminal_rewards([0.0, 1.0]) == [-1.0, 1.0]

    def test_terminal_reward_normalization_can_penalize_draws(self):
        assert normalize_terminal_rewards([1.0, 1.0], draw_value=-1.0) == [-1.0, -1.0]
        assert normalize_terminal_rewards([0.0, 0.0], draw_value=-1.0) == [-1.0, -1.0]

    def test_zero_when_no_change_no_terminal(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=RewardConfig())
        # Same state -> shaping = 0.
        assert pytest.approx(r, abs=1e-6) == 0.0

    def test_ship_loss_penalty_is_negative(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        curr["planets"][0][5] -= 10
        cfg = RewardConfig(
            capture_value_weight=0.0,
            denial_value_weight=0.0,
            planet_diff_weight=0.0,
            ship_diff_weight=0.0,
            ship_loss_weight=0.01,
        )
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=cfg)
        assert r == pytest.approx(-0.1, abs=1e-6)

    def test_invalid_action_penalty_is_negative(self):
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        cfg = RewardConfig(
            capture_value_weight=0.0,
            denial_value_weight=0.0,
            planet_diff_weight=0.0,
            ship_diff_weight=0.0,
            ship_loss_weight=0.0,
            invalid_action_penalty=0.25,
        )
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=cfg,
                                invalid_actions=2)
        assert r == pytest.approx(-0.5, abs=1e-6)

    def test_legacy_planet_diff_increases_reward(self):
        """Old un-weighted planet count differential still works when enabled."""
        prev = _toy_env_obs(player=0)
        curr = _toy_env_obs(player=0)
        curr["planets"][2][1] = 0  # neutral -> mine
        cfg = RewardConfig(
            planet_diff_weight=0.1,
            ship_diff_weight=0.0,
            capture_value_weight=0.0,
            denial_value_weight=0.0,
        )
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=cfg)
        assert r > 0


class TestCaptureValueShaping:
    """Time-decayed, production-weighted capture reward."""

    @staticmethod
    def _two_planet_world(turn: int = 0, *, captured: bool = False, value: int = 5):
        """Tiny world with one neutral planet that may be captured by P0."""
        owner = 0 if captured else -1
        return {
            "planets": [
                # [id, owner, x, y, radius, ships, production]
                [0, 0, 70.0, 70.0, 3.0, 100, 3],
                [1, owner, 30.0, 30.0, 1 + 1.6, 50, value],
            ],
            "fleets": [],
            "initial_planets": [
                [0, 0, 70.0, 70.0, 3.0, 100, 3],
                [1, -1, 30.0, 30.0, 1 + 1.6, 50, value],
            ],
            "comet_planet_ids": [],
            "comets": [],
            "angular_velocity": 0.05,
            "step": turn,
            "player": 0,
        }

    def test_neutral_capture_positive_and_production_weighted(self):
        prev = self._two_planet_world(turn=0, captured=False, value=5)
        curr = self._two_planet_world(turn=0, captured=True, value=5)
        cfg = RewardConfig(
            capture_value_weight=1.0, denial_value_weight=1.0,
            max_turns=100, ship_diff_weight=0.0, planet_diff_weight=0.0,
        )
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=cfg)
        # turn=0 -> time_factor = 1.0; gain = production = 5; denial = 0 (neutral)
        assert r == pytest.approx(5.0, abs=1e-5)

    def test_higher_production_yields_higher_reward(self):
        cfg = RewardConfig(capture_value_weight=1.0, denial_value_weight=1.0,
                           max_turns=100, ship_diff_weight=0.0, planet_diff_weight=0.0)
        r_small = compute_step_reward(
            self._two_planet_world(turn=0, captured=False, value=1),
            self._two_planet_world(turn=0, captured=True, value=1),
            0, terminal=False, env_terminal_reward=0, cfg=cfg,
        )
        r_big = compute_step_reward(
            self._two_planet_world(turn=0, captured=False, value=5),
            self._two_planet_world(turn=0, captured=True, value=5),
            0, terminal=False, env_terminal_reward=0, cfg=cfg,
        )
        assert r_big > r_small > 0

    def test_early_capture_worth_more_than_late(self):
        cfg = RewardConfig(capture_value_weight=1.0, denial_value_weight=1.0,
                           max_turns=100, ship_diff_weight=0.0, planet_diff_weight=0.0)
        r_early = compute_step_reward(
            self._two_planet_world(turn=10, captured=False, value=3),
            self._two_planet_world(turn=10, captured=True, value=3),
            0, terminal=False, env_terminal_reward=0, cfg=cfg,
        )
        r_late = compute_step_reward(
            self._two_planet_world(turn=90, captured=False, value=3),
            self._two_planet_world(turn=90, captured=True, value=3),
            0, terminal=False, env_terminal_reward=0, cfg=cfg,
        )
        # turn=10 -> factor 0.9; turn=90 -> factor 0.1; ratio 9:1
        assert r_early > r_late > 0
        assert r_early / r_late == pytest.approx(9.0, rel=1e-3)

    def test_steal_from_opponent_double_swing(self):
        """Capturing FROM the opponent should be worth 2x capturing neutral."""
        # neutral -> mine
        prev_n = self._two_planet_world(turn=0, captured=False, value=4)
        curr_n = self._two_planet_world(turn=0, captured=True, value=4)
        # opponent -> mine
        prev_o = dict(self._two_planet_world(turn=0, captured=False, value=4))
        prev_o["planets"] = [list(p) for p in prev_o["planets"]]
        prev_o["planets"][1][1] = 1  # opp owns
        curr_o = self._two_planet_world(turn=0, captured=True, value=4)

        cfg = RewardConfig(capture_value_weight=1.0, denial_value_weight=1.0,
                           max_turns=100, ship_diff_weight=0.0, planet_diff_weight=0.0)
        r_neutral = compute_step_reward(prev_n, curr_n, 0, terminal=False,
                                        env_terminal_reward=0, cfg=cfg)
        r_steal = compute_step_reward(prev_o, curr_o, 0, terminal=False,
                                      env_terminal_reward=0, cfg=cfg)
        # capture=denial=1: gain (+4) + denial (+4) vs neutral gain (+4) -> 2x
        assert r_steal == pytest.approx(2.0 * r_neutral, rel=1e-5)

    def test_losing_planet_gives_negative_reward(self):
        """Losing a planet to opponent should be the symmetric negative."""
        prev = self._two_planet_world(turn=0, captured=True, value=3)
        curr = dict(self._two_planet_world(turn=0, captured=True, value=3))
        curr["planets"] = [list(p) for p in curr["planets"]]
        curr["planets"][1][1] = 1   # mine -> opp (steal)
        cfg = RewardConfig(
            capture_value_weight=1.0,
            denial_value_weight=1.0,
            max_turns=100,
            ship_diff_weight=0.0,
            ship_loss_weight=0.0,
            planet_diff_weight=0.0,
        )
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=cfg)
        # gain = -3 (mine lost), denial = -3 (opp gained 3); both weight 1 -> -6
        assert r == pytest.approx(-6.0, abs=1e-5)

    def test_radius_kind_uses_radius_not_production(self):
        prev = self._two_planet_world(turn=0, captured=False, value=5)
        curr = self._two_planet_world(turn=0, captured=True, value=5)
        cfg = RewardConfig(
            capture_value_weight=1.0, denial_value_weight=1.0,
            max_turns=100, capture_value_kind="radius",
            ship_diff_weight=0.0, planet_diff_weight=0.0,
        )
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=cfg)
        # planet[1].radius = 1 + 1.6 = 2.6 (neutral capture: gain only)
        assert r == pytest.approx(2.6, abs=1e-5)

    def test_time_decay_floor_keeps_minimum_signal(self):
        prev = self._two_planet_world(turn=200, captured=False, value=3)
        curr = self._two_planet_world(turn=200, captured=True, value=3)
        cfg = RewardConfig(
            capture_value_weight=1.0, denial_value_weight=1.0,
            max_turns=100, time_decay_floor=0.2,
            ship_diff_weight=0.0, planet_diff_weight=0.0,
        )
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=cfg)
        # turn=200 > max_turns -> raw factor would be 0; floor=0.2 -> reward 3*0.2=0.6
        assert r == pytest.approx(0.6, abs=1e-5)

    def test_no_change_no_capture_signal(self):
        prev = self._two_planet_world(turn=10, captured=False, value=3)
        curr = self._two_planet_world(turn=11, captured=False, value=3)
        cfg = RewardConfig(capture_value_weight=1.0, denial_value_weight=1.0,
                           max_turns=100, ship_diff_weight=0.0, planet_diff_weight=0.0)
        r = compute_step_reward(prev, curr, 0, terminal=False,
                                env_terminal_reward=0, cfg=cfg)
        assert r == pytest.approx(0.0, abs=1e-6)

    # ----- New: independent denial weight tests --------------------------

    def test_denial_weight_zero_means_only_my_growth(self):
        """With denial_weight=0, stealing == capturing neutral (gain only)."""
        prev_o = dict(self._two_planet_world(turn=0, captured=False, value=4))
        prev_o["planets"] = [list(p) for p in prev_o["planets"]]
        prev_o["planets"][1][1] = 1  # opp owns initially
        curr_o = self._two_planet_world(turn=0, captured=True, value=4)
        cfg = RewardConfig(capture_value_weight=1.0, denial_value_weight=0.0,
                           max_turns=100, ship_diff_weight=0.0, planet_diff_weight=0.0)
        r = compute_step_reward(prev_o, curr_o, 0, terminal=False,
                                env_terminal_reward=0, cfg=cfg)
        # Only mine grew by 4 -> reward = 4 (no extra denial bonus)
        assert r == pytest.approx(4.0, abs=1e-5)

    def test_aggressive_denial_amplifies_steal_reward(self):
        """denial_weight = 2 * capture_weight -> stealing 3x more than neutral."""
        prev_n = self._two_planet_world(turn=0, captured=False, value=4)
        curr_n = self._two_planet_world(turn=0, captured=True, value=4)
        prev_o = dict(self._two_planet_world(turn=0, captured=False, value=4))
        prev_o["planets"] = [list(p) for p in prev_o["planets"]]
        prev_o["planets"][1][1] = 1
        curr_o = self._two_planet_world(turn=0, captured=True, value=4)

        cfg = RewardConfig(capture_value_weight=1.0, denial_value_weight=2.0,
                           max_turns=100, ship_diff_weight=0.0, planet_diff_weight=0.0)
        r_neutral = compute_step_reward(prev_n, curr_n, 0, terminal=False,
                                        env_terminal_reward=0, cfg=cfg)
        r_steal = compute_step_reward(prev_o, curr_o, 0, terminal=False,
                                      env_terminal_reward=0, cfg=cfg)
        # neutral: gain*1*4 = 4. steal: gain*1*4 + denial*2*4 = 4 + 8 = 12.
        assert r_neutral == pytest.approx(4.0, abs=1e-5)
        assert r_steal == pytest.approx(12.0, abs=1e-5)
        assert r_steal / r_neutral == pytest.approx(3.0, rel=1e-5)

    def test_denial_only_when_my_planet_destroyed(self):
        """capture_weight=0, denial_weight=1: only opp's losses matter."""
        # neutral capture by ME -> only my gain, no denial -> r=0
        prev_n = self._two_planet_world(turn=0, captured=False, value=4)
        curr_n = self._two_planet_world(turn=0, captured=True, value=4)
        cfg = RewardConfig(capture_value_weight=0.0, denial_value_weight=1.0,
                           max_turns=100, ship_diff_weight=0.0, planet_diff_weight=0.0)
        r_neutral = compute_step_reward(prev_n, curr_n, 0, terminal=False,
                                        env_terminal_reward=0, cfg=cfg)
        assert r_neutral == pytest.approx(0.0, abs=1e-6)

        # steal from opp -> opp shrunk -> denial fires, r=4
        prev_o = dict(self._two_planet_world(turn=0, captured=False, value=4))
        prev_o["planets"] = [list(p) for p in prev_o["planets"]]
        prev_o["planets"][1][1] = 1
        curr_o = self._two_planet_world(turn=0, captured=True, value=4)
        r_steal = compute_step_reward(prev_o, curr_o, 0, terminal=False,
                                      env_terminal_reward=0, cfg=cfg)
        assert r_steal == pytest.approx(4.0, abs=1e-5)


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


# ---------------------------------------------------------------- Metrics

class TestMetrics:
    """Per-parameter training health metrics."""

    def _toy_policy_and_loss(self):
        """Build a loss that touches all logits + value -> all heads grad-aware."""
        torch.manual_seed(0)
        net = OrbitWarsPolicy(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        obs = _toy_obs(p_max=4, f_max=4)
        batch = pack_observations([obs])
        out = net(**batch)
        # Pull every head into the loss directly by summing all logits + value.
        loss = (out.src_logits.sum()
                + out.tgt_logits.sum()
                + out.ship_params.sum()
                + out.value.pow(2).sum())
        return net, loss

    def test_grad_norms_grouped_by_top_module(self):
        net, loss = self._toy_policy_and_loss()
        loss.backward()
        norms = grad_norms_by_module(net)
        # All these top-level groups should receive gradient.
        for expected in ("planet_proj", "global_proj",
                         "type_embed", "edge_proj", "blocks.0", "blocks.1",
                         "src_head", "tgt_query", "tgt_key", "ship_head",
                         "value_head"):
            assert expected in norms, f"missing group: {expected}"
            assert norms[expected] > 0, f"zero grad in {expected}"
        # Note: fleet_proj receives no grad in this toy obs because fleet
        # tokens are all padded (mask=False) and therefore never feed any
        # head's output. That's correct, not a bug.

    def test_weight_norms_match_manual(self):
        net, _ = self._toy_policy_and_loss()
        # planet_proj norm = ||W||^2 + ||b||^2 sqrt
        manual = (net.planet_proj.weight.pow(2).sum() +
                  net.planet_proj.bias.pow(2).sum()).sqrt().item()
        wn = weight_norms_by_module(net)
        assert wn["planet_proj"] == pytest.approx(manual, rel=1e-5)

    def test_weight_snapshot_update_norm_matches_manual(self):
        net, _ = self._toy_policy_and_loss()
        snap = WeightSnapshot(net)
        # Hand-edit one parameter group; rest stay identical.
        with torch.no_grad():
            net.planet_proj.weight.add_(0.1)
        deltas = snap.compute_update_norms(net)
        # Other groups should have ||Δw|| ≈ 0 (they didn't move).
        for k, v in deltas.items():
            if k == "planet_proj":
                assert v > 0
            else:
                assert v == pytest.approx(0.0, abs=1e-6)
        # Update ratio = ||Δw|| / ||w_before||.
        ratios = snap.compute_update_ratios(net)
        assert ratios["planet_proj"] > 0
        # Other groups: ratio = 0 (no update).
        for k, v in ratios.items():
            if k != "planet_proj":
                assert v == pytest.approx(0.0, abs=1e-8)

    def test_grad_norms_empty_when_no_grad(self):
        net = OrbitWarsPolicy(p_max=4, f_max=4, d_model=32, n_heads=2, n_layers=2)
        # No backward called -> no grads.
        norms = grad_norms_by_module(net)
        assert norms == {}
