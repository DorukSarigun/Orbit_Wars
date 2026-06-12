"""Tests for remastered attack + defense campaign reward."""

from __future__ import annotations

import pytest

from reworked_credit import EpisodeLog, LaunchRecord, TurnSnapshot
from reworked_credit import TurnDecision
from reworked_remastered_reward import (
    RemasteredRewardConfig,
    CampaignContrib,
    CampaignLedger,
    OUTCOME_VOID,
    ROLE_DEFENSE,
    capture_reward,
    defense_loss_reward,
    enemy_kill_reward,
    dpp_value,
    attribute_remastered_rewards,
    observe_enemy_kills,
    planet_ownership_events,
    _distribute_kill,
    _stock_at_resolve,
    PlanetCampaign,
)


CFG = RemasteredRewardConfig()


def test_capture_always_positive():
    V = dpp_value(prod=4.0, remaining_turns=250, cfg=CFG)
    R = capture_reward(V, K=35.0, S_end=40.0, delta_steps=80, max_turns=300, cfg=CFG)
    assert R > 0.0
    assert R < V


def test_defense_loss_is_negative_dpp():
    V = dpp_value(prod=5.0, remaining_turns=200, cfg=CFG)
    R = defense_loss_reward(V, K=10.0, S_end=100.0, delta_steps=5, max_turns=300, cfg=CFG)
    assert R < 0.0
    assert abs(R) >= CFG.w_defense_loss * V


def test_fail_reward_absolute_waste_term():
    """w_waste charges a fixed cost per wasted ship, independent of total stock."""
    V = dpp_value(prod=3.0, remaining_turns=200, cfg=CFG)
    # With huge S_end the value-relative w_fail term ~ 0; only waste should bite.
    base = RemasteredRewardConfig(w_fail=0.25, w_waste=0.0)
    waste = RemasteredRewardConfig(w_fail=0.25, w_waste=0.5, waste_norm=50.0)
    from reworked_remastered_reward import fail_reward
    r0 = fail_reward(V, K=20.0, S_end=2000.0, cfg=base)
    r1 = fail_reward(V, K=20.0, S_end=2000.0, cfg=waste)
    assert r0 > r1                      # waste makes it strictly more negative
    assert r1 == pytest.approx(r0 - 0.5 * (20.0 / 50.0))


def test_source_strip_penalizes_threatened_strip_then_loss():
    """Launching from a threatened owned planet that then falls is penalized,
    charged to the launching edge's (record_idx, target slot)."""
    cfg = RemasteredRewardConfig(w_source_strip=1.0, source_strip_window=10,
                                 w_defense_loss=1.0)
    ledger = CampaignLedger()

    log = EpisodeLog(max_turns=300, policy_players={0})
    log.planet_slot[5] = 1
    log.add_decision(player=0, turn=8, record_idx=7)
    # snapshots: own planet 5 with 40 ships at t8, lost to enemy at t10
    log.snapshots = [
        TurnSnapshot(turn=8, owner={5: 0}, ships={5: 40.0},
                     production={5: 4.0}, fleet_ids={99}),
        TurnSnapshot(turn=10, owner={5: 1}, ships={5: 20.0},
                     production={5: 4.0}, fleet_ids=set()),
    ]
    # an outgoing attack launch FROM the threatened planet 5 -> target 9 (slot 3)
    log.launches = [
        LaunchRecord(decision_id=0, player=0, launch_turn=8,
                     src_id=5, tgt_id=9, ships=30, is_attack=True, fleet_id=99,
                     t_local=3, src_under_threat=True),
    ]

    slot_credits, stats = attribute_remastered_rewards(log, ledger, cfg)
    assert stats.n_source_strip == 1
    assert stats.source_strip_reward < 0.0
    # charged to the launching edge slot (record_idx=7, target slot=3)
    assert (7, 3) in slot_credits
    assert slot_credits[(7, 3)] < 0.0


def test_source_strip_skips_when_source_not_threatened():
    cfg = RemasteredRewardConfig(w_source_strip=1.0, source_strip_window=10)
    ledger = CampaignLedger()
    log = EpisodeLog(max_turns=300, policy_players={0})
    log.planet_slot[5] = 1
    log.add_decision(player=0, turn=8, record_idx=7)
    log.snapshots = [
        TurnSnapshot(turn=8, owner={5: 0}, ships={5: 40.0},
                     production={5: 4.0}, fleet_ids={99}),
        TurnSnapshot(turn=10, owner={5: 1}, ships={5: 20.0},
                     production={5: 4.0}, fleet_ids=set()),
    ]
    log.launches = [
        LaunchRecord(decision_id=0, player=0, launch_turn=8,
                     src_id=5, tgt_id=9, ships=30, is_attack=True, fleet_id=99,
                     t_local=3, src_under_threat=False),  # NOT threatened
    ]
    slot_credits, stats = attribute_remastered_rewards(log, ledger, cfg)
    assert stats.n_source_strip == 0
    assert (7, 3) not in slot_credits


def test_source_strip_disabled_by_default():
    """w_source_strip defaults to 0 -> no over-extension penalty."""
    ledger = CampaignLedger()
    log = EpisodeLog(max_turns=300, policy_players={0})
    log.planet_slot[5] = 1
    log.add_decision(player=0, turn=8, record_idx=7)
    log.snapshots = [
        TurnSnapshot(turn=8, owner={5: 0}, ships={5: 40.0},
                     production={5: 4.0}, fleet_ids={99}),
        TurnSnapshot(turn=10, owner={5: 1}, ships={5: 20.0},
                     production={5: 4.0}, fleet_ids=set()),
    ]
    log.launches = [
        LaunchRecord(decision_id=0, player=0, launch_turn=8,
                     src_id=5, tgt_id=9, ships=30, is_attack=True, fleet_id=99,
                     t_local=3, src_under_threat=True),
    ]
    _, stats = attribute_remastered_rewards(log, ledger, CFG)  # default cfg
    assert stats.n_source_strip == 0


def test_episode_end_fails_open_attack_with_wasted_ships():
    """Unresolved attack with wasted ships at game end → fail penalty."""
    ledger = CampaignLedger()
    ledger.note_attack(
        player=0, tgt_id=10, t_decision=5, stock=100.0,
        record_idx=2, t_local=4, ships=20, tgt_was_enemy=True)
    ledger.record_stock(0, 10, 95.0)

    log = EpisodeLog(max_turns=300, policy_players={0})
    log.snapshots = [
        TurnSnapshot(
            turn=5, owner={10: 1}, ships={10: 50.0},
            production={10: 3.0}, fleet_ids={1}),
        TurnSnapshot(
            turn=80, owner={10: 1}, ships={10: 40.0},
            production={10: 3.0}, fleet_ids=set()),
    ]
    log.launches = [
        LaunchRecord(
            decision_id=0, campaign_id=0, player=0, launch_turn=5,
            src_id=1, tgt_id=10, ships=20, is_attack=True, fleet_id=1),
    ]

    slot_credits, stats = attribute_remastered_rewards(log, ledger, CFG)
    assert len(ledger.closed) == 1
    assert ledger.closed[0].outcome == "fail"
    assert ledger.closed[0].K == pytest.approx(20.0)
    assert stats.n_capture == 0 and stats.n_fail == 1 and stats.n_lost == 0
    assert stats.total < 0.0
    assert (2, 4) in slot_credits
    assert slot_credits[(2, 4)] < 0.0


def test_episode_end_voids_open_defense_campaign():
    ledger = CampaignLedger()
    ledger.note_defense(
        player=0, tgt_id=5, t_decision=8, stock=90.0,
        record_idx=2, t_local=1, ships=25)

    log = EpisodeLog(max_turns=300, policy_players={0})
    log.snapshots = [
        TurnSnapshot(
            turn=8, owner={5: 0}, ships={5: 40.0},
            production={5: 4.0}, fleet_ids={10}),
        TurnSnapshot(
            turn=100, owner={5: 0}, ships={5: 55.0},
            production={5: 4.0}, fleet_ids=set()),
    ]
    log.launches = [
        LaunchRecord(
            decision_id=1, campaign_id=1, player=0, launch_turn=8,
            src_id=2, tgt_id=5, ships=25, is_attack=False, fleet_id=10),
    ]

    slot_credits, stats = attribute_remastered_rewards(log, ledger, CFG)
    assert len(ledger.closed) == 1
    assert ledger.closed[0].outcome == OUTCOME_VOID
    assert stats.n_defense_loss == 0 and stats.total == 0.0
    assert slot_credits == {}


def test_multi_turn_attack_split():
    ledger = CampaignLedger()
    ledger.note_attack(player=0, tgt_id=10, t_decision=5, stock=100.0,
                       record_idx=0, t_local=2, ships=20)
    ledger.record_stock(0, 10, 120.0)
    ledger.note_attack(player=0, tgt_id=10, t_decision=8, stock=120.0,
                       record_idx=3, t_local=2, ships=30)

    log = EpisodeLog(max_turns=300, policy_players={0})
    for turn, owner10, ships10, fids in [
        (5, -1, 50.0, {1}), (9, -1, 50.0, {1, 2}), (10, 0, 25.0, set()),
    ]:
        log.snapshots.append(TurnSnapshot(
            turn=turn, owner={10: owner10}, ships={10: ships10},
            production={10: 3.0}, fleet_ids=set(fids)))
    log.launches = [
        LaunchRecord(decision_id=0, campaign_id=0, player=0, launch_turn=5,
                     src_id=1, tgt_id=10, ships=20, is_attack=True, fleet_id=1),
        LaunchRecord(decision_id=0, campaign_id=0, player=0, launch_turn=8,
                     src_id=1, tgt_id=10, ships=30, is_attack=True, fleet_id=2),
    ]
    slot_credits, stats = attribute_remastered_rewards(log, ledger, CFG)
    assert stats.n_capture == 1
    assert abs(slot_credits[(0, 2)] / slot_credits[(3, 2)] - 20.0 / 30.0) < 0.05


def test_post_capture_arrivals_are_defense_not_capture_credit():
    """Fleets landing after capture must not share attack capture credit."""
    ledger = CampaignLedger()
    ledger.note_attack(player=0, tgt_id=10, t_decision=5, stock=100.0,
                       record_idx=0, t_local=2, ships=20)
    ledger.note_attack(player=0, tgt_id=10, t_decision=8, stock=120.0,
                       record_idx=3, t_local=2, ships=30)

    log = EpisodeLog(max_turns=300, policy_players={0})
    for turn, owner10, ships10, fids in [
        (5, -1, 50.0, {1}),
        (9, -1, 50.0, {1, 2}),
        (10, 0, 25.0, {2}),   # capture; fleet 1 arrived, fleet 2 still inbound
        (12, 0, 55.0, set()),  # fleet 2 arrives after capture
    ]:
        log.snapshots.append(TurnSnapshot(
            turn=turn, owner={10: owner10}, ships={10: ships10},
            production={10: 3.0}, fleet_ids=set(fids)))
    log.launches = [
        LaunchRecord(decision_id=0, campaign_id=0, player=0, launch_turn=5,
                     src_id=1, tgt_id=10, ships=20, is_attack=True, fleet_id=1,
                     t_local=2),
        LaunchRecord(decision_id=0, campaign_id=0, player=0, launch_turn=8,
                     src_id=1, tgt_id=10, ships=30, is_attack=True, fleet_id=2,
                     t_local=2),
    ]
    slot_credits, stats = attribute_remastered_rewards(log, ledger, CFG)
    assert stats.n_capture == 1
    assert (0, 2) in slot_credits and slot_credits[(0, 2)] > 0.0
    assert (3, 2) not in slot_credits or abs(slot_credits.get((3, 2), 0.0)) < 1e-9
    # Late wave reclassified as defense (held → void, no extra credit).
    assert not any(
        r.is_attack for r in log.launches if r.fleet_id == 2)


def test_defense_campaign_tracks_reinforce_decisions():
    """Reinforcement launches open a defense campaign; loss applies -dpp."""
    ledger = CampaignLedger()
    ledger.note_defense(player=0, tgt_id=5, t_decision=8, stock=90.0,
                        record_idx=2, t_local=1, ships=25)
    ledger.record_stock(0, 10, 70.0)

    log = EpisodeLog(max_turns=300, policy_players={0})
    log.planet_slot[5] = 1
    for turn, owner5, ships5, fids in [
        (8, 0, 40.0, {10}),
        (10, -1, 30.0, set()),  # lost to enemy at step 10
    ]:
        log.snapshots.append(TurnSnapshot(
            turn=turn, owner={5: owner5}, ships={5: ships5},
            production={5: 4.0}, fleet_ids=set(fids)))
    log.launches = [
        LaunchRecord(decision_id=1, campaign_id=1, player=0, launch_turn=8,
                     src_id=2, tgt_id=5, ships=25, is_attack=False, fleet_id=10),
    ]
    slot_credits, stats = attribute_remastered_rewards(log, ledger, CFG)
    assert stats.n_defense_loss == 1
    assert stats.defense_reward < 0.0
    assert (2, 1) in slot_credits
    assert slot_credits[(2, 1)] < 0.0


def test_passive_defense_loss_without_reinforce():
    """Planet lost with no defense launches — penalty on last decision turn."""
    ledger = CampaignLedger()
    log = EpisodeLog(max_turns=300, policy_players={0})
    log.planet_slot[7] = 3
    log.decisions = [TurnDecision(player=0, turn=9, record_idx=4)]

    log.snapshots.append(TurnSnapshot(
        turn=9, owner={7: 0}, ships={7: 50.0}, production={7: 3.0}, fleet_ids=set()))
    log.snapshots.append(TurnSnapshot(
        turn=10, owner={7: 1}, ships={7: 20.0}, production={7: 3.0}, fleet_ids=set()))

    slot_credits, stats = attribute_remastered_rewards(log, ledger, CFG)
    assert stats.n_defense_loss == 1
    assert slot_credits[(4, 3)] < 0.0


def test_enemy_kill_distributed_to_defense_contrib():
    cfg = RemasteredRewardConfig(w_enemy_kill=1.0)
    contrib = [
        CampaignContrib(record_idx=1, t_local=2, ships=10.0, launch_turn=5),
        CampaignContrib(record_idx=3, t_local=2, ships=30.0, launch_turn=7),
        CampaignContrib(record_idx=9, t_local=2, ships=20.0, launch_turn=12),
    ]
    enemy_stock = 100.0
    part = _distribute_kill(
        contrib, kill_turn=10, enemy_ships=25.0, enemy_stock=enemy_stock, cfg=cfg)
    assert abs(part[(1, 2)] / part[(3, 2)] - 10.0 / 30.0) < 1e-6
    assert (9, 2) not in part
    assert abs(sum(part.values()) - enemy_kill_reward(25.0, enemy_stock, cfg)) < 1e-6


def test_enemy_kill_scales_with_enemy_empire_size():
    cfg = RemasteredRewardConfig(w_enemy_kill=1.0)
    assert enemy_kill_reward(20.0, 200.0, cfg) == pytest.approx(0.1)
    assert enemy_kill_reward(20.0, 50.0, cfg) == pytest.approx(0.4)
    assert enemy_kill_reward(60.0, 50.0, cfg) == pytest.approx(1.0)  # clipped


def test_planet_ownership_events_from_snapshots():
    log = EpisodeLog(max_turns=300, policy_players={0})
    log.snapshots = [
        TurnSnapshot(turn=5, owner={10: -1, 5: 0}, ships={10: 20.0, 5: 40.0},
                     production={10: 3.0, 5: 4.0}, fleet_ids=set()),
        TurnSnapshot(turn=8, owner={10: 0, 5: 0}, ships={10: 15.0, 5: 35.0},
                     production={10: 3.0, 5: 4.0}, fleet_ids=set()),
        TurnSnapshot(turn=12, owner={10: 0, 5: 1}, ships={10: 15.0, 5: 20.0},
                     production={10: 3.0, 5: 4.0}, fleet_ids=set()),
    ]
    caps, losses = planet_ownership_events(log, 0)
    assert len(caps) == 1
    assert caps[0]["planet_id"] == 10 and caps[0]["turn"] == 8
    assert len(losses) == 1
    assert losses[0]["planet_id"] == 5 and losses[0]["turn"] == 12


def test_stock_at_resolve():
    camp = PlanetCampaign(
        campaign_id=0, role="attack", player=0, tgt_id=1, t_decision=5,
        stock_by_turn={5: 80.0, 10: 110.0}, t_resolve=10)
    assert _stock_at_resolve(camp) == 110.0


def _obs_with_fleet(*, planet_owner: int, planet_ships: int, fleet_owner: int,
                    fleet_ships: int, fleet_id: int = 1, planet_id: int = 10,
                    fleets: list | None = None) -> dict:
    if fleets is None:
        fleets = [[fleet_id, fleet_owner, 1.0, 0.0, 3.14, planet_id, fleet_ships]]
    return {
        "planets": [[planet_id, planet_owner, 0.0, 0.0, 5.0, planet_ships, 3]],
        "fleets": fleets,
        "angular_velocity": 0.0,
    }


def test_attack_enemy_kill_credits_assault_contrib(monkeypatch):
    """Enemy fleet destroyed at an assaulted enemy planet → attack campaign credit."""
    monkeypatch.setattr(
        "navigation.predict_all_fleet_targets",
        lambda _fleets, _planets, _omega: {1: {"target_id": 10}},
    )
    cfg = RemasteredRewardConfig(w_enemy_kill=1.0)
    ledger = CampaignLedger()
    ledger.note_attack(
        player=0, tgt_id=10, t_decision=5, stock=100.0,
        record_idx=2, t_local=4, ships=20, tgt_was_enemy=True)
    ledger.note_attack(
        player=0, tgt_id=10, t_decision=7, stock=100.0,
        record_idx=5, t_local=4, ships=40, tgt_was_enemy=True)

    log = EpisodeLog(max_turns=300, policy_players={0})
    log.planet_slot[10] = 4
    log.decisions = [TurnDecision(player=0, turn=7, record_idx=5)]

    # Enemy stock at prev step: garrison 80 + in-flight fleet 25 = 105
    prev_obs = _obs_with_fleet(planet_owner=1, planet_ships=80, fleet_owner=1, fleet_ships=25)
    curr_obs = _obs_with_fleet(planet_owner=1, planet_ships=55, fleet_owner=1, fleet_ships=25,
                               fleets=[])

    slot_credits, total = observe_enemy_kills(
        ledger, log, prev_obs, curr_obs, turn=8, player=0, omega=0.0, cfg=cfg)
    assert total == pytest.approx(enemy_kill_reward(25.0, 105.0, cfg))
    assert abs(slot_credits[(2, 4)] / slot_credits[(5, 4)] - 20.0 / 40.0) < 1e-6


def test_attack_enemy_kill_on_capture_turn(monkeypatch):
    """Capture turn still credits the attack campaign (not defense)."""
    monkeypatch.setattr(
        "navigation.predict_all_fleet_targets",
        lambda _fleets, _planets, _omega: {1: {"target_id": 10}},
    )
    cfg = RemasteredRewardConfig(w_enemy_kill=1.0)
    ledger = CampaignLedger()
    ledger.note_attack(
        player=0, tgt_id=10, t_decision=5, stock=100.0,
        record_idx=1, t_local=3, ships=30, tgt_was_enemy=True)

    log = EpisodeLog(max_turns=300, policy_players={0})
    log.planet_slot[10] = 3

    prev_obs = _obs_with_fleet(planet_owner=1, planet_ships=50, fleet_owner=1, fleet_ships=15)
    curr_obs = _obs_with_fleet(planet_owner=0, planet_ships=8, fleet_owner=1, fleet_ships=15,
                               fleets=[])

    slot_credits, total = observe_enemy_kills(
        ledger, log, prev_obs, curr_obs, turn=9, player=0, omega=0.0, cfg=cfg)
    assert total == pytest.approx(enemy_kill_reward(15.0, 65.0, cfg))
    assert (1, 3) in slot_credits


def _capture_camp(**overrides) -> PlanetCampaign:
    camp = PlanetCampaign(
        campaign_id=0, role="attack", player=0, tgt_id=10, t_decision=5,
        K=0.0, S_end=100.0, outcome="capture", t_resolve=10, prod_at_close=4.0)
    for k, v in overrides.items():
        setattr(camp, k, v)
    return camp


def test_enemy_capture_mult_scales_enemy_owned_capture():
    from reworked_remastered_reward import _campaign_net_reward
    cfg = RemasteredRewardConfig(enemy_capture_mult=1.5)
    r_neutral = _campaign_net_reward(_capture_camp(), 300, cfg)
    r_enemy = _campaign_net_reward(_capture_camp(tgt_was_enemy=True), 300, cfg)
    assert r_neutral > 0.0
    assert r_enemy == pytest.approx(1.5 * r_neutral)


def test_recapture_decay_suppresses_pingpong_farm():
    from reworked_remastered_reward import _campaign_net_reward
    cfg = RemasteredRewardConfig(recapture_decay=0.5)
    r0 = _campaign_net_reward(_capture_camp(), 300, cfg)
    r1 = _campaign_net_reward(_capture_camp(recapture_idx=1), 300, cfg)
    r2 = _campaign_net_reward(_capture_camp(recapture_idx=2), 300, cfg)
    assert r1 == pytest.approx(0.5 * r0)
    assert r2 == pytest.approx(0.25 * r0)


def test_ledger_tracks_recapture_count_per_player_planet():
    ledger = CampaignLedger()
    ledger.note_attack(player=0, tgt_id=10, t_decision=1, stock=50.0,
                       record_idx=0, t_local=2, ships=10)
    c1 = ledger.close(0, 10, "attack", outcome="capture", t_resolve=3, prod=3.0)
    ledger.note_attack(player=0, tgt_id=10, t_decision=8, stock=50.0,
                       record_idx=4, t_local=2, ships=10)
    c2 = ledger.close(0, 10, "attack", outcome="capture", t_resolve=12, prod=3.0)
    # Different planet / non-capture outcomes do not advance the counter.
    ledger.note_attack(player=0, tgt_id=11, t_decision=14, stock=50.0,
                       record_idx=6, t_local=3, ships=10)
    c3 = ledger.close(0, 11, "attack", outcome="capture", t_resolve=16, prod=3.0)
    assert c1.recapture_idx == 0
    assert c2.recapture_idx == 1
    assert c3.recapture_idx == 0


def test_lost_race_counts_inflight_ships_as_casualties():
    """Fleets still flying toward a lost target are wasted — they enter K."""
    ledger = CampaignLedger()
    ledger.note_attack(player=0, tgt_id=10, t_decision=5, stock=100.0,
                       record_idx=0, t_local=2, ships=20)
    ledger.note_attack(player=0, tgt_id=10, t_decision=5, stock=100.0,
                       record_idx=0, t_local=2, ships=30)

    log = EpisodeLog(max_turns=300, policy_players={0})
    for turn, owner10, fids in [
        (5, -1, {1, 2}),
        (10, 1, {2}),      # enemy wins the race; fleet 1 arrived & died
        (12, 1, set()),    # fleet 2 lands on the enemy-held planet later
    ]:
        log.snapshots.append(TurnSnapshot(
            turn=turn, owner={10: owner10}, ships={10: 50.0},
            production={10: 3.0}, fleet_ids=set(fids)))
    log.launches = [
        LaunchRecord(decision_id=0, campaign_id=0, player=0, launch_turn=5,
                     src_id=1, tgt_id=10, ships=20, is_attack=True, fleet_id=1,
                     t_local=2),
        LaunchRecord(decision_id=0, campaign_id=0, player=0, launch_turn=5,
                     src_id=1, tgt_id=10, ships=30, is_attack=True, fleet_id=2,
                     t_local=2),
    ]
    slot_credits, stats = attribute_remastered_rewards(log, ledger, CFG)
    assert stats.n_lost == 1
    lost = [c for c in ledger.closed if c.outcome == "lost"][0]
    assert lost.K == pytest.approx(50.0)   # 20 arrived + 30 in flight
    assert slot_credits[(0, 2)] < 0.0


def test_timeout_terminal_score_sign_and_symmetry():
    from reworked_rollout import _timeout_terminal_score
    obs = {
        "planets": [
            [1, 0, 0.0, 0.0, 5.0, 30, 5],   # P0, prod 5
            [2, 1, 0.0, 0.0, 5.0, 30, 2],   # P1, prod 2
            [3, -1, 0.0, 0.0, 5.0, 30, 4],  # neutral ignored
        ],
        "fleets": [],
    }
    s0 = _timeout_terminal_score(obs, 0)
    s1 = _timeout_terminal_score(obs, 1)
    assert s0 > 0.0 > s1
    assert s0 == pytest.approx(-s1)
    # Dead even -> ~0
    even = {"planets": [[1, 0, 0.0, 0.0, 5.0, 30, 3],
                        [2, 1, 0.0, 0.0, 5.0, 30, 3]], "fleets": []}
    assert _timeout_terminal_score(even, 0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# v13 ship-equity mode
# ---------------------------------------------------------------------------
EQ = RemasteredRewardConfig(ship_equity=True)


def test_equity_enemy_capture_is_double_neutral():
    from reworked_remastered_reward import capture_reward_equity
    V = dpp_value(prod=4.0, remaining_turns=200, cfg=EQ)
    r_neutral = capture_reward_equity(V, K=0.0, siege_turns=0, prod=4.0,
                                      tgt_was_enemy=False, recapture_idx=0, cfg=EQ)
    r_enemy = capture_reward_equity(V, K=0.0, siege_turns=0, prod=4.0,
                                    tgt_was_enemy=True, recapture_idx=0, cfg=EQ)
    assert r_neutral == pytest.approx(V)
    assert r_enemy == pytest.approx(2.0 * V)   # zero-sum swing is intrinsic


def test_equity_pyrrhic_capture_goes_negative():
    """Spending far more ships than the planet is worth nets a loss — by design."""
    from reworked_remastered_reward import capture_reward_equity
    V = dpp_value(prod=1.0, remaining_turns=50, cfg=EQ)   # tiny rock
    R = capture_reward_equity(V, K=200.0, siege_turns=0, prod=1.0,
                              tgt_was_enemy=False, recapture_idx=0, cfg=EQ)
    assert R < 0.0
    assert R == pytest.approx(V - 200.0 / EQ.norm)


def test_equity_siege_costs_defender_regrowth():
    """A dripped 20-turn siege of a prod-5 planet costs 100 regrown ships."""
    from reworked_remastered_reward import capture_reward_equity
    V = dpp_value(prod=5.0, remaining_turns=200, cfg=EQ)
    r_blitz = capture_reward_equity(V, K=30.0, siege_turns=0, prod=5.0,
                                    tgt_was_enemy=False, recapture_idx=0, cfg=EQ)
    r_drip = capture_reward_equity(V, K=30.0, siege_turns=20, prod=5.0,
                                   tgt_was_enemy=False, recapture_idx=0, cfg=EQ)
    assert r_drip == pytest.approx(r_blitz - EQ.eq_siege * 5.0 * 20 / EQ.norm)
    assert r_blitz > r_drip


def test_equity_recapture_decay_discounts_gain_not_cost():
    from reworked_remastered_reward import capture_reward_equity
    V = dpp_value(prod=4.0, remaining_turns=200, cfg=EQ)
    K = 40.0
    r0 = capture_reward_equity(V, K=K, siege_turns=0, prod=4.0,
                               tgt_was_enemy=False, recapture_idx=0, cfg=EQ)
    r1 = capture_reward_equity(V, K=K, siege_turns=0, prod=4.0,
                               tgt_was_enemy=False, recapture_idx=1, cfg=EQ)
    # Gain halves; the casualty cost is identical in both.
    assert r0 == pytest.approx(V - K / EQ.norm)
    assert r1 == pytest.approx(EQ.recapture_decay * V - K / EQ.norm)


def test_equity_fail_prices_ships_absolutely():
    """No K/S_end: 20 wasted ships cost the same for rich and poor empires."""
    from reworked_remastered_reward import fail_reward_equity
    R = fail_reward_equity(K=20.0, siege_turns=0, prod=3.0, cfg=EQ)
    assert R == pytest.approx(-EQ.eq_fail * 20.0 / EQ.norm)
    # Spray that kept the defender producing for 10 turns pays the regrowth too.
    R_spray = fail_reward_equity(K=20.0, siege_turns=10, prod=3.0, cfg=EQ)
    assert R_spray == pytest.approx(R - EQ.eq_siege * 3.0 * 10 / EQ.norm)


def test_equity_defense_loss_is_double_swing_plus_dead_garrison():
    from reworked_remastered_reward import defense_loss_reward_equity
    V = dpp_value(prod=5.0, remaining_turns=200, cfg=EQ)
    R = defense_loss_reward_equity(V, K=25.0, cfg=EQ)
    assert R == pytest.approx(-(2.0 * V + 25.0 / EQ.norm))


def test_equity_enemy_kill_is_absolute_ships():
    """Same kills, same credit — enemy empire size no longer dilutes it."""
    r_small = enemy_kill_reward(kills=30.0, enemy_stock=50.0, cfg=EQ)
    r_big = enemy_kill_reward(kills=30.0, enemy_stock=5000.0, cfg=EQ)
    assert r_small == pytest.approx(r_big) == pytest.approx(30.0 / EQ.norm)
    # Legacy mode keeps the stock-share scaling.
    assert enemy_kill_reward(30.0, 50.0, CFG) > enemy_kill_reward(30.0, 5000.0, CFG)


def test_equity_pingpong_trade_nets_negative():
    """Capture enemy planet (+2V') then lose it back (-2V): with any casualties
    the round trip is net negative, so farming flips cannot pay."""
    from reworked_remastered_reward import (
        capture_reward_equity, defense_loss_reward_equity)
    V = dpp_value(prod=4.0, remaining_turns=200, cfg=EQ)
    gain = capture_reward_equity(V, K=10.0, siege_turns=0, prod=4.0,
                                 tgt_was_enemy=True, recapture_idx=0, cfg=EQ)
    # A few turns later it flips back (V barely changed, garrison died too).
    loss = defense_loss_reward_equity(V, K=5.0, cfg=EQ)
    assert gain + loss < 0.0


def test_equity_first_contact_siege_duration_from_log():
    """_campaign_net_reward derives siege turns from the first arrival."""
    from reworked_remastered_reward import _campaign_net_reward
    log = EpisodeLog(max_turns=300, policy_players=[0])
    camp = PlanetCampaign(campaign_id=1, role="attack", player=0, tgt_id=7,
                          t_decision=10, K=20.0, S_end=100.0)
    camp.outcome = "capture"
    camp.t_resolve = 40
    camp.prod_at_close = 5.0
    # Two waves: first lands t=25, capture lands t=40 -> 15 siege turns.
    for fid, (lt, rt) in enumerate([(10, 25), (28, 40)]):
        log.launches.append(LaunchRecord(
            decision_id=0, player=0, launch_turn=lt, src_id=1, tgt_id=7,
            ships=30, fleet_id=fid, is_attack=True, campaign_id=1, t_local=7,
            resolve_turn=rt))
    R_with_log = _campaign_net_reward(camp, 300, EQ, log)
    R_no_log = _campaign_net_reward(camp, 300, EQ, None)
    assert R_no_log - R_with_log == pytest.approx(EQ.eq_siege * 5.0 * 15 / EQ.norm)


def test_attack_enemy_kill_skips_neutral_target(monkeypatch):
    """Assaulting a neutral planet does not earn enemy-kill credit."""
    monkeypatch.setattr(
        "navigation.predict_all_fleet_targets",
        lambda _fleets, _planets, _omega: {1: {"target_id": 10}},
    )
    cfg = RemasteredRewardConfig(w_enemy_kill=1.0, norm=50.0)
    ledger = CampaignLedger()
    ledger.note_attack(
        player=0, tgt_id=10, t_decision=5, stock=100.0,
        record_idx=1, t_local=2, ships=30, tgt_was_enemy=False)

    log = EpisodeLog(max_turns=300, policy_players={0})
    log.planet_slot[10] = 2

    prev_obs = _obs_with_fleet(planet_owner=-1, planet_ships=20, fleet_owner=1, fleet_ships=12)
    curr_obs = _obs_with_fleet(planet_owner=-1, planet_ships=15, fleet_owner=1, fleet_ships=12,
                               fleets=[])

    slot_credits, total = observe_enemy_kills(
        ledger, log, prev_obs, curr_obs, turn=6, player=0, omega=0.0, cfg=cfg)
    assert total == 0.0
    assert slot_credits == {}
