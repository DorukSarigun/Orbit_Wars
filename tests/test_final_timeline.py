from final_timeline import TimelineEvent, simulate_target_timeline


def test_contested_neutral_temporary_capture_then_enemy_recapture():
    result = simulate_target_timeline(
        initial_owner=-1,
        initial_ships=20,
        production=2,
        player=0,
        horizon=20,
        events=[
            TimelineEvent(eta=10, owner=0, ships=25),
            TimelineEvent(eta=15, owner=1, ships=30),
        ],
    )

    assert result.self_captured
    assert result.first_self_capture_eta == 10
    assert result.first_self_capture_surplus == 5
    assert result.self_lost_after_capture
    assert result.enemy_gift
    assert result.owner_at_horizon == 1


def test_contested_neutral_capture_holds_with_enough_surplus_and_production():
    result = simulate_target_timeline(
        initial_owner=-1,
        initial_ships=20,
        production=2,
        player=0,
        horizon=20,
        events=[
            TimelineEvent(eta=10, owner=0, ships=45),
            TimelineEvent(eta=15, owner=1, ships=30),
        ],
    )

    assert result.self_holds_until_horizon(0)
    assert result.owner_at_horizon == 0
    assert result.garrison_at_horizon > 0


def test_enemy_target_produces_until_arrival():
    result = simulate_target_timeline(
        initial_owner=1,
        initial_ships=20,
        production=3,
        player=0,
        horizon=10,
        events=[TimelineEvent(eta=10, owner=0, ships=49)],
    )

    assert not result.self_captured
    assert result.owner_at_horizon == 1
    assert result.garrison_at_horizon == 1

