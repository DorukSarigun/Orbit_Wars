"""BC extraction from locally recorded env replays (no Kaggle API)."""

from __future__ import annotations

from final_bc import collect_bc_from_replay
from orbit_scripted_agent import OrbitScriptedAgent, OrbitScriptedConfig


def test_collect_bc_from_local_env_replay():
    from kaggle_environments import make

    env = make("orbit_wars", debug=False)
    env.configuration["randomSeed"] = 42
    env.reset(num_agents=2)
    agents = [OrbitScriptedAgent(OrbitScriptedConfig()) for _ in range(2)]
    state = env.step([[], []])

    for turn in range(80):
        actions = [
            agents[p].act(state[p]["observation"], p, current_turn=turn, max_turns=500)
            for p in range(2)
        ]
        state = env.step(actions)
        if env.done:
            break

    replay = {"steps": env.steps}
    ds = collect_bc_from_replay(replay, expert_player=0, max_steps=80)
    assert len(ds.obs) >= 1
    assert len(ds) == len(ds.source_idx) == len(ds.target_idx) == len(ds.amount)
    assert all(0.0 < a < 1.0 for a in ds.amount)
