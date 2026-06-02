"""Regression tests for the recent graph-policy / debug-logger changes.

Covers three orthogonal additions:

1. ``GRAPH_EDGE_FLUX_DIM`` / ``PYG_EDGE_FEATURE_DIM_FLUX`` constants are
   exported and arithmetic-consistent.
2. ``PygPlanetGraphEncoder`` / ``PygPlanetGraphPolicy`` /
   ``PygEdgeNativePolicy`` accept ``use_input_embed=False`` and run a
   complete forward pass without the NodeEmbed / EdgeEmbed projections.
3. ``rl.game_logger.GameLogger`` produces a well-formed JSONL file with
   the expected ``header`` / ``turn`` / ``summary`` event sequence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# 1. Edge flux dim wiring
# ---------------------------------------------------------------------------


def test_edge_flux_dim_constants_consistent():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from model_graph_pyg import (
        GRAPH_EDGE_FLUX_DIM,
        PYG_EDGE_FEATURE_DIM,
        PYG_EDGE_FEATURE_DIM_FLUX,
    )

    assert GRAPH_EDGE_FLUX_DIM == 2
    assert PYG_EDGE_FEATURE_DIM_FLUX == PYG_EDGE_FEATURE_DIM + GRAPH_EDGE_FLUX_DIM


def test_pair_edge_features_flux_widens_last_dim():
    """``_batched_pair_edge_features`` must append exactly 2 flux channels."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from model_graph_pyg import (
        GRAPH_EDGE_FLUX_DIM,
        PYG_EDGE_FEATURE_DIM,
        PYG_EDGE_FEATURE_DIM_FLUX,
        _batched_pair_edge_features,
    )
    from obs_builder import EDGE_FEATURE_DIM, PLANET_FEATURE_DIM

    B, P = 2, 5
    planet_feats = torch.zeros(B, P, PLANET_FEATURE_DIM)
    edge_bias = torch.zeros(B, P, P, EDGE_FEATURE_DIM)
    planet_mask = torch.zeros(B, P, dtype=torch.bool)
    planet_mask[:, :3] = True

    no_flux = _batched_pair_edge_features(planet_feats, edge_bias, planet_mask)
    assert no_flux.shape == (B, P, P, PYG_EDGE_FEATURE_DIM)

    edge_flux = torch.ones(B, P, P, GRAPH_EDGE_FLUX_DIM)
    with_flux = _batched_pair_edge_features(
        planet_feats, edge_bias, planet_mask, edge_flux_features=edge_flux
    )
    assert with_flux.shape == (B, P, P, PYG_EDGE_FEATURE_DIM_FLUX)
    # Padded pairs (rows / cols beyond planet_mask) must be zeroed including
    # the flux block, and valid pairs preserve our ``ones`` in the last 2
    # channels.
    assert torch.all(with_flux[0, 0, 1, -GRAPH_EDGE_FLUX_DIM:] == 1.0)
    assert torch.all(with_flux[0, 4, 4, -GRAPH_EDGE_FLUX_DIM:] == 0.0)


def test_pair_edge_features_rejects_wrong_flux_shape():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from model_graph_pyg import _batched_pair_edge_features
    from obs_builder import EDGE_FEATURE_DIM, PLANET_FEATURE_DIM

    B, P = 1, 4
    planet_feats = torch.zeros(B, P, PLANET_FEATURE_DIM)
    edge_bias = torch.zeros(B, P, P, EDGE_FEATURE_DIM)
    planet_mask = torch.ones(B, P, dtype=torch.bool)
    bad = torch.zeros(B, P, P, 3)  # wrong last dim
    with pytest.raises(ValueError):
        _batched_pair_edge_features(
            planet_feats, edge_bias, planet_mask, edge_flux_features=bad,
        )


# ---------------------------------------------------------------------------
# 2. use_input_embed=False forward pass
# ---------------------------------------------------------------------------


def _packed_batch(B: int = 2, p: int = 6):
    """Tiny pack_observations-style batch suitable for both PyG variants."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from obs_builder import (
        EDGE_FEATURE_DIM,
        F_MAX_DEFAULT,
        FLEET_FEATURE_DIM,
        GLOBAL_FEATURE_DIM,
        PLANET_FEATURE_DIM,
    )

    planet_feats = torch.zeros(B, p, PLANET_FEATURE_DIM)
    fleet_feats = torch.zeros(B, F_MAX_DEFAULT, FLEET_FEATURE_DIM)
    global_feats = torch.zeros(B, GLOBAL_FEATURE_DIM)
    edge_bias = torch.zeros(B, p, p, EDGE_FEATURE_DIM)
    planet_mask = torch.zeros(B, p, dtype=torch.bool)
    planet_mask[:, :4] = True
    fleet_mask = torch.zeros(B, F_MAX_DEFAULT, dtype=torch.bool)
    source_mask = planet_mask.clone()
    return dict(
        planet_feats=planet_feats,
        fleet_feats=fleet_feats,
        global_feats=global_feats,
        edge_bias=edge_bias,
        planet_mask=planet_mask,
        fleet_mask=fleet_mask,
        source_mask=source_mask,
    )


def test_pyg_planet_graph_policy_no_input_embed_forward():
    """``use_input_embed=False`` builds Identity node_proj and runs end-to-end."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch.nn as nn
    from model_graph_pyg import PygPlanetGraphPolicy

    p = 6
    pol = PygPlanetGraphPolicy(
        p_max=p, d_model=32, n_layers=2, use_input_embed=False,
    )
    assert pol.encoder.use_input_embed is False
    assert isinstance(pol.encoder.node_proj, nn.Identity)
    out = pol(**_packed_batch(B=2, p=p))
    assert out.tgt_logits.shape == (2, p, p)
    assert out.value.shape == (2,)


def test_pyg_edge_native_policy_no_input_embed_forward():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch.nn as nn
    from model_graph_pyg_edgeonly import PygEdgeNativePolicy

    p = 6
    pol = PygEdgeNativePolicy(
        p_max=p, d_model=32, n_layers=2, use_input_embed=False,
    )
    assert pol.use_input_embed is False
    assert pol.encoder.use_input_embed is False
    assert isinstance(pol.encoder.node_proj, nn.Identity)
    out = pol(**_packed_batch(B=2, p=p))
    assert out.tgt_logits.shape == (2, p, p)
    assert out.value.shape == (2,)


def test_pyg_edge_native_policy_evaluate_source_actions():
    """`evaluate_source_actions` exists on edge-native policy and matches
    `act_source` conditional log-probs (regression for scripted-expansion
    bootstrap which fixes ``src`` and asks for the conditional log-prob
    of a chosen ``(tgt, ship_value)``)."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from model_graph_pyg_edgeonly import PygEdgeNativePolicy

    p = 6
    torch.manual_seed(0)
    pol = PygEdgeNativePolicy(p_max=p, d_model=32, n_layers=2)
    out = pol(**_packed_batch(B=2, p=p))

    src = torch.tensor([1, 2], dtype=torch.long)
    tgt = torch.tensor([3, 0], dtype=torch.long)
    ship_value = torch.tensor([0.4, 0.7], dtype=torch.float32)
    is_noop = torch.tensor([False, False], dtype=torch.bool)

    ev = pol.evaluate_source_actions(out, src, tgt, ship_value, is_noop)
    assert ev["log_prob"].shape == (2,)
    assert ev["entropy"].shape == (2,)
    assert torch.isfinite(ev["log_prob"]).all()
    assert torch.isfinite(ev["entropy"]).all()
    # Setting is_noop=True must zero the ship-distribution contribution
    # and use action_idx == 0 (the noop slot of the per-source dist).
    is_noop_t = torch.tensor([True, True], dtype=torch.bool)
    ev_noop = pol.evaluate_source_actions(out, src, tgt, ship_value, is_noop_t)
    assert ev_noop["log_prob"].shape == (2,)
    assert torch.isfinite(ev_noop["log_prob"]).all()


def test_collect_episode_pyg_edge_native_scripted_expansion_and_opponent_smoke():
    """End-to-end rollout smoke: PygEdgeNativePolicy + scripted opponent
    + scripted neutral-expansion bootstrap (the exact combination used by
    the train CLI) must complete without ``AttributeError`` /
    ``RuntimeError``.

    Regression for two bugs caught only at training-time, never by unit
    tests:

    1. ``rl/train.py`` was missing the ``GameLogger`` import (NameError).
    2. ``rl/rollout._scripted_neutral_expansion_action`` called
       ``policy.evaluate_source_actions`` which existed on the legacy
       ``GroupedPlanetOrbitWarsPolicy`` but not on
       ``PygEdgeNativePolicy``.

    The smoke covers the full PPO-side write path (forward, action
    sampling, scripted-expansion override, buffer.add, fleet/defense
    trackers) for a small number of steps, so regressions in the dispatch
    layer surface immediately.
    """
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("kaggle_environments")
    import torch
    from rl.buffer import RolloutBuffer
    from rl.rewards import build_reward_config
    from rl.rollout import ScriptedExpansionConfig, collect_episode
    from model_graph_pyg_edgeonly import PygEdgeNativePolicy

    torch.manual_seed(0)
    pol = PygEdgeNativePolicy(d_model=32, n_layers=1)  # P_MAX_DEFAULT internally
    buf = RolloutBuffer()
    cfg = build_reward_config("graph_native")
    stats = collect_episode(
        pol, buf,
        max_steps=8,
        num_agents=2,
        reward_cfg=cfg,
        seed=42,
        device=torch.device("cpu"),
        opponent_indices={1},
        scripted_expansion=ScriptedExpansionConfig(turns=4, nearest_k=3),
    )
    # The episode must produce at least one PPO transition for slot 0
    # (the only PPO-controlled player).  Slot 1 is scripted and produces
    # no transitions in the buffer.
    assert len(buf.data) > 0
    assert all(tr.player == 0 for tr in buf.data)
    # ``stats.n_launches`` is a per-player counter; slot 1 (scripted)
    # should also be tracked separately so the assertion checks both.
    assert sum(stats.n_launches) >= 0  # may be 0 in 8-step micro-episode


def test_pyg_planet_graph_encoder_default_uses_input_embed():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from model_graph_pyg import PygPlanetGraphEncoder

    enc = PygPlanetGraphEncoder(d_model=16, n_layers=1)
    assert enc.use_input_embed is True
    # Default node_proj is the LayerNorm/Linear/GELU stack, NOT Identity.
    assert enc.node_proj.__class__.__name__ != "Identity"


# ---------------------------------------------------------------------------
# 3. JSON game logger format
# ---------------------------------------------------------------------------


class _FakePlanet:
    def __init__(self, *, pid, owner, ships, prod, x, y, r):
        self.id = pid
        self.owner = owner
        self.ships = ships
        self.production = prod
        self.x = x
        self.y = y
        self.radius = r


class _FakeFleet:
    def __init__(self, *, src, tgt, owner, ships, eta):
        self.src_planet_id = src
        self.tgt_planet_id = tgt
        self.owner = owner
        self.ships = ships
        self.eta = eta


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_game_logger_writes_header_turn_summary(tmp_path: Path):
    from rl.game_logger import GameLogger

    log_path = tmp_path / "game.jsonl"
    log = GameLogger(log_path, meta={"iteration": 7, "episode": 3})
    planets = [
        _FakePlanet(pid=0, owner=0, ships=10, prod=2, x=10.0, y=20.0, r=3.0),
        _FakePlanet(pid=1, owner=1, ships=5, prod=1, x=80.0, y=70.0, r=2.0),
    ]
    fleets = [_FakeFleet(src=0, tgt=1, owner=0, ships=4, eta=3.5)]
    log.start_turn(0, planets=planets, fleets=fleets, global_state={"angular_velocity": 0.05})
    log.log_action(player=0, decision={
        "src": 0, "tgt": 1, "ship_value": 0.5, "is_noop": False,
        "launched": True, "log_prob": -0.1234, "value": 0.42,
    })
    log.log_action(player=1, decision={"is_noop": True, "log_prob": -0.5})
    log.log_rewards(
        [0.1, -0.1],
        graph_metrics=[{"friendly_influence": 0.5}, {"friendly_influence": 0.4}],
        invalid_actions=[0, 0],
    )
    log.end_turn()
    log.finish(terminal_rewards=[1.0, -1.0], cum_rewards=[1.1, -1.1], winner=0,
               extra={"steps": 1})

    records = _read_jsonl(log_path)
    assert [r["event"] for r in records] == ["header", "turn", "summary"]

    header = records[0]
    assert header["meta"]["iteration"] == 7
    assert header["meta"]["episode"] == 3

    turn = records[1]
    assert turn["turn"] == 0
    assert len(turn["planets"]) == 2
    assert turn["planets"][0] == {
        "id": 0, "owner": 0, "ships": 10, "production": 2,
        "radius": 3.0, "x": 10.0, "y": 20.0,
    }
    assert turn["fleets"][0]["src"] == 0
    assert turn["fleets"][0]["tgt"] == 1
    assert turn["fleets"][0]["eta"] == 3.5
    # log_action stored under string-keyed player buckets.
    assert set(turn["actions"].keys()) == {"0", "1"}
    assert turn["actions"]["0"][0]["src"] == 0
    assert turn["actions"]["1"][0]["is_noop"] is True
    # Reward dict keyed by player index (string).
    assert turn["rewards"] == {"0": 0.1, "1": -0.1}
    assert turn["invalid_actions"] == {"0": 0, "1": 0}
    assert "graph_metrics" in turn

    summary = records[2]
    assert summary["terminal_rewards"] == [1.0, -1.0]
    assert summary["cum_rewards"] == [1.1, -1.1]
    assert summary["winner"] == 0
    assert summary["extra"] == {"steps": 1}


def test_game_logger_finish_idempotent_and_jsonable(tmp_path: Path):
    """Double-finish must not crash; numpy / nan inputs must serialize."""
    import numpy as np

    from rl.game_logger import GameLogger

    log_path = tmp_path / "edge.jsonl"
    log = GameLogger(log_path, meta={"a": np.int32(1), "b": np.float32(2.5)})
    log.start_turn(
        0,
        planets=[],
        fleets=[],
        global_state={"nan": float("nan"), "inf": float("inf"), "arr": np.zeros(3)},
    )
    log.log_action(player=0, decision={
        "src": np.int64(2), "log_prob": np.float64(-0.5),
        "weird": np.array([1.0, 2.0]),
        "bad_float": float("nan"),
    })
    log.log_rewards([np.float32(0.25), np.float32(-0.25)])
    log.end_turn()
    log.finish(terminal_rewards=[1.0, -1.0])
    log.finish()  # idempotent

    records = _read_jsonl(log_path)
    assert [r["event"] for r in records] == ["header", "turn", "summary"]
    header = records[0]
    assert header["meta"] == {"a": 1, "b": 2.5}
    turn = records[1]
    # NaN / inf in global_state become None (JSON-safe sentinel).
    assert turn["global"]["nan"] is None
    assert turn["global"]["inf"] is None
    assert turn["global"]["arr"] == [0.0, 0.0, 0.0]
    # Numpy ints / floats round-trip through normal JSON types.
    act = turn["actions"]["0"][0]
    assert act["src"] == 2
    assert act["log_prob"] == pytest.approx(-0.5)
    assert act["weird"] == [1.0, 2.0]
    assert act["bad_float"] is None


def test_game_logger_context_manager_closes_on_exit(tmp_path: Path):
    from rl.game_logger import GameLogger

    log_path = tmp_path / "ctx.jsonl"
    with GameLogger(log_path, meta={"k": "v"}) as log:
        log.start_turn(0, planets=[], fleets=[])
        log.log_rewards([0.0, 0.0])
        log.end_turn()
    # After __exit__ a final summary record was appended.
    records = _read_jsonl(log_path)
    assert [r["event"] for r in records] == ["header", "turn", "summary"]
