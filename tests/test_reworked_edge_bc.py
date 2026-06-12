"""Edge-space behavior cloning from raw teacher env launches (orbit teacher).

Covers:
* recovering the (src, tgt) edge from an env launch's heading,
* gate/frac targets restricted to valid edges with finite-Beta clipping,
* the ``EdgeBCSample`` stack used by the supervised BC loss in ``ppo_update``.
"""

from __future__ import annotations

import numpy as np
import torch

from obs_builder import build_observation, parse_planets
from reworked_model import ReworkedEdgePolicy
from reworked_pipeline import pack_reworked_batch
from reworked_rollout import (
    EdgeBCSample,
    P_MAX,
    ReworkedBuffer,
    _edge_valid_mask,
    compute_launch,
    teacher_env_action_to_edge_targets,
)


def _env_three_planets():
    # [id, owner, x, y, radius, ships, production]
    rows = [
        [0, 0, 20.0, 20.0, 1.0, 40, 2],
        [1, -1, 20.0, 70.0, 1.0, 5, 3],
        [2, 1, 80.0, 20.0, 1.0, 3, 1],
    ]
    return {
        "planets": rows,
        "fleets": [],
        "initial_planets": rows,
        "angular_velocity": 0.05,
        "comet_planet_ids": [],
        "comets": [],
        "next_fleet_id": 0,
        "step": 0,
    }


def _packed(env_obs, player=0):
    obs = build_observation(env_obs, player=player,
                            angular_velocity=env_obs["angular_velocity"],
                            p_max=P_MAX, f_max=64, turn=0, max_turns=300)
    batch = pack_reworked_batch([(obs, player)], p_max=P_MAX)
    return obs, batch["states"][0]


def test_recovers_target_from_launch_heading():
    env_obs = _env_three_planets()
    obs, st0 = _packed(env_obs)
    parsed = parse_planets(env_obs)
    av = env_obs["angular_velocity"]
    edge_np = _edge_valid_mask(st0, P_MAX, 0, min_send=1)

    # Teacher sends 12 ships from planet 0 toward planet 1: build the raw env
    # action exactly the way the orbit bot does (heading, not target id).
    src, tgt = parsed[0], parsed[1]
    res = compute_launch(src, tgt, 12, parsed, av)
    env_action = [[src.id, float(res["angle"]), int(res["ships"])]]

    gate_t, frac_t = teacher_env_action_to_edge_targets(
        st0, obs, 0, env_action, parsed, av, edge_np, p_max=P_MAX)

    assert gate_t.shape == (P_MAX, P_MAX)
    assert gate_t.sum() == 1.0
    assert gate_t[0, 1] == 1.0           # recovered the right edge
    # frac = ships / garrison, inside the open (0, 1) Beta support.
    assert 0.0 < frac_t[0, 1] < 1.0
    assert abs(frac_t[0, 1] - res["ships"] / 40.0) < 0.05
    # Targets never land outside the valid-edge mask.
    assert (gate_t[~edge_np] == 0).all()


def test_unmatchable_launch_is_dropped():
    env_obs = _env_three_planets()
    obs, st0 = _packed(env_obs)
    parsed = parse_planets(env_obs)
    av = env_obs["angular_velocity"]
    edge_np = _edge_valid_mask(st0, P_MAX, 0, min_send=1)

    # Launch from a planet we do not own (id 2): no valid edges -> no target.
    env_action = [[2, 0.5, 3]]
    gate_t, _ = teacher_env_action_to_edge_targets(
        st0, obs, 0, env_action, parsed, av, edge_np, p_max=P_MAX)
    assert gate_t.sum() == 0.0


def test_edge_bc_stack_and_logp_finite():
    env_obs = _env_three_planets()
    obs, st0 = _packed(env_obs)
    parsed = parse_planets(env_obs)
    av = env_obs["angular_velocity"]
    edge_np = _edge_valid_mask(st0, P_MAX, 0, min_send=1)
    src, tgt = parsed[0], parsed[1]
    res = compute_launch(src, tgt, 12, parsed, av)
    gate_t, frac_t = teacher_env_action_to_edge_targets(
        st0, obs, 0, [[src.id, float(res["angle"]), int(res["ships"])]],
        parsed, av, edge_np, p_max=P_MAX)

    obs_full = build_observation(env_obs, player=0,
                                 angular_velocity=av,
                                 p_max=P_MAX, f_max=64, turn=0, max_turns=300)
    batch = pack_reworked_batch([(obs_full, 0)], p_max=P_MAX)

    buf = ReworkedBuffer()
    buf.add_bc(EdgeBCSample(
        node_feats=batch["node_feats"][0].numpy(),
        edge_feats=batch["edge_feats"][0].numpy(),
        global_feats=batch["global_feats"][0].numpy(),
        planet_mask=batch["planet_mask"][0].numpy(),
        edge_mask=edge_np,
        gate_target=gate_t,
        frac_target=frac_t,
    ))
    bc = buf.stack_bc_edge()
    assert bc["gate"].shape == (1, P_MAX, P_MAX)
    assert bc["edge_mask"].dtype == torch.bool

    policy = ReworkedEdgePolicy(d_model=32, n_layers=1)
    ev = policy.evaluate_actions(
        node_feats=bc["node_feats"], edge_feats=bc["edge_feats"],
        global_feats=bc["global_feats"], planet_mask=bc["planet_mask"],
        edge_mask=bc["edge_mask"],
        action={"gate": bc["gate"], "frac": bc["frac"]},
    )
    tvm = ev["target_valid"]
    assert bool(tvm.any())
    bc_loss = -(ev["target_logp"] * tvm.float()).sum() / tvm.float().sum()
    assert torch.isfinite(bc_loss)
