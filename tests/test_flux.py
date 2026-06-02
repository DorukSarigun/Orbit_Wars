"""Tests for the flux-augmented credit-assignment pipeline.

Covers:
    * Foundation: ``obs_builder`` populates fleet routing arrays correctly.
    * Helpers: ``rl.flux`` computes flux matrices / node features under
      each parametric mode (``raw``, ``linear``, ``exp_decay``).
    * Phase 1: ``rl.rewards.compute_graph_metrics_summary`` returns a
      13-channel summary when ``enable_flux_metrics`` is on, and
      ``compute_graph_native_reward`` actually uses the new flux weights.
"""

from __future__ import annotations

import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Foundation: fleet routing arrays on Observation
# ---------------------------------------------------------------------------


def _build_real_obs_from_env():
    """Step the kaggle env a few turns to obtain a real observation.

    Skipped if the env or torch deps are unavailable.
    """
    pytest.importorskip("kaggle_environments")
    from kaggle_environments import make
    from obs_builder import build_observation

    env = make("orbit_wars", debug=False)
    env.configuration["randomSeed"] = 7
    state = env.reset(num_agents=2)
    state = env.step([[] for _ in range(2)])  # env initialises planets here
    av = float(state[0].observation.get("angular_velocity", 0.0) or 0.0)

    # Force a launch from one of our planets so a fleet is in flight.
    # Try multiple (src, tgt) pairs because compute_launch can return None
    # for sun-blocked routes or impossible intercepts in a given seed.
    planets = state[0].observation["planets"]  # list of [id, owner, x, y, r, ships, prod]
    mine = [p for p in planets if int(p[1]) == 0]
    others = [p for p in planets if int(p[1]) != 0]
    assert mine and others, "expected at least one owned + one foreign planet"
    from feature_extraction import parse_planets
    from navigation import compute_launch
    parsed = parse_planets(state[0].observation)
    launch = None
    src = tgt = None
    for cand_src in mine:
        for cand_tgt in others:
            sp = next((p for p in parsed if p.id == int(cand_src[0])), None)
            tp = next((p for p in parsed if p.id == int(cand_tgt[0])), None)
            if sp is None or tp is None:
                continue
            cand = compute_launch(sp, tp, max(1, int(cand_src[5]) // 2), parsed, av)
            if cand is not None:
                src, tgt, launch = cand_src, cand_tgt, cand
                break
        if launch is not None:
            break
    if launch is None:
        pytest.skip("no valid launch in this seed (all sun-blocked)")
    # env action format: list-of-moves per player; each move is [src_id, angle, ships]
    a0 = [[int(src[0]), float(launch["angle"]), int(launch["ships"])]]
    state = env.step([a0, []])
    for _ in range(3):
        state = env.step([[], []])

    obs = build_observation(state[0].observation, player=0, angular_velocity=av)
    return obs, av


def test_observation_has_fleet_routing_fields():
    """Foundation: new routing arrays present and well-shaped."""
    obs, _ = _build_real_obs_from_env()
    f_max = obs.fleet_mask.shape[0]
    assert obs.fleet_src_slot.shape == (f_max,)
    assert obs.fleet_tgt_slot.shape == (f_max,)
    assert obs.fleet_eta_raw.shape == (f_max,)
    assert obs.fleet_ships_raw.shape == (f_max,)
    assert obs.fleet_is_self.shape == (f_max,)
    assert obs.fleet_is_enemy.shape == (f_max,)
    assert obs.fleet_src_slot.dtype == np.int32
    assert obs.fleet_tgt_slot.dtype == np.int32
    assert obs.fleet_eta_raw.dtype == np.float32
    assert obs.fleet_ships_raw.dtype == np.float32


def test_routing_populated_for_active_fleet():
    """At least one in-flight fleet must have valid src/ships/ownership."""
    obs, _ = _build_real_obs_from_env()
    active = obs.fleet_mask
    if not active.any():
        pytest.skip("no fleets in flight after smoke launch (rare)")
    assert (obs.fleet_src_slot[active] >= 0).any()
    assert (obs.fleet_ships_raw[active] > 0).any()
    # The launch we triggered belongs to player 0.
    assert obs.fleet_is_self[active].any()
    # Inactive slots stay zero / -1 (Observation default).
    inactive = ~active
    if inactive.any():
        assert (obs.fleet_src_slot[inactive] == -1).all()
        assert (obs.fleet_tgt_slot[inactive] == -1).all()
        assert (obs.fleet_ships_raw[inactive] == 0.0).all()


# ---------------------------------------------------------------------------
# Flux helpers
# ---------------------------------------------------------------------------


def _synthetic_obs(p_max: int = 5, f_max: int = 4):
    """Lightweight obs stub matching what ``rl.flux`` reads."""

    class _O:
        pass

    o = _O()
    o.planet_feats = np.zeros((p_max, 34), dtype=np.float32)
    o.planet_mask = np.array([True] * (p_max - 1) + [False])
    o.fleet_mask = np.array([1, 1, 1, 0], dtype=bool)
    o.fleet_src_slot = np.array([0, 1, 2, -1], dtype=np.int32)
    o.fleet_tgt_slot = np.array([1, 2, 0, -1], dtype=np.int32)
    o.fleet_eta_raw = np.array([0.0, 10.0, 50.0, 0.0], dtype=np.float32)
    o.fleet_ships_raw = np.array([10.0, 20.0, 30.0, 0.0], dtype=np.float32)
    o.fleet_is_self = np.array([1, 0, 0, 0], dtype=bool)
    o.fleet_is_enemy = np.array([0, 1, 1, 0], dtype=bool)
    return o


def test_flux_mode_raw_uses_ship_count():
    from rl.flux import compute_flux_matrices

    obs = _synthetic_obs()
    m = compute_flux_matrices(obs, mode="raw")
    assert m.friendly[0, 1] == pytest.approx(10.0)
    assert m.enemy[1, 2] == pytest.approx(20.0)
    assert m.enemy[2, 0] == pytest.approx(30.0)
    # Inactive fleets contribute zero.
    assert m.friendly.sum() == pytest.approx(10.0)
    assert m.enemy.sum() == pytest.approx(50.0)


def test_flux_mode_linear_zeros_far_fleets():
    """linear: weight = ships * max(0, 1 - eta/eta_norm); eta=50 -> 0."""
    from rl.flux import compute_flux_matrices

    obs = _synthetic_obs()
    m = compute_flux_matrices(obs, mode="linear", eta_norm=50.0)
    assert m.friendly[0, 1] == pytest.approx(10.0)              # eta=0  -> w=1.0
    assert m.enemy[1, 2] == pytest.approx(20.0 * 0.8, rel=1e-6) # eta=10 -> w=0.8
    assert m.enemy[2, 0] == pytest.approx(0.0)                  # eta=50 -> w=0.0


def test_flux_mode_exp_decay_uses_tau():
    from rl.flux import compute_flux_matrices

    obs = _synthetic_obs()
    m = compute_flux_matrices(obs, mode="exp_decay", tau=10.0)
    assert m.friendly[0, 1] == pytest.approx(10.0, rel=1e-5)             # exp(0)=1
    assert m.enemy[1, 2] == pytest.approx(20.0 * math.exp(-1.0), rel=1e-5)
    assert m.enemy[2, 0] == pytest.approx(30.0 * math.exp(-5.0), rel=1e-5)


def test_node_flux_features_layout_and_mask():
    """Channels [friendly_in, friendly_out, enemy_in, enemy_out]; padded zeroed."""
    from rl.flux import compute_node_flux_features, NODE_FLUX_CHANNELS

    obs = _synthetic_obs()
    feat = compute_node_flux_features(obs, mode="raw", ship_norm=1.0)
    assert feat.shape == (obs.planet_feats.shape[0], NODE_FLUX_CHANNELS)
    # Slot 0: friendly_out=10 (sends to 1); enemy_in=30 (receives from 2).
    assert feat[0, 1] == pytest.approx(10.0)
    assert feat[0, 2] == pytest.approx(30.0)
    # Slot 1: friendly_in=10 (recv from 0); enemy_out=20 (sends to 2).
    assert feat[1, 0] == pytest.approx(10.0)
    assert feat[1, 3] == pytest.approx(20.0)
    # Padded slot stays zero.
    assert np.all(feat[-1] == 0)


def test_node_flux_ship_norm_scales_output():
    from rl.flux import compute_node_flux_features

    obs = _synthetic_obs()
    raw = compute_node_flux_features(obs, mode="raw", ship_norm=1.0)
    scaled = compute_node_flux_features(obs, mode="raw", ship_norm=50.0)
    np.testing.assert_allclose(scaled, raw / 50.0)


def test_pair_flux_edge_features_align_with_edge_index():
    from rl.flux import compute_flux_matrices, compute_pair_flux_edge_features

    obs = _synthetic_obs()
    n = 3
    dist = np.array(
        [[0.0, 0.1, 0.2], [0.3, 0.0, 0.4], [0.5, 0.6, 0.0]],
        dtype=np.float32,
    )
    edge_index = np.array(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2], [1, 2, 0, 0, 2, 1, 0, 1, 2]],
        dtype=np.int64,
    )
    mats = compute_flux_matrices(obs, p_max=n, mode="raw")
    edge_attr = compute_pair_flux_edge_features(
        obs, edge_index, dist, n_real=n, mode="raw", ship_norm=1.0, matrices=mats,
    )
    assert edge_attr.shape == (9, 3)
    src, dst = edge_index
    np.testing.assert_allclose(edge_attr[:, 0], dist[src, dst])
    np.testing.assert_allclose(edge_attr[:, 1], mats.friendly[src, dst])
    np.testing.assert_allclose(edge_attr[:, 2], mats.enemy[src, dst])
    assert edge_attr[0, 1] == pytest.approx(10.0)  # friendly 0->1


def test_invalid_flux_mode_raises():
    from rl.flux import compute_flux_matrices

    obs = _synthetic_obs()
    with pytest.raises(ValueError):
        compute_flux_matrices(obs, mode="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Phase 1: reward summary + reward integration
# ---------------------------------------------------------------------------


def test_metric_vector_dim_switches_on_flux_flag():
    """``compute_planet_metric_vectors`` returns 10 vs 14 channels."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from rl.rewards import RewardConfig, compute_planet_metric_vectors

    obs, _ = _build_real_obs_from_env()
    m_off = compute_planet_metric_vectors(obs, cfg=RewardConfig(enable_flux_metrics=False))
    m_on = compute_planet_metric_vectors(obs, cfg=RewardConfig(enable_flux_metrics=True))
    assert m_off is not None and m_off.shape[1] == 10
    assert m_on is not None and m_on.shape[1] == 14


def test_flux_channel_weights_independent():
    """Arrival-credit flux weights are independent configuration knobs."""
    from rl.arrival_credit import build_channel_weights
    from rl.rewards import RewardConfig

    cfg = RewardConfig(
        arrival_credit_flux_friendly_in_weight=3.0,
        arrival_credit_flux_enemy_out_weight=-2.0,
    )
    w = build_channel_weights(cfg)
    assert w[10] == pytest.approx(3.0)
    assert w[13] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# Phase 3: encoder / policy opt-in flux features
# ---------------------------------------------------------------------------


def test_encoder_input_dim_grows_with_flux_flag():
    """``PygPlanetGraphEncoder(enable_flux_features=True)`` ingests 48 cols."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from model_graph_pyg import (
        PYG_PLANET_FEATURE_DIM,
        PYG_PLANET_FEATURE_DIM_FLUX,
        PygPlanetGraphEncoder,
    )

    enc_off = PygPlanetGraphEncoder(d_model=32, n_layers=1)
    enc_on = PygPlanetGraphEncoder(d_model=32, n_layers=1, enable_flux_features=True)
    assert enc_off.planet_dim == PYG_PLANET_FEATURE_DIM
    assert enc_on.planet_dim == PYG_PLANET_FEATURE_DIM_FLUX
    # LayerNorm + Linear: the first projection's in_features tells us
    # what the encoder actually expects on input.
    proj_in_off = enc_off.node_proj[1].in_features
    proj_in_on = enc_on.node_proj[1].in_features
    assert proj_in_off == 44
    assert proj_in_on == 48


def test_observation_to_pyg_data_respects_flux_features():
    """``Data.x`` widens to 48 columns when ``flux_features`` is passed."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from model_graph_pyg import GRAPH_NODE_FLUX_DIM, observation_to_pyg_data

    obs, _ = _build_real_obs_from_env()
    n = int(obs.planet_mask.sum())
    flux = torch.zeros(obs.planet_feats.shape[0], GRAPH_NODE_FLUX_DIM)
    data_off = observation_to_pyg_data(obs)
    data_on = observation_to_pyg_data(obs, flux_features=flux)
    assert data_off.x.shape == (n, 44)
    assert data_on.x.shape == (n, 48)


def test_packed_observations_to_pyg_batch_with_flux_shape_error():
    """Wrong-shape flux_features must fail loudly."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from model_graph_pyg import packed_observations_to_pyg_batch
    from obs_builder import P_MAX_DEFAULT, PLANET_FEATURE_DIM, EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM

    B, P = 2, P_MAX_DEFAULT
    batch = {
        "planet_feats": torch.zeros(B, P, PLANET_FEATURE_DIM),
        "edge_bias": torch.zeros(B, P, P, EDGE_FEATURE_DIM),
        "planet_mask": torch.zeros(B, P, dtype=torch.bool),
        "source_mask": torch.zeros(B, P, dtype=torch.bool),
        "global_feats": torch.zeros(B, GLOBAL_FEATURE_DIM),
    }
    # Mark a few real planets so the helper runs through node-metric path.
    batch["planet_mask"][:, :3] = True
    with pytest.raises(ValueError):
        packed_observations_to_pyg_batch(batch, flux_features=torch.zeros(B, P, 3))


def test_policy_forward_with_flux_features_shape():
    """End-to-end forward through ``PygEdgeNativePolicy`` with flux on."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from model_graph_pyg_edgeonly import PygEdgeNativePolicy
    from obs_builder import (
        EDGE_FEATURE_DIM,
        F_MAX_DEFAULT,
        FLEET_FEATURE_DIM,
        GLOBAL_FEATURE_DIM,
        P_MAX_DEFAULT,
        PLANET_FEATURE_DIM,
    )

    p = P_MAX_DEFAULT
    pol = PygEdgeNativePolicy(p_max=p, d_model=32, n_layers=1, enable_flux_features=True)
    B = 2
    planet_feats = torch.zeros(B, p, PLANET_FEATURE_DIM)
    fleet_feats = torch.zeros(B, F_MAX_DEFAULT, FLEET_FEATURE_DIM)
    global_feats = torch.zeros(B, GLOBAL_FEATURE_DIM)
    edge_bias = torch.zeros(B, p, p, EDGE_FEATURE_DIM)
    planet_mask = torch.zeros(B, p, dtype=torch.bool)
    planet_mask[:, :4] = True
    fleet_mask = torch.zeros(B, F_MAX_DEFAULT, dtype=torch.bool)
    source_mask = planet_mask.clone()

    # Missing flux_features -> loud error.
    with pytest.raises(ValueError):
        pol(planet_feats, fleet_feats, global_feats, edge_bias,
            planet_mask, fleet_mask, source_mask)

    flux = torch.zeros(B, p, 4)
    # Missing edge_flux_features -> loud error (paired with node flux).
    with pytest.raises(ValueError):
        pol(planet_feats, fleet_feats, global_feats, edge_bias,
            planet_mask, fleet_mask, source_mask, flux_features=flux)

    edge_flux = torch.zeros(B, p, p, 2)
    out = pol(planet_feats, fleet_feats, global_feats, edge_bias,
              planet_mask, fleet_mask, source_mask,
              flux_features=flux, edge_flux_features=edge_flux)
    assert out.tgt_logits.shape == (B, p, p)
    assert out.value.shape == (B,)


# ---------------------------------------------------------------------------
# Phase 2: SinkhornEdgeHead
# ---------------------------------------------------------------------------


def test_sinkhorn_log_yields_doubly_stochastic_plan():
    """Unmasked Sinkhorn iterations make rows AND cols sum to 1 in exp-space."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from model_graph_pyg import SinkhornEdgeHead

    torch.manual_seed(0)
    B, P = 2, 6
    raw = torch.randn(B, P, P)
    L = SinkhornEdgeHead.sinkhorn_log(raw, num_iterations=30)
    plan = L.exp()
    row_sums = plan.sum(dim=2)
    col_sums = plan.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)


def test_sinkhorn_log_respects_edge_mask():
    """Masked cells stay at -inf and the remaining sub-matrix is doubly-stochastic."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from model_graph_pyg import SinkhornEdgeHead

    torch.manual_seed(1)
    B, P = 1, 5
    raw = torch.randn(B, P, P)
    mask = torch.ones(B, P, P, dtype=torch.bool)
    # Drop self-loops -- the canonical edge mask for our action space.
    eye = torch.eye(P, dtype=torch.bool)
    mask[:, eye] = False
    L = SinkhornEdgeHead.sinkhorn_log(raw, num_iterations=50, edge_mask=mask)
    plan = L.exp()
    # Masked cells must be exactly zero in exp-space.
    assert (plan[0][eye] == 0).all()
    # Row/col sums equal one over the *non-masked* support.
    row_sums = plan.sum(dim=2)
    col_sums = plan.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)


def test_sinkhorn_zero_iters_is_identity_up_to_temperature():
    """num_iterations=0 must skip the normalization (modulo tau scaling)."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from model_graph_pyg import SinkhornEdgeHead

    head = SinkhornEdgeHead(d_model=8, num_iterations=0, tau=1.0)
    src = torch.randn(2, 4, 8)
    tgt = torch.randn(2, 4, 8)
    edge = torch.randn(2, 4, 4, 8)
    glb = torch.randn(2, 8)
    out_zero = head(src, tgt, edge, glb).squeeze(-1)
    # The same scorer wrapped without Sinkhorn must produce the same value.
    raw = head.scorer(src, tgt, edge, glb).squeeze(-1)
    assert torch.allclose(out_zero, raw, atol=1e-6)


def test_sinkhorn_invalid_args():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from model_graph_pyg import SinkhornEdgeHead

    with pytest.raises(ValueError):
        SinkhornEdgeHead(d_model=8, tau=0.0)
    with pytest.raises(ValueError):
        SinkhornEdgeHead(d_model=8, num_iterations=-1)


def test_policy_with_sinkhorn_forward_and_log_prob():
    """End-to-end: PygEdgeNativePolicy(enable_sinkhorn=True) trains-like."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch
    from model_graph_pyg_edgeonly import PygEdgeNativePolicy
    from obs_builder import (
        EDGE_FEATURE_DIM,
        F_MAX_DEFAULT,
        FLEET_FEATURE_DIM,
        GLOBAL_FEATURE_DIM,
        P_MAX_DEFAULT,
        PLANET_FEATURE_DIM,
    )

    p = P_MAX_DEFAULT
    pol = PygEdgeNativePolicy(
        p_max=p, d_model=32, n_layers=1,
        enable_sinkhorn=True, sinkhorn_iters=4, sinkhorn_tau=0.7,
    )
    B = 2
    planet_feats = torch.zeros(B, p, PLANET_FEATURE_DIM)
    fleet_feats = torch.zeros(B, F_MAX_DEFAULT, FLEET_FEATURE_DIM)
    global_feats = torch.zeros(B, GLOBAL_FEATURE_DIM)
    edge_bias = torch.zeros(B, p, p, EDGE_FEATURE_DIM)
    planet_mask = torch.zeros(B, p, dtype=torch.bool)
    planet_mask[:, :5] = True
    fleet_mask = torch.zeros(B, F_MAX_DEFAULT, dtype=torch.bool)
    source_mask = planet_mask.clone()

    out = pol(planet_feats, fleet_feats, global_feats, edge_bias,
              planet_mask, fleet_mask, source_mask)
    assert out.tgt_logits.shape == (B, p, p)
    # Action + log-prob round-trip works identically to the bilinear path.
    act = pol.act(out, deterministic=False)
    ev = pol.evaluate_actions(
        out, act["src"], act["tgt"], act["ship_value"],
    )
    assert ev["log_prob"].shape == (B,)
    assert ev["entropy"].shape == (B,)
    # Logits over valid edges must be finite for the masked sub-matrix.
    eye = torch.eye(p, dtype=torch.bool)
    valid = (~eye[None]) & planet_mask[:, None, :] & source_mask[:, :, None]
    assert torch.isfinite(out.tgt_logits[valid]).all()


# ---------------------------------------------------------------------------
# Buffer / PPO plumbing
# ---------------------------------------------------------------------------


def _mk_transition(*, with_flux: bool, p_max: int = 8):
    """Construct a single-step ``Transition`` for buffer tests."""
    from obs_builder import (
        EDGE_FEATURE_DIM, FLEET_FEATURE_DIM, GLOBAL_FEATURE_DIM, PLANET_FEATURE_DIM,
    )
    from rl.buffer import Transition

    f_max = 4
    return Transition(
        player=0,
        planet_feats=np.zeros((p_max, PLANET_FEATURE_DIM), dtype=np.float32),
        fleet_feats=np.zeros((f_max, FLEET_FEATURE_DIM), dtype=np.float32),
        global_feats=np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32),
        edge_bias=np.zeros((p_max, p_max, EDGE_FEATURE_DIM), dtype=np.float32),
        planet_mask=np.ones(p_max, dtype=bool),
        fleet_mask=np.zeros(f_max, dtype=bool),
        source_mask=np.ones(p_max, dtype=bool),
        src=0, tgt=1, ship_value=0.5, is_noop=False,
        log_prob=-1.0, value=0.0, reward=0.0, done=False, turn=0,
        flux_features=(np.ones((p_max, 4), dtype=np.float32) if with_flux else None),
    )


def test_buffer_stack_includes_flux_when_present():
    from rl.buffer import RolloutBuffer

    buf = RolloutBuffer()
    for _ in range(3):
        buf.add(_mk_transition(with_flux=True))
    buf.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)
    batch = next(iter(buf.iter_minibatches(batch_size=3, shuffle=False)))
    assert "flux_features" in batch
    assert tuple(batch["flux_features"].shape) == (3, 8, 4)


def test_buffer_stack_omits_flux_when_absent():
    from rl.buffer import RolloutBuffer

    buf = RolloutBuffer()
    for _ in range(3):
        buf.add(_mk_transition(with_flux=False))
    buf.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)
    batch = next(iter(buf.iter_minibatches(batch_size=3, shuffle=False)))
    assert "flux_features" not in batch


def test_buffer_mixed_flux_raises():
    """Mixing flux/no-flux transitions in a batch must fail loudly."""
    from rl.buffer import RolloutBuffer

    buf = RolloutBuffer()
    buf.add(_mk_transition(with_flux=True))
    buf.add(_mk_transition(with_flux=False))
    buf.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)
    with pytest.raises(ValueError):
        list(buf.iter_minibatches(batch_size=2, shuffle=False))


# ---------------------------------------------------------------------------
# Factory + CLI integration
# ---------------------------------------------------------------------------


def test_train_factory_pyg_variants_with_flux_sinkhorn():
    """``_make_policy`` accepts new variants + flags; round-trips checkpoint."""
    pytest.importorskip("torch_geometric")
    import tempfile
    from pathlib import Path
    import torch
    from rl.train import _make_policy, _save_ckpt, load_checkpoint, TrainConfig
    from rl.ppo import PPOConfig
    from rl.rewards import RewardConfig

    pol = _make_policy(
        "pyg_edge_native",
        d_model=32, n_heads=4, n_layers=1,
        attention_type="full", n_inducing_points=32,
        enable_flux_features=True, enable_sinkhorn=True,
        sinkhorn_iters=3, sinkhorn_tau=0.8,
    )
    assert pol.model_variant == "pyg_edge_native"
    assert pol.enable_flux_features is True
    assert pol.enable_sinkhorn is True
    assert pol.sinkhorn_iters == 3
    # Legacy variants must silently drop the new kwargs.
    legacy = _make_policy(
        "planet_only", d_model=16, n_heads=2, n_layers=1, dropout=0.0,
        attention_type="full", n_inducing_points=8,
        enable_flux_features=True, enable_sinkhorn=True,
        sinkhorn_iters=2, sinkhorn_tau=0.5,
    )
    assert legacy.__class__.__name__ == "PlanetOnlyOrbitWarsPolicy"
    # Checkpoint round-trip preserves the flux/sinkhorn flags.
    opt = torch.optim.Adam(pol.parameters(), lr=1e-4)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = Path(f.name)
    try:
        _save_ckpt(
            pol, opt, ckpt_path, iteration=0,
            train_cfg=TrainConfig(model_variant="pyg_edge_native"),
            ppo_cfg=PPOConfig(), reward_cfg=RewardConfig(),
        )
        pol2, _ = load_checkpoint(ckpt_path)
    finally:
        ckpt_path.unlink(missing_ok=True)
    assert pol2.enable_flux_features is True
    assert pol2.enable_sinkhorn is True
    assert pol2.sinkhorn_iters == 3


def test_train_cli_parses_new_flags():
    from rl.train import _build_argparser

    args = _build_argparser().parse_args([
        "--model-variant", "pyg_edge_native",
        "--enable-flux-features",
        "--enable-sinkhorn",
        "--sinkhorn-iters", "7",
        "--sinkhorn-tau", "0.4",
    ])
    assert args.model_variant == "pyg_edge_native"
    assert args.enable_flux_features is True
    assert args.enable_sinkhorn is True
    assert args.sinkhorn_iters == 7
    assert args.sinkhorn_tau == pytest.approx(0.4)
    # Defaults stay backwards compatible.
    defaults = _build_argparser().parse_args([])
    assert defaults.enable_flux_features is False
    assert defaults.enable_sinkhorn is False
    assert defaults.model_variant == "planet_only"


def test_flux_disabled_summary_ignores_weights():
    """When ``enable_flux_metrics=False`` flux channels are not in metric vectors."""
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from rl.arrival_credit import build_channel_weights
    from rl.rewards import RewardConfig, compute_planet_metric_vectors

    obs, _ = _build_real_obs_from_env()
    cfg = RewardConfig(
        enable_flux_metrics=False,
        arrival_credit_flux_friendly_in_weight=10.0,
    )
    mat = compute_planet_metric_vectors(obs, cfg=cfg)
    assert mat is not None and mat.shape[1] == 10
    w = build_channel_weights(cfg)
    assert w.shape[0] == 14
    assert w[10] == pytest.approx(10.0)
