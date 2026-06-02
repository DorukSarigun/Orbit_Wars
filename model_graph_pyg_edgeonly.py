from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    PolicyOutput,
    _beta_concentrations,
    _clamp_for_log_prob,
    _make_ship_dist,
)
from model_graph_pyg import (
    PYG_EDGE_FEATURE_DIM,
    PYG_PLANET_FEATURE_DIM,
    PYG_PLANET_FEATURE_DIM_FLUX,
    BilinearEdgeHead,
    PygPlanetGraphEncoder,
    SinkhornEdgeHead,
    _batched_pair_edge_features,
    packed_observations_to_pyg_batch,
    pyg_nodes_to_padded,
)
from obs_builder import (
    EDGE_FEATURE_DIM,
    F_MAX_DEFAULT,
    GLOBAL_FEATURE_DIM,
    P_MAX_DEFAULT,
    PLANET_FEATURE_DIM,
)


class PygEdgeNativePolicy(nn.Module):
    """PyG GNN policy with a fully edge-native action head.

    Each directed edge (src -> tgt) receives a scalar logit for "launch along
    this edge", plus a single global noop logit.  The action space is
    therefore ``1 + P*P`` categorical choices, identical to the grouped
    planet variants, but the network never factors through an explicit
    source-selection stage.
    """

    def __init__(
        self,
        *,
        p_max: int = P_MAX_DEFAULT,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.0,
        enable_flux_features: bool = False,
        enable_sinkhorn: bool = False,
        sinkhorn_iters: int = 5,
        sinkhorn_tau: float = 1.0,
        use_input_embed: bool = True,
    ):
        """Edge-native planet-graph PPO policy.

        Set ``enable_sinkhorn=True`` to replace the plain
        :class:`BilinearEdgeHead` edge scorer with a
        :class:`SinkhornEdgeHead` that runs ``sinkhorn_iters`` log-domain
        row / col normalization passes before returning logits.  Action
        sampling, log-prob evaluation, and the rest of the PPO
        machinery are unchanged.

        ``enable_flux_features`` widens the encoder input by the
        four node and two edge flux channels (node 43 -> 47, edge
        15 -> 17): ``[friendly_in, friendly_out, enemy_in,
        enemy_out]`` on nodes and ``[friendly_flux_ij, enemy_flux_ij]``
        on directed edges (Phase 3 of the credit-assignment plan).

        ``use_input_embed`` (default ``True``) gates the NodeEmbed /
        EdgeEmbed projections inside the encoder.  When set to
        ``False`` the GINE convs ingest the raw graph features
        directly (matches the ``DirectFcPlanetGnn`` prototype).
        """
        super().__init__()
        self.p_max = p_max
        self.d_model = d_model
        self.model_variant = "pyg_edge_native"
        self.enable_sinkhorn = bool(enable_sinkhorn)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_tau = float(sinkhorn_tau)
        # ``planet_dim`` is exposed as the *raw* feature dim for rollout
        # / build_observation compatibility.  The encoder receives
        # augmented features (PYG_PLANET_FEATURE_DIM = 43, or 47 when
        # ``enable_flux_features`` is on) appended internally by
        # ``packed_observations_to_pyg_batch``.
        self.planet_dim = PLANET_FEATURE_DIM
        self.enable_flux_features = bool(enable_flux_features)
        self.use_input_embed = bool(use_input_embed)
        encoder_planet_dim = (
            PYG_PLANET_FEATURE_DIM_FLUX if enable_flux_features else PYG_PLANET_FEATURE_DIM
        )
        self.encoder_planet_dim = encoder_planet_dim
        self.global_dim = GLOBAL_FEATURE_DIM
        self.edge_dim = EDGE_FEATURE_DIM
        self.f_max = F_MAX_DEFAULT

        self.encoder = PygPlanetGraphEncoder(
            planet_dim=encoder_planet_dim,
            edge_dim=PYG_EDGE_FEATURE_DIM,
            global_dim=GLOBAL_FEATURE_DIM,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            enable_flux_features=enable_flux_features,
            use_input_embed=use_input_embed,
        )

        # Edge-native heads (Bilinear: fuses src/tgt/edge/global with O(B*P) heavy
        # projections instead of O(B*P*P) 4*d_model concat MLP).  Edge projection
        # is shared via self.encoder.edge_proj.
        if enable_sinkhorn:
            # Wraps a BilinearEdgeHead internally plus K log-domain
            # row / col normalization passes.  Returns (B, P, P, 1).
            self.edge_head = SinkhornEdgeHead(
                d_model, num_iterations=sinkhorn_iters, tau=sinkhorn_tau,
            )
        else:
            self.edge_head = BilinearEdgeHead(d_model, output_dim=1)
        self.ship_head = BilinearEdgeHead(d_model, output_dim=2)
        self.noop_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.aux_owner_head = nn.Linear(d_model, 5)

    # --------------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------------
    def forward(
        self,
        planet_feats: torch.Tensor,
        fleet_feats: torch.Tensor,
        global_feats: torch.Tensor,
        edge_bias: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
        source_mask: torch.Tensor,
        flux_features: Optional[torch.Tensor] = None,
        edge_flux_features: Optional[torch.Tensor] = None,
    ) -> PolicyOutput:
        # When the encoder was built with flux features, callers must
        # supply both the node ``(B, P_max, 4)`` and edge
        # ``(B, P_max, P_max, 2)`` tensors; otherwise we fail loudly
        # rather than silently broadcasting zeros (which would mask bugs
        # in the rollout / buffer plumbing).
        if self.enable_flux_features and flux_features is None:
            raise ValueError(
                "PygEdgeNativePolicy was built with enable_flux_features=True "
                "but `flux_features` was not passed to forward()."
            )
        if self.enable_flux_features and edge_flux_features is None:
            raise ValueError(
                "PygEdgeNativePolicy was built with enable_flux_features=True "
                "but `edge_flux_features` was not passed to forward()."
            )
        # 1. Pack observations into a PyG Batch.
        batch_dict: dict[str, torch.Tensor] = {
            "planet_feats": planet_feats,
            "global_feats": global_feats,
            "edge_bias": edge_bias,
            "planet_mask": planet_mask,
            "source_mask": source_mask,
        }
        if self.enable_flux_features:
            batch_dict["flux_features"] = flux_features  # type: ignore[assignment]
            batch_dict["edge_flux_features"] = edge_flux_features  # type: ignore[assignment]
        graph_batch = packed_observations_to_pyg_batch(batch_dict)

        # 2. Encode with GINEConv layers.
        enc = self.encoder(graph_batch)
        planet_tok = pyg_nodes_to_padded(enc["node_tokens"], graph_batch, p_max=self.p_max)
        graph_summary = enc["graph_summary"]  # (B, d_model)

        # 3. Vectorized pairwise edge features (one batched ETA pass), then
        # project via the shared encoder edge_proj for consistency with
        # message-passing.
        edge_pair = _batched_pair_edge_features(
            planet_feats, edge_bias, planet_mask,
            edge_flux_features=edge_flux_features if self.enable_flux_features else None,
        )
        edge_tok = self.encoder.edge_proj(edge_pair)  # (B, P, P, d_model)

        # 4. Edge logits / ship params via Bilinear (or Sinkhorn) edge head.
        if self.enable_sinkhorn:
            # Build a valid-edge mask so Sinkhorn doesn't normalise mass
            # through self-loops or pad / non-source cells.
            eye_mask = torch.eye(self.p_max, dtype=torch.bool, device=planet_feats.device)
            edge_valid = (
                (~eye_mask)[None]
                & planet_mask[:, None, :]
                & source_mask[:, :, None]
            )
            edge_logits = self.edge_head(
                planet_tok, planet_tok, edge_tok, graph_summary,
                edge_mask=edge_valid,
            ).squeeze(-1)
        else:
            edge_logits = self.edge_head(planet_tok, planet_tok, edge_tok, graph_summary).squeeze(-1)

        # 5. Noop logit  ->  (B, 1)
        noop_logit = self.noop_head(graph_summary)  # (B, 1)

        # 6. Ship params  ->  (B, P, P, 2)
        ship_params = self.ship_head(planet_tok, planet_tok, edge_tok, graph_summary)

        # 7. Value  ->  (B,)
        value = self.value_head(graph_summary).squeeze(-1)

        # 8. Aux owner  ->  (B, P_max, 5)
        aux_owner_logits = self.aux_owner_head(planet_tok)

        # 9. Masking
        very_neg = torch.finfo(edge_logits.dtype).min
        eye = torch.eye(self.p_max, dtype=torch.bool, device=planet_feats.device)
        edge_logits = edge_logits.masked_fill(eye[None], very_neg)
        edge_logits = edge_logits.masked_fill((~planet_mask)[:, None, :], very_neg)
        edge_logits = edge_logits.masked_fill((~source_mask)[:, :, None], very_neg)

        # 11. Backward-compatible PolicyOutput.
        # src_logits shape = (B, P_max+1)  ->  [:P_max] = per-source edge summary,
        #                                      [P_max]   = noop
        src_logits = torch.cat([
            edge_logits.max(dim=2).values,  # (B, P_max)
            noop_logit,                     # (B, 1)
        ], dim=1)

        return PolicyOutput(
            src_logits=src_logits,
            tgt_logits=edge_logits,
            ship_params=ship_params,
            value=value,
            tokens=planet_tok,
            aux_owner_logits=aux_owner_logits,
        )

    # --------------------------------------------------------------------------
    # Action sampling (edge-native, compatible with grouped-planet rollout)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, out: PolicyOutput, deterministic: bool = False) -> dict:
        B = out.tgt_logits.shape[0]
        flat_pair = out.tgt_logits.reshape(B, self.p_max * self.p_max)
        noop = out.src_logits[:, self.p_max : self.p_max + 1]
        action_logits = torch.cat([noop, flat_pair], dim=1)
        action_dist = torch.distributions.Categorical(logits=action_logits)

        if deterministic:
            action_idx = action_logits.argmax(dim=-1)
        else:
            action_idx = action_dist.sample()

        is_noop = action_idx == 0
        pair_idx = torch.clamp(action_idx - 1, min=0)
        src = pair_idx // self.p_max
        tgt = pair_idx % self.p_max

        ship_params_sel = out.ship_params[
            torch.arange(B, device=out.tgt_logits.device), src, tgt
        ]
        ship_dist = _make_ship_dist(ship_params_sel)
        if deterministic:
            alpha, beta = _beta_concentrations(ship_params_sel)
            ship_value = alpha / (alpha + beta)
        else:
            ship_value = ship_dist.sample()

        src = torch.where(is_noop, torch.full_like(src, self.p_max), src)
        tgt = torch.where(is_noop, torch.zeros_like(tgt), tgt)
        ship_value = torch.where(is_noop, torch.full_like(ship_value, 0.5), ship_value)
        valid = (~is_noop).float()
        ship_logp = ship_dist.log_prob(_clamp_for_log_prob(ship_value))
        log_prob = action_dist.log_prob(action_idx) + valid * ship_logp

        return {
            "src": src,
            "tgt": tgt,
            "ship_value": ship_value,
            "is_noop": is_noop,
            "log_prob": log_prob,
        }

    @torch.no_grad()
    def act_source(
        self,
        out: PolicyOutput,
        source_slot: torch.Tensor,
        deterministic: bool = False,
    ) -> dict:
        source_slot = source_slot.to(out.src_logits.device).long()
        B = out.src_logits.shape[0]
        noop_sel = out.src_logits[torch.arange(B, device=source_slot.device), source_slot].unsqueeze(-1)
        tgt_logits_sel = out.tgt_logits[torch.arange(B, device=source_slot.device), source_slot]
        action_logits = torch.cat([noop_sel, tgt_logits_sel], dim=1)
        action_dist = torch.distributions.Categorical(logits=action_logits)

        if deterministic:
            action_idx = action_logits.argmax(dim=-1)
        else:
            action_idx = action_dist.sample()

        is_noop = action_idx == 0
        valid = (~is_noop).float()
        tgt = torch.clamp(action_idx - 1, min=0)

        ship_params_sel = out.ship_params[
            torch.arange(B, device=source_slot.device), source_slot, tgt
        ]
        ship_dist = _make_ship_dist(ship_params_sel)
        if deterministic:
            alpha, beta = _beta_concentrations(ship_params_sel)
            ship_value = alpha / (alpha + beta)
        else:
            ship_value = ship_dist.sample()

        ship_logp = ship_dist.log_prob(_clamp_for_log_prob(ship_value))
        log_prob = action_dist.log_prob(action_idx) + valid * ship_logp

        return {
            "src": source_slot,
            "tgt": tgt,
            "ship_value": ship_value,
            "is_noop": is_noop,
            "log_prob": log_prob,
        }

    # --------------------------------------------------------------------------
    # Log-prob / entropy for PPO update
    # --------------------------------------------------------------------------
    def evaluate_actions(
        self,
        out: PolicyOutput,
        src: torch.Tensor,
        tgt: torch.Tensor,
        ship_value: torch.Tensor,
    ) -> dict:
        src = src.long()
        tgt = tgt.long()
        ship_value = ship_value.float()
        is_noop = src == self.p_max
        B = out.tgt_logits.shape[0]

        flat_pair = out.tgt_logits.reshape(B, self.p_max * self.p_max)
        noop = out.src_logits[:, self.p_max : self.p_max + 1]
        action_logits = torch.cat([noop, flat_pair], dim=1)
        action_dist = torch.distributions.Categorical(logits=action_logits)

        src_idx = torch.where(is_noop, torch.zeros_like(src), src)
        tgt_idx = torch.where(is_noop, torch.zeros_like(tgt), tgt)
        pair_idx = src_idx * self.p_max + tgt_idx
        action_idx = torch.where(is_noop, torch.zeros_like(pair_idx), pair_idx + 1)
        log_prob = action_dist.log_prob(action_idx)

        ship_params_sel = out.ship_params[
            torch.arange(B, device=src.device), src_idx, tgt_idx
        ]
        ship_dist = _make_ship_dist(ship_params_sel)
        ship_logp = ship_dist.log_prob(_clamp_for_log_prob(ship_value))
        valid = (~is_noop).float()
        log_prob = log_prob + valid * ship_logp

        return {
            "log_prob": log_prob,
            "entropy": action_dist.entropy(),
        }

    def evaluate_source_actions(
        self,
        out: PolicyOutput,
        src: torch.Tensor,
        tgt: torch.Tensor,
        ship_value: torch.Tensor,
        is_noop: torch.Tensor,
    ) -> dict:
        """Conditional-on-src log_prob / entropy.

        Mirrors :meth:`act_source` exactly so that callers that *fix* the
        source slot (e.g. the scripted neutral-expansion bootstrap in
        ``rl/rollout._scripted_neutral_expansion_action``) get a log
        probability that's consistent with the action the network would
        have sampled.  This is the edge-native counterpart of
        :meth:`GroupedPlanetOrbitWarsPolicy.evaluate_source_actions` so
        the rollout helpers stay polymorphic across policy variants.

        The action distribution is built from
        ``[src_logits[:, src], tgt_logits[:, src, :]]``: the per-source
        noop logit followed by the row of edge logits leaving that
        source.  The continuous ship-fraction term is taken from
        ``ship_params[:, src, tgt]`` and only contributes when
        ``is_noop`` is False.
        """
        src = src.long()
        tgt = tgt.long()
        ship_value = ship_value.float()
        is_noop = is_noop.bool()
        B = out.src_logits.shape[0]
        idx_b = torch.arange(B, device=src.device)
        # Per-source noop logit + per-source target logits, matching
        # act_source's conditional categorical exactly.
        noop_sel = out.src_logits[idx_b, src].unsqueeze(-1)
        tgt_logits_sel = out.tgt_logits[idx_b, src]
        action_logits = torch.cat([noop_sel, tgt_logits_sel], dim=1)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action_idx = torch.where(is_noop, torch.zeros_like(tgt), tgt + 1)

        tgt_idx = torch.where(is_noop, torch.zeros_like(tgt), tgt)
        ship_params_sel = out.ship_params[idx_b, src, tgt_idx]
        ship_dist = _make_ship_dist(ship_params_sel)
        ship_logp = ship_dist.log_prob(_clamp_for_log_prob(ship_value))
        valid = (~is_noop).float()
        log_prob = action_dist.log_prob(action_idx) + valid * ship_logp

        # Population-level entropy: H(action | src) + Pr(launch | src) ·
        # H(ship | src, tgt).  Same convention used by the grouped
        # variant so PPO entropy bonuses stay consistent across heads.
        action_probs = torch.softmax(action_logits, dim=-1)
        p_nonnoop = 1.0 - action_probs[..., 0]
        entropy = action_dist.entropy() + p_nonnoop * ship_dist.entropy()
        value = out.value if out.value is not None else None
        return {"log_prob": log_prob, "entropy": entropy, "value": value}
