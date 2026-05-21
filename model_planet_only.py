from __future__ import annotations

import math

import torch
import torch.nn as nn

from obs_builder import (
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    P_MAX_DEFAULT,
    PLANET_FEATURE_DIM,
)
from model import OrbitWarsPolicy, PolicyOutput, TransformerBlock


class PlanetOnlyOrbitWarsPolicy(OrbitWarsPolicy):
    def __init__(
        self,
        *,
        p_max: int = P_MAX_DEFAULT,
        f_max: int = 0,
        planet_dim: int = PLANET_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.0,
    ):
        nn.Module.__init__(self)
        self.p_max = p_max
        self.f_max = f_max
        self.planet_dim = planet_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.model_variant = "planet_only"

        self.planet_proj = nn.Linear(planet_dim, d_model)
        self.global_proj = nn.Linear(global_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model)
        self.edge_proj = nn.Linear(edge_dim, n_heads)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.src_head = nn.Linear(d_model, 1)
        self.noop_logit = nn.Parameter(torch.zeros(1))
        self.tgt_query = nn.Linear(d_model, d_model)
        self.tgt_key = nn.Linear(d_model, d_model)
        # Ship Beta head: outputs raw (alpha, beta) logits.
        self.ship_head = nn.Linear(d_model, 2)
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.aux_owner_head = nn.Linear(d_model, 3)  # mine / enemy / neutral

        type_ids = torch.zeros(p_max + 1, dtype=torch.long)
        type_ids[-1] = 1
        self.register_buffer("type_ids", type_ids, persistent=False)

    def _build_tokens(
        self,
        planet_feats: torch.Tensor,
        fleet_feats: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> torch.Tensor:
        planet_tok = self.planet_proj(planet_feats)
        global_tok = self.global_proj(global_feats).unsqueeze(1)
        x = torch.cat([planet_tok, global_tok], dim=1)
        return x + self.type_embed(self.type_ids)[None]

    def _build_attn_bias(self, edge_bias: torch.Tensor) -> torch.Tensor:
        planet_bias = edge_bias[:, : self.p_max, : self.p_max]
        global_to_planet = edge_bias[:, -1:, : self.p_max]
        planet_to_global = edge_bias[:, : self.p_max, -1:]
        global_to_global = edge_bias[:, -1:, -1:]
        top = torch.cat([planet_bias, planet_to_global], dim=2)
        bottom = torch.cat([global_to_planet, global_to_global], dim=2)
        compact = torch.cat([top, bottom], dim=1)
        b = self.edge_proj(compact)
        return b.permute(0, 3, 1, 2).contiguous()

    def _build_padding_mask(
        self, planet_mask: torch.Tensor, fleet_mask: torch.Tensor
    ) -> torch.Tensor:
        B = planet_mask.shape[0]
        global_real = torch.ones(B, 1, dtype=torch.bool, device=planet_mask.device)
        real = torch.cat([planet_mask, global_real], dim=1)
        return ~real

    def forward(
        self,
        planet_feats: torch.Tensor,
        fleet_feats: torch.Tensor,
        global_feats: torch.Tensor,
        edge_bias: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> PolicyOutput:
        x = self._build_tokens(planet_feats, fleet_feats, global_feats)
        attn_bias = self._build_attn_bias(edge_bias)
        kpm = self._build_padding_mask(planet_mask, fleet_mask)

        for blk in self.blocks:
            x = blk(x, attn_bias, kpm)
        x = self.norm(x)

        planet_tok = x[:, : self.p_max]
        global_tok = x[:, -1]
        planet_pooled = planet_tok.mean(dim=1)
        value_input = torch.cat([global_tok, planet_pooled], dim=-1)
        value = self.value_head(value_input).squeeze(-1)
        aux_owner_logits = self.aux_owner_head(planet_tok)
        src_raw = self.src_head(planet_tok).squeeze(-1)
        very_neg = torch.finfo(src_raw.dtype).min
        src_raw = src_raw.masked_fill(~source_mask, very_neg)
        B = planet_tok.shape[0]
        noop = self.noop_logit.expand(B, 1)
        src_logits = torch.cat([src_raw, noop], dim=1)

        q = self.tgt_query(planet_tok)
        k = self.tgt_key(planet_tok)
        tgt_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        eye = torch.eye(self.p_max, dtype=torch.bool, device=q.device)
        tgt_logits = tgt_logits.masked_fill(eye[None], very_neg)
        tgt_logits = tgt_logits.masked_fill((~planet_mask)[:, None, :], very_neg)

        ship_params = self.ship_head(planet_tok)        # (B, P, 2)

        return PolicyOutput(
            src_logits=src_logits,
            tgt_logits=tgt_logits,
            ship_params=ship_params,
            value=value,
            tokens=x,
            aux_owner_logits=aux_owner_logits,
        )
