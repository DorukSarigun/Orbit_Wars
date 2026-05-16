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
from model import PolicyOutput
from model_grouped_planet import GroupedPlanetOrbitWarsPolicy, PlanetProjector


class EdgeAwareMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.edge_score = nn.Linear(edge_dim, n_heads)
        self.edge_value = nn.Linear(edge_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_pair: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        edge_scores = self.edge_score(edge_pair).permute(0, 3, 1, 2).contiguous()
        scores = scores + edge_scores
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        if self.dropout > 0.0 and self.training:
            attn = nn.functional.dropout(attn, p=self.dropout)
        content_out = torch.matmul(attn, v)
        edge_v = self.edge_value(edge_pair).reshape(B, T, T, H, Dh).permute(0, 3, 1, 2, 4)
        edge_out = (attn[..., None] * edge_v).sum(dim=-2)
        out = (content_out + edge_out).transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class EdgeAwareTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = EdgeAwareMultiHeadAttention(d_model, n_heads, edge_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_pair: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), edge_pair, key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


class EdgeAwareGroupedPlanetRelationEncoder(nn.Module):
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
        attention_type: str = "full",
        n_inducing_points: int = 32,
        n_pool_seeds: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if attention_type != "full":
            raise ValueError("grouped_planet_edge supports only full attention")
        self.p_max = p_max
        self.f_max = f_max
        self.planet_dim = planet_dim
        self.global_dim = global_dim
        self.edge_dim = edge_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_type = attention_type
        self.n_inducing_points = n_inducing_points
        self.n_pool_seeds = n_pool_seeds
        self.planet_proj = PlanetProjector(planet_dim=planet_dim, d_model=d_model)
        self.blocks = nn.ModuleList([
            EdgeAwareTransformerBlock(d_model, n_heads, edge_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool_seeds = nn.Parameter(torch.randn(n_pool_seeds, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(d_model)
        self.summary_fusion = nn.Sequential(
            nn.Linear(n_pool_seeds * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def encode_planets(
        self,
        planet_feats: torch.Tensor,
        fleet_feats: torch.Tensor,
        global_feats: torch.Tensor,
        edge_bias: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self.planet_proj(planet_feats)
        edge_pair = edge_bias[:, : self.p_max, : self.p_max, :]
        kpm = ~planet_mask
        for blk in self.blocks:
            x = blk(x, edge_pair, kpm)
        x = self.norm(x)
        B = x.shape[0]
        seeds = self.pool_seeds.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(
            query=seeds,
            key=x,
            value=x,
            key_padding_mask=kpm,
            need_weights=False,
        )
        pooled = self.pool_norm(seeds + pooled)
        global_summary = self.summary_fusion(pooled.reshape(B, self.n_pool_seeds * self.d_model))
        planet_tok = x * planet_mask[:, :, None].to(x.dtype)
        return {
            "planet_tokens": planet_tok,
            "pool_seeds": pooled,
            "global_summary": global_summary,
        }


class EdgeAwareGroupedPlanetOrbitWarsPolicy(GroupedPlanetOrbitWarsPolicy):
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
        attention_type: str = "full",
        n_inducing_points: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__(
            p_max=p_max,
            f_max=f_max,
            planet_dim=planet_dim,
            global_dim=global_dim,
            edge_dim=edge_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            attention_type=attention_type,
            n_inducing_points=n_inducing_points,
            dropout=dropout,
        )
        self.model_variant = "grouped_planet_edge"
        self.encoder = EdgeAwareGroupedPlanetRelationEncoder(
            p_max=p_max,
            f_max=f_max,
            planet_dim=planet_dim,
            global_dim=global_dim,
            edge_dim=edge_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            attention_type=attention_type,
            n_inducing_points=n_inducing_points,
            dropout=dropout,
        )
        self.pair_edge_proj = nn.Sequential(
            nn.Linear(edge_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.pair_edge_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.ship_head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

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
        enc = self.encoder.encode_planets(
            planet_feats=planet_feats,
            fleet_feats=fleet_feats,
            global_feats=global_feats,
            edge_bias=edge_bias,
            planet_mask=planet_mask,
            fleet_mask=fleet_mask,
        )
        planet_tok = enc["planet_tokens"]
        global_summary = self.summary_head(torch.cat([
            enc["global_summary"],
            self.global_feat_proj(global_feats),
        ], dim=-1))
        global_per_planet = global_summary[:, None, :].expand(-1, self.p_max, -1)
        src_input = torch.cat([planet_tok, global_per_planet], dim=-1)
        source_launch = self.src_head(src_input).squeeze(-1)
        noop_raw = self.noop_head(src_input).squeeze(-1)
        very_neg = torch.finfo(noop_raw.dtype).min
        source_launch = source_launch.masked_fill(~source_mask, very_neg)
        noop_raw = noop_raw.masked_fill(~source_mask, very_neg)
        noop = noop_raw.max(dim=1, keepdim=True).values
        src_logits = torch.cat([noop_raw, noop], dim=1)

        edge_pair = edge_bias[:, : self.p_max, : self.p_max, :]
        edge_tok = self.pair_edge_proj(edge_pair)
        q = self.tgt_query(planet_tok)
        k = self.tgt_key(planet_tok)
        tgt_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        tgt_logits = tgt_logits + source_launch[:, :, None]
        tgt_logits = tgt_logits + self.pair_edge_score(edge_tok).squeeze(-1)
        eye = torch.eye(self.p_max, dtype=torch.bool, device=q.device)
        tgt_logits = tgt_logits.masked_fill(eye[None], very_neg)
        tgt_logits = tgt_logits.masked_fill((~planet_mask)[:, None, :], very_neg)
        tgt_logits = tgt_logits.masked_fill((~source_mask)[:, :, None], very_neg)

        src_pair = planet_tok[:, :, None, :].expand(-1, -1, self.p_max, -1)
        tgt_pair = planet_tok[:, None, :, :].expand(-1, self.p_max, -1, -1)
        global_pair = global_summary[:, None, None, :].expand(-1, self.p_max, self.p_max, -1)
        ship_input = torch.cat([src_pair, tgt_pair, edge_tok, global_pair], dim=-1)
        ship_params = self.ship_head(ship_input)
        value = self.value_head(global_summary).squeeze(-1)
        aux_owner_logits = self.aux_owner_head(planet_tok)

        return PolicyOutput(
            src_logits=src_logits,
            tgt_logits=tgt_logits,
            ship_params=ship_params,
            value=value,
            tokens=planet_tok,
            aux_owner_logits=aux_owner_logits,
        )
