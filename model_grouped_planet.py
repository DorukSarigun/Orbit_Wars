from __future__ import annotations

import math

import torch
import torch.nn as nn

from obs_builder import (
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    P_MAX_DEFAULT,
    PLANET_FEATURE_DIM,
    PF_IDX_CAPTURE_VALUE,
    PF_IDX_COS_THETA,
    PF_IDX_DEFENSE_DEFICIT,
    PF_IDX_INCOMING_ENEMY_LARGEST_ETA,
    PF_IDX_INCOMING_ENEMY_LARGEST_SHIPS,
    PF_IDX_INCOMING_ENEMY_MIN_ETA,
    PF_IDX_INCOMING_ENEMY_SOONEST_SHIPS,
    PF_IDX_INCOMING_ENEMY_TOTAL,
    PF_IDX_INCOMING_NET_THREAT,
    PF_IDX_INCOMING_SELF_B0,
    PF_IDX_INCOMING_SELF_MIN_ETA,
    PF_IDX_INCOMING_SELF_TOTAL,
    PF_IDX_IS_STATIC,
    PF_IDX_LOG_SHIPS,
    PF_IDX_OWNER_SELF,
    PF_IDX_PRODUCTION,
    PF_IDX_R_SUN,
    PF_IDX_RADIUS,
    PF_IDX_SHIPS,
)
from model import PolicyOutput, TransformerBlock


class CrossMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Q, _ = query.shape
        K = key_value.shape[1]
        H, Dh = self.n_heads, self.d_head
        q = self.q_proj(query).reshape(B, Q, H, Dh).transpose(1, 2)
        kv = self.kv_proj(key_value).reshape(B, K, 2, H, Dh).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        if self.dropout > 0.0 and self.training:
            attn = nn.functional.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, Q, self.d_model)
        return self.out_proj(out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.key_value_norm = nn.LayerNorm(d_model)
        self.attn = CrossMultiHeadAttention(d_model, n_heads, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = query + self.attn(
            self.query_norm(query),
            self.key_value_norm(key_value),
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.ff(self.ff_norm(query))
        return query


class InducedSetAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_inducing_points: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(n_inducing_points, d_model) * 0.02)
        self.induce = CrossAttentionBlock(d_model, n_heads, dropout)
        self.project = CrossAttentionBlock(d_model, n_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B = x.shape[0]
        inducing = self.inducing_points.unsqueeze(0).expand(B, -1, -1)
        h = self.induce(inducing, x, key_padding_mask=key_padding_mask)
        return self.project(x, h)


class PlanetFeatureGroupProjector(nn.Module):
    def __init__(self, planet_dim: int = PLANET_FEATURE_DIM, d_model: int = 256):
        super().__init__()
        if planet_dim < PLANET_FEATURE_DIM:
            raise ValueError(f"planet_dim={planet_dim} is smaller than PLANET_FEATURE_DIM={PLANET_FEATURE_DIM}")
        self.planet_dim = int(planet_dim)
        self.n_groups = 6
        self.owner = nn.Sequential(nn.Linear(5, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.economy = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.geometry = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.dynamics = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.incoming = nn.Sequential(nn.Linear(15, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.value = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        owner = x[..., PF_IDX_OWNER_SELF:PF_IDX_OWNER_SELF + 5]
        economy = torch.cat([
            x[..., PF_IDX_SHIPS:PF_IDX_SHIPS + 1],
            x[..., PF_IDX_PRODUCTION:PF_IDX_PRODUCTION + 1],
            x[..., PF_IDX_RADIUS:PF_IDX_RADIUS + 1],
            x[..., PF_IDX_LOG_SHIPS:PF_IDX_LOG_SHIPS + 1],
        ], dim=-1)
        geometry = x[..., PF_IDX_R_SUN:PF_IDX_COS_THETA + 2]
        dynamics = x[..., PF_IDX_IS_STATIC:PF_IDX_IS_STATIC + 4]
        incoming = x[..., PF_IDX_INCOMING_SELF_B0:PF_IDX_INCOMING_ENEMY_LARGEST_ETA + 1]
        value = x[..., PF_IDX_INCOMING_NET_THREAT:PF_IDX_DEFENSE_DEFICIT + 1]
        return torch.stack([
            self.owner(owner),
            self.economy(economy),
            self.geometry(geometry),
            self.dynamics(dynamics),
            self.incoming(incoming),
            self.value(value),
        ], dim=2)


class GroupedPlanetRelationEncoder(nn.Module):
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
        super().__init__()
        if attention_type not in ("full", "induced"):
            raise ValueError(f"attention_type must be 'full' or 'induced', got {attention_type!r}")
        self.p_max = p_max
        self.f_max = f_max
        self.planet_dim = planet_dim
        self.global_dim = global_dim
        self.edge_dim = edge_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_type = attention_type
        self.n_inducing_points = n_inducing_points
        self.planet_proj = PlanetFeatureGroupProjector(planet_dim=planet_dim, d_model=d_model)
        self.n_planet_groups = self.planet_proj.n_groups
        self.group_embed = nn.Embedding(self.n_planet_groups, d_model)
        self.edge_proj = nn.Linear(edge_dim, n_heads)
        block_cls = TransformerBlock if attention_type == "full" else InducedSetAttentionBlock
        if attention_type == "full":
            self.blocks = nn.ModuleList([
                block_cls(d_model, n_heads, dropout=dropout)
                for _ in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                block_cls(d_model, n_heads, n_inducing_points=n_inducing_points, dropout=dropout)
                for _ in range(n_layers)
            ])
        self.norm = nn.LayerNorm(d_model)
        self.group_fusion = nn.Sequential(
            nn.Linear(self.n_planet_groups * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.pool_seeds = nn.Parameter(torch.randn(self.n_planet_groups, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(d_model)
        self.summary_fusion = nn.Sequential(
            nn.Linear(self.n_planet_groups * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def _build_tokens(
        self,
        planet_feats: torch.Tensor,
        fleet_feats: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> torch.Tensor:
        B = planet_feats.shape[0]
        group_tok = self.planet_proj(planet_feats)
        group_tok = group_tok + self.group_embed.weight[None, None]
        return group_tok.reshape(B, self.p_max * self.n_planet_groups, self.d_model)

    def _build_attn_bias(self, edge_bias: torch.Tensor) -> torch.Tensor:
        G = self.n_planet_groups
        planet_pair = edge_bias[:, : self.p_max, : self.p_max]
        planet_pair = planet_pair[:, :, None, :, None, :].expand(-1, -1, G, -1, G, -1)
        planet_pair = planet_pair.reshape(edge_bias.shape[0], self.p_max * G, self.p_max * G, edge_bias.shape[-1])
        b = self.edge_proj(planet_pair)
        return b.permute(0, 3, 1, 2).contiguous()

    def _build_padding_mask(
        self, planet_mask: torch.Tensor, fleet_mask: torch.Tensor
    ) -> torch.Tensor:
        B = planet_mask.shape[0]
        group_real = planet_mask[:, :, None].expand(-1, -1, self.n_planet_groups)
        return ~group_real.reshape(B, self.p_max * self.n_planet_groups)

    def _pool_groups(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        seeds = self.pool_seeds.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_attn(
            query=seeds,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.pool_norm(seeds + pooled)
        global_summary = self.summary_fusion(pooled.reshape(B, self.n_planet_groups * self.d_model))
        return pooled, global_summary

    def encode_planets(
        self,
        planet_feats: torch.Tensor,
        fleet_feats: torch.Tensor,
        global_feats: torch.Tensor,
        edge_bias: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B = planet_feats.shape[0]
        x = self._build_tokens(planet_feats, fleet_feats, global_feats)
        attn_bias = self._build_attn_bias(edge_bias) if self.attention_type == "full" else None
        kpm = self._build_padding_mask(planet_mask, fleet_mask)

        for blk in self.blocks:
            x = blk(x, attn_bias, kpm)
        x = self.norm(x)
        group_summary, global_summary = self._pool_groups(x, kpm)

        group_tok = x[:, : self.p_max * self.n_planet_groups]
        group_tok = group_tok.reshape(B, self.p_max, self.n_planet_groups, self.d_model)
        planet_tok = self.group_fusion(group_tok.reshape(B, self.p_max, self.n_planet_groups * self.d_model))
        group_mask = planet_mask[:, :, None, None].to(group_tok.dtype)
        planet_mask_f = planet_mask[:, :, None].to(planet_tok.dtype)
        group_tok = group_tok * group_mask
        planet_tok = planet_tok * planet_mask_f
        return {
            "group_tokens": group_tok,
            "planet_tokens": planet_tok,
            "group_summary_tokens": group_summary,
            "global_summary": global_summary,
        }

    def forward(
        self,
        planet_feats: torch.Tensor,
        fleet_feats: torch.Tensor,
        global_feats: torch.Tensor,
        edge_bias: torch.Tensor,
        planet_mask: torch.Tensor,
        fleet_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.encode_planets(
            planet_feats=planet_feats,
            fleet_feats=fleet_feats,
            global_feats=global_feats,
            edge_bias=edge_bias,
            planet_mask=planet_mask,
            fleet_mask=fleet_mask,
        )


class GroupedPlanetOrbitWarsPolicy(nn.Module):
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
        n_ship_bins: int = 5,
        attention_type: str = "full",
        n_inducing_points: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p_max = p_max
        self.f_max = f_max
        self.planet_dim = planet_dim
        self.global_dim = global_dim
        self.edge_dim = edge_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_ship_bins = n_ship_bins
        self.model_variant = "grouped_planet"
        self.attention_type = attention_type
        self.n_inducing_points = n_inducing_points
        self.encoder = GroupedPlanetRelationEncoder(
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
        self.global_feat_proj = nn.Sequential(
            nn.Linear(global_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.summary_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.src_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.noop_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.tgt_query = nn.Linear(d_model, d_model)
        self.tgt_key = nn.Linear(d_model, d_model)
        self.ship_head = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_ship_bins),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.aux_owner_head = nn.Linear(d_model, 5)

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
        B = planet_tok.shape[0]
        global_per_planet = global_summary[:, None, :].expand(-1, self.p_max, -1)
        src_input = torch.cat([planet_tok, global_per_planet], dim=-1)
        source_launch = self.src_head(src_input).squeeze(-1)
        noop_raw = self.noop_head(src_input).squeeze(-1)
        very_neg = torch.finfo(noop_raw.dtype).min
        source_launch = source_launch.masked_fill(~source_mask, very_neg)
        noop_raw = noop_raw.masked_fill(~source_mask, very_neg)
        noop = noop_raw.max(dim=1, keepdim=True).values
        src_logits = torch.cat([noop_raw, noop], dim=1)

        q = self.tgt_query(planet_tok)
        k = self.tgt_key(planet_tok)
        tgt_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        tgt_logits = tgt_logits + source_launch[:, :, None]
        eye = torch.eye(self.p_max, dtype=torch.bool, device=q.device)
        tgt_logits = tgt_logits.masked_fill(eye[None], very_neg)
        tgt_logits = tgt_logits.masked_fill((~planet_mask)[:, None, :], very_neg)
        tgt_logits = tgt_logits.masked_fill((~source_mask)[:, :, None], very_neg)

        src_pair = planet_tok[:, :, None, :].expand(-1, -1, self.p_max, -1)
        tgt_pair = planet_tok[:, None, :, :].expand(-1, self.p_max, -1, -1)
        global_pair = global_summary[:, None, None, :].expand(-1, self.p_max, self.p_max, -1)
        ship_input = torch.cat([src_pair, tgt_pair, global_pair], dim=-1)
        ship_logits = self.ship_head(ship_input)
        value = self.value_head(global_summary).squeeze(-1)
        aux_owner_logits = self.aux_owner_head(planet_tok)
        tokens = enc["group_tokens"].reshape(B, self.p_max * self.encoder.n_planet_groups, self.d_model)

        return PolicyOutput(
            src_logits=src_logits,
            tgt_logits=tgt_logits,
            ship_logits=ship_logits,
            value=value,
            tokens=tokens,
            aux_owner_logits=aux_owner_logits,
        )

    @torch.no_grad()
    def act(
        self,
        out: PolicyOutput,
        deterministic: bool = False,
    ) -> dict:
        B = out.tgt_logits.shape[0]
        flat_pair = out.tgt_logits.reshape(B, self.p_max * self.p_max)
        noop = out.src_logits[:, self.p_max:self.p_max + 1]
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

        ship_logits_sel = out.ship_logits[torch.arange(B, device=out.tgt_logits.device), src, tgt]
        ship_dist = torch.distributions.Categorical(logits=ship_logits_sel)
        if deterministic:
            ship = ship_logits_sel.argmax(dim=-1)
        else:
            ship = ship_dist.sample()

        src = torch.where(is_noop, torch.full_like(src, self.p_max), src)
        tgt = torch.where(is_noop, torch.zeros_like(tgt), tgt)
        ship = torch.where(is_noop, torch.zeros_like(ship), ship)
        valid = (~is_noop).float()
        log_prob = action_dist.log_prob(action_idx) + valid * ship_dist.log_prob(ship)
        return {
            "src": src,
            "tgt": tgt,
            "ship_bin": ship,
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

        ship_logits_sel = out.ship_logits[torch.arange(B, device=source_slot.device), source_slot, tgt]
        if deterministic:
            ship = ship_logits_sel.argmax(dim=-1)
        else:
            masked_ship = ship_logits_sel.clone()
            masked_ship[is_noop] = float("-inf")
            masked_ship[is_noop, 0] = 0.0
            ship = torch.distributions.Categorical(logits=masked_ship).sample()

        ship_logp = torch.distributions.Categorical(logits=ship_logits_sel).log_prob(ship)
        log_prob = action_dist.log_prob(action_idx) + valid * ship_logp
        return {
            "src": source_slot,
            "tgt": tgt,
            "ship_bin": ship,
            "is_noop": is_noop,
            "log_prob": log_prob,
        }

    def evaluate_actions(
        self,
        out: PolicyOutput,
        src: torch.Tensor,
        tgt: torch.Tensor,
        ship_bin: torch.Tensor,
    ) -> dict:
        src = src.long()
        tgt = tgt.long()
        ship_bin = ship_bin.long()
        is_noop = src == self.p_max
        B = out.tgt_logits.shape[0]
        flat_pair = out.tgt_logits.reshape(B, self.p_max * self.p_max)
        noop = out.src_logits[:, self.p_max:self.p_max + 1]
        action_logits = torch.cat([noop, flat_pair], dim=1)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        src_idx = torch.where(is_noop, torch.zeros_like(src), src)
        tgt_idx = torch.where(is_noop, torch.zeros_like(tgt), tgt)
        action_idx = torch.where(is_noop, torch.zeros_like(src), src_idx * self.p_max + tgt_idx + 1)

        ship_logits_sel = out.ship_logits[torch.arange(B, device=src.device), src_idx, tgt_idx]
        ship_dist = torch.distributions.Categorical(logits=ship_logits_sel)

        valid = (~is_noop).float()
        log_prob = action_dist.log_prob(action_idx) + valid * ship_dist.log_prob(ship_bin)
        entropy = action_dist.entropy() + valid * ship_dist.entropy()
        return {"log_prob": log_prob, "entropy": entropy, "value": out.value}

    def evaluate_source_actions(
        self,
        out: PolicyOutput,
        src: torch.Tensor,
        tgt: torch.Tensor,
        ship_bin: torch.Tensor,
        is_noop: torch.Tensor,
    ) -> dict:
        src = src.long()
        tgt = tgt.long()
        ship_bin = ship_bin.long()
        is_noop = is_noop.bool()
        B = out.src_logits.shape[0]
        noop_sel = out.src_logits[torch.arange(B, device=src.device), src].unsqueeze(-1)
        tgt_logits_sel = out.tgt_logits[torch.arange(B, device=src.device), src]
        action_logits = torch.cat([noop_sel, tgt_logits_sel], dim=1)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action_idx = torch.where(is_noop, torch.zeros_like(tgt), tgt + 1)

        tgt_idx = torch.where(is_noop, torch.zeros_like(tgt), tgt)
        ship_logits_sel = out.ship_logits[torch.arange(B, device=src.device), src, tgt_idx]
        ship_dist = torch.distributions.Categorical(logits=ship_logits_sel)

        valid = (~is_noop).float()
        log_prob = action_dist.log_prob(action_idx) + valid * ship_dist.log_prob(ship_bin)
        entropy = action_dist.entropy() + valid * ship_dist.entropy()
        return {"log_prob": log_prob, "entropy": entropy, "value": out.value}
