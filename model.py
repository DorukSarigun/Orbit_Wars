"""
Orbit Wars - Set Transformer policy / value network.

Token layout (length T = P_max + F_max + 1):
    [planet_0, ..., planet_{P-1}, fleet_0, ..., fleet_{F-1}, GLOBAL]

The first P_max slots are planets, the next F_max are fleets, and the
final slot holds a global summary token (CLS-style) computed from the
global feature vector. This ordering must match `obs_builder`.

Action factorization (matches `obs_builder.decode_action`):
    p(a | s) = p_src(slot) * p_tgt(slot | src) * p_ships(bin | src, tgt)

A NO-OP action is exposed via an extra synthetic source slot (index P_max),
so the source policy has P_max + 1 logits.

Edge bias: the (T, T, edge_dim) tensor from the observation is projected
to an (n_heads, T, T) additive bias tensor and added to attention scores
before softmax. This gives the model geometric priors without forcing it
to learn distance from raw coordinates.

Usage
-----
    >>> net = OrbitWarsPolicy()
    >>> out = net(obs_batch)             # see `forward` for shapes
    >>> action_dist = net.distribution(out, obs_batch)
    >>> src, tgt, bin_, log_prob, entropy = action_dist.sample()

For PPO/A2C, `out["value"]` is the critic output. For SAC/DQN you'd
restructure the heads, but the encoder is reusable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from obs_builder import (
    EDGE_FEATURE_DIM,
    F_MAX_DEFAULT,
    FLEET_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    P_MAX_DEFAULT,
    PLANET_FEATURE_DIM,
)


# =============================================================================
# Transformer block with additive attention bias
# =============================================================================
class BiasedMultiHeadAttention(nn.Module):
    """Standard MHA, but accepts a per-head additive bias of shape
    (B, n_heads, T, T) which is added to QK^T / sqrt(d) before softmax."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, d_model)
        attn_bias: Optional[torch.Tensor],    # (B, H, T, T) or None
        key_padding_mask: Optional[torch.Tensor],  # (B, T) bool, True=PAD
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        qkv = self.qkv(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # each (B, H, T, Dh)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,T,T)
        if attn_bias is not None:
            scores = scores + attn_bias

        if key_padding_mask is not None:
            # Mask out PAD keys: rows can attend, but never TO a padded key.
            mask = key_padding_mask[:, None, None, :]  # (B,1,1,T)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        # Numerical safety: a row that's all -inf produces NaN softmax. Replace
        # with zeros so output for that (PAD) query token is well-defined.
        attn = torch.nan_to_num(attn, nan=0.0)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)                      # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with biased attention."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = BiasedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x, attn_bias, key_padding_mask):
        x = x + self.attn(self.norm1(x), attn_bias, key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


# =============================================================================
# Full policy / value network
# =============================================================================
_BETA_MIN_CONCENTRATION = 1.0
_BETA_MAX_CONCENTRATION = 30.0
_BETA_SAMPLE_EPS = 1e-6


def _beta_concentrations(ship_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw (alpha, beta) logits to Beta concentrations in [1, MAX].

    Adding _BETA_MIN_CONCENTRATION after softplus guarantees both
    concentration parameters are >= 1, which keeps the Beta unimodal
    (avoids the bimodal U-shape that arises when alpha,beta < 1 and
    encourages either-extreme launches).

    The soft cap at _BETA_MAX_CONCENTRATION (via tanh) prevents the
    distribution from collapsing into a delta function during PPO
    updates: unbounded concentrations lead to log-prob ratios that
    explode beyond the clip range and effectively destroy the trust
    region (observed: KL > 100, clip_fraction > 0.9, polL spikes >>
    1e3).  With the cap, log p_Beta(sigma) is bounded across the
    [1, MAX] x [1, MAX] grid, so importance ratios stay well-behaved.
    """
    raw_alpha = ship_params[..., 0]
    raw_beta = ship_params[..., 1]
    alpha_unbounded = F.softplus(raw_alpha)
    beta_unbounded = F.softplus(raw_beta)
    span = _BETA_MAX_CONCENTRATION - _BETA_MIN_CONCENTRATION
    alpha = _BETA_MIN_CONCENTRATION + span * torch.tanh(alpha_unbounded / span)
    beta = _BETA_MIN_CONCENTRATION + span * torch.tanh(beta_unbounded / span)
    return alpha, beta


def _make_ship_dist(ship_params: torch.Tensor) -> torch.distributions.Beta:
    alpha, beta = _beta_concentrations(ship_params)
    return torch.distributions.Beta(alpha, beta)


def _clamp_for_log_prob(value: torch.Tensor) -> torch.Tensor:
    """Clamp Beta samples away from {0, 1} before computing log_prob.

    Beta is defined on the open interval (0, 1); values exactly at the
    boundary produce -inf log_prob. We clamp by EPS to keep gradients
    well-defined while preserving the action's float identity for
    downstream decoding.
    """
    return value.clamp(min=_BETA_SAMPLE_EPS, max=1.0 - _BETA_SAMPLE_EPS)


@dataclass
class PolicyOutput:
    src_logits: torch.Tensor       # (B, P_max + 1)  last slot = NO-OP
    tgt_logits: torch.Tensor       # (B, P_max, P_max)  per-source target logits
    ship_params: torch.Tensor      # (B, P_max, 2)  per-source Beta(alpha, beta) raw logits
    value: torch.Tensor            # (B,)
    tokens: torch.Tensor           # (B, T, d_model) for debugging/inspection
    aux_owner_logits: Optional[torch.Tensor] = None  # (B, P_max, 3) optional auxiliary head


class OrbitWarsPolicy(nn.Module):
    """Set Transformer with masked pointer policy and value head.

    Action factorization:
        src_slot   ~ Categorical(src_logits)        # P_max + 1 (incl. NO-OP)
        tgt_slot   ~ Categorical(tgt_logits[src])   # P_max
        ship_value ~ Beta(alpha, beta)              # continuous in (0, 1)

    The Beta sample is decoded by ``obs_builder.decode_action_with_launch``
    into an absolute ship request via ``ship_value * SHIP_REQUEST_CAP``;
    overflow above the source garrison is reported back so the rollout
    can apply a learning signal that discourages over-allocation.
    """

    def __init__(
        self,
        *,
        p_max: int = P_MAX_DEFAULT,
        f_max: int = F_MAX_DEFAULT,
        planet_dim: int = PLANET_FEATURE_DIM,
        fleet_dim: int = FLEET_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.p_max = p_max
        self.f_max = f_max
        self.planet_dim = planet_dim
        self.d_model = d_model
        self.n_heads = n_heads

        # Per-token-type input projections.
        self.planet_proj = nn.Linear(planet_dim, d_model)
        self.fleet_proj = nn.Linear(fleet_dim, d_model)
        self.global_proj = nn.Linear(global_dim, d_model)

        # Token-type embedding (planet=0, fleet=1, global=2). Helps the model
        # tell heterogeneous tokens apart even after projection to d_model.
        self.type_embed = nn.Embedding(3, d_model)

        # Edge feature projection -> per-head additive bias.
        self.edge_proj = nn.Linear(edge_dim, n_heads)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ---- Heads ----
        # Source logit: per planet slot. Plus a learned NO-OP logit.
        self.src_head = nn.Linear(d_model, 1)
        self.noop_logit = nn.Parameter(torch.zeros(1))

        # Target logit: pointer-style attention from each source planet token
        # to every (potential target) planet token. Implemented as a single
        # linear scoring head: score(src, tgt) = src @ W @ tgt^T / sqrt(d).
        self.tgt_query = nn.Linear(d_model, d_model)
        self.tgt_key = nn.Linear(d_model, d_model)

        # Ship Beta head: per planet slot, outputs raw (alpha, beta)
        # logits which are mapped through softplus + 1.0 in
        # _beta_concentrations to produce a unimodal Beta distribution
        # on (0, 1) per source planet.
        self.ship_head = nn.Linear(d_model, 2)

        # Value head: pooled global token through MLP.
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Pre-build static type-id tensor (planet, fleet, global).
        type_ids = torch.zeros(p_max + f_max + 1, dtype=torch.long)
        type_ids[p_max:p_max + f_max] = 1
        type_ids[-1] = 2
        self.register_buffer("type_ids", type_ids, persistent=False)

    # ------------------------------------------------------------------ utils
    def _build_tokens(
        self,
        planet_feats: torch.Tensor,    # (B, P, planet_dim)
        fleet_feats: torch.Tensor,     # (B, F, fleet_dim)
        global_feats: torch.Tensor,    # (B, global_dim)
    ) -> torch.Tensor:
        B = planet_feats.shape[0]
        planet_tok = self.planet_proj(planet_feats)
        fleet_tok = self.fleet_proj(fleet_feats)
        global_tok = self.global_proj(global_feats).unsqueeze(1)  # (B, 1, d)
        x = torch.cat([planet_tok, fleet_tok, global_tok], dim=1)  # (B, T, d)
        x = x + self.type_embed(self.type_ids)[None]
        return x

    def _build_attn_bias(self, edge_bias: torch.Tensor) -> torch.Tensor:
        # edge_bias: (B, T, T, edge_dim) -> (B, n_heads, T, T)
        b = self.edge_proj(edge_bias)
        return b.permute(0, 3, 1, 2).contiguous()

    def _build_padding_mask(
        self, planet_mask: torch.Tensor, fleet_mask: torch.Tensor
    ) -> torch.Tensor:
        # True = PAD (to be masked out). Global token is always real.
        B = planet_mask.shape[0]
        pad_global = torch.zeros(B, 1, dtype=torch.bool, device=planet_mask.device)
        real = torch.cat([planet_mask, fleet_mask, ~pad_global], dim=1)
        return ~real  # True where padding

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        planet_feats: torch.Tensor,    # (B, P, planet_dim)
        fleet_feats: torch.Tensor,     # (B, F, fleet_dim)
        global_feats: torch.Tensor,    # (B, global_dim)
        edge_bias: torch.Tensor,       # (B, T, T, edge_dim)
        planet_mask: torch.Tensor,     # (B, P) bool, True=real
        fleet_mask: torch.Tensor,      # (B, F) bool, True=real
        source_mask: torch.Tensor,     # (B, P) bool, True=can launch from
    ) -> PolicyOutput:
        x = self._build_tokens(planet_feats, fleet_feats, global_feats)
        attn_bias = self._build_attn_bias(edge_bias)
        kpm = self._build_padding_mask(planet_mask, fleet_mask)

        for blk in self.blocks:
            x = blk(x, attn_bias, kpm)
        x = self.norm(x)                               # (B, T, d)

        planet_tok = x[:, : self.p_max]                # (B, P, d)
        global_tok = x[:, -1]                          # (B, d)

        # ---- Source logits (P + 1 with NO-OP appended) ----
        src_raw = self.src_head(planet_tok).squeeze(-1)   # (B, P)
        # Mask invalid sources to -inf BEFORE concatenating NO-OP slot, so
        # the policy can't accidentally launch from an empty planet.
        very_neg = torch.finfo(src_raw.dtype).min
        src_raw = src_raw.masked_fill(~source_mask, very_neg)
        B = planet_tok.shape[0]
        noop = self.noop_logit.expand(B, 1)
        src_logits = torch.cat([src_raw, noop], dim=1)    # (B, P+1)

        # ---- Target logits (per-source pointer attention) ----
        q = self.tgt_query(planet_tok)                  # (B, P, d)
        k = self.tgt_key(planet_tok)                    # (B, P, d)
        tgt_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        # Mask self-targets and padded planets.
        eye = torch.eye(self.p_max, dtype=torch.bool, device=q.device)
        tgt_logits = tgt_logits.masked_fill(eye[None], very_neg)
        tgt_pad = ~planet_mask                          # (B, P) True=pad
        tgt_logits = tgt_logits.masked_fill(tgt_pad[:, None, :], very_neg)

        # ---- Ship Beta params (per source) ----
        ship_params = self.ship_head(planet_tok)        # (B, P, 2)

        # ---- Value ----
        value = self.value_head(global_tok).squeeze(-1) # (B,)

        return PolicyOutput(
            src_logits=src_logits,
            tgt_logits=tgt_logits,
            ship_params=ship_params,
            value=value,
            tokens=x,
        )

    # ----------------------------------------------------- action sampling API
    @torch.no_grad()
    def act(
        self,
        out: PolicyOutput,
        deterministic: bool = False,
    ) -> dict:
        """Sample (or argmax) one action per batch element from a policy
        output. Returns dict with src, tgt, ship_value (float in [0,1]),
        is_noop, log_prob.

        The factorization conditions tgt and ship_value on the chosen src.
        For deterministic mode the Beta head returns its mean
        ``alpha / (alpha + beta)``.
        """
        src_dist = torch.distributions.Categorical(logits=out.src_logits)
        if deterministic:
            src = out.src_logits.argmax(dim=-1)
        else:
            src = src_dist.sample()
        is_noop = src == self.p_max

        # For NO-OP rows we still need *some* tgt/ship choice for shape
        # consistency; callers must check is_noop. Clamp src to a valid
        # planet index for indexing.
        src_idx = torch.where(is_noop, torch.zeros_like(src), src)

        gather_idx = src_idx[:, None, None].expand(-1, 1, self.p_max)
        tgt_logits_sel = out.tgt_logits.gather(1, gather_idx).squeeze(1)  # (B, P)
        tgt_dist = torch.distributions.Categorical(logits=tgt_logits_sel)
        if deterministic:
            tgt = tgt_logits_sel.argmax(dim=-1)
        else:
            tgt = tgt_dist.sample()

        ship_params_sel = out.ship_params.gather(
            1, src_idx[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)  # (B, 2)
        ship_dist = _make_ship_dist(ship_params_sel)
        if deterministic:
            alpha, beta = _beta_concentrations(ship_params_sel)
            ship_value = alpha / (alpha + beta)
        else:
            ship_value = ship_dist.sample()

        log_prob = src_dist.log_prob(src)
        # Don't credit tgt/ship for NO-OP rows (they're degenerate).
        valid = (~is_noop).float()
        ship_logp = ship_dist.log_prob(_clamp_for_log_prob(ship_value))
        log_prob = log_prob + valid * (tgt_dist.log_prob(tgt) + ship_logp)

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
        src_raw = out.src_logits[:, : self.p_max]
        noop_raw = out.src_logits[:, self.p_max].unsqueeze(-1).expand_as(src_raw)
        launch_logits = torch.stack(
            [noop_raw, src_raw],
            dim=-1,
        )
        launch_sel = launch_logits[torch.arange(B, device=src_raw.device), source_slot]
        launch_dist = torch.distributions.Categorical(logits=launch_sel)
        if deterministic:
            launch = launch_sel.argmax(dim=-1)
        else:
            launch = launch_dist.sample()
        is_noop = launch == 0
        valid = (~is_noop).float()

        # Only sample target/ship if launching (avoid wasted computation on NO-OP).
        # Use zeros for NO-OP rows; they'll be masked out of log_prob anyway.
        gather_idx = source_slot[:, None, None].expand(-1, 1, self.p_max)
        tgt_logits_sel = out.tgt_logits.gather(1, gather_idx).squeeze(1)
        if deterministic:
            tgt = tgt_logits_sel.argmax(dim=-1)
        else:
            # Mask NO-OP rows so they don't affect sampling (we'll overwrite below)
            masked_logits = tgt_logits_sel.clone()
            masked_logits[is_noop] = float("-inf")
            masked_logits[is_noop, 0] = 0.0  # Ensure valid distribution for NO-OP rows
            tgt = torch.distributions.Categorical(logits=masked_logits).sample()

        ship_params_sel = out.ship_params.gather(
            1, source_slot[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)  # (B, 2)
        ship_dist = _make_ship_dist(ship_params_sel)
        if deterministic:
            alpha, beta = _beta_concentrations(ship_params_sel)
            ship_value = alpha / (alpha + beta)
        else:
            ship_value = ship_dist.sample()

        # Conditional log_prob: only add tgt/ship terms when not NO-OP.
        tgt_logp = torch.distributions.Categorical(logits=tgt_logits_sel).log_prob(tgt)
        ship_logp = ship_dist.log_prob(_clamp_for_log_prob(ship_value))
        log_prob = launch_dist.log_prob(launch) + valid * (tgt_logp + ship_logp)

        return {
            "src": source_slot,
            "tgt": tgt,
            "ship_value": ship_value,
            "is_noop": is_noop,
            "log_prob": log_prob,
        }

    # ----------------------------------------- PPO re-evaluation (with grad)
    def evaluate_actions(
        self,
        out: "PolicyOutput",
        src: torch.Tensor,         # (B,) long
        tgt: torch.Tensor,         # (B,) long
        ship_value: torch.Tensor,  # (B,) float in [0, 1]
    ) -> dict:
        """Re-evaluate stored actions under the current policy distribution.

        Used by PPO to compute the new log-prob and entropy of actions that
        were sampled by an *older* policy snapshot. Gradients flow back
        through the logits (no `no_grad` here).

        Returns dict with `log_prob` (B,), `entropy` (B,), and `value` (B,)
        passthrough for convenience.
        """
        is_noop = src == self.p_max
        src_idx = torch.where(is_noop, torch.zeros_like(src), src)

        src_dist = torch.distributions.Categorical(logits=out.src_logits)

        gather_idx = src_idx[:, None, None].expand(-1, 1, self.p_max)
        tgt_logits_sel = out.tgt_logits.gather(1, gather_idx).squeeze(1)
        tgt_dist = torch.distributions.Categorical(logits=tgt_logits_sel)

        ship_params_sel = out.ship_params.gather(
            1, src_idx[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        ship_dist = _make_ship_dist(ship_params_sel)

        valid = (~is_noop).float()
        ship_logp = ship_dist.log_prob(_clamp_for_log_prob(ship_value.float()))
        log_prob = src_dist.log_prob(src) + valid * (
            tgt_dist.log_prob(tgt) + ship_logp
        )
        # Entropy: H_src + Pr(not NOOP) * (H_tgt + H_ship). The conditional
        # weight is the *current policy's* probability of choosing a launch
        # action (last src slot is NO-OP), not the realized 0/1 flag from
        # the buffer. This is the population-level entropy of the joint
        # discrete-continuous policy and gives a stable entropy bonus.
        src_probs = torch.softmax(out.src_logits, dim=-1)
        p_nonnoop = 1.0 - src_probs[..., self.p_max]
        entropy = src_dist.entropy() + p_nonnoop * (
            tgt_dist.entropy() + ship_dist.entropy()
        )

        return {"log_prob": log_prob, "entropy": entropy, "value": out.value}

    def evaluate_source_actions(
        self,
        out: "PolicyOutput",
        src: torch.Tensor,
        tgt: torch.Tensor,
        ship_value: torch.Tensor,
        is_noop: torch.Tensor,
    ) -> dict:
        src = src.long()
        tgt = tgt.long()
        ship_value = ship_value.float()
        is_noop = is_noop.bool()

        B = out.src_logits.shape[0]
        src_raw = out.src_logits[:, : self.p_max]
        noop_raw = out.src_logits[:, self.p_max].unsqueeze(-1).expand_as(src_raw)
        launch_logits = torch.stack(
            [noop_raw, src_raw],
            dim=-1,
        )
        launch_sel = launch_logits[torch.arange(B, device=src.device), src]
        launch_dist = torch.distributions.Categorical(logits=launch_sel)
        launch = (~is_noop).long()

        gather_idx = src[:, None, None].expand(-1, 1, self.p_max)
        tgt_logits_sel = out.tgt_logits.gather(1, gather_idx).squeeze(1)
        tgt_dist = torch.distributions.Categorical(logits=tgt_logits_sel)

        ship_params_sel = out.ship_params.gather(
            1, src[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        ship_dist = _make_ship_dist(ship_params_sel)

        valid = (~is_noop).float()
        ship_logp = ship_dist.log_prob(_clamp_for_log_prob(ship_value))
        log_prob = launch_dist.log_prob(launch) + valid * (
            tgt_dist.log_prob(tgt) + ship_logp
        )
        # Per-source entropy: H(launch) + Pr(launch) * (H_tgt + H_ship).
        # `launch_sel` columns are [noop_logit, src_logit] for this row;
        # softmax index 1 is the launch probability under the current
        # policy. Using this marginal (instead of the realized launch
        # flag from the buffer) keeps the entropy bonus a population
        # average and prevents the realized-flag bias from skewing the
        # gradient when the policy's noop rate shifts during PPO epochs.
        launch_probs = torch.softmax(launch_sel, dim=-1)
        p_launch = launch_probs[..., 1]
        entropy = launch_dist.entropy() + p_launch * (
            tgt_dist.entropy() + ship_dist.entropy()
        )

        return {"log_prob": log_prob, "entropy": entropy, "value": out.value}


# =============================================================================
# Convenience: tensor packing from a list of Observations
# =============================================================================
def pack_observations(obs_list: list, device: Optional[torch.device] = None) -> dict:
    """Stack a list of `Observation` dataclasses into a single batched
    tensor dict, ready for `OrbitWarsPolicy.forward(**batch)`."""
    import numpy as np

    def stack(field):
        return torch.from_numpy(np.stack([getattr(o, field) for o in obs_list]))

    batch = {
        "planet_feats": stack("planet_feats"),
        "fleet_feats": stack("fleet_feats"),
        "global_feats": stack("global_feats"),
        "edge_bias": stack("edge_bias"),
        "planet_mask": stack("planet_mask"),
        "fleet_mask": stack("fleet_mask"),
        "source_mask": stack("source_mask"),
    }
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch
