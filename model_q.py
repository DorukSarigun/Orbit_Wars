"""Q-network for OrbitWars (DQN / Dueling DQN).

Architecture: same Set Transformer encoder as `OrbitWarsPolicy`, but
heads emit *Q-values* instead of logits. The action factorization is
identical to the policy's:

    Q_total(s, src, tgt, ship) =
        Q_src(s, src)
      + 1{src != NOOP} * (Q_tgt(s, src, tgt) + Q_ship(s, src, ship))

This is the "Branching DQN" formulation (Tavakoli et al., 2018):
each action dimension has its own head and we sum them, conditional on
the chosen source. Greedy action selection is decomposed:

    src*  = argmax_a Q_src(s, a)
    tgt*  = argmax_a Q_tgt(s, src*, a)
    ship* = argmax_a Q_ship(s, src*, a)

Dueling variant (Wang et al., 2016) decomposes each branch as:

    Q_b(s, a) = V(s) + A_b(s, a) - mean_a A_b(s, a)

with a single shared V(s) (read from the global token).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import torch
import torch.nn as nn

from model import TransformerBlock  # encoder block reused
from obs_builder import (
    EDGE_FEATURE_DIM, F_MAX_DEFAULT, FLEET_FEATURE_DIM,
    GLOBAL_FEATURE_DIM, P_MAX_DEFAULT, PLANET_FEATURE_DIM,
)


VERY_NEG = -1e9


@dataclass
class QOutput:
    q_src:  torch.Tensor            # (B, P_max + 1)         Q over sources (incl NOOP)
    q_tgt:  torch.Tensor            # (B, P_max, P_max)      Q(src, tgt)
    q_ship: torch.Tensor            # (B, P_max, n_ship_bins) Q(src, ship_bin)
    tokens: torch.Tensor            # (B, T, d_model)         for inspection
    v_state: Optional[torch.Tensor] # (B,)  dueling only; None otherwise


class OrbitWarsQNet(nn.Module):
    """Branching (optionally dueling) Q-network sharing OrbitWarsPolicy's encoder."""

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
        n_ship_bins: int = 5,
        dropout: float = 0.0,
        dueling: bool = False,
    ):
        super().__init__()
        self.p_max = p_max
        self.f_max = f_max
        self.n_ship_bins = n_ship_bins
        self.d_model = d_model
        self.n_heads = n_heads
        self.dueling = dueling

        # --- Encoder (mirrors OrbitWarsPolicy) ---
        self.planet_proj = nn.Linear(planet_dim, d_model)
        self.fleet_proj = nn.Linear(fleet_dim, d_model)
        self.global_proj = nn.Linear(global_dim, d_model)
        self.type_embed = nn.Embedding(3, d_model)
        self.edge_proj = nn.Linear(edge_dim, n_heads)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # --- Q heads (or A heads in dueling mode) ---
        # Source: per-planet scalar + a learned NO-OP scalar.
        self.q_src_head = nn.Linear(d_model, 1)
        self.q_noop = nn.Parameter(torch.zeros(1))

        # Target: pointer-style score q(src, tgt).
        self.q_tgt_query = nn.Linear(d_model, d_model)
        self.q_tgt_key = nn.Linear(d_model, d_model)

        # Ship: per-planet n_ship_bins.
        self.q_ship_head = nn.Linear(d_model, n_ship_bins)

        # Dueling: shared V(s) head from global token.
        if dueling:
            self.v_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

        type_ids = torch.zeros(p_max + f_max + 1, dtype=torch.long)
        type_ids[p_max:p_max + f_max] = 1
        type_ids[-1] = 2
        self.register_buffer("type_ids", type_ids, persistent=False)

    # ---------------------------------------------------------------- helpers
    def _encode(
        self,
        planet_feats, fleet_feats, global_feats, edge_bias,
        planet_mask, fleet_mask,
    ) -> torch.Tensor:
        B = planet_feats.shape[0]
        planet_tok = self.planet_proj(planet_feats)
        fleet_tok  = self.fleet_proj(fleet_feats)
        global_tok = self.global_proj(global_feats).unsqueeze(1)
        x = torch.cat([planet_tok, fleet_tok, global_tok], dim=1)
        x = x + self.type_embed(self.type_ids)[None]

        attn_bias = self.edge_proj(edge_bias).permute(0, 3, 1, 2).contiguous()
        pad_global = torch.zeros(B, 1, dtype=torch.bool, device=planet_mask.device)
        real = torch.cat([planet_mask, fleet_mask, ~pad_global], dim=1)
        kpm = ~real

        for block in self.blocks:
            x = block(x, attn_bias, kpm)
        return self.norm(x)

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        planet_feats, fleet_feats, global_feats, edge_bias,
        planet_mask, fleet_mask, source_mask,
    ) -> QOutput:
        tokens = self._encode(planet_feats, fleet_feats, global_feats, edge_bias,
                              planet_mask, fleet_mask)
        planet_tok = tokens[:, :self.p_max]                  # (B, P, d)
        global_tok = tokens[:, -1]                           # (B, d)

        # Raw advantages / Q-values per branch.
        a_src_per = self.q_src_head(planet_tok).squeeze(-1)  # (B, P)
        a_src_noop = self.q_noop.expand(a_src_per.shape[0], 1)
        a_src = torch.cat([a_src_per, a_src_noop], dim=1)    # (B, P+1)

        q = self.q_tgt_query(planet_tok)
        k = self.q_tgt_key(planet_tok)
        a_tgt = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.d_model)  # (B, P, P)

        a_ship = self.q_ship_head(planet_tok)                # (B, P, n_bins)

        if self.dueling:
            v = self.v_head(global_tok).squeeze(-1)          # (B,)

            # Dueling decomposition per branch. Subtract the *valid* mean
            # so masking later doesn't change the centering.
            q_src = v[:, None] + a_src - a_src.mean(dim=-1, keepdim=True)
            q_tgt = v[:, None, None] + a_tgt - a_tgt.mean(dim=-1, keepdim=True)
            q_ship = v[:, None, None] + a_ship - a_ship.mean(dim=-1, keepdim=True)
            v_state = v
        else:
            q_src = a_src
            q_tgt = a_tgt
            q_ship = a_ship
            v_state = None

        return QOutput(q_src=q_src, q_tgt=q_tgt, q_ship=q_ship,
                       tokens=tokens, v_state=v_state)

    # ----------------------------------------------------- masked argmax
    @staticmethod
    def select_actions(
        q_out: QOutput,
        source_mask: torch.Tensor,     # (B, P) bool
        planet_mask: torch.Tensor,     # (B, P) bool
        *,
        epsilon: float = 0.0,
        rng: Optional[torch.Generator] = None,
    ) -> dict:
        """Select per-branch greedy actions with mask + ε-greedy noise.

        Returns dict with 1-D long tensors src/tgt/ship_bin and bool is_noop.
        Self-target is forbidden by setting Q(src, src)=very_neg.
        """
        B, P = source_mask.shape
        n_bins = q_out.q_ship.shape[-1]
        p_max_plus_one = q_out.q_src.shape[-1]
        device = q_out.q_src.device

        # ---- src ----
        q_src = q_out.q_src.clone()
        # NO-OP slot is always valid; planet slots gated by source_mask.
        valid_src = torch.cat([source_mask,
                               torch.ones(B, 1, dtype=torch.bool, device=device)],
                              dim=1)
        q_src = q_src.masked_fill(~valid_src, VERY_NEG)
        greedy_src = q_src.argmax(dim=-1)

        if epsilon > 0.0:
            rand_unit = torch.rand(B, generator=rng, device=device)
            do_random = rand_unit < epsilon
            # Sample uniformly from the *valid* sources (NOOP always valid).
            valid_float = valid_src.float()
            rand_src = torch.multinomial(valid_float, 1, generator=rng).squeeze(-1)
            src = torch.where(do_random, rand_src, greedy_src)
        else:
            src = greedy_src

        is_noop = src == (p_max_plus_one - 1)
        src_idx = torch.where(is_noop, torch.zeros_like(src), src)

        # ---- tgt ----
        gather_t = src_idx[:, None, None].expand(-1, 1, P)
        q_tgt = q_out.q_tgt.gather(1, gather_t).squeeze(1)   # (B, P)
        # Block self-target.
        self_mask = torch.zeros_like(q_tgt, dtype=torch.bool)
        self_mask.scatter_(1, src_idx[:, None], True)
        # Block padding.
        pad_t = ~planet_mask
        q_tgt = q_tgt.masked_fill(self_mask | pad_t, VERY_NEG)
        greedy_tgt = q_tgt.argmax(dim=-1)
        if epsilon > 0.0:
            valid_tgt = ~(self_mask | pad_t)
            valid_f = valid_tgt.float()
            # If a row has no valid target (rare: only one planet alive),
            # keep argmax to avoid NaN sampling.
            row_has_valid = valid_f.sum(dim=-1) > 0
            sampled = torch.where(
                row_has_valid,
                torch.multinomial(valid_f.clamp(min=1e-8), 1,
                                  generator=rng).squeeze(-1),
                greedy_tgt,
            )
            do_random = torch.rand(B, generator=rng, device=device) < epsilon
            tgt = torch.where(do_random & row_has_valid, sampled, greedy_tgt)
        else:
            tgt = greedy_tgt

        # ---- ship ----
        gather_s = src_idx[:, None, None].expand(-1, 1, n_bins)
        q_ship = q_out.q_ship.gather(1, gather_s).squeeze(1)
        greedy_ship = q_ship.argmax(dim=-1)
        if epsilon > 0.0:
            do_random = torch.rand(B, generator=rng, device=device) < epsilon
            rand_ship = torch.randint(0, n_bins, (B,), generator=rng, device=device)
            ship = torch.where(do_random, rand_ship, greedy_ship)
        else:
            ship = greedy_ship

        return {
            "src": src, "tgt": tgt, "ship_bin": ship, "is_noop": is_noop,
        }

    # -------------------------------------------------- joint Q evaluation
    def joint_q(
        self,
        q_out: QOutput,
        src: torch.Tensor,         # (B,) long
        tgt: torch.Tensor,         # (B,) long
        ship_bin: torch.Tensor,    # (B,) long
    ) -> torch.Tensor:
        """Joint Q(s, a) for the stored composite action.

        For NO-OP rows only Q_src contributes; tgt/ship branches are
        masked out, mirroring the policy's log-prob factorization.
        """
        is_noop = src == self.p_max
        src_idx = torch.where(is_noop, torch.zeros_like(src), src)

        q_src_a = q_out.q_src.gather(1, src[:, None]).squeeze(1)   # (B,)

        gather_t = src_idx[:, None, None].expand(-1, 1, self.p_max)
        tgt_row = q_out.q_tgt.gather(1, gather_t).squeeze(1)       # (B, P)
        q_tgt_a = tgt_row.gather(1, tgt[:, None]).squeeze(1)

        gather_s = src_idx[:, None, None].expand(-1, 1, self.n_ship_bins)
        ship_row = q_out.q_ship.gather(1, gather_s).squeeze(1)     # (B, n_bins)
        q_ship_a = ship_row.gather(1, ship_bin[:, None]).squeeze(1)

        valid = (~is_noop).float()
        return q_src_a + valid * (q_tgt_a + q_ship_a)

    # -------------------------------------------------- max joint Q (for targets)
    def max_joint_q(
        self,
        q_out: QOutput,
        source_mask: torch.Tensor,    # (B, P) bool
        planet_mask: torch.Tensor,    # (B, P) bool
    ) -> torch.Tensor:
        """Greedy max over the factored action space.

        max_a Q_total = max_src [Q_src(src) +
                                 (1{src != NOOP}) * (max_tgt Q_tgt(src, tgt)
                                                     + max_ship Q_ship(src, ship))]

        We maximize per-branch which is *exact* given the additive
        factorization (sum of independents).
        """
        B, P = source_mask.shape
        device = q_out.q_src.device

        # Per-source max over tgt (mask self + padding).
        q_tgt = q_out.q_tgt.clone()                                # (B, P, P)
        eye = torch.eye(P, dtype=torch.bool, device=device).unsqueeze(0)
        q_tgt = q_tgt.masked_fill(eye, VERY_NEG)
        pad_t = (~planet_mask).unsqueeze(1).expand(-1, P, -1)
        q_tgt = q_tgt.masked_fill(pad_t, VERY_NEG)
        max_tgt_per_src = q_tgt.max(dim=-1).values                 # (B, P)
        # If row has no valid target, treat the contribution as 0
        # rather than -inf (otherwise that source becomes unselectable
        # even if it's the only valid one).
        no_valid = ~planet_mask.any(dim=-1, keepdim=True)
        max_tgt_per_src = torch.where(no_valid.expand(-1, P),
                                      torch.zeros_like(max_tgt_per_src),
                                      max_tgt_per_src)

        # Per-source max over ship_bin.
        max_ship_per_src = q_out.q_ship.max(dim=-1).values         # (B, P)

        per_src_total = q_out.q_src[:, :P] + max_tgt_per_src + max_ship_per_src
        # Mask invalid sources; NO-OP only contributes Q_src(NOOP).
        per_src_total = per_src_total.masked_fill(~source_mask, VERY_NEG)
        noop_q = q_out.q_src[:, P:P + 1]                           # (B, 1)
        all_q = torch.cat([per_src_total, noop_q], dim=1)          # (B, P+1)
        return all_q.max(dim=-1).values
