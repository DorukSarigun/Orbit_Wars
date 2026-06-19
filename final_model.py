"""Neural node policy for final RL training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical

from final_obs import EDGE_FEATURE_DIM, GLOBAL_FEATURE_DIM, NODE_FEATURE_DIM, P_MAX


_NEG = -1.0e9


@dataclass(frozen=True)
class FinalPolicyAction:
    target_idx: torch.Tensor
    source_idx: torch.Tensor
    amount: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class FinalStrategicScores:
    defense_demand: torch.Tensor
    support_demand: torch.Tensor
    attack_value: torch.Tensor
    expand_value: torch.Tensor
    donor_capacity: torch.Tensor
    reserve_value: torch.Tensor
    evacuation_urgency: torch.Tensor
    frontier_anchor_value: torch.Tensor


class FinalNodePolicy(nn.Module):
    """Edge-aware GNN policy: choose target, source and amount fraction."""

    def __init__(
        self,
        *,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        global_dim: int = GLOBAL_FEATURE_DIM,
        p_max: int = P_MAX,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.global_dim = int(global_dim)
        self.p_max = int(p_max)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)

        self.node_proj = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.edge_proj = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        msg_in = 3 * d_model
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(msg_in, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_layers)
        ])
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(msg_in, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            for _ in range(n_layers)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_layers)
        ])
        self.node_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.global_proj = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, d_model),
            nn.GELU(),
        )
        self.strategic_head = nn.Linear(d_model, 8)
        self.target_residual_head = nn.Linear(d_model, 1)
        self.source_residual_head = nn.Linear(d_model, 1)
        self.amount_head = nn.Linear(d_model, 2)
        self.value_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def encode(self, node, edge, global_features, valid_mask, edge_mask):
        h = self.node_proj(node)
        e = self.edge_proj(edge)
        valid_node = valid_mask.bool().unsqueeze(-1)
        valid_edge = edge_mask.bool().unsqueeze(-1)
        for msg_mlp, gate_mlp, update_mlp, norm in zip(
            self.message_mlps,
            self.gate_mlps,
            self.update_mlps,
            self.node_norms,
        ):
            src_h = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)
            dst_h = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)
            edge_context = torch.cat([src_h, dst_h, e], dim=-1)
            msg = msg_mlp(edge_context)
            gate = torch.sigmoid(gate_mlp(edge_context))
            msg = msg * gate * valid_edge
            degree = valid_edge.float().sum(dim=1).clamp_min(1.0)
            incoming = msg.sum(dim=1) / degree
            h = norm(h + update_mlp(torch.cat([h, incoming], dim=-1)))
            h = h * valid_node
        mask_f = valid_mask.float().unsqueeze(-1)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        g = self.global_proj(global_features)
        value = self.value_head(torch.cat([pooled, g], dim=-1)).squeeze(-1)
        return h, value

    def strategic_scores(self, h: torch.Tensor) -> FinalStrategicScores:
        raw = torch.sigmoid(self.strategic_head(h))
        return FinalStrategicScores(
            defense_demand=raw[..., 0],
            support_demand=raw[..., 1],
            attack_value=raw[..., 2],
            expand_value=raw[..., 3],
            donor_capacity=raw[..., 4],
            reserve_value=raw[..., 5],
            evacuation_urgency=raw[..., 6],
            frontier_anchor_value=raw[..., 7],
        )

    def logits(self, node, edge, global_features, valid_mask, edge_mask, source_mask, target_mask):
        h, value = self.encode(node, edge, global_features, valid_mask, edge_mask)
        scores = self.strategic_scores(h)
        # Target selection is score-driven: enemy/neutral attack potential plus a
        # small learned residual.  Source selection is donor capacity minus
        # reserve/evac pressure plus residual.
        target_logits = (
            2.0 * scores.attack_value
            + 1.0 * scores.expand_value
            + self.target_residual_head(h).squeeze(-1)
        )
        source_logits = (
            2.0 * scores.donor_capacity
            - 1.0 * scores.reserve_value
            - 0.5 * scores.evacuation_urgency
            + self.source_residual_head(h).squeeze(-1)
        )
        target_logits = target_logits.masked_fill(~target_mask.bool(), _NEG)
        source_logits = source_logits.masked_fill(~source_mask.bool(), _NEG)
        return h, target_logits, source_logits, value

    def score_tensors(self, node, edge, global_features, valid_mask, edge_mask) -> tuple[FinalStrategicScores, torch.Tensor]:
        h, value = self.encode(node, edge, global_features, valid_mask, edge_mask)
        return self.strategic_scores(h), value

    @staticmethod
    def _safe_categorical(logits):
        # If a mask removes everything, Categorical would produce NaNs.  Fall
        # back to slot 0; rollout will turn that into a no-op if invalid.
        invalid = torch.isneginf(logits).all(dim=-1) if torch.isinf(logits).any() else (logits <= _NEG / 2).all(dim=-1)
        if invalid.any():
            logits = logits.clone()
            logits[invalid] = _NEG
            logits[invalid, 0] = 0.0
        return Categorical(logits=logits)

    def _amount_dist(self, h, source_idx):
        src_h = h.gather(1, source_idx.view(-1, 1, 1).expand(-1, 1, h.size(-1))).squeeze(1)
        raw = self.amount_head(src_h)
        alpha = F.softplus(raw[:, 0]) + 1.0
        beta = F.softplus(raw[:, 1]) + 1.0
        return Beta(alpha, beta)

    def act(self, node, edge, global_features, valid_mask, edge_mask, source_mask, target_mask, *, greedy: bool = False):
        h, target_logits, source_logits, value = self.logits(
            node, edge, global_features, valid_mask, edge_mask, source_mask, target_mask)
        target_dist = self._safe_categorical(target_logits)
        source_dist = self._safe_categorical(source_logits)
        if greedy:
            target_idx = target_logits.argmax(dim=-1)
            source_idx = source_logits.argmax(dim=-1)
        else:
            target_idx = target_dist.sample()
            source_idx = source_dist.sample()
        amount_dist = self._amount_dist(h, source_idx)
        amount = amount_dist.mean if greedy else amount_dist.sample()
        amount = amount.clamp(1e-4, 1.0 - 1e-4)
        logprob = (
            target_dist.log_prob(target_idx)
            + source_dist.log_prob(source_idx)
            + amount_dist.log_prob(amount)
        )
        entropy = target_dist.entropy() + source_dist.entropy() + amount_dist.entropy()
        return FinalPolicyAction(
            target_idx=target_idx,
            source_idx=source_idx,
            amount=amount,
            logprob=logprob,
            entropy=entropy,
            value=value,
        )

    def evaluate_actions(
        self,
        node,
        edge,
        global_features,
        valid_mask,
        edge_mask,
        source_mask,
        target_mask,
        target_idx,
        source_idx,
        amount,
    ):
        h, target_logits, source_logits, value = self.logits(
            node, edge, global_features, valid_mask, edge_mask, source_mask, target_mask)
        target_dist = self._safe_categorical(target_logits)
        source_dist = self._safe_categorical(source_logits)
        amount_dist = self._amount_dist(h, source_idx.long())
        amount = amount.clamp(1e-4, 1.0 - 1e-4)
        logprob = (
            target_dist.log_prob(target_idx.long())
            + source_dist.log_prob(source_idx.long())
            + amount_dist.log_prob(amount)
        )
        entropy = target_dist.entropy() + source_dist.entropy() + amount_dist.entropy()
        return logprob, entropy, value
