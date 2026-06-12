"""Reworked graph encoder for the Orbit Wars policy (clean rewrite).

This module is the network counterpart of :mod:`reworked_features`.  It is a
**fresh, self-contained** GNN encoder that consumes the redesigned, compact
feature set and produces per-planet node tokens + a graph summary that the
action / value heads (designed separately) will sit on top of.

Design choices (see ``docs/reworked_features_reference.md`` and the chat
design notes):

* **Compact inputs.** Node features are ``REWORKED_NODE_FEATURE_DIM = 23``
  (10 raw ``standard_node_features`` + 6 derived metrics + 7 v12 extras),
  edge features are ``REWORKED_EDGE_FEATURE_DIM = 9`` (distance, crosses_sun,
  friendly/enemy flux + v12 kinematics: eta, orbital phase sin/cos,
  ships-at-arrival, race margin).  Global features keep the 14-D layout.
* **Input embedding.** Node / edge / global blocks are each projected with a
  ``LayerNorm -> Linear -> GELU -> LayerNorm`` stem to ``d_model``.
* **Message passing.** ``n_layers`` of a **dense edge-attention GNN** layer:
  edge-conditioned multi-head attention aggregates incoming messages per target
  node, followed by an **edge update** that refreshes ``edge_tokens`` from the
  new node states.  The graph is fully connected and tiny (<=25 planets), so a
  masked dense formulation stays dependency-free (no PyG / ``edge_index``).
* **Permutation equivariance.** Node tokens ``(B,P,d)`` and edge tokens
  ``(B,P,P,d)`` transform consistently under planet relabelling; graph-level
  summaries pool to permutation-invariant scalars.

Action heads are intentionally **not** implemented here — the encoder exposes
``node_tokens``, ``edge_tokens`` and ``graph_summary`` so the heads can be
designed and bolted on separately.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # keep import side-effect-free if run as a loose script
    from .reworked_features import (
        STANDARD_NODE_FEATURE_DIM,
        REWORKED_EDGE_FEATURE_DIM,
    )
except ImportError:  # pragma: no cover - flat layout / direct execution
    from reworked_features import (  # type: ignore
        STANDARD_NODE_FEATURE_DIM,
        REWORKED_EDGE_FEATURE_DIM,
    )

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------
# Derived per-node metrics appended to the 10 raw standard channels:
#   discounted_production_potential, closeness_centrality,
#   potential_threat, potential_support, friendly_compactness,
#   capture_affordability
NUM_DERIVED_NODE_METRICS: int = 6
# v12 extras appended after the derived block (and after the optional
# eta_value_rate channel, which keeps legacy 16/17-dim checkpoints sliceable):
#   friendly_inbound_total, enemy_inbound_total, enemy_min_eta,
#   defense_deficit, recapture_count, comet_remaining_life, frontier_distance
NUM_EXTRA_NODE_METRICS: int = 7
REWORKED_NODE_FEATURE_DIM: int = (
    STANDARD_NODE_FEATURE_DIM + NUM_DERIVED_NODE_METRICS + NUM_EXTRA_NODE_METRICS
)  # 23
REWORKED_GLOBAL_FEATURE_DIM: int = 14  # mirrors obs_builder.GLOBAL_FEATURE_DIM

_NEG_INF = float("-inf")


def _input_stem(in_dim: int, d_model: int) -> nn.Sequential:
    """``LayerNorm -> Linear -> GELU -> LayerNorm`` projection to ``d_model``."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, d_model),
        nn.GELU(),
        nn.LayerNorm(d_model),
    )


class DenseEdgeAttnGINELayer(nn.Module):
    r"""Edge-conditioned attention + edge update on a dense directed graph.

    **Node update** (per target ``i``, attend over sources ``j``):

    .. math::

        m_{j\to i} = \mathrm{ReLU}(x_j + W_e e_{j\to i}),\quad
        \alpha_{j\to i} = \mathrm{softmax}_j\!\big(
            \langle W_q x_i,\, W_k m_{j\to i}\rangle / \sqrt{d_h}\big),\quad
        x_i' = \mathrm{MLP}\!\big((1+\epsilon)\,x_i + \sum_j \alpha_{j\to i}\, W_v m_{j\to i}\big)

    **Edge update** (per directed pair ``j\to i``):

    .. math::

        e_{j\to i}' = \mathrm{LayerNorm}\!\big(
            e_{j\to i} + \mathrm{MLP}([x_j', x_i', e_{j\to i}])\big)

    All operations are content-only (no index parameters), so node tokens
    permute as ``h[:, \pi]`` and edge tokens as ``e[:, \pi, \pi]`` under a
    planet relabelling ``\pi``.
    """

    def __init__(self, d_model: int, *, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads

        self.eps = nn.Parameter(torch.zeros(()))
        self.edge_lin = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.node_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.node_norm = nn.LayerNorm(d_model)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.edge_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,            # (B, P, d)
        edge_tokens: torch.Tensor,  # (B, P, P, d)  indexed [b, src, dst]
        pair_mask: torch.Tensor,    # (B, P, P)     valid source j -> target i, no diagonal
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, P, d = h.shape
        H, hd = self.n_heads, self.head_dim
        neg = torch.finfo(h.dtype).min

        we = self.edge_lin(edge_tokens)                              # (B, j, i, d)
        msg = F.relu(h.unsqueeze(2) + we)                            # (B, j, i, d)

        q = self.q_proj(h).view(B, 1, P, H, hd)                      # (B, 1, i, H, hd)
        k = self.k_proj(msg).view(B, P, P, H, hd)                    # (B, j, i, H, hd)
        v = self.v_proj(msg).view(B, P, P, H, hd)

        scores = (q * k).sum(dim=-1) / (hd ** 0.5)                   # (B, j, i, H)
        m = pair_mask.unsqueeze(-1).to(torch.bool)                   # (B, j, i, 1)
        scores = scores.masked_fill(~m, neg)
        attn = F.softmax(scores, dim=1)                              # softmax over sources j
        attn = attn * pair_mask.unsqueeze(-1).to(attn.dtype)
        attn = self.dropout(attn)

        agg = (attn.unsqueeze(-1) * v).sum(dim=1)                    # (B, i, H, hd)
        agg = self.out_proj(agg.reshape(B, P, d))
        updated = self.node_mlp((1.0 + self.eps) * h + agg)
        h_new = self.node_norm(h + self.dropout(updated))

        # Edge update from endpoint node states + previous edge embedding.
        hi = h_new.unsqueeze(1).expand(-1, P, -1, -1)               # (B, j, i, d) target
        hj = h_new.unsqueeze(2).expand(-1, -1, P, -1)               # (B, j, i, d) source
        edge_in = torch.cat([hj, hi, edge_tokens], dim=-1)           # (B, j, i, 3d)
        edge_delta = self.edge_mlp(edge_in)
        edge_new = self.edge_norm(edge_tokens + self.dropout(edge_delta))
        edge_new = edge_new * pair_mask.unsqueeze(-1).to(edge_new.dtype)

        return h_new, edge_new


class ReworkedPlanetGraphEncoder(nn.Module):
    """Encoder: compact planet features -> node tokens + graph summary.

    Parameters
    ----------
    node_dim, edge_dim, global_dim:
        Input feature widths (default to the reworked layout 16 / 4 / 14).
    d_model:
        Hidden width.
    n_layers:
        Number of dense edge-attention message-passing layers.
    n_heads:
        Attention heads inside each layer (``d_model`` must be divisible).
    dropout:
        Dropout on attention weights and residual branches.
    """

    def __init__(
        self,
        *,
        node_dim: int = REWORKED_NODE_FEATURE_DIM,
        edge_dim: int = REWORKED_EDGE_FEATURE_DIM,
        global_dim: int = REWORKED_GLOBAL_FEATURE_DIM,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = int(n_heads)

        self.node_embed = _input_stem(node_dim, d_model)
        self.edge_embed = _input_stem(edge_dim, d_model)
        self.global_embed = _input_stem(global_dim, d_model)

        self.layers = nn.ModuleList(
            DenseEdgeAttnGINELayer(d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        )
        # mean-pool ++ max-pool ++ global token  ->  d_model
        self.summary = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        node_feats: torch.Tensor,            # (B, P, node_dim)
        edge_feats: torch.Tensor,            # (B, P, P, edge_dim)
        global_feats: torch.Tensor,          # (B, global_dim)
        planet_mask: Optional[torch.Tensor] = None,  # (B, P) bool; None -> all valid
    ) -> dict[str, torch.Tensor]:
        """Return ``node_tokens (B,P,d)``, ``edge_tokens (B,P,P,d)``, ``graph_summary (B,d)``."""
        B, P, _ = node_feats.shape
        device = node_feats.device
        if planet_mask is None:
            planet_mask = torch.ones((B, P), dtype=torch.bool, device=device)
        else:
            planet_mask = planet_mask.to(dtype=torch.bool)

        h = self.node_embed(node_feats)
        edge_tokens = self.edge_embed(edge_feats)

        # Pair mask for message passing: valid source j, exclude diagonal j == i.
        eye = torch.eye(P, dtype=torch.bool, device=device).unsqueeze(0)        # (1, P, P)
        pair_mask = (planet_mask[:, :, None] & ~eye).to(dtype=h.dtype)          # (B, j, i)

        for layer in self.layers:
            h, edge_tokens = layer(h, edge_tokens, pair_mask)

        # Zero out padded node tokens so they cannot leak into pooling.
        node_valid = planet_mask.unsqueeze(-1).to(dtype=h.dtype)               # (B, P, 1)
        h = h * node_valid

        # Masked mean / max pooling over valid planets.
        counts = node_valid.sum(dim=1).clamp_min(1.0)                          # (B, 1)
        mean_pool = h.sum(dim=1) / counts                                      # (B, d)
        masked_for_max = h.masked_fill(~planet_mask.unsqueeze(-1), _NEG_INF)
        max_pool = masked_for_max.max(dim=1).values
        # Guard fully-empty graphs (no valid planet) against -inf.
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)

        global_tok = self.global_embed(global_feats)                           # (B, d)
        graph_summary = self.summary(torch.cat([mean_pool, max_pool, global_tok], dim=-1))

        return {
            "node_tokens": h,
            "edge_tokens": edge_tokens,
            "graph_summary": graph_summary,
        }


class ReworkedHierarchicalPolicy(nn.Module):
    r"""Hierarchical policy: RL picks *targets*, a scripted allocator routes ships.

    Per turn the RL agent selects, independently for every planet, whether it
    is an attack target and/or a defend target — i.e. two **multi-label**
    (per-planet Bernoulli) heads, so *several* targets of each kind can be
    chosen in one turn (or none):

    * **attack set** — independent Bernoulli over ``{non-self planets}``
      (enemy planets *or* neutral planets, the latter being expansion).
    * **defend set** — independent Bernoulli over ``{own planets}``.

    The empty set is the natural "noop" (no planet selected), so no explicit
    noop logit is needed.

    On top of *which* targets, the agent also decides their **funding
    priority** under the shared ship budget — itself a strategic call (fund the
    attack or save the planet first when ships are scarce).  A shared
    ``priority_head`` emits one scalar per planet (common scale across attack
    and defend), and a **Plackett-Luce** ranking is sampled over the selected
    set (:func:`plackett_luce_sample`).  Both the Bernoulli selections and the
    PL ranking have closed-form log-probs that sum into the PPO objective, so
    the priority head trains directly from advantage.

    The agent does **not** pick the exact source planets / ship counts — that
    logistics problem is delegated to a graph-algorithm allocator.  But it does
    set one further *strategic* lever: a per-own-planet **commit fraction**
    ``Beta(alpha, beta)`` saying how much of each planet's garrison the
    allocator may draw (strip a safe interior base, keep an exposed frontier
    garrisoned).  The allocator then funds the selected targets in the sampled
    priority order, drawing only within each source's committed budget, using
    the reworked metrics (``potential_threat`` / ``potential_support`` /
    ``friendly_compactness`` / edge flux + ETA).

    Finally the agent sets a board-level **defense posture** — a global
    ``potential_weight`` and ``horizon`` (two ``Beta`` draws, see
    :func:`sample_defense_posture`) controlling how proactively the allocator
    reserves against *nearby enemy garrisons that could strike soon* versus only
    the already-launched in-flight fleets.  This is the global counterpart of
    the per-planet commit lever.

    Total per-turn action log-prob::

        log_prob = Σ_g Bernoulli(attack/defend selection)
                 + log_prob_PL(priority order over selected targets)
                 + Σ_{g∈own} Beta(commit_fraction_g)
                 + Beta(potential_weight) + Beta(horizon)   # global posture

    Permutation behaviour (verified in tests):

    * ``attack_logits`` / ``defend_logits`` / ``priority_logits`` and the
      ``commit_alpha`` / ``commit_beta`` params (per-planet) are **equivariant**;
    * the ``value`` and the global ``posture_*`` Beta params are **invariant**;
    * the PL ranking log-prob is **invariant** under a joint relabelling of the
      scores and the sampled order.

    Heads only consume the encoder's equivariant ``node_tokens`` and invariant
    ``graph_summary`` (no index-based parameters), so the whole policy is
    permutation-symmetric by construction.
    """

    def __init__(
        self,
        *,
        node_dim: int = REWORKED_NODE_FEATURE_DIM,
        edge_dim: int = REWORKED_EDGE_FEATURE_DIM,
        global_dim: int = REWORKED_GLOBAL_FEATURE_DIM,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
        aux_owner: bool = True,
    ):
        super().__init__()
        self.encoder = ReworkedPlanetGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            global_dim=global_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.node_dim = int(node_dim)
        self.d_model = d_model

        # Per-planet target heads: input = [node_token, broadcast graph_summary].
        def _node_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

        # Multi-label (per-planet Bernoulli) target heads.
        self.attack_head = _node_head()
        self.defend_head = _node_head()
        # Shared priority head: one scalar per planet on a *common* scale for
        # both attack and defend candidates, so the Plackett-Luce ranking can
        # trade off "capture this enemy planet" vs "save my planet" under one
        # shared ship budget.
        self.priority_head = _node_head()
        # Per-planet commit fraction (strategic risk lever): how much of each
        # OWN planet's garrison the allocator may draw.  A Beta(alpha, beta)
        # per planet -> 2 outputs.  Lets the agent strip a safe interior base
        # (high commit) while keeping an exposed frontier garrisoned (low).
        self.commit_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        # Global defense-posture head (invariant): how proactively to defend.
        # Reads the graph summary and emits two Beta(alpha, beta) distributions
        # (-> 4 params) for the allocator's ``potential_weight`` and ``horizon``
        # knobs (see :func:`sample_defense_posture`).  This is the board-level
        # analogue of the per-planet ``commit_head`` -- a single strategic lever
        # trading reactive (count only in-flight enemies) vs proactive (also
        # reserve against nearby enemy garrisons that could strike soon) defense.
        self.posture_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )
        # Critic (invariant).
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        # Optional auxiliary per-node owner prediction (5 classes: self / 3 enemy / neutral).
        self.aux_owner_head = nn.Linear(d_model, 5) if aux_owner else None

    def forward(
        self,
        node_feats: torch.Tensor,                  # (B, P, node_dim)
        edge_feats: torch.Tensor,                  # (B, P, P, edge_dim)
        global_feats: torch.Tensor,                # (B, global_dim)
        planet_mask: torch.Tensor,                 # (B, P) bool
        attack_mask: torch.Tensor,                 # (B, P) bool: valid attack targets (non-self, real)
        defend_mask: torch.Tensor,                 # (B, P) bool: valid defend targets (own planets)
    ) -> dict[str, torch.Tensor]:
        """Return per-planet ``attack_logits`` / ``defend_logits`` / ``priority_logits`` ``(B, P)``, value, tokens.

        Invalid planets are pushed to a large negative logit: non-attack /
        non-defend candidates get ``sigmoid -> 0`` (never selected, 0 Bernoulli
        log-prob contribution); non-candidates get ``priority -> -inf`` so they
        can never enter a Plackett-Luce ranking.
        """
        enc = self.encoder(node_feats, edge_feats, global_feats, planet_mask)
        tok = enc["node_tokens"]                  # (B, P, d)
        summary = enc["graph_summary"]            # (B, d)
        B, P, _ = tok.shape

        s_bc = summary[:, None, :].expand(-1, P, -1)
        node_in = torch.cat([tok, s_bc], dim=-1)  # (B, P, 2d)

        attack_logits = self.attack_head(node_in).squeeze(-1)   # (B, P)
        defend_logits = self.defend_head(node_in).squeeze(-1)   # (B, P)
        priority_logits = self.priority_head(node_in).squeeze(-1)  # (B, P)

        attack_bool = attack_mask.to(torch.bool)
        defend_bool = defend_mask.to(torch.bool)
        neg = torch.finfo(attack_logits.dtype).min
        attack_logits = attack_logits.masked_fill(~attack_bool, neg)
        defend_logits = defend_logits.masked_fill(~defend_bool, neg)
        # Priority is only meaningful for candidates (anything that could be a
        # target).  Non-candidates -> -inf so they can never enter a ranking.
        candidate_bool = attack_bool | defend_bool
        priority_logits = priority_logits.masked_fill(~candidate_bool, neg)

        # Per-planet commit fraction Beta params (>= 1 -> unimodal, stable).
        commit_raw = self.commit_head(node_in)                          # (B, P, 2)
        commit_params = F.softplus(commit_raw) + 1.0
        commit_alpha = commit_params[..., 0]
        commit_beta = commit_params[..., 1]
        # Non-source planets (not own) get a neutral Beta(1, 1); they are
        # ignored by the allocator and the masked log-prob.
        ones = torch.ones_like(commit_alpha)
        commit_alpha = torch.where(defend_bool, commit_alpha, ones)
        commit_beta = torch.where(defend_bool, commit_beta, ones)

        # Global defense-posture Beta params (>= 1 -> unimodal, stable).  Built
        # from the permutation-invariant graph summary, so they are invariant.
        posture_params = F.softplus(self.posture_head(summary)) + 1.0    # (B, 4)
        posture_pw_alpha = posture_params[:, 0]
        posture_pw_beta = posture_params[:, 1]
        posture_h_alpha = posture_params[:, 2]
        posture_h_beta = posture_params[:, 3]

        value = self.value_head(summary).squeeze(-1)                    # (B,)

        out = {
            "attack_logits": attack_logits,
            "defend_logits": defend_logits,
            "priority_logits": priority_logits,
            "commit_alpha": commit_alpha,
            "commit_beta": commit_beta,
            "posture_pw_alpha": posture_pw_alpha,
            "posture_pw_beta": posture_pw_beta,
            "posture_h_alpha": posture_h_alpha,
            "posture_h_beta": posture_h_beta,
            "value": value,
            "node_tokens": tok,
        }
        if self.aux_owner_head is not None:
            out["aux_owner_logits"] = self.aux_owner_head(tok)          # (B, P, 5)
        return out

    # ------------------------------------------------------------------ act
    @torch.no_grad()
    def act(
        self,
        *,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        global_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        attack_mask: torch.Tensor,
        defend_mask: torch.Tensor,
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Sample a full hierarchical action and its summed log-probability.

        Returns ``(action, log_prob (B,), value (B,))`` where ``action`` holds
        every sampled component needed to (a) drive the allocator and (b)
        re-evaluate the log-prob in PPO:

        * ``attack_sel`` / ``defend_sel`` ``(B, P)`` float 0/1 Bernoulli picks,
        * ``order`` ``(B, P)`` long Plackett-Luce ranking (-1 padded),
        * ``commit_frac`` ``(B, P)`` Beta commit fractions (0 off-source),
        * ``posture_u`` ``(B, 2)`` normalised posture samples,
        * ``potential_weight`` / ``horizon`` ``(B,)`` scaled posture knobs.
        """
        out = self.forward(node_feats, edge_feats, global_feats, planet_mask,
                            attack_mask, defend_mask)
        attack_sel, a_logp = bernoulli_sample_log_prob(
            out["attack_logits"], attack_mask, deterministic=deterministic, generator=generator)
        defend_sel, d_logp = bernoulli_sample_log_prob(
            out["defend_logits"], defend_mask, deterministic=deterministic, generator=generator)
        selected = (attack_sel > 0.5) | (defend_sel > 0.5)
        order, pl_logp = plackett_luce_sample(out["priority_logits"], selected, generator=generator)
        commit_frac, c_logp = sample_commit_fractions(
            out["commit_alpha"], out["commit_beta"], defend_mask)
        pw, hz, u, p_logp = sample_defense_posture(
            out["posture_pw_alpha"], out["posture_pw_beta"],
            out["posture_h_alpha"], out["posture_h_beta"])

        log_prob = a_logp + d_logp + pl_logp + c_logp + p_logp
        action = {
            "attack_sel": attack_sel,
            "defend_sel": defend_sel,
            "order": order,
            "commit_frac": commit_frac,
            "posture_u": u,
            "potential_weight": pw,
            "horizon": hz,
        }
        return action, log_prob, out["value"]

    # ------------------------------------------------------- evaluate (PPO)
    def evaluate_actions(
        self,
        *,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        global_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        attack_mask: torch.Tensor,
        defend_mask: torch.Tensor,
        action: dict,
    ) -> dict:
        """Re-evaluate a stored hierarchical ``action`` under current params.

        Returns ``{"log_prob" (B,), "value" (B,), "entropy" (B,)}``.  Entropy is
        the sum of the Bernoulli (attack/defend) and Beta (commit/posture)
        entropies — a tractable lower bound used for the PPO entropy bonus (the
        Plackett-Luce term is omitted).
        """
        out = self.forward(node_feats, edge_feats, global_feats, planet_mask,
                            attack_mask, defend_mask)
        a_logp = bernoulli_log_prob(out["attack_logits"], action["attack_sel"], attack_mask)
        d_logp = bernoulli_log_prob(out["defend_logits"], action["defend_sel"], defend_mask)
        selected = (action["attack_sel"] > 0.5) | (action["defend_sel"] > 0.5)
        pl_logp = plackett_luce_log_prob(out["priority_logits"], action["order"], selected)
        c_logp = commit_fraction_log_prob(
            out["commit_alpha"], out["commit_beta"], action["commit_frac"], defend_mask)
        p_logp = defense_posture_log_prob(
            out["posture_pw_alpha"], out["posture_pw_beta"],
            out["posture_h_alpha"], out["posture_h_beta"], action["posture_u"])
        log_prob = a_logp + d_logp + pl_logp + c_logp + p_logp

        # All terms are per-dimension MEANS (board-size invariant); the two
        # global posture Betas are averaged so each of the four heads
        # contributes on the same ~O(0.5 nat) scale.
        posture_ent = 0.5 * (
            torch.distributions.Beta(out["posture_pw_alpha"], out["posture_pw_beta"]).entropy()
            + torch.distributions.Beta(out["posture_h_alpha"], out["posture_h_beta"]).entropy()
        )
        candidate_bool = attack_mask.to(torch.bool) | defend_mask.to(torch.bool)
        entropy = (
            bernoulli_entropy(out["attack_logits"], attack_mask)
            + bernoulli_entropy(out["defend_logits"], defend_mask)
            + _plackett_luce_entropy_bonus(out["priority_logits"], candidate_bool)
            + _beta_entropy(out["commit_alpha"], out["commit_beta"], defend_mask)
            + posture_ent
        )
        result = {"log_prob": log_prob, "value": out["value"], "entropy": entropy}
        if self.aux_owner_head is not None:
            result["aux_owner_logits"] = out["aux_owner_logits"]
        return result


def bernoulli_sample_log_prob(
    logits: torch.Tensor,            # (B, P)
    mask: torch.Tensor,              # (B, P) bool: valid candidates
    *,
    deterministic: bool = False,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent Bernoulli selection over masked candidates + summed log-prob.

    Invalid (masked-out) entries are forced to 0 and contribute 0 to the
    log-prob.  Returns ``(sample (B, P) float 0/1, log_prob (B,))``.
    """
    mask_f = mask.to(logits.dtype)
    probs = torch.sigmoid(logits) * mask_f
    if deterministic:
        sample = (probs > 0.5).to(logits.dtype)
        # Training samples Bernoulli draws; logits often sit below 0 so every
        # candidate has sigmoid < 0.5.  Threshold-only decoding would pick NO
        # targets -> zero launches in watch / frozen-opponent play.  If nothing
        # clears 0.5, fall back to the single best masked candidate (still
        # deterministic and reproducible).
        none_sel = (sample.sum(dim=1) < 0.5) & (mask_f.sum(dim=1) > 0.5)
        if none_sel.any():
            best = probs.argmax(dim=1, keepdim=True)
            pick = torch.zeros_like(sample).scatter_(1, best, 1.0)
            sample = torch.where(none_sel.unsqueeze(1), pick, sample)
    else:
        sample = torch.bernoulli(probs, generator=generator)
    logp = -F.binary_cross_entropy_with_logits(logits, sample, reduction="none")
    return sample, (logp * mask_f).sum(dim=1)


def bernoulli_log_prob(
    logits: torch.Tensor,            # (B, P)
    sample: torch.Tensor,            # (B, P) float 0/1
    mask: torch.Tensor,              # (B, P) bool
) -> torch.Tensor:
    """Summed masked Bernoulli log-prob of a stored ``sample`` (PPO eval)."""
    mask_f = mask.to(logits.dtype)
    logp = -F.binary_cross_entropy_with_logits(logits, sample, reduction="none")
    return (logp * mask_f).sum(dim=1)


def bernoulli_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """**Mean** Bernoulli entropy over masked candidates ``(B,)``.

    Averaged (not summed) over the valid set so the regulariser scale is
    board-size invariant (~O(0.5 nat) regardless of how many planets exist).
    A summed entropy grows with the number of targets, which makes the entropy
    bonus dwarf the (advantage-driven) policy gradient and pins the policy at
    maximum randomness — observed as a flat H that never anneals.
    """
    mask_f = mask.to(logits.dtype)
    ent = torch.distributions.Bernoulli(logits=logits.clamp(-30.0, 30.0)).entropy()
    return (ent * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def _beta_entropy(alpha: torch.Tensor, beta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """**Mean** Beta entropy over masked (own) planets ``(B,)`` (see above)."""
    alpha, beta = _sanitize_beta_params(alpha, beta)
    mask_f = mask.to(alpha.dtype)
    ent = torch.distributions.Beta(alpha, beta).entropy()
    return (ent * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def plackett_luce_sample(
    scores: torch.Tensor,            # (B, P) priority scores (logits)
    valid_mask: torch.Tensor,        # (B, P) bool: items in the set to be ranked
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Sample a Plackett-Luce ranking over the valid items of each row.

    Plackett-Luce = repeated softmax sampling **without replacement**: draw the
    rank-1 item from ``softmax(scores)`` over the valid set, remove it, draw
    rank-2 from the remaining, and so on.  The log-probability of the resulting
    order ``π`` is the closed form

    .. math::

        \log p(\pi) = \sum_{k} \Big( s_{\pi_k}
            - \log \sum_{j \in R_k} e^{s_j} \Big)

    where ``R_k`` is the set of items not yet placed at step ``k``.

    Parameters
    ----------
    scores:
        ``(B, P)`` priority logits.  Entries where ``valid_mask`` is ``False``
        are ignored (treated as ``-inf``).
    valid_mask:
        ``(B, P)`` boolean mask of the items to rank (e.g. the Bernoulli-
        selected attack/defend targets).
    generator:
        Optional torch RNG for reproducible sampling.

    Returns
    -------
    (order, log_prob):
        ``order`` is ``(B, P)`` long: ``order[b, k]`` is the planet index placed
        at rank ``k``; positions past the number of valid items are ``-1``.
        ``log_prob`` is ``(B,)`` — the log-probability of the sampled order.
    """
    B, P = scores.shape
    device = scores.device
    neg = torch.finfo(scores.dtype).min

    available = valid_mask.to(torch.bool).clone()
    order = torch.full((B, P), -1, dtype=torch.long, device=device)
    log_prob = scores.new_zeros(B)

    for k in range(P):
        has = available.any(dim=1)                                   # (B,)
        if not bool(has.any()):
            break
        masked = torch.where(available, scores, torch.full_like(scores, neg))
        # Rows with nothing available would give all -inf -> NaN softmax; feed
        # them a dummy uniform row and discard their draw via ``has``.
        safe = torch.where(has[:, None], masked, torch.zeros_like(masked))
        probs = torch.softmax(safe, dim=1)
        idx = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)  # (B,)

        lse = torch.logsumexp(masked, dim=1)                         # (B,)
        chosen = scores.gather(1, idx[:, None]).squeeze(1)           # (B,)
        term = chosen - lse
        log_prob = log_prob + torch.where(has, term, torch.zeros_like(term))

        order[:, k] = torch.where(has, idx, torch.full_like(idx, -1))
        picked = torch.zeros_like(available)
        picked.scatter_(1, idx[:, None], True)
        available = available & ~(picked & has[:, None])

    return order, log_prob


def plackett_luce_log_prob(
    scores: torch.Tensor,            # (B, P)
    order: torch.Tensor,             # (B, P) long, -1 padded
    valid_mask: torch.Tensor,        # (B, P) bool
) -> torch.Tensor:
    r"""Log-probability of a given Plackett-Luce ``order`` under ``scores``.

    The PPO-evaluation counterpart of :func:`plackett_luce_sample`: recompute
    the log-prob of a stored ranking with fresh ``scores`` (e.g. after a policy
    update).  See that function for the closed form.

    Returns ``(B,)``.
    """
    B, P = scores.shape
    neg = torch.finfo(scores.dtype).min

    available = valid_mask.to(torch.bool).clone()
    log_prob = scores.new_zeros(B)

    for k in range(P):
        idx = order[:, k]                                            # (B,) or -1
        active = idx >= 0
        if not bool(active.any()):
            break
        masked = torch.where(available, scores, torch.full_like(scores, neg))
        lse = torch.logsumexp(masked, dim=1)                         # (B,)
        safe_idx = idx.clamp_min(0)
        chosen = scores.gather(1, safe_idx[:, None]).squeeze(1)      # (B,)
        term = chosen - lse
        log_prob = log_prob + torch.where(active, term, torch.zeros_like(term))

        picked = torch.zeros_like(available)
        picked.scatter_(1, safe_idx[:, None], True)
        available = available & ~(picked & active[:, None])

    return log_prob


def _plackett_luce_entropy_bonus(
    scores: torch.Tensor,            # (B, P)
    valid_mask: torch.Tensor,      # (B, P) bool
) -> torch.Tensor:
    r"""Mean categorical entropy over valid priority candidates ``(B,)``.

    Tractable exploration bonus for the Plackett-Luce head: one softmax over
    the candidate set (mirrors the rank-1 draw).  Averaged over candidates so
    the scale matches the Bernoulli / Beta entropy terms.
    """
    neg = torch.finfo(scores.dtype).min
    m = valid_mask.to(torch.bool)
    masked = torch.where(m, scores, torch.full_like(scores, neg))
    has = m.any(dim=1)
    safe = torch.where(has[:, None], masked, torch.zeros_like(masked))
    logp = F.log_softmax(safe, dim=1)
    probs = logp.exp()
    ent = -(probs * logp).sum(dim=1)
    return ent / m.sum(dim=1).to(scores.dtype).clamp_min(1.0)


def aux_owner_cross_entropy(
    logits: torch.Tensor,          # (B, P, 5)
    node_feats: torch.Tensor,        # (B, P, node_dim)
    planet_mask: torch.Tensor,       # (B, P) bool
) -> torch.Tensor:
    """Supervised owner prediction from the first 5 standard one-hot channels."""
    labels = node_feats[..., :5].argmax(dim=-1)                       # (B, P)
    valid = planet_mask.to(torch.bool).reshape(-1)
    if not bool(valid.any()):
        return logits.new_zeros(())
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1])[valid],
        labels.reshape(-1)[valid],
    )


_COMMIT_EPS = 1e-4
_BETA_PARAM_MAX = 50.0   # cap Beta concentration (prevents log_prob blow-ups in PPO)


def _sanitize_beta_params(
    alpha: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Finite, strictly-positive Beta parameters for stable distributions."""
    lo = 1.0 + _COMMIT_EPS
    a = torch.nan_to_num(alpha, nan=lo, posinf=_BETA_PARAM_MAX, neginf=lo)
    b = torch.nan_to_num(beta, nan=lo, posinf=_BETA_PARAM_MAX, neginf=lo)
    return a.clamp(lo, _BETA_PARAM_MAX), b.clamp(lo, _BETA_PARAM_MAX)


def sample_commit_fractions(
    alpha: torch.Tensor,             # (B, P)
    beta: torch.Tensor,              # (B, P)
    source_mask: torch.Tensor,       # (B, P) bool: own planets (sources)
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Sample per-planet commit fractions and their masked log-prob.

    Draws ``frac_g ~ Beta(alpha_g, beta_g)`` for every planet, then returns the
    fractions (non-source planets forced to 0 — they cannot be drawn from) and
    the summed Beta log-prob over source planets only.

    Returns ``(fractions (B, P), log_prob (B,))``.
    """
    alpha, beta = _sanitize_beta_params(alpha, beta)
    dist = torch.distributions.Beta(alpha, beta)
    frac = dist.rsample() if alpha.requires_grad else dist.sample()
    frac_clamped = frac.clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)
    mask = source_mask.to(torch.bool)
    logp = (dist.log_prob(frac_clamped) * mask).sum(dim=1).clamp(-40.0, 40.0)
    fractions = torch.where(mask, frac_clamped, torch.zeros_like(frac_clamped))
    return fractions, logp


def commit_fraction_log_prob(
    alpha: torch.Tensor,             # (B, P)
    beta: torch.Tensor,              # (B, P)
    fractions: torch.Tensor,         # (B, P)
    source_mask: torch.Tensor,       # (B, P) bool
) -> torch.Tensor:
    """Masked Beta log-prob of stored commit ``fractions`` (PPO evaluation)."""
    alpha, beta = _sanitize_beta_params(alpha, beta)
    dist = torch.distributions.Beta(alpha, beta)
    frac = fractions.clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)
    mask = source_mask.to(torch.bool)
    return (dist.log_prob(frac) * mask).sum(dim=1).clamp(-40.0, 40.0)


# ---------------------------------------------------------------------------
# Defense posture (global potential_weight / horizon)
# ---------------------------------------------------------------------------
# The two Beta heads emit normalised samples ``u in [0, 1]``; these affine maps
# turn them into the allocator's actual knobs.  ``potential_weight`` scales how
# much nearby enemy garrison mass adds to the defend demand (0 = react only to
# in-flight fleets); ``horizon`` is the ETA window (turns) within which such
# garrisons are considered a threat.  See ``reworked_features.incoming_enemy_estimate``.
POTENTIAL_WEIGHT_MAX = 2.0
HORIZON_MIN = 3.0
HORIZON_MAX = 40.0


def _scale_posture(
    u_pw: torch.Tensor, u_h: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine map the two normalised Beta samples to allocator units."""
    potential_weight = POTENTIAL_WEIGHT_MAX * u_pw
    horizon = HORIZON_MIN + (HORIZON_MAX - HORIZON_MIN) * u_h
    return potential_weight, horizon


def sample_defense_posture(
    pw_alpha: torch.Tensor,          # (B,)
    pw_beta: torch.Tensor,           # (B,)
    h_alpha: torch.Tensor,           # (B,)
    h_beta: torch.Tensor,            # (B,)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Sample the global ``potential_weight`` / ``horizon`` posture.

    Draws normalised ``u_pw ~ Beta(pw_alpha, pw_beta)`` and
    ``u_h ~ Beta(h_alpha, h_beta)`` in ``[0, 1]``, scales them to allocator
    units via :func:`_scale_posture`, and returns the summed Beta log-prob.

    Because the scaling is a *fixed* affine map, its change-of-variables term is
    constant and cancels in the PPO importance ratio; we therefore keep the
    log-prob in the normalised ``u``-space (exactly mirroring ``commit``), and
    return the normalised samples ``u`` for storage / PPO re-evaluation.

    Returns
    -------
    (potential_weight (B,), horizon (B,), u (B, 2), log_prob (B,))
        ``u = [u_pw, u_h]`` are the normalised samples to store in the buffer.
    """
    dist_pw = torch.distributions.Beta(pw_alpha, pw_beta)
    dist_h = torch.distributions.Beta(h_alpha, h_beta)
    u_pw = dist_pw.rsample() if pw_alpha.requires_grad else dist_pw.sample()
    u_h = dist_h.rsample() if h_alpha.requires_grad else dist_h.sample()
    u_pw = u_pw.clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)
    u_h = u_h.clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)

    log_prob = dist_pw.log_prob(u_pw) + dist_h.log_prob(u_h)            # (B,)
    potential_weight, horizon = _scale_posture(u_pw, u_h)
    u = torch.stack([u_pw, u_h], dim=-1)                               # (B, 2)
    return potential_weight, horizon, u, log_prob


def defense_posture_log_prob(
    pw_alpha: torch.Tensor,          # (B,)
    pw_beta: torch.Tensor,           # (B,)
    h_alpha: torch.Tensor,           # (B,)
    h_beta: torch.Tensor,            # (B,)
    u: torch.Tensor,                 # (B, 2) stored normalised samples
) -> torch.Tensor:
    """Beta log-prob of a stored posture sample ``u`` (PPO evaluation).

    ``u`` are the normalised ``[u_pw, u_h]`` returned by
    :func:`sample_defense_posture`.  Returns ``(B,)``.
    """
    dist_pw = torch.distributions.Beta(pw_alpha, pw_beta)
    dist_h = torch.distributions.Beta(h_alpha, h_beta)
    u_pw = u[..., 0].clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)
    u_h = u[..., 1].clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)
    return dist_pw.log_prob(u_pw) + dist_h.log_prob(u_h)


def posture_from_norm(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map stored normalised posture ``u (..., 2)`` back to allocator units.

    Convenience for turning a buffered ``u = [u_pw, u_h]`` into
    ``(potential_weight, horizon)`` without re-sampling.
    """
    return _scale_posture(u[..., 0], u[..., 1])


# ===========================================================================
# v8 launch policy: source-centric, allocator-free
# ===========================================================================
# Each OWN planet (source) chooses, in one turn:
#   * commit_h ~ Beta(alpha, beta)              -- garrison fraction to deploy
#   * split_h  ~ Dirichlet(conc_{h, valid t})   -- distribution over targets
# and launches ``floor(commit_h * garrison_h * split_{h,t})`` to every target
# (dropping sub-``min_send`` slivers).  Because the split sums to 1 the per-source
# committed mass never exceeds the garrison -> feasibility is automatic and no
# min-cost-flow allocator is needed.  Attack vs. reinforcement is derived from
# the target's owner, so no separate attack/defend heads are required.

_DIR_CONC_EPS = 1e-3   # concentration floor (keeps Beta/Gamma well-defined)
_DIR_CONC_MAX = 50.0   # cap Dirichlet concentration (prevents PPO log_prob spikes)
_DIR_EPS = 1e-6        # numerical floor for logs / denominators
_LOGP_CLAMP = 40.0     # per-term log-prob clamp for PPO stability


def _sanitize_dirichlet_conc(conc: torch.Tensor) -> torch.Tensor:
    """Finite, strictly-positive Dirichlet concentrations."""
    c = torch.nan_to_num(conc, nan=_DIR_CONC_EPS, posinf=_DIR_CONC_MAX, neginf=_DIR_CONC_EPS)
    return c.clamp(_DIR_CONC_EPS, _DIR_CONC_MAX)


def _dirichlet_row_logp(
    conc: torch.Tensor,   # (B, P, P) concentration (rows = sources, cols = targets)
    x: torch.Tensor,      # (B, P, P) simplex weights per source row
    valid: torch.Tensor,  # (B, P, P) bool: valid (source, target) pairs
) -> torch.Tensor:
    r"""Per-row masked Dirichlet log-density ``(B, P)``.

    .. math::
        \log p(x \mid a) = \log\Gamma(\textstyle\sum_i a_i)
            - \sum_i \log\Gamma(a_i) + \sum_i (a_i - 1)\log x_i

    summed only over the valid columns of each source row.  Invalid columns are
    excluded (they contribute nothing to either the normaliser or the kernel).
    """
    vf = valid.to(conc.dtype)
    a_sum = (conc * vf).sum(dim=2)                                  # (B, P)
    sum_lgamma = (torch.lgamma(conc.clamp_min(_DIR_EPS)) * vf).sum(dim=2)
    xv = x.clamp_min(_DIR_EPS)
    kernel = ((conc - 1.0) * torch.log(xv) * vf).sum(dim=2)        # (B, P)
    return (torch.lgamma(a_sum.clamp_min(_DIR_EPS)) - sum_lgamma + kernel).clamp(
        -_LOGP_CLAMP, _LOGP_CLAMP)


def masked_dirichlet_log_prob(
    conc: torch.Tensor,   # (B, P, P)
    x: torch.Tensor,      # (B, P, P)
    valid: torch.Tensor,  # (B, P, P) bool
) -> torch.Tensor:
    """Summed masked Dirichlet log-prob over source rows ``(B,)`` (PPO eval)."""
    conc = _sanitize_dirichlet_conc(conc)
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    row = _dirichlet_row_logp(conc, x, valid)                      # (B, P)
    row_valid = valid.any(dim=2)                                   # (B, P) source rows
    return (row * row_valid.to(row.dtype)).sum(dim=1).clamp(-_LOGP_CLAMP, _LOGP_CLAMP)


def masked_dirichlet_sample_log_prob(
    conc: torch.Tensor,   # (B, P, P)
    valid: torch.Tensor,  # (B, P, P) bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a per-row Dirichlet split over valid targets + its log-prob.

    Draws via the Gamma construction: ``g_t ~ Gamma(conc_t, 1)``, zero the
    invalid columns, then normalise each row over its valid columns.  Returns
    ``(weights (B, P, P) with 0 on invalid cols, log_prob (B,))``.
    """
    conc = _sanitize_dirichlet_conc(conc)
    conc_pos = conc.clamp_min(_DIR_EPS)
    dist = torch.distributions.Gamma(conc_pos, torch.ones_like(conc_pos))
    g = dist.rsample() if conc.requires_grad else dist.sample()
    vf = valid.to(conc.dtype)
    g = g * vf
    denom = g.sum(dim=2, keepdim=True).clamp_min(_DIR_EPS)
    w = g / denom
    return w, masked_dirichlet_log_prob(conc, w, valid)


def masked_dirichlet_mean(
    conc: torch.Tensor,   # (B, P, P)
    valid: torch.Tensor,  # (B, P, P) bool
) -> torch.Tensor:
    """Deterministic Dirichlet mean ``a / sum(a)`` per row (0 on invalid cols)."""
    conc = _sanitize_dirichlet_conc(conc)
    vf = valid.to(conc.dtype)
    a = conc * vf
    a0 = a.sum(dim=2, keepdim=True).clamp_min(_DIR_EPS)
    return a / a0


def masked_dirichlet_entropy(
    conc: torch.Tensor,   # (B, P, P)
    valid: torch.Tensor,  # (B, P, P) bool
) -> torch.Tensor:
    r"""**Mean** masked Dirichlet entropy over source rows ``(B,)``.

    Closed form ``H(a) = \log B(a) + (a_0 - K)\psi(a_0)
    - \sum_i (a_i - 1)\psi(a_i)`` restricted to the valid columns (``K`` of
    them).  Each row's entropy is divided by its support size ``K`` and then
    averaged over source rows, so the regulariser stays ~O(1) per dimension and
    board-size invariant -- mirroring the per-dimension *mean* convention of
    :func:`bernoulli_entropy` / :func:`_beta_entropy` (a raw Dirichlet entropy
    scales with the number of targets and would otherwise dwarf the policy
    gradient / pin the policy at maximum spread).
    """
    conc = _sanitize_dirichlet_conc(conc)
    vf = valid.to(conc.dtype)
    a0 = (conc * vf).sum(dim=2)                                    # (B, P)
    K = vf.sum(dim=2)                                              # (B, P)
    sum_lgamma = (torch.lgamma(conc.clamp_min(_DIR_EPS)) * vf).sum(dim=2)
    log_beta = sum_lgamma - torch.lgamma(a0.clamp_min(_DIR_EPS))
    dig_a0 = torch.digamma(a0.clamp_min(_DIR_EPS))
    sum_term = ((conc - 1.0) * torch.digamma(conc.clamp_min(_DIR_EPS)) * vf).sum(dim=2)
    H = log_beta + (a0 - K) * dig_a0 - sum_term                   # (B, P)
    H = H / K.clamp_min(1.0)                                      # per-dimension scale
    row_valid = valid.any(dim=2).to(conc.dtype)                   # (B, P)
    denom = row_valid.sum(dim=1).clamp_min(1.0)
    return (H * row_valid).sum(dim=1) / denom


class ReworkedLaunchPolicy(nn.Module):
    r"""Allocator-free, source-centric launch policy (v8).

    The agent decides launches **directly**: for every own planet (source) it
    samples a garrison commit fraction ``Beta(alpha, beta)`` and a target
    distribution ``Dirichlet`` over the reachable planets, then sends
    ``floor(commit * garrison * split_t)`` ships to each target.  This replaces
    the hierarchical ``attack/defend`` Bernoulli + Plackett-Luce ordering +
    per-source commit + global posture heads of :class:`ReworkedHierarchicalPolicy`
    and removes the min-cost-flow allocator entirely (feasibility is automatic
    because the per-source split is a simplex).

    Heads only read the encoder's equivariant ``node_tokens`` / ``edge_tokens``
    and invariant ``graph_summary`` (no index parameters), so the policy is
    permutation-symmetric.  The Dirichlet concentration is an **additive** edge
    score (source bias + target bias + low-rank interaction + edge term) to
    avoid materialising a ``(B, P, P, 4d)`` concatenation.

    Per-turn action log-prob::

        log_prob = Sum_{h in own} [ Beta(commit_h) + Dirichlet(split_h) ]
    """

    def __init__(
        self,
        *,
        node_dim: int = REWORKED_NODE_FEATURE_DIM,
        edge_dim: int = REWORKED_EDGE_FEATURE_DIM,
        global_dim: int = REWORKED_GLOBAL_FEATURE_DIM,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
        aux_owner: bool = True,
        interaction_rank: int = 16,
    ):
        super().__init__()
        self.encoder = ReworkedPlanetGraphEncoder(
            node_dim=node_dim, edge_dim=edge_dim, global_dim=global_dim,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
        )
        self.node_dim = int(node_dim)
        self.d_model = d_model
        r = int(interaction_rank)

        # Per-source commit fraction Beta(alpha, beta).
        self.commit_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 2),
        )
        # Dirichlet concentration over targets, built additively (memory-safe):
        #   conc[h, t] = softplus( src_bias_h + tgt_bias_t
        #                          + <qs_h, qt_t> + edge_term_{h,t} ) + eps
        self.dir_src = nn.Linear(2 * d_model, 1)
        self.dir_tgt = nn.Linear(2 * d_model, 1)
        self.dir_src_int = nn.Linear(2 * d_model, r)
        self.dir_tgt_int = nn.Linear(2 * d_model, r)
        self.dir_edge = nn.Linear(d_model, 1)

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        self.aux_owner_head = nn.Linear(d_model, 5) if aux_owner else None

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        global_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        source_mask: torch.Tensor,        # (B, P) bool: own planets that can send
    ) -> dict:
        enc = self.encoder(node_feats, edge_feats, global_feats, planet_mask)
        tok = torch.nan_to_num(enc["node_tokens"], nan=0.0, posinf=0.0, neginf=0.0)
        summary = torch.nan_to_num(enc["graph_summary"], nan=0.0, posinf=0.0, neginf=0.0)
        edge_tok = torch.nan_to_num(enc["edge_tokens"], nan=0.0, posinf=0.0, neginf=0.0)
        B, P, _ = tok.shape

        s_bc = summary[:, None, :].expand(-1, P, -1)
        node_in = torch.cat([tok, s_bc], dim=-1)  # (B, P, 2d)

        commit_params = F.softplus(self.commit_head(node_in)) + 1.0   # (B, P, 2)
        ca = commit_params[..., 0]
        cb = commit_params[..., 1]
        src_bool = source_mask.to(torch.bool)
        ones = torch.ones_like(ca)
        commit_alpha = torch.where(src_bool, ca, ones)   # non-source -> Beta(1,1)
        commit_beta = torch.where(src_bool, cb, ones)
        commit_alpha, commit_beta = _sanitize_beta_params(commit_alpha, commit_beta)

        s = self.dir_src(node_in).squeeze(-1)            # (B, P) source bias
        t = self.dir_tgt(node_in).squeeze(-1)            # (B, P) target bias
        qs = self.dir_src_int(node_in)                   # (B, P, r)
        qt = self.dir_tgt_int(node_in)                   # (B, P, r)
        inter = torch.einsum("bhr,btr->bht", qs, qt)     # (B, P, P)
        e = self.dir_edge(edge_tok).squeeze(-1)          # (B, P, P)
        dir_logit = s[:, :, None] + t[:, None, :] + inter + e
        dir_conc = _sanitize_dirichlet_conc(F.softplus(dir_logit) + _DIR_CONC_EPS)

        value = self.value_head(summary).squeeze(-1)      # (B,)
        out = {
            "commit_alpha": commit_alpha,
            "commit_beta": commit_beta,
            "dir_conc": dir_conc,
            "value": value,
            "node_tokens": tok,
        }
        if self.aux_owner_head is not None:
            out["aux_owner_logits"] = self.aux_owner_head(tok)   # (B, P, 5)
        return out

    @torch.no_grad()
    def act(
        self,
        *,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        global_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        source_mask: torch.Tensor,        # (B, P) bool
        dest_mask: torch.Tensor,          # (B, P, P) bool: valid (source, target)
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Sample (or take the mode of) a full launch action + summed log-prob.

        Returns ``(action, log_prob (B,), value (B,))`` with
        ``action = {"commit" (B, P), "split_w" (B, P, P)}``.
        """
        out = self.forward(node_feats, edge_feats, global_feats, planet_mask, source_mask)
        ca, cb = out["commit_alpha"], out["commit_beta"]
        valid = dest_mask.to(torch.bool)
        src_bool = source_mask.to(torch.bool)

        if deterministic:
            commit = (ca / (ca + cb)).clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)
            commit = torch.where(src_bool, commit, torch.zeros_like(commit))
            c_logp = commit_fraction_log_prob(ca, cb, commit, source_mask)
            w = masked_dirichlet_mean(out["dir_conc"], valid)
            d_logp = masked_dirichlet_log_prob(out["dir_conc"], w, valid)
        else:
            commit, c_logp = sample_commit_fractions(ca, cb, source_mask)
            w, d_logp = masked_dirichlet_sample_log_prob(out["dir_conc"], valid)

        log_prob = (c_logp + d_logp).clamp(-_LOGP_CLAMP, _LOGP_CLAMP)
        action = {"commit": commit, "split_w": w}
        return action, log_prob, out["value"]

    def evaluate_actions(
        self,
        *,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        global_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        source_mask: torch.Tensor,
        dest_mask: torch.Tensor,
        action: dict,
    ) -> dict:
        """Re-evaluate a stored launch ``action`` under current params (PPO)."""
        out = self.forward(node_feats, edge_feats, global_feats, planet_mask, source_mask)
        valid = dest_mask.to(torch.bool)
        c_logp = commit_fraction_log_prob(
            out["commit_alpha"], out["commit_beta"], action["commit"], source_mask)
        d_logp = masked_dirichlet_log_prob(out["dir_conc"], action["split_w"], valid)
        log_prob = (c_logp + d_logp).clamp(-_LOGP_CLAMP, _LOGP_CLAMP)
        entropy = (
            _beta_entropy(out["commit_alpha"], out["commit_beta"], source_mask)
            + masked_dirichlet_entropy(out["dir_conc"], valid)
        )
        result = {"log_prob": log_prob, "value": out["value"], "entropy": entropy}
        if self.aux_owner_head is not None:
            result["aux_owner_logits"] = out["aux_owner_logits"]
        return result


def _edge_per_target_mean(
    edge_terms: torch.Tensor,   # (B, P, P) indexed [source h, target t]
    edge_mask: torch.Tensor,    # (B, P, P) bool
) -> torch.Tensor:
    """Mean per-edge quantity over valid sources for each target ``(B, P)``.

    Keeps log-prob / entropy board-size invariant when several sources fire at
    the same target (raw sums would grow with in-degree).
    """
    m = edge_mask.to(edge_terms.dtype)
    n_src = m.sum(dim=1).clamp_min(1.0)
    return (edge_terms * m).sum(dim=1) / n_src


class ReworkedEdgePolicy(nn.Module):
    r"""Directed-edge, allocator-free launch policy (v9, per-target credit).

    Every valid directed edge ``h -> t`` is an **independent** decision:

    * ``gate ~ Bernoulli(sigmoid(gate_logit_{h,t}))`` -- send along this edge?
    * ``frac ~ Beta(a_{h,t}, b_{h,t})`` -- fraction of source ``h``'s garrison to
      commit (only counts when ``gate = 1``).

    Ships sent ``= floor(frac * garrison_h)``, feasibility-scaled per source at
    execution time (no hard per-source simplex).  Over-extension is therefore
    not *forbidden* but *taught*: the source-loss credit term penalises stripping
    a planet that subsequently falls, so a deliberate sacrifice (capture
    elsewhere worth more) stays available to the policy.

    Because the per-edge log-prob factorises fully over edges, the joint
    log-prob partitions cleanly under **any** grouping.  We group by **target**:
    all edges ``(. -> t)`` form target ``t``'s campaign, so its
    capture/defence/fail/source-loss credit attaches to a single per-target
    advantage.  Different sources to the same target share the learning signal
    (same purpose); different targets are independent.  ``act`` / ``evaluate``
    therefore return log-prob and entropy **per target** ``(B, P)`` -- not
    summed -- which the per-campaign PPO objective consumes directly.
    The critic mirrors this with an equivariant **per-target value head**
    ``V_t = f(node_t, summary)`` — one scalar per planet slot that permutes
    with the board (not a single invariant graph value).

    The gate uses an additive low-rank edge score (source bias + target bias +
    low-rank interaction + edge term) so target selection is expressive; the
    magnitude Beta uses a cheaper additive (source + target + edge) score.  All
    heads read only equivariant node/edge tokens + the invariant summary, so the
    policy is permutation-symmetric and index-free.
    """

    def __init__(
        self,
        *,
        node_dim: int = REWORKED_NODE_FEATURE_DIM,
        edge_dim: int = REWORKED_EDGE_FEATURE_DIM,
        global_dim: int = REWORKED_GLOBAL_FEATURE_DIM,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
        aux_owner: bool = True,
        interaction_rank: int = 16,
    ):
        super().__init__()
        self.encoder = ReworkedPlanetGraphEncoder(
            node_dim=node_dim, edge_dim=edge_dim, global_dim=global_dim,
            d_model=d_model, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
        )
        self.node_dim = int(node_dim)
        self.d_model = d_model
        r = int(interaction_rank)

        # Per-edge gate (Bernoulli) logit, built additively (memory-safe):
        #   gate_logit[h, t] = src_h + tgt_t + <qs_h, qt_t> + edge_term_{h,t}
        self.gate_src = nn.Linear(2 * d_model, 1)
        self.gate_tgt = nn.Linear(2 * d_model, 1)
        self.gate_src_int = nn.Linear(2 * d_model, r)
        self.gate_tgt_int = nn.Linear(2 * d_model, r)
        self.gate_edge = nn.Linear(d_model, 1)

        # Per-edge magnitude Beta(alpha, beta) -- additive (source + target +
        # edge) pre-activations, softplus'd to keep concentrations >= 1.
        self.mag_a_src = nn.Linear(2 * d_model, 1)
        self.mag_a_tgt = nn.Linear(2 * d_model, 1)
        self.mag_a_edge = nn.Linear(d_model, 1)
        self.mag_b_src = nn.Linear(2 * d_model, 1)
        self.mag_b_tgt = nn.Linear(2 * d_model, 1)
        self.mag_b_edge = nn.Linear(d_model, 1)

        # Per-target value head (equivariant): one scalar V_t per planet slot.
        # Reads ``[node_token_t, broadcast graph_summary]`` — no index params, so
        # relabelling planets permutes the output vector the same way as
        # ``gate_logit`` / ``target_logp``.  Aligns the critic with per-target
        # campaign advantages in PPO.
        self.target_value_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        self.aux_owner_head = nn.Linear(d_model, 5) if aux_owner else None

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        global_feats: torch.Tensor,
        planet_mask: torch.Tensor,
    ) -> dict:
        enc = self.encoder(node_feats, edge_feats, global_feats, planet_mask)
        tok = torch.nan_to_num(enc["node_tokens"], nan=0.0, posinf=0.0, neginf=0.0)
        summary = torch.nan_to_num(enc["graph_summary"], nan=0.0, posinf=0.0, neginf=0.0)
        edge_tok = torch.nan_to_num(enc["edge_tokens"], nan=0.0, posinf=0.0, neginf=0.0)
        B, P, _ = tok.shape
        pm = planet_mask.to(torch.bool)

        s_bc = summary[:, None, :].expand(-1, P, -1)
        node_in = torch.cat([tok, s_bc], dim=-1)  # (B, P, 2d)

        gs = self.gate_src(node_in).squeeze(-1)          # (B, P)
        gt = self.gate_tgt(node_in).squeeze(-1)          # (B, P)
        gqs = self.gate_src_int(node_in)                 # (B, P, r)
        gqt = self.gate_tgt_int(node_in)                 # (B, P, r)
        ginter = torch.einsum("bhr,btr->bht", gqs, gqt)  # (B, P, P)
        ge = self.gate_edge(edge_tok).squeeze(-1)        # (B, P, P)
        gate_logit = gs[:, :, None] + gt[:, None, :] + ginter + ge

        ma = (self.mag_a_src(node_in).squeeze(-1)[:, :, None]
              + self.mag_a_tgt(node_in).squeeze(-1)[:, None, :]
              + self.mag_a_edge(edge_tok).squeeze(-1))
        mb = (self.mag_b_src(node_in).squeeze(-1)[:, :, None]
              + self.mag_b_tgt(node_in).squeeze(-1)[:, None, :]
              + self.mag_b_edge(edge_tok).squeeze(-1))
        mag_alpha = F.softplus(ma) + 1.0                 # (B, P, P)
        mag_beta = F.softplus(mb) + 1.0
        mag_alpha, mag_beta = _sanitize_beta_params(mag_alpha, mag_beta)

        target_value = self.target_value_head(node_in).squeeze(-1)       # (B, P)
        target_value = target_value.masked_fill(~pm, 0.0)
        out = {
            "gate_logit": gate_logit,
            "mag_alpha": mag_alpha,
            "mag_beta": mag_beta,
            "target_value": target_value,
            "node_tokens": tok,
        }
        if self.aux_owner_head is not None:
            out["aux_owner_logits"] = self.aux_owner_head(tok)
        return out

    @torch.no_grad()
    def act(
        self,
        *,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        global_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        edge_mask: torch.Tensor,          # (B, P, P) bool: valid directed edges
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[dict, torch.Tensor, torch.Tensor]:
        """Sample per-edge gate + magnitude; return per-TARGET log-prob + values.

        Returns ``(action, target_logp (B, P), target_value (B, P))`` with
        ``action = {"gate" (B, P, P), "frac" (B, P, P)}``.  ``target_logp[b, t]``
        is the **mean** per-edge log-prob over sources landing on target ``t``.
        ``target_value[b, t]`` is the equivariant critic for campaigning on ``t``.
        """
        out = self.forward(node_feats, edge_feats, global_feats, planet_mask)
        gl = out["gate_logit"]
        a, b = _sanitize_beta_params(out["mag_alpha"], out["mag_beta"])
        m = edge_mask.to(gl.dtype)
        probs = torch.sigmoid(gl) * m
        dist = torch.distributions.Beta(a, b)

        if deterministic:
            gate = (probs > 0.5).to(gl.dtype)
            # Avoid a no-op turn: if a board gated everything off but has valid
            # edges, fire the single most-probable edge (reproducible).
            none_sel = (gate.flatten(1).sum(dim=1) < 0.5) & (m.flatten(1).sum(dim=1) > 0.5)
            if bool(none_sel.any()):
                flat = probs.flatten(1)
                best = flat.argmax(dim=1, keepdim=True)
                pick = torch.zeros_like(flat).scatter_(1, best, 1.0).view_as(gate)
                gate = torch.where(none_sel[:, None, None], pick, gate)
            frac = (a / (a + b)).clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)
        else:
            gate = torch.bernoulli(probs, generator=generator)
            frac = dist.sample().clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)

        gate = gate * m
        gate_logp = -F.binary_cross_entropy_with_logits(gl, gate, reduction="none") * m
        mag_logp = dist.log_prob(frac.clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)) * gate
        edge_logp = gate_logp + mag_logp
        target_logp = _edge_per_target_mean(edge_logp, edge_mask).clamp(
            -_LOGP_CLAMP, _LOGP_CLAMP)
        action = {"gate": gate, "frac": frac}
        return action, target_logp, out["target_value"]

    def evaluate_actions(
        self,
        *,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        global_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        action: dict,
    ) -> dict:
        """Re-evaluate a stored edge ``action`` (PPO), per target.

        Returns ``target_logp`` / ``entropy_target`` / ``target_value`` ``(B, P)``
        and the ``target_valid`` mask ``(B, P)``.
        """
        out = self.forward(node_feats, edge_feats, global_feats, planet_mask)
        gl = out["gate_logit"]
        a, b = _sanitize_beta_params(out["mag_alpha"], out["mag_beta"])
        m = edge_mask.to(gl.dtype)
        dist = torch.distributions.Beta(a, b)

        gate = action["gate"].to(gl.dtype) * m
        frac = action["frac"].clamp(_COMMIT_EPS, 1.0 - _COMMIT_EPS)
        gate_logp = -F.binary_cross_entropy_with_logits(gl, gate, reduction="none") * m
        mag_logp = dist.log_prob(frac) * gate
        edge_logp = gate_logp + mag_logp
        target_logp = _edge_per_target_mean(edge_logp, edge_mask).clamp(
            -_LOGP_CLAMP, _LOGP_CLAMP)

        bern_ent = torch.distributions.Bernoulli(logits=gl.clamp(-30.0, 30.0)).entropy()
        beta_ent = dist.entropy()
        probs = torch.sigmoid(gl)
        edge_ent = (bern_ent + probs * beta_ent) * m
        ent_target = _edge_per_target_mean(edge_ent, edge_mask)
        target_valid = m.sum(dim=1) > 0.5
        result = {
            "target_logp": target_logp,
            "target_value": out["target_value"],
            "entropy_target": ent_target,
            "target_valid": target_valid,
        }
        if self.aux_owner_head is not None:
            result["aux_owner_logits"] = out["aux_owner_logits"]
        return result


__all__ = [
    "NUM_DERIVED_NODE_METRICS",
    "NUM_EXTRA_NODE_METRICS",
    "REWORKED_NODE_FEATURE_DIM",
    "REWORKED_GLOBAL_FEATURE_DIM",
    "DenseEdgeAttnGINELayer",
    "ReworkedPlanetGraphEncoder",
    "ReworkedHierarchicalPolicy",
    "ReworkedLaunchPolicy",
    "ReworkedEdgePolicy",
    "masked_dirichlet_log_prob",
    "masked_dirichlet_sample_log_prob",
    "masked_dirichlet_mean",
    "masked_dirichlet_entropy",
    "plackett_luce_sample",
    "plackett_luce_log_prob",
    "_plackett_luce_entropy_bonus",
    "aux_owner_cross_entropy",
    "_edge_per_target_mean",
    "sample_commit_fractions",
    "commit_fraction_log_prob",
    "bernoulli_sample_log_prob",
    "bernoulli_log_prob",
    "bernoulli_entropy",
    "POTENTIAL_WEIGHT_MAX",
    "HORIZON_MIN",
    "HORIZON_MAX",
    "sample_defense_posture",
    "defense_posture_log_prob",
    "posture_from_norm",
]
