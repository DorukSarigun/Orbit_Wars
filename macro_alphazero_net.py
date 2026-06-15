"""Neural (policy prior + value) network for the macro MCTS (AlphaZero hooks).

This module wires a small, permutation-aware Torch network into the abstract
macro search of :mod:`orbit_macro_mcts_agent`:

* **Value head** -- reads abstract per-planet features + a board-level summary
  and predicts the abstract return ``V`` (same scale as
  :meth:`MacroTimedMCTS._score`).  Used as the MCTS leaf evaluation, blended
  with the heuristic rollout via ``value_blend``.
* **Prior head** -- reads per-candidate macro-action features
  (``MACRO_FEATURE_NAMES``) and emits a prior logit per action, used inside
  PUCT selection.

The network operates on the *abstract* state, so it is usable at **every** MCTS
node (root and internal leaves) without a representation gap -- the consistent
choice for AlphaZero-style search over the abstract model.

Architecture
------------
Planet processing is a **graph neural network** (``ReworkedPlanetGraphEncoder``,
a dependency-free dense edge-attention GNN) rather than a per-planet MLP + pool.
It runs over the *abstract* state, so it stays queryable at every MCTS node:

* **Node features** -- the abstract per-planet vector (ownership / ships /
  production / inbound pressure / candidacy), ``PLANET_FEATURE_DIM`` wide.
* **Edge features** -- static directed route geometry from the orbit graph
  (reachable / ETA / distance / sun-crossing), ``EDGE_FEATURE_DIM`` wide, so the
  GNN sees the board *topology* (which planet can reach which, and how fast).
* **Value head** -- reads the permutation-invariant graph summary and predicts
  the abstract return ``V`` (same scale as ``MacroTimedMCTS._score``).
* **Prior head** -- per candidate macro action, concatenates the hand-crafted
  action features with the *target planet's* GNN node token and the graph
  summary, so priors are conditioned on the learned board representation.

The base search module stays Torch-free for Kaggle packaging.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # self-contained dense edge-attention GNN (no PyG / edge_index)
    from reworked_model import ReworkedPlanetGraphEncoder
except ImportError:  # pragma: no cover - package layout
    from .reworked_model import ReworkedPlanetGraphEncoder  # type: ignore

from orbit_graph import OrbitGraph
from orbit_macro_mcts_agent import (
    MACRO_FEATURE_NAMES,
    LeafValueFn,
    MacroMCTSConfig,
    MacroPriorProvider,
    MacroPriorRequest,
    MacroSearchResult,
    MacroTimedMCTS,
    OrbitMacroMCTSAgent,
)
from orbit_scripted_agent import OrbitScriptedConfig
from orbit_timed_mcts import _TimedState


# Per-planet abstract features (permutation-equivariant, pooled for value).
PLANET_FEATURE_NAMES: tuple[str, ...] = (
    "own_root",
    "own_enemy",
    "neutral",
    "ships_norm",
    "prod_norm",
    "inbound_hostile_norm",
    "inbound_friendly_norm",
    "min_inbound_eta_norm",
    "is_candidate",
)
GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "turn_frac",
    "root_owned_frac",
    "root_ships_share",
    "root_prod_share",
    "enemy_ships_share",
    "enemy_prod_share",
)
# Directed per-edge route geometry (static; from the orbit graph).
EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "reachable",
    "eta_norm",
    "distance",
    "crosses_sun",
)

PLANET_FEATURE_DIM = len(PLANET_FEATURE_NAMES)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)
ACTION_FEATURE_DIM = len(MACRO_FEATURE_NAMES)

# Cap for the normalised ETA edge feature (also the value for unreachable legs).
_ETA_NORM_CLIP = 4.0


# --------------------------------------------------------------------------- #
# Abstract-state featurisation (numpy; shared by inference + training).
# --------------------------------------------------------------------------- #
def abstract_state_arrays(
    mcts: MacroTimedMCTS, state: _TimedState
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(planet_feats [P, F], global_feats [G])`` for an abstract state."""
    owners = state.owner_map()
    ships = state.ship_map()
    root = int(mcts.root_player)
    horizon = max(1.0, float(mcts.horizon_turns))

    inbound_hostile: dict[int, float] = {}
    inbound_friendly: dict[int, float] = {}
    min_eta: dict[int, float] = {}
    for e in state.pending:
        tgt = int(e.target)
        rel = max(0, int(e.arrival_turn) - int(state.turn))
        if int(e.owner) == root:
            inbound_friendly[tgt] = inbound_friendly.get(tgt, 0.0) + float(e.ships)
        elif int(e.owner) >= 0:
            inbound_hostile[tgt] = inbound_hostile.get(tgt, 0.0) + float(e.ships)
        prev = min_eta.get(tgt)
        min_eta[tgt] = rel if prev is None else min(prev, rel)

    ship_by_owner: dict[int, float] = {}
    prod_by_owner: dict[int, float] = {}
    total_ships = 0.0
    total_prod = 0.0
    n_root_owned = 0
    for pid, owner in owners.items():
        s = float(ships.get(pid, 0))
        pr = float(mcts._prod.get(pid, 0))
        total_ships += s
        total_prod += pr
        if int(owner) == root:
            n_root_owned += 1
        if int(owner) >= 0:
            ship_by_owner[owner] = ship_by_owner.get(owner, 0.0) + s
            prod_by_owner[owner] = prod_by_owner.get(owner, 0.0) + pr

    ids = sorted(owners.keys())
    feats = np.zeros((len(ids), PLANET_FEATURE_DIM), dtype=np.float32)
    for i, pid in enumerate(ids):
        owner = int(owners[pid])
        feats[i] = (
            1.0 if owner == root else 0.0,
            1.0 if owner >= 0 and owner != root else 0.0,
            1.0 if owner < 0 else 0.0,
            float(ships.get(pid, 0)) / 100.0,
            float(mcts._prod.get(pid, 0)) / 10.0,
            inbound_hostile.get(pid, 0.0) / 100.0,
            inbound_friendly.get(pid, 0.0) / 100.0,
            (min_eta[pid] / horizon) if pid in min_eta else 1.0,
            1.0 if pid in mcts.candidates else 0.0,
        )

    root_ships = ship_by_owner.get(root, 0.0)
    root_prod = prod_by_owner.get(root, 0.0)
    enemy_ships = max((v for o, v in ship_by_owner.items() if o != root), default=0.0)
    enemy_prod = max((v for o, v in prod_by_owner.items() if o != root), default=0.0)
    glob = np.array([
        float(mcts.current_turn) / max(1.0, float(mcts.max_turns)),
        float(n_root_owned) / max(1, len(ids)),
        root_ships / max(1e-6, total_ships),
        root_prod / max(1e-6, total_prod),
        enemy_ships / max(1e-6, total_ships),
        enemy_prod / max(1e-6, total_prod),
    ], dtype=np.float32)
    return feats, glob


def abstract_state_ids(state: _TimedState) -> list[int]:
    """Planet ids in the same order as :func:`abstract_state_arrays` rows."""
    return sorted(state.owner_map().keys())


def abstract_edge_arrays(mcts: MacroTimedMCTS, ids: list[int]) -> np.ndarray:
    """Return directed edge features ``[P, P, EDGE_FEATURE_DIM]`` indexed ``[src, dst]``.

    Route geometry is static (depends only on the orbit graph), so the same
    tensor is valid for every abstract state of one search.  ``edge[i, j]`` is
    the directed leg ``ids[i] -> ids[j]`` (source -> destination), matching the
    ``edge_tokens[b, src, dst]`` convention of the GNN encoder.
    """
    G = mcts.og.G
    horizon = max(1.0, float(mcts.horizon_turns))
    n = len(ids)
    ef = np.zeros((n, n, EDGE_FEATURE_DIM), dtype=np.float32)
    for i, src in enumerate(ids):
        for j, dst in enumerate(ids):
            if i == j or not G.has_edge(src, dst):
                continue
            data = G.edges[src, dst]
            eta = float(data.get("eta", math.inf))
            if math.isfinite(eta):
                eta_norm = min(eta / horizon, _ETA_NORM_CLIP)
            else:
                eta_norm = _ETA_NORM_CLIP
            ef[i, j] = (
                1.0 if data.get("reachable", False) else 0.0,
                eta_norm,
                float(data.get("distance", 0.0)),
                1.0 if data.get("crosses_sun", False) else 0.0,
            )
    return ef


def action_feature_vector(features: Mapping[str, float]) -> np.ndarray:
    """Pack a macro-action feature dict into the fixed ``MACRO_FEATURE_NAMES`` order."""
    return np.array(
        [float(features.get(name, 0.0)) for name in MACRO_FEATURE_NAMES],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------- #
# Network.
# --------------------------------------------------------------------------- #
class MacroAlphaZeroNet(nn.Module):
    """GNN ``f(s) -> (prior logits, value)`` over the abstract macro graph.

    The planet stack is a dense edge-attention GNN (``ReworkedPlanetGraphEncoder``)
    that consumes node + directed-edge + global features and emits per-planet
    node tokens and a permutation-invariant graph summary.  The value head reads
    the summary; the prior head reads, per action, ``[action_feats, target node
    token, graph summary]``.
    """

    def __init__(
        self,
        d_model: int = 64,
        *,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.arch = "gnn"
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)

        self.encoder = ReworkedPlanetGraphEncoder(
            node_dim=PLANET_FEATURE_DIM,
            edge_dim=EDGE_FEATURE_DIM,
            global_dim=GLOBAL_FEATURE_DIM,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            dropout=float(dropout),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        # Learned token used as the "target" embedding for actions without a
        # planet target (e.g. PASS).
        self.pass_token = nn.Parameter(torch.zeros(self.d_model))
        self.prior_head = nn.Sequential(
            nn.Linear(ACTION_FEATURE_DIM + 2 * self.d_model, self.d_model), nn.GELU(),
            nn.Linear(self.d_model, self.d_model), nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    # -- encoding ---------------------------------------------------------- #
    def encode(
        self,
        planet_feats: torch.Tensor,   # (B, P, Fp)
        edge_feats: torch.Tensor,     # (B, P, P, Fe)
        global_feats: torch.Tensor,   # (B, G)
        planet_mask: Optional[torch.Tensor] = None,  # (B, P)
    ) -> dict[str, torch.Tensor]:
        mask = None if planet_mask is None else planet_mask.to(torch.bool)
        return self.encoder(planet_feats, edge_feats, global_feats, mask)

    # -- value ------------------------------------------------------------- #
    def value_from_encoding(self, enc: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.value_head(enc["graph_summary"]).squeeze(-1)

    def forward_value(
        self,
        planet_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        global_feats: torch.Tensor,
        planet_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                # (B,)
        return self.value_from_encoding(
            self.encode(planet_feats, edge_feats, global_feats, planet_mask)
        )

    # -- prior ------------------------------------------------------------- #
    def gather_target_tokens(
        self,
        node_tokens: torch.Tensor,    # (B, P, d)
        target_idx: torch.Tensor,     # (B, A) long; -1 -> pass_token
    ) -> torch.Tensor:                # (B, A, d)
        b, _, d = node_tokens.shape
        a = target_idx.shape[1]
        safe = target_idx.clamp_min(0)
        gathered = torch.gather(node_tokens, 1, safe.unsqueeze(-1).expand(b, a, d))
        is_pass = (target_idx < 0).unsqueeze(-1)
        pass_tok = self.pass_token.view(1, 1, d).expand(b, a, d)
        return torch.where(is_pass, pass_tok, gathered)

    def prior_from_encoding(
        self,
        enc: Mapping[str, torch.Tensor],
        action_feats: torch.Tensor,   # (B, A, Fa)
        target_idx: torch.Tensor,     # (B, A) long
        action_mask: Optional[torch.Tensor] = None,  # (B, A)
    ) -> torch.Tensor:                # (B, A)
        node_tokens = enc["node_tokens"]
        summary = enc["graph_summary"]
        b, a, _ = action_feats.shape
        tgt_tokens = self.gather_target_tokens(node_tokens, target_idx)
        summary_exp = summary.unsqueeze(1).expand(b, a, summary.shape[-1])
        ctx = torch.cat([action_feats, tgt_tokens, summary_exp], dim=-1)
        logits = self.prior_head(ctx).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits

    def forward_prior(
        self,
        planet_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        global_feats: torch.Tensor,
        planet_mask: torch.Tensor,
        action_feats: torch.Tensor,
        target_idx: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                # (B, A)
        enc = self.encode(planet_feats, edge_feats, global_feats, planet_mask)
        return self.prior_from_encoding(enc, action_feats, target_idx, action_mask)


class MacroAlphaZeroNetMLP(nn.Module):
    """Legacy compact MLP net (kept so pre-GNN checkpoints still load / run).

    Identical to the original architecture: a per-planet MLP with mean/max
    pooling for the value head and an action-feature-only prior head.  The GNN
    (:class:`MacroAlphaZeroNet`) is the default for new training; this class is
    retained for backward compatibility with old ``best.pt`` files.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.arch = "mlp"
        self.d_model = int(d_model)

        self.planet_mlp = nn.Sequential(
            nn.Linear(PLANET_FEATURE_DIM, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * d_model + GLOBAL_FEATURE_DIM, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.prior_head = nn.Sequential(
            nn.Linear(ACTION_FEATURE_DIM, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward_value(
        self,
        planet_feats: torch.Tensor,   # (B, P, Fp)
        global_feats: torch.Tensor,   # (B, G)
        planet_mask: torch.Tensor,    # (B, P) bool / float
    ) -> torch.Tensor:                # (B,)
        mask = planet_mask.to(planet_feats.dtype).unsqueeze(-1)
        h = self.planet_mlp(planet_feats) * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean_pool = h.sum(dim=1) / denom
        neg_inf = torch.finfo(h.dtype).min
        masked = h.masked_fill(mask <= 0.0, neg_inf)
        max_pool = masked.max(dim=1).values
        max_pool = torch.nan_to_num(max_pool, neginf=0.0)
        summary = torch.cat([mean_pool, max_pool, global_feats], dim=-1)
        return self.value_head(summary).squeeze(-1)

    def forward_prior(
        self,
        action_feats: torch.Tensor,   # (B, A, Fa)
        action_mask: Optional[torch.Tensor] = None,  # (B, A)
    ) -> torch.Tensor:                # (B, A)
        logits = self.prior_head(action_feats).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits


def build_net(arch: str = "gnn", **kwargs) -> nn.Module:
    """Factory: ``arch="gnn"`` -> :class:`MacroAlphaZeroNet`, ``"mlp"`` -> legacy."""
    arch = str(arch).lower()
    if arch == "mlp":
        return MacroAlphaZeroNetMLP(d_model=int(kwargs.get("d_model", 64)))
    if arch == "gnn":
        return MacroAlphaZeroNet(
            d_model=int(kwargs.get("d_model", 64)),
            n_layers=int(kwargs.get("n_layers", 2)),
            n_heads=int(kwargs.get("n_heads", 4)),
            dropout=float(kwargs.get("dropout", 0.0)),
        )
    raise ValueError(f"unknown arch: {arch!r} (expected 'gnn' or 'mlp')")


# --------------------------------------------------------------------------- #
# Providers binding a net to a specific search (root_player / board geometry).
# --------------------------------------------------------------------------- #
def _make_mlp_providers(
    net: "MacroAlphaZeroNetMLP",
    mcts: MacroTimedMCTS,
    net_device: torch.device,
) -> tuple[MacroPriorProvider, LeafValueFn]:
    """Legacy MLP providers (action-feature prior + pooled value)."""

    def prior_provider(request: MacroPriorRequest) -> Mapping[tuple[str, int], float]:
        keys = [a.key() for a in request.actions]
        if not keys:
            return {}
        feats = np.stack([
            action_feature_vector(request.features_by_key[k]) for k in keys
        ])[None]
        with torch.no_grad():
            t = torch.from_numpy(feats).to(net_device)
            logits = net.forward_prior(t)[0].cpu().numpy()
        return {k: float(logits[i]) for i, k in enumerate(keys)}

    def leaf_value_fn(state: _TimedState, root_player: int) -> float:
        planet_feats, glob = abstract_state_arrays(mcts, state)
        with torch.no_grad():
            pf = torch.from_numpy(planet_feats[None]).to(net_device)
            gf = torch.from_numpy(glob[None]).to(net_device)
            mask = torch.ones((1, pf.shape[1]), dtype=torch.bool, device=net_device)
            v = net.forward_value(pf, gf, mask)[0].item()
        return float(v)

    return prior_provider, leaf_value_fn


def make_macro_providers(
    net: MacroAlphaZeroNet,
    mcts: MacroTimedMCTS,
    *,
    device: Optional[torch.device] = None,
) -> tuple[MacroPriorProvider, LeafValueFn]:
    """Build ``(prior_provider, leaf_value_fn)`` closures bound to ``mcts``."""
    device = device or torch.device("cpu")
    net_device = next(net.parameters()).device

    if getattr(net, "arch", "gnn") == "mlp":
        return _make_mlp_providers(net, mcts, net_device)

    # Route geometry is static across a search, so cache the edge tensor and the
    # planet-id -> row index map once (owner-map keys are constant per search).
    base_ids = abstract_state_ids(mcts._initial_state)
    base_edge = torch.from_numpy(abstract_edge_arrays(mcts, base_ids)[None]).to(net_device)
    base_idx = {pid: i for i, pid in enumerate(base_ids)}

    def _encode(state: _TimedState):
        planet_feats, glob = abstract_state_arrays(mcts, state)
        ids = abstract_state_ids(state)
        pf = torch.from_numpy(planet_feats[None]).to(net_device)
        gf = torch.from_numpy(glob[None]).to(net_device)
        if ids == base_ids:
            ef = base_edge
            idx_map = base_idx
        else:  # defensive: rebuild if the planet set ever changes
            ef = torch.from_numpy(abstract_edge_arrays(mcts, ids)[None]).to(net_device)
            idx_map = {pid: i for i, pid in enumerate(ids)}
        mask = torch.ones((1, pf.shape[1]), dtype=torch.bool, device=net_device)
        return net.encode(pf, ef, gf, mask), idx_map

    def prior_provider(request: MacroPriorRequest) -> Mapping[tuple[str, int], float]:
        keys = [a.key() for a in request.actions]
        if not keys:
            return {}
        state = request.state
        if state is None:  # no graph context available -> flat priors
            return {k: 0.0 for k in keys}
        feats = np.stack([
            action_feature_vector(request.features_by_key[k]) for k in keys
        ])[None]  # (1, A, Fa)
        with torch.no_grad():
            enc, idx_map = _encode(state)
            target_idx = np.array(
                [[idx_map.get(int(a.target), -1) for a in request.actions]],
                dtype=np.int64,
            )
            af = torch.from_numpy(feats).to(net_device)
            ti = torch.from_numpy(target_idx).to(net_device)
            logits = net.prior_from_encoding(enc, af, ti)[0].cpu().numpy()
        return {k: float(logits[i]) for i, k in enumerate(keys)}

    def leaf_value_fn(state: _TimedState, root_player: int) -> float:
        with torch.no_grad():
            enc, _ = _encode(state)
            v = net.value_from_encoding(enc)[0].item()
        return float(v)

    return prior_provider, leaf_value_fn


def attach_net(
    net: MacroAlphaZeroNet,
    mcts: MacroTimedMCTS,
    *,
    device: Optional[torch.device] = None,
) -> MacroTimedMCTS:
    """Attach a net's prior + value providers onto an existing search."""
    prior, value = make_macro_providers(net, mcts, device=device)
    mcts.prior_provider = prior
    mcts.leaf_value_fn = value
    return mcts


# --------------------------------------------------------------------------- #
# Agent: macro MCTS steered by a trained net (real-env play / evaluation).
# --------------------------------------------------------------------------- #
class MacroAlphaZeroAgent(OrbitMacroMCTSAgent):
    """Scripted agent whose macro search uses a trained AlphaZero net."""

    def __init__(
        self,
        net: MacroAlphaZeroNet,
        cfg: OrbitScriptedConfig | None = None,
        macro_cfg: MacroMCTSConfig | None = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(cfg=cfg, macro_cfg=macro_cfg)
        self.net = net.eval()
        self.device = device or torch.device("cpu")

    def _build_macro_mcts(self, og: OrbitGraph, *, current_turn: int, max_turns: int) -> MacroTimedMCTS:
        mcts = MacroTimedMCTS(
            og,
            self.macro_cfg,
            root_player=int(og.player),
            current_turn=int(current_turn),
            max_turns=int(max_turns),
        )
        attach_net(self.net, mcts, device=self.device)
        return mcts


def save_net(net: nn.Module, path: str, *, meta: Optional[dict] = None) -> None:
    payload = {
        "state_dict": net.state_dict(),
        "arch": getattr(net, "arch", "gnn"),
        "d_model": net.d_model,
        "meta": meta or {},
    }
    if getattr(net, "arch", "gnn") == "gnn":
        payload["n_layers"] = net.n_layers
        payload["n_heads"] = net.n_heads
    torch.save(payload, path)


def _detect_arch(ckpt: dict) -> str:
    """Resolve the checkpoint architecture (explicit field, else from keys)."""
    arch = ckpt.get("arch")
    if arch:
        return str(arch).lower()
    sd = ckpt.get("state_dict", {})
    if any(k.startswith("encoder.") for k in sd):
        return "gnn"
    if any(k.startswith("planet_mlp.") for k in sd):
        return "mlp"
    return "gnn"


def load_net(path: str, *, device: Optional[torch.device] = None) -> nn.Module:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = _detect_arch(ckpt)
    net = build_net(
        arch,
        d_model=int(ckpt.get("d_model", 64)),
        n_layers=int(ckpt.get("n_layers", 2)),
        n_heads=int(ckpt.get("n_heads", 4)),
    )
    _load_state_dict_compatible(net, ckpt["state_dict"])
    net.to(device)
    return net.eval()


def _load_state_dict_compatible(net: nn.Module, state_dict: dict) -> list[str]:
    """Load weights, skipping tensors whose shape changed (re-init those layers).

    Lets a checkpoint warm-start a net even after the action-feature schema
    grows (``ACTION_FEATURE_DIM`` change re-shapes only the prior head's first
    layer): the encoder + value head load as usual and just the mismatched
    prior-head tensors are left at their fresh initialisation.
    """
    own = net.state_dict()
    filtered: dict = {}
    skipped: list[str] = []
    for k, v in state_dict.items():
        if k in own and tuple(own[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            skipped.append(k)
    net.load_state_dict(filtered, strict=False)
    if skipped:
        import warnings
        warnings.warn(
            "load_net: re-initialised "
            f"{len(skipped)} tensor(s) with changed/absent shape (warm-start): "
            + ", ".join(sorted(skipped)[:6])
            + (" ..." if len(skipped) > 6 else ""),
            stacklevel=2,
        )
    return skipped


__all__ = [
    "PLANET_FEATURE_NAMES",
    "GLOBAL_FEATURE_NAMES",
    "EDGE_FEATURE_NAMES",
    "PLANET_FEATURE_DIM",
    "GLOBAL_FEATURE_DIM",
    "EDGE_FEATURE_DIM",
    "ACTION_FEATURE_DIM",
    "abstract_state_arrays",
    "abstract_state_ids",
    "abstract_edge_arrays",
    "action_feature_vector",
    "MacroAlphaZeroNet",
    "MacroAlphaZeroNetMLP",
    "build_net",
    "make_macro_providers",
    "attach_net",
    "MacroAlphaZeroAgent",
    "save_net",
    "load_net",
]
