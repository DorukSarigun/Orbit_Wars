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

Design note
-----------
The recommendation was to reuse the heavy ``ReworkedPlanetGraphEncoder``.  In
practice that encoder needs the full 16-D reworked observation features, which
internal abstract leaves do not have.  Forcing it would create a root/leaf
representation gap.  Instead this is a compact net over abstract-state features
(same permutation-invariant pooling spirit), trainable end-to-end and queryable
at any node.  The base search module stays Torch-free for Kaggle packaging.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from orbit_graph import OrbitGraph
from orbit_macro_mcts_agent import (
    MACRO_FEATURE_NAMES,
    LeafValueFn,
    MacroMCTSConfig,
    MacroPriorProvider,
    MacroPriorRequest,
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

PLANET_FEATURE_DIM = len(PLANET_FEATURE_NAMES)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURE_NAMES)
ACTION_FEATURE_DIM = len(MACRO_FEATURE_NAMES)


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
    """Compact ``f(s) -> (prior logits, value)`` over abstract macro features."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = int(d_model)

        self.planet_mlp = nn.Sequential(
            nn.Linear(PLANET_FEATURE_DIM, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.GELU(),
        )
        # mean-pool + max-pool over planets, concatenated with the global vector.
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
        mask = planet_mask.to(planet_feats.dtype).unsqueeze(-1)  # (B, P, 1)
        h = self.planet_mlp(planet_feats) * mask                 # (B, P, d)
        denom = mask.sum(dim=1).clamp_min(1.0)                    # (B, 1)
        mean_pool = h.sum(dim=1) / denom                         # (B, d)
        neg_inf = torch.finfo(h.dtype).min
        masked = h.masked_fill(mask <= 0.0, neg_inf)
        max_pool = masked.max(dim=1).values                      # (B, d)
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


# --------------------------------------------------------------------------- #
# Providers binding a net to a specific search (root_player / board geometry).
# --------------------------------------------------------------------------- #
def make_macro_providers(
    net: MacroAlphaZeroNet,
    mcts: MacroTimedMCTS,
    *,
    device: Optional[torch.device] = None,
) -> tuple[MacroPriorProvider, LeafValueFn]:
    """Build ``(prior_provider, leaf_value_fn)`` closures bound to ``mcts``."""
    device = device or torch.device("cpu")
    net_device = next(net.parameters()).device

    def prior_provider(request: MacroPriorRequest) -> Mapping[tuple[str, int], float]:
        keys = [a.key() for a in request.actions]
        if not keys:
            return {}
        feats = np.stack([
            action_feature_vector(request.features_by_key[k]) for k in keys
        ])[None]  # (1, A, Fa)
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

    def _attack_priority_bonus(
        self,
        og: OrbitGraph,
        plan,
        defense_states,
        *,
        omega,
        current_turn,
        max_turns,
    ):
        del plan, defense_states, omega
        if int(self.macro_cfg.timed.simulations) <= 0 or float(self.macro_cfg.bonus_weight) == 0.0:
            return None
        mcts = MacroTimedMCTS(
            og,
            self.macro_cfg,
            root_player=int(og.player),
            current_turn=int(current_turn),
            max_turns=int(max_turns),
        )
        attach_net(self.net, mcts, device=self.device)
        bonus = mcts.search()
        return bonus or None


def save_net(net: MacroAlphaZeroNet, path: str, *, meta: Optional[dict] = None) -> None:
    torch.save(
        {"state_dict": net.state_dict(), "d_model": net.d_model, "meta": meta or {}},
        path,
    )


def load_net(path: str, *, device: Optional[torch.device] = None) -> MacroAlphaZeroNet:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = MacroAlphaZeroNet(d_model=int(ckpt.get("d_model", 64)))
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    return net.eval()


__all__ = [
    "PLANET_FEATURE_NAMES",
    "GLOBAL_FEATURE_NAMES",
    "PLANET_FEATURE_DIM",
    "GLOBAL_FEATURE_DIM",
    "ACTION_FEATURE_DIM",
    "abstract_state_arrays",
    "action_feature_vector",
    "MacroAlphaZeroNet",
    "make_macro_providers",
    "attach_net",
    "MacroAlphaZeroAgent",
    "save_net",
    "load_net",
]
