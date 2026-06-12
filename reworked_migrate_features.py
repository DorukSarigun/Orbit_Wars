"""Migrate a pre-v12 reworked checkpoint to the v12 feature layout.

v12 grew the inputs with channels appended strictly at the end:

* node: 16 (or 17 with the eta-value channel) -> 23 (24)
  ``[+friendly_inbound, +enemy_inbound, +enemy_min_eta, +defense_deficit,
  +recapture_count, +comet_remaining_life, +frontier_distance]``
* edge: 4 -> 9
  ``[+eta, +dphi_sin, +dphi_cos, +ships_at_arrival, +race_margin]``

Only the two input stems touch the feature dim, so migration = expanding

* ``encoder.node_embed.0`` / ``encoder.edge_embed.0`` (LayerNorm): new
  channels get ``weight=1, bias=0``;
* ``encoder.node_embed.1`` / ``encoder.edge_embed.1`` (Linear): new input
  columns get **zero** weights, so at load time the new channels contribute
  nothing and behaviour matches the source checkpoint (up to the input
  LayerNorm statistics, which now normalise over the wider channel set —
  the warm start is therefore *approximate*, not bit-exact).

The Adam state is dropped (its moment tensors keep the old shapes).

Usage::

    python -m reworked_migrate_features \
        --in checkpoints_remastered_selfplay/latest.pt \
        --out checkpoints_remastered_v12/migrated.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .reworked_model import REWORKED_NODE_FEATURE_DIM
    from .reworked_features import REWORKED_EDGE_FEATURE_DIM
except ImportError:  # pragma: no cover - flat layout
    from reworked_model import REWORKED_NODE_FEATURE_DIM  # type: ignore
    from reworked_features import REWORKED_EDGE_FEATURE_DIM  # type: ignore


def _expand_layernorm(sd: dict, prefix: str, new_dim: int) -> int:
    """Grow ``{prefix}.weight/.bias`` (LayerNorm) to ``new_dim``; returns old."""
    w, b = sd[f"{prefix}.weight"], sd[f"{prefix}.bias"]
    old = int(w.shape[0])
    if old >= new_dim:
        return old
    pad = new_dim - old
    sd[f"{prefix}.weight"] = torch.cat([w, torch.ones(pad, dtype=w.dtype)])
    sd[f"{prefix}.bias"] = torch.cat([b, torch.zeros(pad, dtype=b.dtype)])
    return old


def _expand_linear_in(sd: dict, prefix: str, new_in: int) -> int:
    """Grow ``{prefix}.weight`` (Linear) input columns to ``new_in`` with zeros."""
    w = sd[f"{prefix}.weight"]
    old = int(w.shape[1])
    if old >= new_in:
        return old
    pad = torch.zeros(w.shape[0], new_in - old, dtype=w.dtype)
    sd[f"{prefix}.weight"] = torch.cat([w, pad], dim=1)
    return old


def migrate_checkpoint(in_path: str | Path, out_path: str | Path) -> dict:
    """Expand input stems to the v12 dims; returns the migrated payload."""
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    arch = dict(ckpt.get("arch", {}))
    sd = ckpt["policy_state_dict"]

    old_node = int(arch.get("node_dim", 16))
    old_edge = int(arch.get("edge_dim", 4))
    has_vrate = bool(arch.get("eta_value_feature", False)) or old_node in (17,)
    new_node = REWORKED_NODE_FEATURE_DIM + (1 if has_vrate else 0)
    new_edge = REWORKED_EDGE_FEATURE_DIM

    if old_node >= new_node and old_edge >= new_edge:
        raise SystemExit(
            f"Checkpoint already at node_dim={old_node}, edge_dim={old_edge} "
            f"-- nothing to migrate.")

    _expand_layernorm(sd, "encoder.node_embed.0", new_node)
    _expand_linear_in(sd, "encoder.node_embed.1", new_node)
    _expand_layernorm(sd, "encoder.edge_embed.0", new_edge)
    _expand_linear_in(sd, "encoder.edge_embed.1", new_edge)
    # Final stem LayerNorm (index 3) is over d_model -- untouched.

    arch["node_dim"] = new_node
    arch["edge_dim"] = new_edge
    ckpt["arch"] = arch
    # Adam moments keep the old parameter shapes; a fresh optimizer is safer.
    ckpt.pop("optimizer_state_dict", None)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    print(f"Migrated node {old_node}->{new_node}, edge {old_edge}->{new_edge}: "
          f"{in_path} -> {out_path}")
    return ckpt


def _smoke_test(out_path: str | Path) -> None:
    """Rebuild the migrated policy and run one forward pass."""
    try:
        from .reworked_train import load_checkpoint
    except ImportError:  # pragma: no cover
        from reworked_train import load_checkpoint  # type: ignore
    policy, ckpt = load_checkpoint(out_path)
    arch = ckpt["arch"]
    B, P = 2, 8
    out = policy.forward(
        torch.randn(B, P, arch["node_dim"]),
        torch.randn(B, P, P, arch["edge_dim"]),
        torch.randn(B, arch["global_dim"]),
        torch.ones(B, P, dtype=torch.bool),
    )
    keys = sorted(out.keys())
    print(f"Smoke test OK ({type(policy).__name__}, forward keys: {keys})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Migrate a checkpoint to v12 features")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--no-smoke", action="store_true",
                   help="skip the post-migration forward smoke test")
    args = p.parse_args(argv)
    migrate_checkpoint(args.in_path, args.out_path)
    if not args.no_smoke:
        _smoke_test(args.out_path)


if __name__ == "__main__":
    main()
