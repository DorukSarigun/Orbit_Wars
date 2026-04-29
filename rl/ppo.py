"""PPO update step (clipped surrogate objective).

Standard implementation: K epochs over shuffled minibatches, with
clipped policy ratio, clipped value loss option, and an entropy bonus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import OrbitWarsPolicy
from .buffer import RolloutBuffer
from .metrics import grad_norms_by_module


@dataclass
class PPOConfig:
    clip_range: float = 0.2
    value_clip_range: Optional[float] = 0.2  # None disables value clipping
    epochs: int = 4
    batch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    n_updates: int
    # Per-top-module L2 grad norm captured on the LAST minibatch of the
    # last epoch (representative snapshot). Empty dict if no updates ran.
    last_grad_norms: dict = None  # type: ignore[assignment]


def ppo_update(
    policy: OrbitWarsPolicy,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    cfg: Optional[PPOConfig] = None,
    *,
    device: Optional[torch.device] = None,
) -> PPOStats:
    """Run a full PPO update over the rollout buffer."""
    cfg = cfg or PPOConfig()
    device = device or next(policy.parameters()).device

    pol_losses, val_losses, ents, kls, clips = [], [], [], [], []
    n_updates = 0
    last_grad_norms: dict = {}

    for _ in range(cfg.epochs):
        for batch in buffer.iter_minibatches(cfg.batch_size, device=device):
            out = policy(
                planet_feats=batch["planet_feats"],
                fleet_feats=batch["fleet_feats"],
                global_feats=batch["global_feats"],
                edge_bias=batch["edge_bias"],
                planet_mask=batch["planet_mask"],
                fleet_mask=batch["fleet_mask"],
                source_mask=batch["source_mask"],
            )
            ev = policy.evaluate_source_actions(
                out,
                src=batch["src"],
                tgt=batch["tgt"],
                ship_bin=batch["ship_bin"],
                is_noop=batch["is_noop"],
            )
            new_logp = ev["log_prob"]
            entropy = ev["entropy"].mean()
            new_value = ev["value"]

            adv = batch["advantages"]
            old_logp = batch["old_log_prob"]
            ratio = torch.exp(new_logp - old_logp)

            # Clipped surrogate objective.
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()

            # Value loss (optionally clipped).
            returns = batch["returns"]
            old_value = batch["old_value"]
            if cfg.value_clip_range is not None:
                v_clipped = old_value + torch.clamp(
                    new_value - old_value,
                    -cfg.value_clip_range, cfg.value_clip_range,
                )
                v_loss_unclipped = (new_value - returns).pow(2)
                v_loss_clipped = (v_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * F.mse_loss(new_value, returns)

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Capture per-module grad norm BEFORE clipping (so we see the
            # raw signal, not the clipped one).
            last_grad_norms = grad_norms_by_module(policy)
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = 0.5 * (old_logp - new_logp).pow(2).mean().item()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()

            pol_losses.append(policy_loss.item())
            val_losses.append(value_loss.item())
            ents.append(entropy.item())
            kls.append(approx_kl)
            clips.append(clip_frac)
            n_updates += 1

    def _mean(xs):
        return float(sum(xs) / max(1, len(xs)))

    return PPOStats(
        policy_loss=_mean(pol_losses),
        value_loss=_mean(val_losses),
        entropy=_mean(ents),
        approx_kl=_mean(kls),
        clip_fraction=_mean(clips),
        n_updates=n_updates,
        last_grad_norms=last_grad_norms,
    )
