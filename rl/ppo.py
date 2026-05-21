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

from model import OrbitWarsPolicy, _beta_concentrations
from .buffer import RolloutBuffer
from .metrics import grad_norms_by_module


@dataclass
class PPOConfig:
    clip_range: float = 0.2
    value_clip_range: Optional[float] = 0.2  # None disables value clipping
    epochs: int = 4
    batch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    max_grad_norm: float = 0.5
    # KL-based early stopping. If the average approx_kl on a minibatch
    # exceeds `target_kl`, stop the remaining epochs for the current
    # rollout. Set to None to disable. 0.05 is a conservative default
    # well below where the trust region breaks (KL >> 1 produces
    # clip_fraction near 1.0 and gradient explosion in observed runs).
    target_kl: Optional[float] = 0.05


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
    # Importance ratio quantiles (5th, 50th, 95th percentile) averaged
    # across all minibatches. Useful for diagnosing trust-region health:
    # a wide spread (e.g. q05 < 0.1 or q95 > 10) indicates the policy is
    # changing too aggressively per update.
    ratio_q05: float = 0.0
    ratio_q50: float = 0.0
    ratio_q95: float = 0.0
    # Mean Beta concentrations on the chosen (src, tgt) pair, averaged
    # across non-noop samples. Tracking these is the cheapest way to
    # detect the ship-head collapsing into a delta function.
    ship_alpha_mean: float = 0.0
    ship_beta_mean: float = 0.0
    # Number of full epochs actually run before KL early stopping (if any).
    epochs_run: int = 0


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
    ratio_q05s, ratio_q50s, ratio_q95s = [], [], []
    alpha_means, beta_means = [], []
    n_updates = 0
    epochs_run = 0
    last_grad_norms: dict = {}

    for _epoch in range(cfg.epochs):
        epoch_kls: list[float] = []
        stop_ppo = False
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
                ship_value=batch["ship_value"],
                is_noop=batch["is_noop"],
            )
            new_logp = ev["log_prob"]
            entropy = ev["entropy"].mean()
            new_value = ev["value"]

            adv = batch["advantages"]
            old_logp = batch["old_log_prob"]
            log_ratio = new_logp - old_logp
            if not (
                torch.isfinite(log_ratio).all()
                and torch.isfinite(new_value).all()
                and torch.isfinite(entropy).all()
            ):
                stop_ppo = True
                break
            with torch.no_grad():
                approx_kl = 0.5 * log_ratio.pow(2).mean().item()
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                kls.append(approx_kl)
                epoch_kls.append(approx_kl)
                stop_ppo = True
                break
            ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))

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
                clip_frac = ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()
                # Importance ratio quantiles — diagnose trust-region health.
                if ratio.numel() >= 20:
                    qs = torch.quantile(
                        ratio.detach(),
                        torch.tensor([0.05, 0.5, 0.95], device=ratio.device),
                    )
                    ratio_q05s.append(qs[0].item())
                    ratio_q50s.append(qs[1].item())
                    ratio_q95s.append(qs[2].item())
                # Beta head mean concentrations on the chosen pair (non-noop only).
                B = out.ship_params.shape[0]
                src_idx = batch["src"].long().clamp(max=out.ship_params.shape[1] - 1)
                if out.ship_params.dim() == 4:
                    tgt_idx = batch["tgt"].long().clamp(max=out.ship_params.shape[2] - 1)
                    chosen_params = out.ship_params[
                        torch.arange(B, device=out.ship_params.device),
                        src_idx,
                        tgt_idx,
                    ]
                else:
                    chosen_params = out.ship_params[
                        torch.arange(B, device=out.ship_params.device),
                        src_idx,
                    ]
                a_chosen, b_chosen = _beta_concentrations(chosen_params)
                non_noop_mask = (~batch["is_noop"].bool())
                if non_noop_mask.any():
                    alpha_means.append(a_chosen[non_noop_mask].mean().item())
                    beta_means.append(b_chosen[non_noop_mask].mean().item())

            pol_losses.append(policy_loss.item())
            val_losses.append(value_loss.item())
            ents.append(entropy.item())
            kls.append(approx_kl)
            clips.append(clip_frac)
            epoch_kls.append(approx_kl)
            n_updates += 1

        epochs_run += 1
        if stop_ppo:
            break

        # KL early stopping: if the average approx_kl on this epoch's
        # minibatches blew past `target_kl`, the policy has already
        # drifted too far from the rollout snapshot. Stop now so the
        # next iteration collects fresh on-policy data.
        if cfg.target_kl is not None and epoch_kls:
            mean_epoch_kl = sum(epoch_kls) / len(epoch_kls)
            if mean_epoch_kl > cfg.target_kl:
                break

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
        ratio_q05=_mean(ratio_q05s),
        ratio_q50=_mean(ratio_q50s),
        ratio_q95=_mean(ratio_q95s),
        ship_alpha_mean=_mean(alpha_means),
        ship_beta_mean=_mean(beta_means),
        epochs_run=epochs_run,
    )
