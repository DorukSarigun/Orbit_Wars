"""DQN / Double DQN update step for the branching Q-network.

Reference: Mnih et al. 2015 (DQN), van Hasselt et al. 2016 (Double DQN),
Wang et al. 2016 (Dueling), Tavakoli et al. 2018 (Branching DQN).

Loss: factored MSE on the joint Q given the additive decomposition

    Q_total(s, src, tgt, ship) =
        Q_src(s, src) + 1{src!=NOOP} * (Q_tgt(s, src, tgt) + Q_ship(s, src, ship))

Target: y = r + gamma * (1 - done) * max_a' Q_target_total(s', a')

Double DQN:
    a* = argmax_a Q_online_total(s', a)
    y  = r + gamma * (1 - done) * Q_target_total(s', a*)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_q import OrbitWarsQNet


@dataclass
class DQNConfig:
    gamma: float = 0.99
    double_dqn: bool = True
    huber_delta: float = 1.0       # use SmoothL1 by default; 0.0 -> MSE
    max_grad_norm: float = 10.0


def _compute_target_max_q(
    online_net: OrbitWarsQNet,
    target_net: OrbitWarsQNet,
    next_batch: dict,
    *,
    cfg: DQNConfig,
) -> torch.Tensor:
    """Return max_a' Q_target(s', a') (or Double DQN variant)."""
    src_mask_next = next_batch["next_source_mask"]
    plan_mask_next = next_batch["next_planet_mask"]

    with torch.no_grad():
        target_out = target_net(
            planet_feats=next_batch["next_planet_feats"],
            fleet_feats=next_batch["next_fleet_feats"],
            global_feats=next_batch["next_global_feats"],
            edge_bias=next_batch["next_edge_bias"],
            planet_mask=plan_mask_next,
            fleet_mask=next_batch["next_fleet_mask"],
            source_mask=src_mask_next,
        )

        if cfg.double_dqn:
            online_out = online_net(
                planet_feats=next_batch["next_planet_feats"],
                fleet_feats=next_batch["next_fleet_feats"],
                global_feats=next_batch["next_global_feats"],
                edge_bias=next_batch["next_edge_bias"],
                planet_mask=plan_mask_next,
                fleet_mask=next_batch["next_fleet_mask"],
                source_mask=src_mask_next,
            )
            # Argmax with the online net (per-branch).
            actions = OrbitWarsQNet.select_actions(
                online_out, src_mask_next, plan_mask_next, epsilon=0.0,
            )
            return target_net.joint_q(
                target_out, actions["src"], actions["tgt"], actions["ship_bin"],
            )
        else:
            return target_net.max_joint_q(target_out, src_mask_next, plan_mask_next)


def dqn_update(
    online_net: OrbitWarsQNet,
    target_net: OrbitWarsQNet,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    *,
    cfg: Optional[DQNConfig] = None,
) -> dict:
    """One DQN gradient step on a sampled minibatch."""
    cfg = cfg or DQNConfig()

    online_out = online_net(
        planet_feats=batch["planet_feats"],
        fleet_feats=batch["fleet_feats"],
        global_feats=batch["global_feats"],
        edge_bias=batch["edge_bias"],
        planet_mask=batch["planet_mask"],
        fleet_mask=batch["fleet_mask"],
        source_mask=batch["source_mask"],
    )
    q_taken = online_net.joint_q(
        online_out, batch["src"], batch["tgt"], batch["ship_bin"],
    )

    max_q_next = _compute_target_max_q(online_net, target_net, batch, cfg=cfg)
    target = batch["reward"] + cfg.gamma * (1.0 - batch["done"]) * max_q_next

    if cfg.huber_delta > 0:
        loss = F.smooth_l1_loss(q_taken, target.detach(), beta=cfg.huber_delta)
    else:
        loss = F.mse_loss(q_taken, target.detach())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(online_net.parameters(), cfg.max_grad_norm)
    optimizer.step()

    with torch.no_grad():
        td_err = (q_taken - target).abs().mean().item()
        q_mean = q_taken.mean().item()
    return {
        "dqn_loss": float(loss.item()),
        "td_error": float(td_err),
        "q_mean": float(q_mean),
        "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
    }


def soft_update_target(target_net: nn.Module, online_net: nn.Module, *, tau: float) -> None:
    """Polyak averaging: theta_target <- tau * theta_online + (1-tau) * theta_target."""
    with torch.no_grad():
        for p_t, p_o in zip(target_net.parameters(), online_net.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p_o.data, alpha=tau)


def hard_update_target(target_net: nn.Module, online_net: nn.Module) -> None:
    target_net.load_state_dict(online_net.state_dict())
