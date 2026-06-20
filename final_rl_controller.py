"""Controller adapter for trained final neural checkpoints."""

from __future__ import annotations

from typing import Any

import torch

from final_obs import build_final_observation
from final_rollout import _action_to_launch, _obs_to_tensors
from final_train_rl import load_checkpoint


class FinalRLController:
    def __init__(
        self,
        ckpt_path: str,
        *,
        device: str = "cpu",
        greedy: bool = True,
        include_owned_targets: bool | None = None,
    ):
        self.device = torch.device(device)
        self.policy, self.ckpt = load_checkpoint(ckpt_path, self.device)
        self.policy.eval()
        self.greedy = bool(greedy)
        # Default to whatever the checkpoint was trained with (BC stores it in
        # args); fall back to the legacy non-owned-target behaviour.
        if include_owned_targets is None:
            ckpt_args = self.ckpt.get("args", {}) if isinstance(self.ckpt, dict) else {}
            include_owned_targets = bool(ckpt_args.get("include_owned_targets", False))
        self.include_owned_targets = bool(include_owned_targets)

    def act(self, env_obs: Any, player: int, *, current_turn: int, max_turns: int) -> list[list]:
        obs = build_final_observation(
            env_obs, int(player), include_owned_targets=self.include_owned_targets,
        )
        tensors = _obs_to_tensors(obs, self.device)
        with torch.no_grad():
            action = self.policy.act(**tensors, greedy=self.greedy)
        launches, _reason = _action_to_launch(
            env_obs,
            obs,
            action,
            int(player),
            current_turn=int(current_turn),
            max_turns=int(max_turns),
        )
        return launches

