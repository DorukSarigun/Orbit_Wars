"""Online self-play PPO training stack for OrbitWarsPolicy.

Modules
-------
- rewards : per-step reward shaping + terminal win/loss
- rollout : self-play episode collector (runs both sides through one policy)
- buffer  : on-policy rollout buffer with GAE advantages
- ppo     : clipped surrogate update step
- train   : main training loop entry point
- watch   : load a checkpoint and play/render one self-play game
"""

from .rewards import REWARD_SYSTEMS, RewardConfig, build_reward_config, compute_step_reward
from .buffer import RolloutBuffer
from .rollout import collect_episode, EpisodeStats
from .ppo import ppo_update, PPOConfig
from .replay import ReplayBuffer, DQNTransition
from .dqn import DQNConfig, dqn_update, soft_update_target, hard_update_target
