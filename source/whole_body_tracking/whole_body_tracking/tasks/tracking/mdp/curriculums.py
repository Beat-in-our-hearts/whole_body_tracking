from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def adaptive_sampling_ratio(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "motion_global_anchor_ori",
    max_ratio: float = 0.8,
    threshold: float = 0.9,
    episode_num: int = 1,
    operation: Literal["add", "scale"] = "scale",
    scale_ratio: float = 2.0,
    delta_ratio: float = 5e-2,
) -> torch.Tensor:
    # use episode alive length to adjust the sampling ratio between uniform and adaptive sampling
    command_term = env.command_manager.get_term("motion")
    
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % (env.max_episode_length * episode_num) == 0:
        if reward > reward_term.weight * threshold:
            if operation == "add":
                command_term.cfg.adaptive_uniform_ratio = min(
                    max_ratio, command_term.cfg.adaptive_uniform_ratio + delta_ratio
                )
            elif operation == "scale":
                command_term.cfg.adaptive_uniform_ratio = min(
                    max_ratio, command_term.cfg.adaptive_uniform_ratio * scale_ratio
                )
    
    return torch.tensor(command_term.cfg.adaptive_uniform_ratio, device=env.device)
    