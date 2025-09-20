import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
from ..envs import ManagerBasedSafeRLEnv

class RslSafeRlVecEnvWrapper(RslRlVecEnvWrapper):
    
    def __init__(self, env: ManagerBasedSafeRLEnv, clip_actions: float | None = None):
        super().__init__(env, clip_actions)
    
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs_dict, reward, cost, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, reward, cost, dones, extras