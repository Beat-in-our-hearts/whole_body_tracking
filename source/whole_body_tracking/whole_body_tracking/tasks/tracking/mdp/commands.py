from __future__ import annotations

import math
import os
from os.path import dirname, join

import numpy as np
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Union

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from whole_body_tracking.utils.motion_dataset import Motion_Dataset
from whole_body_tracking.utils.motion_dataloader import Motion_Dataloader

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


    def motion_robot_joint_pos_vel(self, interval: int, frames: int):
        """
        get `frames=M` future frame state, every frame has `interval=T`
        """
        # [N, 1] -> [N, M] broadcasting
        offsets = interval * torch.arange(frames, dtype=self.time_steps.dtype, device=self.time_steps.device)
        self.future_time_steps = self.time_steps.unsqueeze(-1) + offsets
        # clamp not big than the time_step_total 
        self.future_time_steps = torch.clamp(self.future_time_steps, max=self.motion.time_step_total-1)
        
        return torch.cat([self.motion.joint_pos[self.future_time_steps], 
                          self.motion.joint_vel[self.future_time_steps]], dim=-1)
        
@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


class MultiMotionCommand(CommandTerm):
    """Multi-motion tracking command with global bins sampling strategy.
    
    Key differences from old implementation:
    1. Global bins: All motions' bins are concatenated into a single global bin buffer
    2. Direct global sampling: Sample global bin index → uniform sample within bin → compute motion_id + time_step
    3. Vectorized operations: Similar to motion_buffer design, use offsets for efficient indexing
    
    Sampling workflow:
    1. Compute global bin probabilities (adaptive + uniform)
    2. Sample global bin indices using multinomial
    3. Uniform sample local timesteps within selected bins
    4. Use searchsorted to find motion_id from global timesteps
    5. Compute local time_steps from global timesteps
    """
    
    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # Initialize dataset and dataloader
        print(f"[MultiMotionCommand] Loading dataset from: {cfg.dataset_dirs}")
        self.dataset = Motion_Dataset(
            dataset_dirs=cfg.dataset_dirs,
            robot_name=cfg.robot_name,
            splits=cfg.splits,
        )
        
        self.dataloader = Motion_Dataloader(
            dataset=self.dataset,
            body_indexes=self.body_indexes,
            device=self.device,
        )
        
        # Environment state: which motion and timestep each env is at
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.global_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Track if system has started (scalar, not per-env)
        self._has_started = False
        
        # Relative pose storage for tracking
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # === Global Bins Setup ===
        self.bin_size = int(1 / (env.cfg.decimation * env.cfg.sim.dt))
        self.bin_count = int(self.dataloader.time_step_total // self.bin_size) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)

        # Metrics
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        
        print(f"[MultiMotionCommand] Initialization complete:")
        print(f"  - Loaded {self.dataloader.num_motions} motions")
        print(f"  - Total frames: {self.dataloader.time_step_total}")
        print(f"  - Total bins: {self.bin_count}")

    @property
    def command(self) -> torch.Tensor:
        """Command tensor for observation."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """Target joint positions for all environments."""
        return self.dataloader.motion_buffer.joint_pos[self.global_time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Target joint velocities for all environments."""
        return self.dataloader.motion_buffer.joint_vel[self.global_time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Target body positions in world frame."""
        return self.dataloader.motion_buffer.body_pos_w[self.global_time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Target body quaternions in world frame."""
        return self.dataloader.motion_buffer.body_quat_w[self.global_time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Target body linear velocities in world frame."""
        return self.dataloader.motion_buffer.body_lin_vel_w[self.global_time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Target body angular velocities in world frame."""
        return self.dataloader.motion_buffer.body_ang_vel_w[self.global_time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Target anchor body position in world frame."""
        return self.dataloader.motion_buffer.body_pos_w[self.global_time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Target anchor body quaternion in world frame."""
        return self.dataloader.motion_buffer.body_quat_w[self.global_time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """Target anchor body linear velocity in world frame."""
        return self.dataloader.motion_buffer.body_lin_vel_w[self.global_time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """Target anchor body angular velocity in world frame."""
        return self.dataloader.motion_buffer.body_ang_vel_w[self.global_time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        """Current robot joint positions."""
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        """Current robot joint velocities."""
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """Current robot body positions in world frame."""
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """Current robot body quaternions in world frame."""
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """Current robot body linear velocities in world frame."""
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """Current robot body angular velocities in world frame."""
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Current robot anchor body position in world frame."""
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Current robot anchor body quaternion in world frame."""
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        """Current robot anchor body linear velocity in world frame."""
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        """Current robot anchor body angular velocity in world frame."""
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    def _update_metrics(self):
        """Update tracking error metrics."""
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Adaptive sampling using global bins.
        
        Workflow:
        1. Update global bin failure counts based on terminated environments
        2. Compute sampling probabilities (failure-weighted + uniform)
        3. Sample global bin indices
        4. Uniform sample within selected bins to get global timesteps
        5. Reverse lookup: global_timestep → motion_id + local time_step
        """
        # Ensure env_ids is a tensor
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        
        # === Step 1: Update bin failure statistics ===
        episode_failed = self._env.termination_manager.terminated[env_ids_tensor]
        
        if torch.any(episode_failed):
            failed_envs = env_ids_tensor[episode_failed]
            failed_global_time_steps = self.global_time_steps[failed_envs]
            failed_bins = failed_global_time_steps // self.bin_size
            self._current_bin_failed += torch.bincount(failed_bins, minlength=self.bin_count).float()
        
        # === Step 2: Compute sampling probabilities ===
        clip_bin_failed_count = torch.minimum(self.bin_failed_count, self.bin_failed_count.sum() / self.cfg.adaptive_cap)
        failed_sampling_probabilities = (clip_bin_failed_count + 1e-12) / (clip_bin_failed_count.sum() + 1e-12 * self.bin_count)
        sampling_probabilities = self.cfg.adaptive_uniform_ratio * failed_sampling_probabilities + \
                                (1 - self.cfg.adaptive_uniform_ratio) / self.bin_count
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        
        # === Step 3: Sample global bins ===
        sampled_global_bins = torch.multinomial(sampling_probabilities, len(env_ids_tensor), replacement=True)  # [M]
        
        # === Step 4: Uniform sample within bins to get global timesteps ===
        bin_starts = sampled_global_bins * self.bin_size  # [M]
        bin_ends = torch.minimum(
            (sampled_global_bins + 1) * self.bin_size,
            torch.tensor(self.dataloader.time_step_total-1, dtype=torch.long, device=self.device)
        )  # [M]

        # Uniform sample within each bin
        random_offsets = sample_uniform(0.0, 1.0, (len(env_ids_tensor),), device=self.device)
        self.global_time_steps[env_ids_tensor] = (bin_starts + random_offsets * (bin_ends - bin_starts)).long()
        
        # === Step 5: Reverse lookup motion_id and time_steps ===
        new_motion_ids = torch.searchsorted(
            self.dataloader.motion_offsets,
            self.global_time_steps[env_ids_tensor].float(),
            right=False
        ) - 1
        # new_motion_ids = torch.clamp(new_motion_ids, 0, self.dataloader.num_motions - 1)
        
        new_time_steps = self.global_time_steps[env_ids_tensor] - self.dataloader.motion_offsets[new_motion_ids]
        
        # check
        # new_motion_lengths = self.dataloader.motion_lengths[new_motion_ids]
        # new_time_steps = torch.clamp(new_time_steps, min=0)
        # new_time_steps = torch.minimum(new_time_steps, new_motion_lengths - 1)
        
        self.motion_ids[env_ids_tensor] = new_motion_ids
        self.time_steps[env_ids_tensor] = new_time_steps
        
        # === Metrics ===
        H = -(failed_sampling_probabilities * (failed_sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = failed_sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample motion commands for given environments."""
        if len(env_ids) == 0:
            return
        
        self._adaptive_sampling(env_ids)

        # === Initialize robot state from sampled motion data with noise. ===
        # Get motion data for resampled environments
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # Add pose noise
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        
        # Add velocity noise
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        # Get joint positions and velocities
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        # Add joint position noise and clip to limits
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        
        # Write to simulation
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """Update command each timestep."""
        # Increment time steps (both local and global)
        self.time_steps += 1
        self.global_time_steps += 1
        
        # Find environments that exceeded motion length or buffer boundary
        env_ids = torch.where(self.time_steps >= self.dataloader.motion_lengths[self.motion_ids])[0]
        self._resample_command(env_ids)

        # Update relative poses for tracking
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # Update bin failure counts with exponential moving average
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()
        
        # Mark system as started after first update completes
        self._has_started = True

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])

@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for multi-motion command with global bins sampling."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    # Dataset configuration
    dataset_dirs: list[str] = MISSING
    """List of dataset directories containing NPZ motion files."""
    
    robot_name: str = MISSING
    """Robot name for dataset filtering."""
    
    splits: list[Union[str, list[str]]] = MISSING
    """Dataset splits configuration. Flexible format supporting:
    - Single split per dataset: ["train", "val", "test"]
    - Combined splits per dataset: [["train", "walk_subset"], "test"]
    - Mixed format: ["train", ["train", "walk_subset"], "test"]
    
    Examples:
        splits=["train", "val"]  # Use "train" for dataset_dirs[0], "val" for dataset_dirs[1]
        splits=[["train", "walk"], "test"]  # Combine "train"+"walk" for dataset_dirs[0], "test" for dataset_dirs[1]
    """

    # Body configuration
    anchor_body_name: str = MISSING
    """Name of the anchor body (usually root or pelvis)."""
    
    body_names: list[str] = MISSING
    """List of body names to track."""

    # Initialization noise ranges
    pose_range: dict[str, tuple[float, float]] = {}
    """Pose noise ranges for x, y, z, roll, pitch, yaw."""
    
    velocity_range: dict[str, tuple[float, float]] = {}
    """Velocity noise ranges for x, y, z, roll, pitch, yaw."""
    
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    """Joint position noise range."""

    # Adaptive sampling parameters
    adaptive_kernel_size: int = 1
    """Kernel size for convolution smoothing of bin probabilities."""
    
    adaptive_lambda: float = 0.8
    """Exponential decay factor for kernel weights."""
    
    adaptive_uniform_ratio: float = 0.1
    """Ratio of uniform sampling mixed with failure-based sampling."""
    
    adaptive_cap: int = 200
    """Cap for bin failure counts to prevent extreme probabilities."""
    
    adaptive_alpha: float = 0.001
    """EMA smoothing factor for bin failure counts."""

    # Evaluation-specific parameters
    eval_target_attempts: int = 128
    """Number of evaluation attempts per motion (used by EvalMultiMotionCommand)."""

    # Visualization
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)



class EvalMultiMotionCommand(MultiMotionCommand):
    """Evaluation-specific command with attempt-based weighted sampling.
    
    This class extends MultiMotionCommand for systematic evaluation where:
    1. Sampling weights prioritize motions needing more evaluation attempts
    2. All resampling resets to timestep 0 (always start from motion beginning)
    3. Tracks success/failure statistics for each motion
    4. Automatically balances evaluation progress across all motions
    """
    
    cfg: MultiMotionCommandCfg  # Reuse same config, no separate EvalMultiMotionCommandCfg
    
    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        # ============ Evaluation-specific tracking ============
        self.num_motions = len(self.dataset)
        
        # Track evaluation progress for each motion
        # attempt_count: motions that have been sampled/assigned (including in-progress)
        # completed_count: motions that have finished (success or failure)
        self.eval_motion_attempt_count = torch.zeros(self.num_motions, dtype=torch.long, device=self.device)
        self.eval_motion_completed_count = torch.zeros(self.num_motions, dtype=torch.long, device=self.device)
        self.eval_motion_success_count = torch.zeros(self.num_motions, dtype=torch.long, device=self.device)
        self.eval_motion_failure_count = torch.zeros(self.num_motions, dtype=torch.long, device=self.device)
        self.eval_motion_failure_steps = torch.zeros(self.num_motions, dtype=torch.float, device=self.device)

        print(f"[EvalMultiMotionCommand] Initialized for evaluation")
        print(f"  Number of motions: {self.num_motions}")
        print(f"  Target attempts per motion: {cfg.eval_target_attempts}")
        print(f"  Total environments: {self.num_envs}")
    
    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Evaluation-specific adaptive sampling.
        
        Unlike training mode (which samples global bins), evaluation mode:
        1. Tracks success/failure statistics for completed attempts
        2. Samples motions directly based on remaining attempts (not bins)
        3. Always resets time_steps to 0 (start from motion beginning)
        4. Increments attempt count AFTER sampling new motion
        
        This ensures balanced evaluation coverage across all motions.
        """
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device) \
            if not isinstance(env_ids, torch.Tensor) else env_ids
        num_envs = len(env_ids)
        
        # ============ Track success/failure statistics BEFORE resampling ============
        # Get motion properties for all environments being resampled
        motion_ids_batch = self.motion_ids[env_ids_tensor]
        current_timesteps_batch = self.time_steps[env_ids_tensor]
        motion_lengths_batch = self.dataloader.motion_lengths[motion_ids_batch]
        
        # Check termination reasons (vectorized)
        is_failed_batch = self._env.termination_manager.terminated[env_ids_tensor]
        is_motion_complete_batch = current_timesteps_batch >= (motion_lengths_batch - 1)

        # CRITICAL: Only count statistics for attempts that have actually started
        # This avoids false positives at initialization
        has_started_mask = torch.ones(len(env_ids_tensor), dtype=torch.bool, device=self.device) if self._has_started else torch.zeros(len(env_ids_tensor), dtype=torch.bool, device=self.device)

        # Separate failed and completed environments
        failed_mask = has_started_mask & is_failed_batch & ~is_motion_complete_batch
        completed_mask = has_started_mask & is_motion_complete_batch & ~is_failed_batch
        
        # Update failure statistics and increment completed count
        if failed_mask.any():
            failed_motion_ids = motion_ids_batch[failed_mask]
            failed_timesteps = current_timesteps_batch[failed_mask]
            
            self.eval_motion_failure_count += torch.bincount(
                failed_motion_ids, minlength=self.num_motions
            )
            self.eval_motion_completed_count += torch.bincount(
                failed_motion_ids, minlength=self.num_motions
            )
            # Track steps reached before failure for completion rate calculation
            self.eval_motion_failure_steps += torch.bincount(
                failed_motion_ids, weights=failed_timesteps.float(), minlength=self.num_motions
            )
        
        # Update success statistics and increment completed count
        if completed_mask.any():
            completed_motion_ids = motion_ids_batch[completed_mask]
            
            self.eval_motion_success_count += torch.bincount(
                completed_motion_ids, minlength=self.num_motions
            )
            self.eval_motion_completed_count += torch.bincount(
                completed_motion_ids, minlength=self.num_motions
            )
        
        # ============ Track bin failures for global bins (optional, for analysis) ============
        episode_failed = self._env.termination_manager.terminated[env_ids_tensor]
        # Only track bin failures for episodes that have actually started
        episode_failed = episode_failed & self._has_started
        
        if torch.any(episode_failed):
            failed_envs = env_ids_tensor[episode_failed]
            failed_global_time_steps = self.global_time_steps[failed_envs]
            failed_bins = (failed_global_time_steps.float() / self.bin_size).long()
            failed_bins = torch.clamp(failed_bins, 0, self.bin_count - 1)
            
            self._current_bin_failed += torch.bincount(
                failed_bins, minlength=self.bin_count
            ).float()
        
        # ============ Sample new motions using remaining-attempts weights ============
        # Compute weights: prioritize motions needing more COMPLETED attempts
        # Use completed_count to ensure we keep sampling until motions are actually finished
        remaining_attempts = self.cfg.eval_target_attempts - self.eval_motion_attempt_count
        weights = remaining_attempts.float() + 1e-4  # Add epsilon to avoid zero weights
        weights = torch.clamp(weights, min=1e-4)
        
        # Normalize weights for multinomial sampling
        weights_normalized = weights / weights.sum()
        
        # Sample motion IDs based on remaining completed attempts
        sampled_motion_ids = torch.multinomial(weights_normalized, num_envs, replacement=True)
        
        # ============ CRITICAL: Increment attempt count AFTER sampling ============
        self.eval_motion_attempt_count += torch.bincount(
            sampled_motion_ids, minlength=self.num_motions
        )
        
        self.motion_ids[env_ids_tensor] = sampled_motion_ids
        
        # ============ Always reset timesteps to 0 ============
        # In evaluation, we always start from the beginning of the motion
        self.time_steps[env_ids_tensor] = 0
        
        # Update global_time_steps based on motion_offsets
        self.global_time_steps[env_ids_tensor] = self.dataloader.motion_offsets[sampled_motion_ids]
    
    def check_eval_complete(self) -> bool:
        """Check if all motions have reached target evaluation attempts and completed.
        
        Returns:
            True if all motions have completed at least eval_target_attempts times
        """
        return torch.all(self.eval_motion_completed_count >= self.cfg.eval_target_attempts).item()
    
    def get_incomplete_motions(self) -> torch.Tensor:
        """Get motion IDs that haven't reached target completed attempts.
        
        Returns:
            Tensor of motion IDs that need more completed evaluation attempts
        """
        return torch.where(self.eval_motion_completed_count < self.cfg.eval_target_attempts)[0]
    
    def get_eval_results(self) -> dict:
        """Get evaluation results for all motions.
        
        Returns:
            Dictionary mapping motion_id to evaluation statistics:
            {
                motion_id: {
                    'motion_name': str,
                    'completed': int,
                    'successes': int,
                    'failures': int,
                    'success_rate': float,
                    'completion_rate': float,  # Average completion percentage (0-1)
                    'avg_steps': float,
                    'motion_length': int,
                }
            }
        """
        results = {}

        for motion_id in range(self.num_motions):
            attempts = self.eval_motion_attempt_count[motion_id].item()
            completed = self.eval_motion_completed_count[motion_id].item()
            successes = self.eval_motion_success_count[motion_id].item()
            failures = self.eval_motion_failure_count[motion_id].item()
            failure_steps = self.eval_motion_failure_steps[motion_id].item()
            motion_length = self.dataloader.motion_lengths[motion_id].item()
            
            success_rate = successes / completed if completed > 0 else 0.0
            
            # Completion rate: average percentage of motion completed across all attempts
            # For successes: they completed full motion_length
            # For failures: steps reached before failure
            total_completed_steps = successes * motion_length + failure_steps
            completion_rate = total_completed_steps / (completed * motion_length) if completed > 0 else 0.0
            
            motion_info = self.dataset[motion_id]
            
            results[motion_id] = {
                'motion_name': motion_info['motion_name'],
                'completed': completed,
                'successes': successes,
                'failures': failures,
                'success_rate': success_rate,
                'completion_rate': completion_rate,
                'motion_length': motion_length,
                'quantity': motion_info['quantity'],
            }
        
        return results
    
    def print_progress(self):
        """Print evaluation progress to console."""
        incomplete = self.get_incomplete_motions()
        num_complete = self.num_motions - len(incomplete)
        
        # Attempt statistics (including in-progress)
        min_attempts = self.eval_motion_attempt_count.min().item()
        max_attempts = self.eval_motion_attempt_count.max().item()
        avg_attempts = self.eval_motion_attempt_count.float().mean().item()
        total_attempts = self.eval_motion_attempt_count.sum().item()
        
        # Completed statistics
        min_completed = self.eval_motion_completed_count.min().item()
        max_completed = self.eval_motion_completed_count.max().item()
        avg_completed = self.eval_motion_completed_count.float().mean().item()
        total_completed = self.eval_motion_completed_count.sum().item()
        
        # In-progress count
        total_in_progress = total_attempts - total_completed
        
        # Current time_steps statistics across all environments
        min_timesteps = self.time_steps.min().item()
        max_timesteps = self.time_steps.max().item()
        avg_timesteps = self.time_steps.float().mean().item()

        print(f"[Eval Progress] {num_complete}/{self.num_motions} motions complete")
        print(f"  Attempts (assigned): min={min_attempts}, max={max_attempts}, avg={avg_attempts:.1f}, total={total_attempts}")
        print(f"  Completed (finished): min={min_completed}, max={max_completed}, avg={avg_completed:.1f}, total={total_completed}")
        print(f"  In-progress: {total_in_progress}")
        print(f"  Current time_steps: min={min_timesteps}, max={max_timesteps}, avg={avg_timesteps:.1f}")
        print(f"  Incomplete motions: {len(incomplete)}")


