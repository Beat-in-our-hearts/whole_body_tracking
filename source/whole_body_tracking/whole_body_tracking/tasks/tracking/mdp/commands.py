from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

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

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

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
    """Multi-motion tracking command with dataset-based sampling.
    
    This class extends MotionCommand to support multiple motion clips loaded from
    a dataset, with flexible weighted sampling strategies for curriculum learning
    and adaptive training.
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
            split=cfg.split,
        )
        
        self.dataloader = Motion_Dataloader(
            dataset=self.dataset,
            body_indexes=self.body_indexes.tolist(),
            device=self.device,
        )
        
        # Track which motion each environment is using
        self.motion_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Relative pose storage
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Bin-based adaptive sampling for each motion
        # Use concatenation + offset indexing for efficient vectorized operations
        motion_bin_counts_list = []
        
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            motion_length = sample['length']
            # Calculate bin count for this motion based on its length
            bin_count = int(motion_length // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
            motion_bin_counts_list.append(bin_count)
        
        # Store bin counts as tensor for vectorized access
        self.motion_bin_counts = torch.tensor(motion_bin_counts_list, dtype=torch.long, device=self.device)
        
        # Concatenate all bins into single tensors (similar to motion_buffer design)
        total_bins = self.motion_bin_counts.sum().item()
        self.motion_bin_failed_counts = torch.zeros(total_bins, dtype=torch.float, device=self.device)
        self.motion_current_bin_failed = torch.zeros(total_bins, dtype=torch.float, device=self.device)
        
        # Compute offsets for each motion's bins (cumulative sum)
        self.motion_bin_offsets = torch.cat([
            torch.tensor([0], device=self.device),
            torch.cumsum(self.motion_bin_counts, dim=0)[:-1]
        ], dim=0)  # [num_motions], offsets[i] = starting bin index for motion i
        
        # Adaptive sampling kernel
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()
        
        # Track motion usage for diversity
        self.agent_motion_usage = torch.zeros(self.num_envs, len(self.dataset), dtype=torch.float, device=self.device)
        
        # Curriculum learning state
        self.curriculum_progress = 0.0  # 0.0 to 1.0

        # Metrics
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
        self.metrics["curriculum_progress"] = torch.zeros(self.num_envs, device=self.device)
    
    def _batch_index_motions(self, motion_data_key: str) -> torch.Tensor:
        """Batch index motion data for all environments using vectorized operations.
        
        This method uses the dataloader's efficient concatenation + offset indexing
        to retrieve motion data for all environments in a single operation, eliminating
        all Python loops and leveraging GPU parallelism.
        
        Args:
            motion_data_key: Key for the motion data to retrieve (e.g., 'joint_pos', 'body_pos_w')
            
        Returns:
            Tensor containing the motion data for all environments at their current timesteps
        """
        return self.dataloader.batch_index(
            motion_ids=self.motion_indices,
            time_steps=self.time_steps,
            data_key=motion_data_key
        )

    def _compute_sampling_weights(self, num_samples: int) -> torch.Tensor:
        """Compute uniform sampling weights for all motions.
        
        Returns:
            weights: [num_motions] tensor of uniform sampling weights
        """
        num_motions = len(self.dataset)
        
        # Uniform weights for all motions
        weights = torch.ones(num_motions, device=self.device) / num_motions
        
        return weights

    def _adaptive_sampling(self, sampled_motion_indices: torch.Tensor) -> torch.Tensor:
        """Vectorized adaptive timestep sampling for multiple motions.
        
        This method performs bin-based adaptive sampling for a batch of environments,
        where each environment may use a different motion with different bin configurations.
        Uses padding technique to enable full vectorization without any for-loops.
        
        Args:
            sampled_motion_indices: [num_envs] tensor of motion indices
            
        Returns:
            sampled_time_steps: [num_envs] tensor of timestep indices
        """
        num_envs = sampled_motion_indices.shape[0]
        
        # Get motion properties for all environments (vectorized)
        motion_lengths = self.dataloader.motion_lengths[sampled_motion_indices]  # [num_envs]
        bin_counts = self.motion_bin_counts[sampled_motion_indices]  # [num_envs]
        bin_offsets = self.motion_bin_offsets[sampled_motion_indices]  # [num_envs]
        max_bin_count = bin_counts.max().item()
        
        # ============ Step 1: Gather failure counts using advanced indexing ============
        # Create index matrix: [num_envs, max_bin_count] where each row contains global bin indices
        bin_index_offsets = torch.arange(max_bin_count, device=self.device).unsqueeze(0)  # [1, max_bin_count]
        global_bin_indices = bin_offsets.unsqueeze(1) + bin_index_offsets  # [num_envs, max_bin_count]
        
        # Clamp indices to avoid out-of-bounds (padding will be masked out later)
        global_bin_indices = torch.clamp(global_bin_indices, 0, self.motion_bin_failed_counts.shape[0] - 1)
        
        # Gather failure counts for all bins (vectorized)
        padded_failure_counts = self.motion_bin_failed_counts[global_bin_indices]  # [num_envs, max_bin_count]
        
        # Add uniform ratio component
        uniform_component = self.cfg.adaptive_uniform_ratio / bin_counts.float().unsqueeze(1)  # [num_envs, 1]
        padded_probs = padded_failure_counts + uniform_component  # [num_envs, max_bin_count]
        
        # ============ Step 2: Apply smoothing kernel (vectorized) ============
        if self.cfg.adaptive_kernel_size > 1:
            # Pad probabilities for convolution: [num_envs, 1, max_bin_count]
            padded_probs_3d = padded_probs.unsqueeze(1)
            
            # Replicate padding for kernel
            padded_probs_3d = torch.nn.functional.pad(
                padded_probs_3d,
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            
            # Apply 1D convolution with kernel for all environments at once
            # Use groups=1 to apply same kernel across all environments
            kernel_3d = self.kernel.view(1, 1, -1)  # [1, 1, kernel_size]
            padded_probs_3d = torch.nn.functional.conv1d(padded_probs_3d, kernel_3d)  # [num_envs, 1, max_bin_count]
            
            # Reshape back to [num_envs, max_bin_count]
            padded_probs = padded_probs_3d.squeeze(1)
        
        # ============ Step 3: Mask out padding regions ============
        # Create mask: True for valid bins, False for padding
        mask = torch.arange(max_bin_count, device=self.device).unsqueeze(0) < bin_counts.unsqueeze(1)
        padded_probs = padded_probs * mask.float()  # Zero out padding
        
        # ============ Step 4: Normalize probabilities (vectorized) ============
        prob_sums = padded_probs.sum(dim=1, keepdim=True)  # [num_envs, 1]
        padded_probs = padded_probs / (prob_sums + 1e-12)  # Add epsilon for numerical stability
        
        # ============ Step 5: Batch multinomial sampling ============
        sampled_bins = torch.multinomial(padded_probs, 1).squeeze(1)  # [num_envs]
        
        # ============ Step 6: Convert bins to timesteps (vectorized) ============
        random_jitter = torch.rand(num_envs, device=self.device)  # [0, 1)
        sampled_time_steps = (
            ((sampled_bins.float() + random_jitter) / bin_counts.float()) * (motion_lengths.float() - 1)
        ).long()
        sampled_time_steps = torch.clamp(sampled_time_steps, 0, motion_lengths - 1)
        
        return sampled_time_steps

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample motions for specified environments (fully vectorized implementation)."""
        if len(env_ids) == 0:
            return
        
        # Convert to tensor for vectorized operations
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
        num_envs = len(env_ids)
        
        # ============ Vectorized bin failure tracking ============
        episode_failed = self._env.termination_manager.terminated[env_ids_tensor]
        if torch.any(episode_failed):
            # Get failed environment indices (boolean indexing)
            failed_mask = episode_failed
            failed_env_ids = env_ids_tensor[failed_mask]
            
            # Vectorized: get motion indices and properties for all failed envs
            failed_motion_indices = self.motion_indices[failed_env_ids]  # [N_failed]
            failed_time_steps = self.time_steps[failed_env_ids]  # [N_failed]
            failed_motion_lengths = self.dataloader.motion_lengths[failed_motion_indices]  # [N_failed]
            failed_bin_counts = self.motion_bin_counts[failed_motion_indices]  # [N_failed]
            
            # Vectorized: calculate bin indices for all failed envs
            current_bin_indices = torch.clamp(
                (failed_time_steps * failed_bin_counts) // torch.clamp(failed_motion_lengths, min=1),
                0, failed_bin_counts - 1
            )  # [N_failed]
            
            # Vectorized: compute global bin indices
            global_bin_indices = self.motion_bin_offsets[failed_motion_indices] + current_bin_indices  # [N_failed]
            
            # Vectorized: update failure counts using scatter_add (atomic operation)
            self.motion_current_bin_failed.scatter_add_(
                0, 
                global_bin_indices, 
                torch.ones_like(global_bin_indices, dtype=torch.float)
            )
        
        # ============ Sample new motions ============
        weights = self._compute_sampling_weights(num_envs)
        sampled_indices = self.dataloader.sample(n=num_envs, weights=weights)
        self.motion_indices[env_ids_tensor] = sampled_indices
        
        # ============ Vectorized usage tracking ============
        # Update usage counts using advanced indexing
        self.agent_motion_usage[env_ids_tensor, sampled_indices] += 1
        
        # ============ Vectorized timestep sampling ============
        if self.cfg.use_adaptive:
            # Use vectorized adaptive sampling method
            sampled_time_steps = self._adaptive_sampling(sampled_indices)
        else:
            # Vectorized uniform sampling within each motion
            motion_lengths = self.dataloader.motion_lengths[sampled_indices]  # [num_envs]
            random_fractions = torch.rand(num_envs, device=self.device)  # [0, 1)
            sampled_time_steps = (random_fractions * motion_lengths.float()).long()
            sampled_time_steps = torch.clamp(sampled_time_steps, 0, motion_lengths - 1)
        
        # Vectorized: update time_steps for all environments
        self.time_steps[env_ids_tensor] = sampled_time_steps
        
        # Initialize robot state from sampled motions
        self._init_robot_state(env_ids)
        
        # Update curriculum progress
        if self.cfg.use_curriculum:
            # Progress based on total environment steps
            total_steps = self._env.episode_length_buf.float().mean()
            self.curriculum_progress = min(1.0, total_steps / self.cfg.curriculum_total_steps)
        
        # Compute and log sampling metrics
        H = -(weights * (weights + 1e-12).log()).sum()
        H_norm = H / math.log(len(self.dataset))
        pmax, imax = weights.max(dim=0)
        
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["curriculum_progress"][:] = self.curriculum_progress

    def _init_robot_state(self, env_ids: Sequence[int]):
        """Initialize robot state from sampled motion timesteps using vectorized dataloader access."""
        if len(env_ids) == 0:
            return
        
        # Convert env_ids to tensor if it's a list
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
        
        # Get motion states for all environments using batch_index
        motion_ids_batch = self.motion_indices[env_ids_tensor]
        time_steps_batch = self.time_steps[env_ids_tensor]
        
        # Batch retrieve all motion data
        body_pos_w_batch = self.dataloader.batch_index(motion_ids_batch, time_steps_batch, 'body_pos_w')  # [N, num_bodies, 3]
        body_quat_w_batch = self.dataloader.batch_index(motion_ids_batch, time_steps_batch, 'body_quat_w')  # [N, num_bodies, 4]
        body_lin_vel_w_batch = self.dataloader.batch_index(motion_ids_batch, time_steps_batch, 'body_lin_vel_w')  # [N, num_bodies, 3]
        body_ang_vel_w_batch = self.dataloader.batch_index(motion_ids_batch, time_steps_batch, 'body_ang_vel_w')  # [N, num_bodies, 3]
        joint_pos_batch = self.dataloader.batch_index(motion_ids_batch, time_steps_batch, 'joint_pos')  # [N, num_joints]
        joint_vel_batch = self.dataloader.batch_index(motion_ids_batch, time_steps_batch, 'joint_vel')  # [N, num_joints]
        
        # Extract root (first body) states
        root_pos = body_pos_w_batch[:, 0, :].clone()  # [N, 3]
        root_ori = body_quat_w_batch[:, 0, :].clone()  # [N, 4]
        root_lin_vel = body_lin_vel_w_batch[:, 0, :].clone()  # [N, 3]
        root_ang_vel = body_ang_vel_w_batch[:, 0, :].clone()  # [N, 3]
        joint_pos = joint_pos_batch.clone()  # [N, num_joints]
        joint_vel = joint_vel_batch.clone()  # [N, num_joints]
        
        # Add pose noise
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)
        
        # Add velocity noise
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]
        
        # Add joint position noise and clip to limits
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids_tensor]  # [N, num_joints, 2]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])
        
        # Write to simulation (batch operation)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )

    @property
    def command(self) -> torch.Tensor:
        """Get current command (joint positions and velocities)."""
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """Get joint positions for all environments."""
        return self._batch_index_motions('joint_pos')

    @property
    def joint_vel(self) -> torch.Tensor:
        """Get joint velocities for all environments."""
        return self._batch_index_motions('joint_vel')

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Get body positions for all environments."""
        return self._batch_index_motions('body_pos_w') + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Get body quaternions for all environments."""
        return self._batch_index_motions('body_quat_w')

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Get body linear velocities for all environments."""
        return self._batch_index_motions('body_lin_vel_w')

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Get body angular velocities for all environments."""
        return self._batch_index_motions('body_ang_vel_w')

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Get anchor position for all environments."""
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Get anchor quaternion for all environments."""
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """Get anchor linear velocity for all environments."""
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """Get anchor angular velocity for all environments."""
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

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

    def _update_command(self):
        """Update command to next timestep and resample if needed."""
        self.time_steps += 1
        
        # Vectorized check: find environments that have exceeded their motion length
        # Get motion lengths for each environment
        motion_lengths = self.dataloader.motion_lengths[self.motion_indices]  # [num_envs]
        needs_resample = self.time_steps >= motion_lengths  # [num_envs], bool
        
        env_ids = torch.where(needs_resample)[0]
        # if len(env_ids) > 0:
            # Note: parent class _resample() will call our _resample_command()
            # But we also need to reset time_left to prevent double resampling
            # self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
        self._resample_command(env_ids)

        # Update relative poses
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # Update adaptive failure tracking for all motions (vectorized)
        # EMA update: new = alpha * current + (1 - alpha) * old
        self.motion_bin_failed_counts = (
            self.cfg.adaptive_alpha * self.motion_current_bin_failed +
            (1 - self.cfg.adaptive_alpha) * self.motion_bin_failed_counts
        )
        self.motion_current_bin_failed.zero_()

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
    """Configuration for multi-motion command with dataset-based sampling."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING
    
    # Dataset configuration
    dataset_dirs: list[str] = MISSING
    robot_name: str = MISSING
    split: str = "train"

    # Body tracking configuration
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    # Initialization noise ranges
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_total_steps: float = 1e8  # Total steps to complete curriculum
    
    # Adaptive sampling
    use_adaptive: bool = True
    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001  # EMA decay for bin failure tracking
    
    # Visualization
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

