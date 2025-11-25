"""Unified motion dataloader for loading and sampling paired robot and SMPL-X motion data.

This module provides a PyTorch-style dataloader with weighted sampling support and
efficient vectorized batch indexing for dual-source motion data (robot + SMPL-X).

Key features:
- Preloads all motion data and concatenates into single tensors for efficient batch indexing
- Handles data length alignment between robot and SMPL-X sources (takes minimum length)
- Supports weighted sampling based on motion quantity or custom weights
- Provides global index mapping via motion_offsets for fast tensor indexing
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, Optional

import numpy as np
import torch

from whole_body_tracking.utils.unify_motion_dataset import Unify_Motion_Dataset


class Unify_Motion_Dataloader:
    """Dataloader for paired robot and SMPL-X motion data with weighted sampling.
    
    Preloads all motion data from both sources and concatenates into single tensors,
    enabling efficient vectorized batch access via global indexing.
    
    Args:
        dataset: Unify_Motion_Dataset instance providing paired robot/SMPL-X data
        body_indexes: Indices for filtering robot body data (only affects robot tensors)
        device: Device to load tensors on ("cuda" or "cpu")
        
    Attributes:
        motion_buffer: MotionBuffer instance containing concatenated motion tensors
        motion_lengths: [num_motions] aligned frame counts after length matching
        motion_offsets: [num_motions] global starting indices for each motion
        motion_fps: [num_motions] frame rates for each motion
        time_step_total: Total frames in concatenated buffer
    """
    
    class MotionBuffer:
        """Stores concatenated motion data from robot and SMPL-X sources.
        
        Robot tensors are concatenated with optional body_indexes filtering via
        property decorators. SMPL-X pose_body is stored directly without filtering.
        
        Attributes:
            joint_pos: [total_frames, num_joints] - Joint positions (robot)
            joint_vel: [total_frames, num_joints] - Joint velocities (robot)
            body_pos_w: [total_frames, num_bodies_filtered, 3] - Body positions (filtered by body_indexes)
            body_quat_w: [total_frames, num_bodies_filtered, 4] - Body quaternions (filtered)
            body_lin_vel_w: [total_frames, num_bodies_filtered, 3] - Body linear velocities (filtered)
            body_ang_vel_w: [total_frames, num_bodies_filtered, 3] - Body angular velocities (filtered)
            smplx_pose_body: [total_frames, pose_body_dim] - SMPL-X body pose (no filtering)
        """

        def __init__(self, body_indexes: Sequence[int]):
            """Initialize empty motion buffer.
            
            Args:
                body_indexes: Indices for filtering body-related tensors from robot data
            """
            # Robot data tensors
            self.joint_pos: Optional[torch.Tensor] = None
            self.joint_vel: Optional[torch.Tensor] = None
            self._body_pos_w: Optional[torch.Tensor] = None
            self._body_quat_w: Optional[torch.Tensor] = None
            self._body_lin_vel_w: Optional[torch.Tensor] = None
            self._body_ang_vel_w: Optional[torch.Tensor] = None
            
            # SMPL-X data tensor
            self._smplx_pose_body: Optional[torch.Tensor] = None
            
            self.body_indexes = body_indexes
        
        @property
        def body_pos_w(self) -> torch.Tensor:
            return self._body_pos_w[:, self.body_indexes]

        @property
        def body_quat_w(self) -> torch.Tensor:
            return self._body_quat_w[:, self.body_indexes]

        @property
        def body_lin_vel_w(self) -> torch.Tensor:
            return self._body_lin_vel_w[:, self.body_indexes]

        @property
        def body_ang_vel_w(self) -> torch.Tensor:
            return self._body_ang_vel_w[:, self.body_indexes]
        
        @property
        def smplx_pose_body(self) -> torch.Tensor:
            return self._smplx_pose_body

    def __init__(
        self,
        dataset: Unify_Motion_Dataset,
        body_indexes: Sequence[int],
        device: str = "cuda",
    ):
        """Initialize the unified motion dataloader.
        
        Args:
            dataset: Unify_Motion_Dataset instance
            body_indexes: Sequence of body indices for filtering robot data
            device: Device to load tensors on ("cuda" or "cpu")
        """
        self.dataset = dataset
        self.device = device
        self.num_motions = len(dataset)
        self._body_indexes = body_indexes
        
        # Initialize motion buffer
        self.motion_buffer = self.MotionBuffer(self._body_indexes)
        
        # Motion metadata (populated in _preload_and_concatenate)
        self.motion_lengths: torch.Tensor = None  # [num_motions]
        self.motion_offsets: torch.Tensor = None  # [num_motions]
        self.motion_fps: torch.Tensor = None      # [num_motions]
        self.time_step_total: int = 0
        
        # Track data alignment statistics
        self._num_misaligned: int = 0
        self._total_truncated_frames: int = 0
        
        print(f"[Unify_Motion_Dataloader] Loading and concatenating {self.num_motions} paired motions...")
        
        # Load all motions and concatenate into single tensors
        self._preload_and_concatenate()
        
        print(f"[Unify_Motion_Dataloader] Initialization complete. Total frames: {self.time_step_total}")
        if self._num_misaligned > 0:
            print(f"[Unify_Motion_Dataloader] Data alignment: {self._num_misaligned} motions had length mismatches")
            print(f"[Unify_Motion_Dataloader] Total frames truncated: {self._total_truncated_frames}")

    def _preload_and_concatenate(self) -> None:
        """Preload all motions and concatenate into single tensors with offset tracking.
        
        This method:
        1. Loads all motion pairs from the dataset
        2. Detects and handles length mismatches (takes minimum of robot and SMPL-X lengths)
        3. Truncates both sources to aligned length
        4. Concatenates all data along time dimension
        5. Computes offsets for global indexing
        """
        # Temporary lists for collecting data
        robot_joint_pos_list = []
        robot_joint_vel_list = []
        robot_body_pos_w_list = []
        robot_body_quat_w_list = []
        robot_body_lin_vel_w_list = []
        robot_body_ang_vel_w_list = []
        smplx_pose_body_list = []
        
        lengths = []
        fps_list = []
        
        # Load all motion pairs
        for i in range(self.num_motions):
            sample = self.dataset[i]
            motion_name = sample["motion_name"]
            
            # Extract robot motion data
            robot_motion = sample["robot"]["motion"]
            robot_fps = sample["robot"]["fps"]
            robot_len = robot_motion["joint_pos"].shape[0]
            
            # Extract SMPL-X motion data
            smplx_motion = sample["smplx"]["motion"]
            smplx_fps = sample["smplx"]["fps"]
            smplx_len = smplx_motion["pose_body"].shape[0]
            
            # Validate FPS consistency
            if round(robot_fps) != round(smplx_fps):
                raise ValueError(
                    f"Motion {motion_name} has mismatched FPS: robot={robot_fps}, smplx={smplx_fps}"
                )
            
            # Handle length mismatch by taking minimum
            aligned_len = min(robot_len, smplx_len)
            if aligned_len < robot_len or aligned_len < smplx_len:
                self._num_misaligned += 1
                self._total_truncated_frames += max(robot_len, smplx_len) - aligned_len
            
            # Extract and truncate robot data
            robot_joint_pos_list.append(
                torch.tensor(robot_motion["joint_pos"][:aligned_len], dtype=torch.float32, device=self.device)
            )
            robot_joint_vel_list.append(
                torch.tensor(robot_motion["joint_vel"][:aligned_len], dtype=torch.float32, device=self.device)
            )
            robot_body_pos_w_list.append(
                torch.tensor(robot_motion["body_pos_w"][:aligned_len], dtype=torch.float32, device=self.device)
            )
            robot_body_quat_w_list.append(
                torch.tensor(robot_motion["body_quat_w"][:aligned_len], dtype=torch.float32, device=self.device)
            )
            robot_body_lin_vel_w_list.append(
                torch.tensor(robot_motion["body_lin_vel_w"][:aligned_len], dtype=torch.float32, device=self.device)
            )
            robot_body_ang_vel_w_list.append(
                torch.tensor(robot_motion["body_ang_vel_w"][:aligned_len], dtype=torch.float32, device=self.device)
            )
            
            # Extract and truncate SMPL-X data
            smplx_pose_body_list.append(
                torch.tensor(smplx_motion["pose_body"][:aligned_len], dtype=torch.float32, device=self.device)
            )
            
            lengths.append(aligned_len)
            fps_list.append(robot_fps)
        
        # Concatenate all sequences along time dimension
        self.motion_buffer.joint_pos = torch.cat(robot_joint_pos_list, dim=0)
        self.motion_buffer.joint_vel = torch.cat(robot_joint_vel_list, dim=0)
        self.motion_buffer._body_pos_w = torch.cat(robot_body_pos_w_list, dim=0)
        self.motion_buffer._body_quat_w = torch.cat(robot_body_quat_w_list, dim=0)
        self.motion_buffer._body_lin_vel_w = torch.cat(robot_body_lin_vel_w_list, dim=0)
        self.motion_buffer._body_ang_vel_w = torch.cat(robot_body_ang_vel_w_list, dim=0)
        self.motion_buffer._smplx_pose_body = torch.cat(smplx_pose_body_list, dim=0)
        
        # Compute offsets for each motion (cumulative sum of lengths)
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_offsets = torch.cat([
            torch.tensor([0], dtype=torch.long, device=self.device),
            torch.cumsum(self.motion_lengths, dim=0)[:-1]
        ], dim=0)
        
        # Store FPS for each motion
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)
        
        # Store total buffer length
        self.time_step_total = self.motion_buffer.joint_pos.shape[0]
        
        # Print concatenation results
        print(f"[Unify_Motion_Dataloader] Concatenated tensors:")
        print(f"  Robot joint_pos: {self.motion_buffer.joint_pos.shape}")
        print(f"  Robot joint_vel: {self.motion_buffer.joint_vel.shape}")
        print(f"  Robot body_pos_w (filtered): {self.motion_buffer.body_pos_w.shape}")
        print(f"  Robot body_quat_w (filtered): {self.motion_buffer.body_quat_w.shape}")
        print(f"  SMPL-X pose_body: {self.motion_buffer.smplx_pose_body.shape}")
        print(f"  Total frames: {self.time_step_total}")
        print(f"  Motion lengths: {self.motion_lengths.shape}, range: [{self.motion_lengths.min()}, {self.motion_lengths.max()}]")
        print(f"  Motion offsets: {self.motion_offsets.shape}")

    def get_motion_length(self, motion_id: int) -> int:
        """Get aligned frame count of a specific motion.
        
        Args:
            motion_id: Index of the motion
            
        Returns:
            Number of frames in the motion (after alignment)
        """
        return self.motion_lengths[motion_id].item()
    
    def get_motion_fps(self, motion_id: int) -> float:
        """Get frame rate of a specific motion.
        
        Args:
            motion_id: Index of the motion
            
        Returns:
            Frames per second of the motion
        """
        return self.motion_fps[motion_id].item()
    
    def sample(
        self,
        n: int,
        weights: Optional[torch.Tensor | list] = None,
    ) -> torch.Tensor:
        """Sample n motion indices with optional weights.
        
        Args:
            n: Number of motion clips to sample
            weights: Optional [num_motions] tensor or list of sampling weights.
                    If None, uniform sampling is used.
                    Weights will be normalized internally.
        
        Returns:
            motion_indices: Tensor[n], sampled motion indices in dataset
            
        Example:
            # Uniform sampling
            indices = dataloader.sample(10)
            
            # Weighted sampling based on quantity
            weights = [0.85 if q==1 else 0.10 if q==2 else 0.05 
                      for q in dataset.quantities]
            indices = dataloader.sample(10, weights=weights)
        """
        if weights is None:
            # Uniform sampling
            weights = torch.ones(self.num_motions, device=self.device)
        else:
            # Convert to tensor if needed
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            else:
                weights = weights.to(self.device)
            
            # Validate shape
            if weights.shape[0] != self.num_motions:
                raise ValueError(
                    f"Weights shape mismatch: expected [{self.num_motions}], got {weights.shape}"
                )
        
        # Ensure positive weights
        weights = torch.clamp(weights, min=1e-8)
        
        # Normalize
        weights = weights / weights.sum()
        
        # Sample
        motion_indices = torch.multinomial(weights, n, replacement=True)
        
        return motion_indices


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Test Unify_Motion_Dataloader")
    parser.add_argument(
        "--robot_dataset",
        type=str,
        nargs="+",
        help="Robot dataset in format: path:split1,split2 (repeatable)",
    )
    parser.add_argument(
        "--smplx_dataset",
        type=str,
        nargs="+",
        help="SMPL-X dataset directories (must match robot_dataset count)",
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default="g1",
        help="Robot name",
    )
    parser.add_argument(
        "--body_indexes",
        type=int,
        nargs="+",
        default=None,
        help="Body indices for filtering (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()
    
    # Parse robot_dataset format: path:split1,split2
    def parse_robot_dataset(items):
        if not items:
            raise ValueError("--robot_dataset is required")
        out = {}
        for item in items:
            if ":" not in item:
                raise ValueError("Use format: path:split1,split2")
            root, splits_str = item.split(":", 1)
            out[root] = splits_str.split(",")
        return out
    
    robot_dataset = parse_robot_dataset(args.robot_dataset)
    smplx_dataset = args.smplx_dataset or []
    
    if len(robot_dataset) != len(smplx_dataset):
        raise ValueError(
            f"Number of robot_dataset entries ({len(robot_dataset)}) must match "
            f"number of smplx_dataset entries ({len(smplx_dataset)})"
        )
    
    # Default body_indexes: all indices from 0 to some reasonable max
    if args.body_indexes is None:
        args.body_indexes = list(range(15))  # Default: first 15 bodies
    
    # Create dataset
    print("Creating Unify_Motion_Dataset...")
    dataset = Unify_Motion_Dataset(
        robot_dataset=robot_dataset,
        smplx_dataset=smplx_dataset,
        robot_name=args.robot_name,
    )
    
    print(f"Dataset created with {len(dataset)} motion pairs\n")
    
    # Create dataloader
    print("Creating Unify_Motion_Dataloader...")
    dataloader = Unify_Motion_Dataloader(
        dataset=dataset,
        body_indexes=args.body_indexes,
        device=args.device,
    )
    
    print("\n" + "="*60)
    print("Test 1: Uniform Sampling")
    print("="*60)
    indices = dataloader.sample(n=5)
    print(f"Sampled motion indices: {indices}")
    
    print("\n" + "="*60)
    print("Test 2: Motion Metadata Access")
    print("="*60)
    for i in range(min(3, len(dataset))):
        length = dataloader.get_motion_length(i)
        fps = dataloader.get_motion_fps(i)
        print(f"Motion {i}: length={length} frames, fps={fps}")
    
    print("\n" + "="*60)
    print("Test 3: Direct Buffer Access")
    print("="*60)
    motion_ids = dataloader.sample(n=3)
    time_steps = torch.tensor([10, 20, 15], device=args.device)
    
    # Compute global indices
    global_indices = dataloader.motion_offsets[motion_ids] + time_steps
    
    # Direct buffer access
    joint_pos = dataloader.motion_buffer.joint_pos[global_indices]
    body_pos_w = dataloader.motion_buffer.body_pos_w[global_indices]
    smplx_pose_body = dataloader.motion_buffer.smplx_pose_body[global_indices]
    
    print(f"Direct buffer access for 3 motions:")
    print(f"  joint_pos shape: {joint_pos.shape}")
    print(f"  body_pos_w shape (filtered): {body_pos_w.shape}")
    print(f"  smplx_pose_body shape: {smplx_pose_body.shape}")
    
    print("\n" + "="*60)
    print("Test 4: Weighted Sampling")
    print("="*60)
    weights = [0.85 if q == 1 else 0.10 if q == 2 else 0.05 for q in dataset.quantities]
    sampled_indices = dataloader.sample(n=20, weights=weights)
    sampled_quantities = [dataset.quantities[idx.item()] for idx in sampled_indices]
    print(f"Sampled indices: {sampled_indices}")
    print(f"Sampled quantities: {sampled_quantities}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
