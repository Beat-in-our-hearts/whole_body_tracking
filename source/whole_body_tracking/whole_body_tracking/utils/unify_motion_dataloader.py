"""Unified motion dataloader extending Motion_Dataloader with SMPL-X data support.

This module provides Unify_Motion_Dataloader which extends Motion_Dataloader to handle
paired robot and SMPL-X motion data with efficient batch indexing.
"""
from collections.abc import Sequence
from typing import Any, Dict, Optional
import torch
import numpy as np

from whole_body_tracking.utils.motion_dataloader import Motion_Dataloader
from whole_body_tracking.utils.unify_motion_dataset import Unify_Motion_Dataset


class Unify_Motion_Dataloader(Motion_Dataloader):
    """Extended dataloader for paired robot and SMPL-X motion data.
    
    Inherits from Motion_Dataloader and adds support for:
    - smplx_pose_body: [total_frames, 126] - SMPL-X body pose flattened
    - robot_keypoints_trans: [total_frames, 15] - Robot keypoints translation flattened
    - robot_keypoints_rot: [total_frames, 30] - Robot keypoints rotation in 6D flattened
    
    Note: Extended keys are already flattened in Unify_Motion_Dataset.__getitem__()
    
    Args:
        dataset: Unify_Motion_Dataset instance (paired robot + SMPL-X)
        body_indexes: Indices for filtering robot body data
        device: Device to load tensors on
        
    Example:
        >>> dataset = Unify_Motion_Dataset(robot_map, smplx_map, robot_name="g1")
        >>> dataloader = Unify_Motion_Dataloader(dataset, body_indexes=[0,1,2,...], device="cuda")
        >>> 
        >>> # Use inherited sampling method
        >>> indices = dataloader.sample(n=32)
        >>> time_steps = torch.tensor([10, 20, 15, ...], device="cuda")
        >>> global_indices = dataloader.motion_offsets[indices] + time_steps
        >>> 
        >>> # Access all data including extended keys
        >>> joint_pos = dataloader.motion_buffer.joint_pos[global_indices]
        >>> smplx_data = dataloader.motion_buffer.smplx_pose_body[global_indices]
        >>> keypoint_trans = dataloader.motion_buffer.robot_keypoints_trans[global_indices]
    """
    
    class UnifyMotionBuffer(Motion_Dataloader.MotionBuffer):
        """Extended MotionBuffer with SMPL-X and extended data support.
        
        Inherits all robot motion buffers from parent and adds:
        - smplx_pose_body: SMPL-X body pose data
        - robot_keypoints_trans: Robot keypoints translation
        - robot_keypoints_rot: Robot keypoints rotation in 6D
        """
        
        def __init__(self, body_indexes: Sequence[int]):
            super().__init__(body_indexes)
            self._smplx_pose_body = None
            self._robot_keypoints_trans = None
            self._robot_keypoints_rot = None
        
        @property
        def smplx_pose_body(self) -> torch.Tensor | None:
            """SMPL-X body pose in 6D representation."""
            return self._smplx_pose_body
        
        @property
        def robot_keypoints_trans(self) -> torch.Tensor | None:
            """Robot keypoints SE3 translation data."""
            return self._robot_keypoints_trans
        
        @property
        def robot_keypoints_rot(self) -> torch.Tensor | None:
            """Robot keypoints SE3 rotation in 6D representation."""
            return self._robot_keypoints_rot
    
    
    def __init__(
        self,
        dataset: Unify_Motion_Dataset,
        body_indexes: Sequence[int],
        device: str = "cuda",
    ):
        """Initialize Unify_Motion_Dataloader.
        
        Args:
            dataset: Unify_Motion_Dataset instance with paired data
            body_indexes: Sequence of body indices for filtering
            device: Device to load tensors on
        """
        # Set the dataset before calling parent __init__
        self.dataset = dataset
        self.device = device
        self.num_motions = len(dataset)
        self._body_indexes = body_indexes
        
        # Use extended buffer class
        self.motion_buffer = self.UnifyMotionBuffer(self._body_indexes)
        
        # Initialize metadata (will be populated in _preload_and_concatenate)
        self.motion_lengths: torch.Tensor = None
        self.motion_offsets: torch.Tensor = None
        self.motion_fps: torch.Tensor = None
        self.time_step_total: int = 0
        
        print(f"[Unify_Motion_Dataloader] Loading and concatenating {self.num_motions} paired motions...")
        
        # Load all motions and concatenate
        self._preload_and_concatenate()
        
        print(f"[Unify_Motion_Dataloader] Initialization complete. Total frames: {self.time_step_total}")
    
    
    def _preload_and_concatenate(self) -> None:
        """Preload all paired motions and concatenate with extended data handling.
        
        Extends parent method to also handle SMPL-X and extended robot data.
        """
        # Base data lists (inherited from parent)
        data_lists = {
            'joint_pos': [],
            'joint_vel': [],
            'body_pos_w': [],
            'body_quat_w': [],
            'body_lin_vel_w': [],
            'body_ang_vel_w': [],
        }
        
        # Extended data lists for SMPL-X and robot keypoints
        # Note: These are already flattened by Unify_Motion_Dataset
        extended_lists = {
            'smplx_pose_body': [],
            'robot_keypoints_trans': [],
            'robot_keypoints_rot': [],
        }
        
        lengths = []
        fps_list = []
        
        # Load all motion pairs
        for i in range(self.num_motions):
            item = self.dataset[i]
            robot_motion = item["motion"]
            robot_len = item["length"]
            robot_item = item
            
            # Load base data (reuse from Motion_Dataset interface)
            for key in data_lists.keys():
                data_lists[key].append(
                    torch.tensor(robot_motion[key], dtype=torch.float32, device=self.device)
                )
            
            # Load extended data (must exist from extend_datasets.py preprocessing)
            # Note: Data is already flattened by Unify_Motion_Dataset.__getitem__()
            for ext_key in extended_lists.keys():
                extended_lists[ext_key].append(
                    torch.tensor(robot_motion[ext_key], dtype=torch.float32, device=self.device)
                )
            
            lengths.append(robot_len)
            fps_list.append(robot_item["fps"])
        
        # Concatenate base data (robot motion)
        self.motion_buffer.joint_pos = torch.cat(data_lists['joint_pos'], dim=0)
        self.motion_buffer.joint_vel = torch.cat(data_lists['joint_vel'], dim=0)
        self.motion_buffer._body_pos_w = torch.cat(data_lists['body_pos_w'], dim=0)
        self.motion_buffer._body_quat_w = torch.cat(data_lists['body_quat_w'], dim=0)
        self.motion_buffer._body_lin_vel_w = torch.cat(data_lists['body_lin_vel_w'], dim=0)
        self.motion_buffer._body_ang_vel_w = torch.cat(data_lists['body_ang_vel_w'], dim=0)
        
        # Concatenate extended data (must exist from extend_datasets.py preprocessing)
        self.motion_buffer._smplx_pose_body = torch.cat(extended_lists['smplx_pose_body'], dim=0)
        self.motion_buffer._robot_keypoints_trans = torch.cat(extended_lists['robot_keypoints_trans'], dim=0)
        self.motion_buffer._robot_keypoints_rot = torch.cat(extended_lists['robot_keypoints_rot'], dim=0)
        
        # Compute motion metadata
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_offsets = torch.cat([
            torch.tensor([0], dtype=torch.long, device=self.device),
            torch.cumsum(self.motion_lengths, dim=0)[:-1]
        ], dim=0)
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)
        self.time_step_total = self.motion_buffer.joint_pos.shape[0]
        
        # Print buffer info
        print(f"[Unify_Motion_Dataloader] Concatenated tensors:")
        print(f"  joint_pos: {self.motion_buffer.joint_pos.shape}")
        print(f"  joint_vel: {self.motion_buffer.joint_vel.shape}")
        print(f"  body_pos_w: {self.motion_buffer.body_pos_w.shape}")
        print(f"  smplx_pose_body: {self.motion_buffer._smplx_pose_body.shape}")
        print(f"  robot_keypoints_trans: {self.motion_buffer._robot_keypoints_trans.shape}")
        print(f"  robot_keypoints_rot: {self.motion_buffer._robot_keypoints_rot.shape}")
        print(f"  total_frames: {self.time_step_total}")
        print(f"  motion_lengths: {self.motion_lengths.shape}, range: [{self.motion_lengths.min()}, {self.motion_lengths.max()}]")
        print(f"  motion_offsets: {self.motion_offsets.shape}")
