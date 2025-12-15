"""Unified motion dataset for loading extended motion data from NPZ files.

This module provides `Unify_Motion_Dataset` which extends Motion_Dataset to load
NPZ files that have been pre-processed by extend_datasets.py.

These extended NPZ files contain additional keys beyond the standard motion data:
- smplx_pose_body: (N, 21, 6) SMPL-X body pose in 6D rotation representation
- smplx_pose_body_global_rot: (N, 21, 6) SMPL-X body pose with global rotation
- robot_keypoints_trans: (N, 5, 3) robot keypoints translation relative to root
- robot_keypoints_rot: (N, 5, 6) robot keypoints rotation in 6D representation

The class simply extends Motion_Dataset's __getitem__ to expose these new keys
when available in the NPZ file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
from whole_body_tracking.utils.motion_dataset import Motion_Dataset


class Unify_Motion_Dataset(Motion_Dataset):
    """Dataset that loads extended motion data with SMPL-X and keypoint information.

    Extends Motion_Dataset to access additional keys in NPZ files that have been
    pre-processed by extend_datasets.py with SMPL-X data and robot keypoint SE3 data.

    Args:
        dataset_dirs: List of dataset directory paths
        robot_name: robot folder name
        splits: List of dataset splits corresponding to each dataset_dir
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        robot_name: str,
        splits: List[Union[str, List[str]]],
    ) -> None:
        # Simply call parent with same parameters
        super().__init__(
            dataset_dirs=dataset_dirs,
            robot_name=robot_name,
            splits=splits,
        )
        print(f"[Unify_Motion_Dataset] Extended motion dataset loaded with {len(self.npz_paths)} clips")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load extended motion data from NPZ file.
        
        Overrides parent to expose additional extended keys (smplx_pose_body,
        robot_keypoints_trans, etc.) when available in the NPZ file.
        """
        npz_path = self.npz_paths[idx]
        
        # Load motion data
        data = np.load(npz_path)
        
        # Extract standard motion data (same as parent)
        motion = {
            "joint_pos": data["joint_pos"],
            "joint_vel": data["joint_vel"],
            "body_pos_w": data["body_pos_w"],
            "body_quat_w": data["body_quat_w"],
            "body_lin_vel_w": data["body_lin_vel_w"],
            "body_ang_vel_w": data["body_ang_vel_w"],
        }
        
        # Add extended keys if available
        extended_keys = [
            "smplx_pose_body",
            "smplx_pose_body_global_rot",
            "robot_keypoints_trans",
            "robot_keypoints_rot",
        ]
        flatten_keys = set(extended_keys)
        
        for key in extended_keys:
            if key in data:
                arr = data[key]
                # Flatten last two dimensions for all extended keys
                if key in flatten_keys and arr.ndim >= 2:
                    arr = arr.reshape(arr.shape[0], -1)
                motion[key] = arr
        
        fps = int(data["fps"][0])
        length = motion["joint_pos"].shape[0]
        duration = length / fps
        
        return {
            "motion": motion,
            "fps": fps,
            "length": length,
            "duration": duration,
            "npz_path": str(npz_path),
            "motion_name": self.motion_names[idx],
            "quantity": self.quantities[idx],
            "dataset_source": self.dataset_sources[idx],
        }
