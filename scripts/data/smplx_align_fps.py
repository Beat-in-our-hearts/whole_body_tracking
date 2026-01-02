#!/usr/bin/env python3
"""
SMPLX Frame Rate Alignment Script
Supports frame interpolation, frame downsampling, and video frame rate synchronization
"""

import numpy as np
import argparse
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SMPLXFrameAligner:
    """SMPLX Frame Rate Alignment Processor Class"""
    
    # Temporal sequence parameters that need to be processed
    TEMPORAL_PARAMS = [
        'root_orient', 'trans', 'poses', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye'
    ]
    
    # Metadata parameters that should NOT be modified (except mocap_frame_rate)
    METADATA_PARAMS = [
        'gender', 'surface_model_type', 'betas'
    ]
    
    def __init__(self, interpolation_method: str = 'cubic'):
        """
        Initialize the frame aligner.
        
        Args:
            interpolation_method: Interpolation method ('linear' or 'cubic')
        """
        self.interpolation_method = interpolation_method
        if interpolation_method not in ['linear', 'cubic']:
            raise ValueError("interpolation_method must be 'linear' or 'cubic'")
    
    def load_smplx_data(self, npz_path: str) -> Dict[str, np.ndarray]:
        """
        Load SMPLX .npz file.
        
        Args:
            npz_path: Path to .npz file
            
        Returns:
            Dictionary containing all SMPLX parameters
        """
        data = np.load(npz_path, allow_pickle=True)
        smplx_dict = {key: data[key] for key in data.files}
        logger.info(f"Loaded SMPLX data: {npz_path}")
        logger.info(f"Parameters: {list(smplx_dict.keys())}")
        
        # Show parameter shapes for debugging
        shapes_info = ", ".join([f"{k}:{v.shape}" if isinstance(v, np.ndarray) else f"{k}:{v}" 
                                for k, v in smplx_dict.items()])
        logger.debug(f"Shapes: {shapes_info}")
        
        # Validate required parameters
        self._validate_smplx_data(smplx_dict)
        
        return smplx_dict
    
    def _validate_smplx_data(self, smplx_data: Dict) -> None:
        """
        Validate that the data contains all required parameters.
        
        Args:
            smplx_data: Dictionary of SMPLX parameters
            
        Raises:
            ValueError: If required parameters are missing
        """
        required_params = set(self.TEMPORAL_PARAMS + self.METADATA_PARAMS + ['mocap_frame_rate'])
        available_params = set(smplx_data.keys())
        
        missing_params = required_params - available_params
        if missing_params:
            logger.warning(f"Missing parameters: {missing_params}")
    
    def save_smplx_data(self, smplx_data: Dict[str, np.ndarray], output_path: str) -> None:
        """
        Save SMPLX data as .npz file.
        Flattens all temporal sequence data from (N, D1, D2, ...) to (N, D) format.
        
        Args:
            smplx_data: Dictionary of SMPLX parameters
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten temporal sequence data while preserving metadata
        flattened_data = {}
        for key, value in smplx_data.items():
            if isinstance(value, np.ndarray):
                # Temporal sequence data: flatten from (N, D1, D2, ...) to (N, D)
                if value.ndim >= 2:
                    # Get the number of frames (first dimension)
                    num_frames = value.shape[0]
                    # Reshape to (num_frames, -1)
                    flattened_data[key] = value.reshape(num_frames, -1)
                    if value.shape != flattened_data[key].shape:
                        logger.debug(f"Flattened {key}: {value.shape} -> {flattened_data[key].shape}")
                    else:
                        flattened_data[key] = value
                else:
                    # Scalar or 1D data, keep as-is
                    flattened_data[key] = value
            else:
                # Non-array data (scalars), keep as-is
                flattened_data[key] = value
        
        np.savez(output_path, **flattened_data)
        logger.info(f"Saved processed data: {output_path}")
    
    def _get_num_frames(self, smplx_data: Dict[str, np.ndarray]) -> int:
        """Get the total number of frames from temporal data."""
        # Use first available temporal parameter to determine frame count
        for param_name in self.TEMPORAL_PARAMS:
            if param_name in smplx_data:
                data = smplx_data[param_name]
                if isinstance(data, np.ndarray) and data.ndim > 0:
                    return data.shape[0]
        
        raise ValueError(f"Cannot determine number of frames. Temporal parameters: {self.TEMPORAL_PARAMS}")
    
    def _has_temporal_dimension(self, data: np.ndarray, num_frames: int) -> bool:
        """Check if data has temporal dimension."""
        return data.shape[0] == num_frames
    
    def _quaternion_slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation between two quaternions (wxyz format).
        
        Args:
            q1: Quaternion 1 in wxyz format (4,)
            q2: Quaternion 2 in wxyz format (4,)
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated quaternion in wxyz format (4,)
        """
        # Ensure quaternions are normalized
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute dot product
        dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
        
        # If dot product is negative, negate one quaternion to take shorter path
        if dot_product < 0.0:
            q2 = -q2
            dot_product = -dot_product
        
        # Clamp dot product to avoid numerical issues with acos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Compute angle between quaternions
        theta = np.arccos(dot_product)
        
        # If angle is very small, use linear interpolation
        if theta < 1e-6:
            return q1 * (1 - t) + q2 * t
        
        # SLERP formula
        sin_theta = np.sin(theta)
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        return w1 * q1 + w2 * q2
    
    def _interpolate_rotations(self, rotations: np.ndarray, x_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
        """
        Interpolate rotations using spherical linear interpolation (SLERP).
        Handles multi-dimensional rotation arrays by processing each rotation separately.
        
        Args:
            rotations: Array of shape (num_frames, 3), (num_frames, 1, 3), (num_frames, n_joints, 3),
                      or (num_frames, 6) for flattened pose_eye containing rotation vectors
            x_old: Original time indices [0, 1]
            x_new: Target time indices [0, 1]
            
        Returns:
            Interpolated rotations with same shape as input but with len(x_new) frames
        """
        try:
            # Handle different input shapes
            if rotations.ndim == 2 and rotations.shape[1] == 3:
                # Shape: (num_frames, 3) - single rotation
                rot_objs = Rotation.from_rotvec(rotations)
                slerp = Slerp(x_old, rot_objs)
                interpolated_rots = slerp(x_new)
                return interpolated_rots.as_rotvec()
            
            elif rotations.ndim == 2 and rotations.shape[1] in [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99]:
                # Shape: (num_frames, n*3) - flattened multiple rotations (e.g., pose_eye: 6 = 2*3)
                num_frames, total_dims = rotations.shape
                n_joints = total_dims // 3
                num_new_frames = len(x_new)
                interpolated = np.zeros((num_new_frames, total_dims))
                
                # Process each joint separately
                for j in range(n_joints):
                    start_idx = j * 3
                    end_idx = start_idx + 3
                    joint_rotations = rotations[:, start_idx:end_idx]  # (num_frames, 3)
                    rot_objs = Rotation.from_rotvec(joint_rotations)
                    slerp = Slerp(x_old, rot_objs)
                    interpolated[:, start_idx:end_idx] = slerp(x_new).as_rotvec()
                
                return interpolated
            
            elif rotations.ndim == 3 and rotations.shape[2] == 3:
                # Shape: (num_frames, n_joints, 3) - multiple rotations per frame
                num_frames, n_joints, _ = rotations.shape
                num_new_frames = len(x_new)
                interpolated = np.zeros((num_new_frames, n_joints, 3))
                
                # Process each joint separately
                for j in range(n_joints):
                    joint_rotations = rotations[:, j, :]  # (num_frames, 3)
                    rot_objs = Rotation.from_rotvec(joint_rotations)
                    slerp = Slerp(x_old, rot_objs)
                    interpolated[:, j, :] = slerp(x_new).as_rotvec()
                
                return interpolated
            
            elif rotations.ndim == 4 and rotations.shape[3] == 3:
                # Shape: (num_frames, n_groups, n_joints, 3) - e.g., hand poses (30, 2, 3)
                num_frames, n_groups, n_joints, _ = rotations.shape
                num_new_frames = len(x_new)
                interpolated = np.zeros((num_new_frames, n_groups, n_joints, 3))
                
                # Process each rotation separately
                for g in range(n_groups):
                    for j in range(n_joints):
                        joint_rotations = rotations[:, g, j, :]  # (num_frames, 3)
                        rot_objs = Rotation.from_rotvec(joint_rotations)
                        slerp = Slerp(x_old, rot_objs)
                        interpolated[:, g, j, :] = slerp(x_new).as_rotvec()
                
                return interpolated
            
            else:
                # Unsupported shape, fallback to linear
                logger.warning(f"Unsupported rotation shape: {rotations.shape}, using linear interpolation")
                f = interp1d(x_old, rotations, axis=0, kind='linear', fill_value='extrapolate')
                return f(x_new)
        
        except Exception as e:
            logger.warning(f"SLERP interpolation failed: {e}, falling back to linear interpolation")
            f = interp1d(x_old, rotations, axis=0, kind='linear', fill_value='extrapolate')
            return f(x_new)
    
    def _reconstruct_poses(self, original_data: Dict[str, np.ndarray], 
                          processed_data: Dict[str, np.ndarray],
                          original_frames: int, new_frames: int) -> None:
        """
        Reconstruct 'poses' parameter from interpolated components.
        
        Handles different poses shapes:
        - (N, 165): Flat SMPLX format
        - (N, 55, 3): SMPLX multi-dimensional format with 55 joints
        
        Args:
            original_data: Original SMPLX data dictionary
            processed_data: Processed data dictionary (modified in place)
            original_frames: Number of original frames
            new_frames: Number of new frames after interpolation
        """
        original_poses = original_data['poses']
        
        # Check if poses is multi-dimensional (N, 55, 3) format
        if original_poses.ndim == 3 and original_poses.shape[2] == 3:
            # Multi-dimensional format: reconstruct from individual pose components
            num_joints = original_poses.shape[1]  # Usually 55 for SMPLX
            
            # If we have individual pose parameters, reconstruct from them
            if 'root_orient' in processed_data and 'pose_body' in processed_data:
                reconstructed_poses = np.zeros((new_frames, num_joints, 3))
                
                # Index mapping for standard SMPLX (55 joints total)
                # root_orient (1), pose_body (21), pose_hand (30), pose_jaw (1), pose_eye (2)
                idx = 0
                
                # 1. root_orient (1 joint)
                if 'root_orient' in processed_data:
                    root_orient = processed_data['root_orient']
                    # Handle both (N, 3) and (N, 1, 3) shapes
                    if root_orient.ndim == 3 and root_orient.shape[1] == 1:
                        reconstructed_poses[:, idx:idx+1, :] = root_orient
                    elif root_orient.ndim == 2:
                        reconstructed_poses[:, idx:idx+1, :] = root_orient.reshape(new_frames, 1, 3)
                    else:
                        reconstructed_poses[:, idx:idx+1, :] = root_orient.reshape(new_frames, 1, 3)
                    idx += 1
                
                # 2. pose_body (21 joints)
                if 'pose_body' in processed_data:
                    pose_body = processed_data['pose_body']
                    if pose_body.ndim == 3:
                        reconstructed_poses[:, idx:idx+pose_body.shape[1], :] = pose_body
                    idx += 21
                
                # 3. pose_hand (30 joints) - might be stored as (N, 30, 2, 3) or flattened
                if 'pose_hand' in processed_data:
                    pose_hand = processed_data['pose_hand']
                    if pose_hand.ndim == 3 and pose_hand.shape[1] == 30:
                        # Shape (N, 30, 3)
                        reconstructed_poses[:, idx:idx+30, :] = pose_hand
                    elif pose_hand.ndim == 4 and pose_hand.shape[1] == 2 and pose_hand.shape[2] == 15:
                        # Shape (N, 2, 15, 3) - left and right hands with 15 joints each
                        reconstructed_poses[:, idx:idx+15, :] = pose_hand[:, 0, :, :]
                        reconstructed_poses[:, idx+15:idx+30, :] = pose_hand[:, 1, :, :]
                    idx += 30
                
                # 4. pose_jaw (1 joint)
                if 'pose_jaw' in processed_data:
                    pose_jaw = processed_data['pose_jaw']
                    # Handle both (N, 3) and (N, 1, 3) shapes
                    if pose_jaw.ndim == 3 and pose_jaw.shape[1] == 1:
                        reconstructed_poses[:, idx:idx+1, :] = pose_jaw
                    elif pose_jaw.ndim == 2:
                        reconstructed_poses[:, idx:idx+1, :] = pose_jaw.reshape(new_frames, 1, 3)
                    else:
                        reconstructed_poses[:, idx:idx+1, :] = pose_jaw.reshape(new_frames, 1, 3)
                    idx += 1
                
                # 5. pose_eye (2 joints)
                if 'pose_eye' in processed_data:
                    pose_eye = processed_data['pose_eye']
                    if pose_eye.ndim == 3:  # (N, 2, 3)
                        reconstructed_poses[:, idx:idx+pose_eye.shape[1], :] = pose_eye
                    elif pose_eye.ndim == 2:  # (N, 6) - flattened
                        reconstructed_poses[:, idx:idx+2, :] = pose_eye.reshape(new_frames, 2, 3)
                    idx += 2
                
                processed_data['poses'] = reconstructed_poses
                logger.debug(f"Reconstructed poses from components: {original_poses.shape} -> {reconstructed_poses.shape}")
            else:
                # Fallback: interpolate poses directly
                logger.info("Fallback: interpolating poses directly")
                x_old = np.linspace(0, 1, original_frames)
                x_new = np.linspace(0, 1, new_frames)
                
                # Flatten, interpolate, then reshape
                flat_poses = original_poses.reshape(original_frames, -1)
                f = interp1d(x_old, flat_poses, axis=0, kind=self.interpolation_method, fill_value='extrapolate')
                interpolated_flat = f(x_new)
                processed_data['poses'] = interpolated_flat.reshape(new_frames, *original_poses.shape[1:])
        
        elif original_poses.ndim == 2 and original_poses.shape[1] == 165:
            # Flat SMPLX format: reconstruct from individual components
            reconstructed_poses = np.zeros((new_frames, 165))
            
            root_orient_idx = (0, 3)
            pose_body_idx = (3, 66)
            pose_hand_idx = (66, 156)
            pose_jaw_idx = (156, 159)
            pose_eye_idx = (159, 165)
            
            if 'root_orient' in processed_data:
                root_orient = processed_data['root_orient']
                if root_orient.ndim == 3:
                    root_orient = root_orient.squeeze(axis=1)
                reconstructed_poses[:, root_orient_idx[0]:root_orient_idx[1]] = root_orient
            
            if 'pose_body' in processed_data:
                pose_body = processed_data['pose_body']
                if pose_body.ndim == 3:
                    pose_body = pose_body.reshape(new_frames, -1)
                reconstructed_poses[:, pose_body_idx[0]:pose_body_idx[1]] = pose_body
            
            if 'pose_hand' in processed_data:
                pose_hand = processed_data['pose_hand']
                if pose_hand.ndim > 2:
                    pose_hand = pose_hand.reshape(new_frames, -1)
                reconstructed_poses[:, pose_hand_idx[0]:pose_hand_idx[1]] = pose_hand
            
            if 'pose_jaw' in processed_data:
                pose_jaw = processed_data['pose_jaw']
                if pose_jaw.ndim == 3:
                    pose_jaw = pose_jaw.squeeze(axis=1)
                reconstructed_poses[:, pose_jaw_idx[0]:pose_jaw_idx[1]] = pose_jaw
            
            if 'pose_eye' in processed_data:
                pose_eye = processed_data['pose_eye']
                if pose_eye.ndim == 3:
                    pose_eye = pose_eye.reshape(new_frames, -1)
                reconstructed_poses[:, pose_eye_idx[0]:pose_eye_idx[1]] = pose_eye
            
            processed_data['poses'] = reconstructed_poses
            logger.debug(f"Reconstructed poses from components: {original_poses.shape} -> {reconstructed_poses.shape}")
        
        else:
            # Unknown format, try direct interpolation
            logger.warning(f"Unknown poses format: {original_poses.shape}, using direct interpolation")
            x_old = np.linspace(0, 1, original_frames)
            x_new = np.linspace(0, 1, new_frames)
            
            flat_poses = original_poses.reshape(original_frames, -1)
            f = interp1d(x_old, flat_poses, axis=0, kind=self.interpolation_method, fill_value='extrapolate')
            interpolated_flat = f(x_new)
            processed_data['poses'] = interpolated_flat.reshape(new_frames, *original_poses.shape[1:])
    
    def interpolate(self, smplx_data: Dict[str, np.ndarray], 
                   original_fps: float, target_fps: float) -> Dict[str, np.ndarray]:
        """
        Interpolate temporal SMPLX data to increase frame rate.
        
        Args:
            smplx_data: Dictionary of SMPLX parameters
            original_fps: Original frame rate
            target_fps: Target frame rate
            
        Returns:
            Dictionary with interpolated temporal data and unchanged metadata
        """
        if target_fps <= original_fps:
            logger.warning(f"Target fps {target_fps} is less than or equal to original fps {original_fps}, no interpolation needed")
            return smplx_data
        
        try:
            num_frames = self._get_num_frames(smplx_data)
        except ValueError as e:
            logger.error(f"Cannot interpolate: {e}")
            return smplx_data
        
        scale = target_fps / original_fps
        new_num_frames = int(np.round(num_frames * scale))
        
        logger.info(f"Interpolating: {original_fps}fps -> {target_fps}fps")
        logger.info(f"Frame count: {num_frames} -> {new_num_frames}")
        
        # Time indices for original and target
        x_old = np.linspace(0, 1, num_frames)
        x_new = np.linspace(0, 1, new_num_frames)
        
        processed_data = {}
        
        # Copy metadata (unchanged)
        for param in self.METADATA_PARAMS:
            if param in smplx_data:
                processed_data[param] = smplx_data[param]
                logger.debug(f"Copied metadata: {param}")
        
        # Update mocap_frame_rate
        processed_data['mocap_frame_rate'] = target_fps
        logger.info(f"Updated mocap_frame_rate: {original_fps} -> {target_fps}")
        
        # Interpolate temporal parameters
        for param in self.TEMPORAL_PARAMS:
            if param not in smplx_data:
                logger.debug(f"Skipped missing temporal parameter: {param}")
                continue
            
            # Skip 'poses' as it's reconstructed from other interpolated components
            if param == 'poses':
                logger.debug(f"Skipped {param} (will be reconstructed from interpolated components)")
                continue
            
            value = smplx_data[param]
            
            # Preserve original shape for later
            original_shape = value.shape
            
            # Special handling: don't reshape multi-dimensional rotation data
            # Let _interpolate_rotations handle it directly
            if param in ['root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']:
                # Rotation data - keep original shape for SLERP
                if value.ndim > 1:
                    # Multi-dimensional, pass as-is to SLERP handler
                    pass
                elif value.ndim == 1:
                    value = value.reshape(-1, 1)
            else:
                # Position data - standard reshaping
                if value.ndim == 1:
                    value = value.reshape(-1, 1)
                elif value.ndim > 4:
                    value = value.reshape(num_frames, -1)
            
            try:
                # Use SLERP for all rotation data, linear for position data
                if param in ['root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']:
                    # All pose parameters are rotations (axis-angle format) - use SLERP
                    result = self._interpolate_rotations(value, x_old, x_new)
                else:
                    # Position data (trans) - use linear or cubic interpolation
                    f = interp1d(x_old, value, axis=0, kind=self.interpolation_method, 
                               fill_value='extrapolate')
                    result = f(x_new)
                
                # Reshape back to original shape pattern (only if we flattened it)
                if len(original_shape) == 1 and param not in ['root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']:
                    result = result.squeeze(axis=1) if result.ndim > 1 else result
                elif len(original_shape) > 2 and param not in ['root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']:
                    result = result.reshape((new_num_frames,) + original_shape[1:])
                
                processed_data[param] = result
                logger.debug(f"Interpolated parameter: {param}, shape: {original_shape} -> {result.shape}")
            except Exception as e:
                logger.warning(f"Failed to interpolate parameter {param}: {e}, replicating first frame")
                # Replicate first frame as fallback
                processed_data[param] = np.repeat(value[0:1], new_num_frames, axis=0).reshape(
                    (new_num_frames,) + original_shape[1:] if len(original_shape) > 1 else (new_num_frames,)
                )
        
        # Reconstruct 'poses' from interpolated components
        if 'poses' in smplx_data:
            self._reconstruct_poses(smplx_data, processed_data, num_frames, new_num_frames)
        
        return processed_data
    
    def downsample(self, smplx_data: Dict[str, np.ndarray], 
                  original_fps: float, target_fps: float) -> Dict[str, np.ndarray]:
        """
        Downsample temporal SMPLX data to decrease frame rate.
        
        Args:
            smplx_data: Dictionary of SMPLX parameters
            original_fps: Original frame rate
            target_fps: Target frame rate
            
        Returns:
            Dictionary with downsampled temporal data and unchanged metadata
        """
        if target_fps >= original_fps:
            logger.warning(f"Target fps {target_fps} is greater than or equal to original fps {original_fps}, no downsampling needed")
            return smplx_data
        
        try:
            num_frames = self._get_num_frames(smplx_data)
        except ValueError as e:
            logger.error(f"Cannot downsample: {e}")
            return smplx_data
        
        step = int(np.round(original_fps / target_fps))
        new_num_frames = num_frames // step
        
        logger.info(f"Downsampling: {original_fps}fps -> {target_fps}fps")
        logger.info(f"Frame count: {num_frames} -> {new_num_frames} (sampling interval: {step})")
        
        processed_data = {}
        
        # Copy metadata (unchanged)
        for param in self.METADATA_PARAMS:
            if param in smplx_data:
                processed_data[param] = smplx_data[param]
                logger.debug(f"Copied metadata: {param}")
        
        # Update mocap_frame_rate
        processed_data['mocap_frame_rate'] = target_fps
        logger.info(f"Updated mocap_frame_rate: {original_fps} -> {target_fps}")
        
        # Downsample temporal parameters
        for param in self.TEMPORAL_PARAMS:
            if param not in smplx_data:
                logger.debug(f"Skipped missing temporal parameter: {param}")
                continue
            
            # Skip 'poses' as it's reconstructed from other downsampled components
            if param == 'poses':
                logger.debug(f"Skipped {param} (will be reconstructed from downsampled components)")
                continue
            
            value = smplx_data[param]
            downsampled_value = value[::step]
            processed_data[param] = downsampled_value
            logger.debug(f"Downsampled parameter: {param}, shape: {value.shape} -> {downsampled_value.shape}")
        
        # Reconstruct 'poses' from downsampled components
        if 'poses' in smplx_data:
            self._reconstruct_poses(smplx_data, processed_data, num_frames, new_num_frames)
        
        return processed_data
    
    def align_to_fps(self, smplx_data: Dict[str, np.ndarray], 
                    original_fps: float, target_fps: float) -> Dict[str, np.ndarray]:
        """
        Automatically align SMPLX data to target frame rate (intelligently choose interpolation or downsampling).
        
        Args:
            smplx_data: Dictionary of SMPLX parameters
            original_fps: Original frame rate
            target_fps: Target frame rate
            
        Returns:
            Dictionary of aligned SMPLX parameters
        """
        if abs(original_fps - target_fps) < 0.01:
            logger.info(f"Frame rate is already the same: {original_fps}fps")
            return smplx_data
        
        if target_fps > original_fps:
            return self.interpolate(smplx_data, original_fps, target_fps)
        else:
            return self.downsample(smplx_data, original_fps, target_fps)
    
    def get_sequence_info(self, smplx_data: Dict[str, np.ndarray]) -> Dict:
        """
        Get sequence information.
        
        Args:
            smplx_data: Dictionary of SMPLX parameters
            
        Returns:
            Dictionary containing sequence information
        """
        try:
            num_frames = self._get_num_frames(smplx_data)
        except ValueError:
            num_frames = 'Unknown'
        
        info = {
            'num_frames': num_frames,
            'parameters': list(smplx_data.keys()),
            'mocap_frame_rate': smplx_data.get('mocap_frame_rate', 'Unknown'),
            'shapes': {key: value.shape if isinstance(value, np.ndarray) else str(value) 
                      for key, value in smplx_data.items()}
        }
        return info


def process_single_file(input_path: str, output_path: str, 
                       original_fps: float, target_fps: float,
                       interpolation_method: str = 'cubic') -> None:
    """
    Process a single SMPLX file.
    
    Args:
        input_path: Path to input .npz file
        output_path: Path to output .npz file
        original_fps: Original frame rate
        target_fps: Target frame rate
        interpolation_method: Interpolation method
    """
    aligner = SMPLXFrameAligner(interpolation_method=interpolation_method)
    
    # Load data
    smplx_data = aligner.load_smplx_data(input_path)
    
    # Print sequence information
    info = aligner.get_sequence_info(smplx_data)
    logger.info(f"Sequence info: num_frames={info['num_frames']}, mocap_frame_rate={info['mocap_frame_rate']}")
    
    # Align frame rate
    aligned_data = aligner.align_to_fps(smplx_data, original_fps, target_fps)
    
    # Save result
    aligner.save_smplx_data(aligned_data, output_path)
    
    # Print processing result
    new_info = aligner.get_sequence_info(aligned_data)
    logger.info(f"Processing complete: {info['num_frames']} -> {new_info['num_frames']} frames, mocap_frame_rate={new_info['mocap_frame_rate']}")


def process_directory(input_dir: str, output_dir: str,
                     original_fps: float, target_fps: float,
                     interpolation_method: str = 'cubic',
                     file_pattern: str = '*.npz',
                     recursive: bool = False) -> None:
    """
    Batch process SMPLX files in a directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        original_fps: Original frame rate
        target_fps: Target frame rate
        interpolation_method: Interpolation method
        file_pattern: File matching pattern
        recursive: Whether to search subdirectories recursively
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use recursive glob pattern if recursive is True
    glob_pattern = f"**/{file_pattern}" if recursive else file_pattern
    npz_files = list(input_path.glob(glob_pattern))
    
    if not npz_files:
        search_mode = "recursively" if recursive else ""
        logger.warning(f"No {file_pattern} files found {search_mode} in {input_dir}")
        return
    
    logger.info(f"Found {len(npz_files)} files to process")
    if recursive:
        logger.info("Searching subdirectories recursively")
    
    success_count = 0
    failed_count = 0
    
    for i, npz_file in enumerate(npz_files, 1):
        logger.info(f"\nProcessing file [{i}/{len(npz_files)}]: {npz_file.name}")
        
        # Maintain directory structure if recursive
        if recursive:
            relative_path = npz_file.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_file = output_path / npz_file.name
        
        try:
            process_single_file(str(npz_file), str(output_file), 
                              original_fps, target_fps, interpolation_method)
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing file {npz_file.name}: {e}")
            failed_count += 1
            continue
    
    logger.info(f"\nBatch processing complete: {success_count} succeeded, {failed_count} failed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='SMPLX Frame Rate Alignment Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Interpolate single file (30fps -> 60fps)
  python smplx_align_fps.py -i motion.npz -o motion_60fps.npz --original-fps 30 --target-fps 60
  
  # Downsample single file (60fps -> 30fps)
  python smplx_align_fps.py -i motion.npz -o motion_30fps.npz --original-fps 60 --target-fps 30
  
  # Batch process directory (non-recursive)
  python smplx_align_fps.py -d input_dir/ -o output_dir/ --original-fps 30 --target-fps 60
  
  # Batch process directory recursively (search subdirectories)
  python smplx_align_fps.py -d input_dir/ -o output_dir/ --original-fps 30 --target-fps 60 --recursive
  
  # Use linear interpolation with recursive search
  python smplx_align_fps.py -i motion.npz -o motion_60fps.npz --original-fps 30 --target-fps 60 --method linear
  
  # Batch process with custom file pattern and recursive search
  python smplx_align_fps.py -d input_dir/ -o output_dir/ --original-fps 30 --target-fps 60 --pattern "*.npz" --recursive
        '''
    )
    
    # Input/Output arguments
    parser.add_argument('-i', '--input', type=str, default=None,
                       help='Path to input .npz file (for single file processing)')
    parser.add_argument('-d', '--directory', type=str, default=None,
                       help='Input directory path (for batch processing)')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output path (file path or directory path)')
    
    # Frame rate arguments
    parser.add_argument('--original-fps', type=float, required=True,
                       help='Original frame rate (e.g., 30, 60)')
    parser.add_argument('--target-fps', type=float, required=True,
                       help='Target frame rate (e.g., 30, 60, 120)')
    
    # Interpolation method argument
    parser.add_argument('--method', type=str, default='cubic',
                       choices=['linear', 'cubic'],
                       help='Interpolation method (default: cubic)')
    
    # Batch processing arguments
    parser.add_argument('--pattern', type=str, default='*.npz',
                       help='File matching pattern (for batch processing, default: *.npz)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Recursively search subdirectories for files')
    
    args = parser.parse_args()
    
    # Validate input
    if args.input is None and args.directory is None:
        parser.error('Must specify either -i (single file) or -d (directory)')
    
    if args.input is not None and args.directory is not None:
        parser.error('Cannot specify both -i and -d')
    
    if args.original_fps <= 0 or args.target_fps <= 0:
        parser.error('Frame rate must be greater than 0')
    
    # Process single file
    if args.input is not None:
        if not Path(args.input).exists():
            logger.error(f"Input file does not exist: {args.input}")
            return
        
        process_single_file(args.input, args.output, 
                          args.original_fps, args.target_fps, args.method)
    
    # Batch process directory
    else:
        if not Path(args.directory).exists():
            logger.error(f"Input directory does not exist: {args.directory}")
            return
        
        process_directory(args.directory, args.output,
                         args.original_fps, args.target_fps,
                         args.method, args.pattern, args.recursive)


if __name__ == '__main__':
    main()
