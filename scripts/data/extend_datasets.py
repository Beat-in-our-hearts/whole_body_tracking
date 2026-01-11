"""
    extend_datasets.py
    features functions to extend existing datasets by adding new data.
    
    arguments:
       - npz_dataset_path: Path to the existing .npz dataset file.
       - disable_smplx_6d_data: Boolean flag indicating whether to include SMPL-X 6D rotation representation data.
       - disable_pico_data: Boolean flag indicating whether to include keypoints SE3 data from npz files.
       - smplx_dataset_path: Path to the SMPL-X dataset file.
       - enable_smplx_global_rotation: Boolean flag whether to translate smplx body_pose to global rotation representation.
       - output_npz_datasets_path: Path to save the extended .npz dataset file.
       - robot_body_names_yaml: Path to the YAML file containing robot body names.
       - robot_root_body_name: Name of the robot's root body in the YAML file.
        
    new_npz_file keys:
         - `original_data_keys`: The original dataset content.
         - ...
         - smplx_pose_body: np.ndarray of shape (N, 6, 21) containing SMPL-X body pose in 6D rotation representation.
            here relative parent-child joint rotations.
         - smplx_pose_body_global_rot: np.ndarray of shape (N, 6, 21) containing SMPL-X body pose in 6D rotation representation including global root rotation.
            here relative the `pelvis` joint.
         - mocap_keypoints_rot: np.ndarray of shape (N, 5, 6) containing SE3 rotation data in 6D representation for keypoints from PICO dataset. 
            here relative to the root body.   
         - robot_keypoints_trans: np.ndarray of shape (N, 5, 3) containing SE3 translation data for robot keypoints.
            here relative to the robot root body.
         - robot_keypoints_rot: np.ndarray of shape (N, 5, 6) containing SE3 rotation data in 6D representation for robot keypoints.
            here relative to the robot root body.
    NOTE: 
    1. why 6d rotation representation?
        paper: https://arxiv.org/pdf/1812.07035 
        Title: On the Continuity of Rotation Representations in Neural Networks
            - 6D rotation representation is continuous and avoids singularities, making it suitable for neural
    2. why use `enable_smplx_global_rotation``
        smplx root rotation is important for capturing the overall orientation of the human body in 3D space.
        but smplx pose_body only contains relative rotations of body joints.
        need translate relative rotations to global rotations
"""

import numpy as np
import os
import argparse
import yaml
import torch
import sys

# Add scripts to path for importing utils package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.my_math import (
    local_smplx_axis_angle_6d,
    global_smplx_axis_angle_6d,
)

from utils.mathutils import (
    matrix_from_quat,
    subtract_frame_transforms,
)

arg_parser = argparse.ArgumentParser(description="Extend existing dataset with SMPL-X 6D data and PICO keypoints SE3 data.")
arg_parser.add_argument("--npz_dataset_path", type=str, required=True, help="Path to the existing .npz dataset file.")
arg_parser.add_argument("--smplx_dataset_path", type=str, required=False, help="Path to the SMPL-X dataset file.")
arg_parser.add_argument("--output_npz_datasets_path", type=str, required=True, help="Path to save the extended .npz dataset file.")

arg_parser.add_argument("--disable_smplx_6d_data", action="store_true", help="Flag to include SMPL-X 6D rotation representation data.")
arg_parser.add_argument("--disable_pico_data", action="store_true", help="Flag to include keypoints SE3 data from npz files.")
arg_parser.add_argument("--enable_smplx_global_rotation", action="store_true", help="Flag to include SMPL-X root rotation data.")

# arg_parser.add_argument("--robot_body_names_yaml", type=str, required=True, help="Path to the YAML file containing robot body names.")
# arg_parser.add_argument("--robot_root_body_name", type=str, default="pelvis", help="Name of the robot's root body in the YAML file.")
args = arg_parser.parse_args()

# robot_names_file = args.robot_body_names_yaml
# robot_names = yaml.safe_load(open(robot_names_file, 'r'))
# robot_body_names = robot_names['bodies']['names']
# robot_joint_names = robot_names['joints']['names']

def single_file_extend(
    npz_file_path: str,
    disable_smplx_6d_data: bool,
    disable_pico_data: bool,
    smplx_dataset_path: str = None,
    enable_smplx_global_rotation: bool = False,
    output_npz_datasets_path: str = None,
):
    """
    Extend a single NPZ dataset file with SMPL-X 6D data and robot keypoints SE3 data.
    
    Args:
        npz_file_path: Path to the existing .npz dataset file.
        disable_smplx_6d_data: Flag to include SMPL-X 6D rotation data.
        disable_pico_data: Flag to include robot keypoints SE3 data.
        smplx_dataset_path: Path to the SMPL-X dataset file.
        enable_smplx_global_rotation: Flag to include SMPL-X global rotation data.
        output_npz_datasets_path: Path to save the extended .npz dataset file.
    """
    # Load npz dataset
    print(f"Loading NPZ dataset from {npz_file_path}...")
    npz_data = np.load(npz_file_path, allow_pickle=True)
    extended_data = dict(npz_data)
    
    N = extended_data['body_pos_w'].shape[0]
    print(f"Dataset has {N} frames")
    
    # Process SMPL-X data
    if not disable_smplx_6d_data and smplx_dataset_path:
        print(f"Loading SMPL-X dataset from {smplx_dataset_path}...")
        smplx_data = np.load(smplx_dataset_path, allow_pickle=True)
        
        # Get pose_body and reshape from (N, 63) to (N, 21, 3)
        pose_body = smplx_data['pose_body']
        if pose_body.shape[0] >= N:
            pose_body = pose_body[:N]
        else: # padding using last frame
            last_frame = pose_body[-1:]
            num_padding = N - pose_body.shape[0]
            padding_frames = np.repeat(last_frame, num_padding, axis=0)
            pose_body = np.concatenate([pose_body, padding_frames], axis=0)
            
        pose_body = pose_body.reshape(-1, 21, 3)
        pose_body_torch = torch.from_numpy(pose_body).float()
        
        # Compute 6D rotation representation (local)
        print("Converting SMPL-X pose_body to 6D representation...")
        smplx_pose_body_6d = local_smplx_axis_angle_6d(pose_body_torch).cpu().numpy()
        extended_data['smplx_pose_body'] = smplx_pose_body_6d # (N, 21, 6)
        print(f"[INFO] SMPL-X local 6D data added to extended dataset: shape {smplx_pose_body_6d.shape}")
        
        # Compute global rotation if requested
        if enable_smplx_global_rotation:
            print("Computing SMPL-X global rotations...")
            smplx_pose_body_global_6d = global_smplx_axis_angle_6d(pose_body_torch).cpu().numpy()
            extended_data['smplx_pose_body_global_rot'] = smplx_pose_body_global_6d # (N, 21, 6)
            print(f"[INFO] SMPL-X global 6D data added to extended dataset: shape {smplx_pose_body_global_6d.shape}")
    
    # Process robot keypoints SE3 data
    if not disable_pico_data:
        print("Extracting robot keypoints SE3 data...")
        
        # Get body data from npz
        body_pos_w = extended_data['body_pos_w']  # (N, num_bodies, 3)
        body_quat_w = extended_data['body_quat_w']  # (N, num_bodies, 4)
        
        # Pelvis is body 0 (root)
        # 5 PICO keypoints: [body_9(torso_link), left_wrist_roll_link(24), right_wrist_roll_link(25), left_ankle_pitch_link(14), right_ankle_pitch_link(15)]
        keypoint_indices = [9, 24, 25, 14, 15]
        
        # Get pelvis SE3
        pelvis_pos = body_pos_w[:, 0, :]  # (N, 3)
        pelvis_quat = body_quat_w[:, 0, :]  # (N, 4)
        
        # Get keypoints positions and quaternions
        keypoint_pos = body_pos_w[:, keypoint_indices, :]  # (N, 5, 3)
        keypoint_quat = body_quat_w[:, keypoint_indices, :]  # (N, 5, 4)
        
        # Transform to pelvis frame using subtract_frame_transforms from mathutils
        # Expand pelvis to (N, 1, *) for broadcasting with keypoints (N, 5, *)
        t01_torch = torch.from_numpy(pelvis_pos).float()  # (N, 3)
        q01_torch = torch.from_numpy(pelvis_quat).float()  # (N, 4)
        t02_torch = torch.from_numpy(keypoint_pos).float()  # (N, 5, 3)
        q02_torch = torch.from_numpy(keypoint_quat).float()  # (N, 5, 4)
        
        pos_in_pelvis = torch.zeros_like(t02_torch)
        quat_in_pelvis = torch.zeros_like(q02_torch)
        for i in range(pos_in_pelvis.shape[1]):
            pos_in_pelvis[:, i, :], quat_in_pelvis[:, i, :] = subtract_frame_transforms(t01_torch, q01_torch, t02_torch[:, i, :], q02_torch[:, i, :])
        
        pos_in_pelvis = pos_in_pelvis.cpu().numpy()  # (N, 5, 3)
        quat_in_pelvis = quat_in_pelvis.cpu().numpy()  # (N, 5, 4)
        
        # Convert quaternions to 6D representation: quat → matrix_from_quat → extract first 2 columns
        quat_pelvis_torch = torch.from_numpy(quat_in_pelvis).float()  # (N, 5, 4)
        rot_mats = matrix_from_quat(quat_pelvis_torch)  # (N, 5, 3, 3)
        quat_in_pelvis_6d = rot_mats[..., :2].reshape(N, 5, 6).cpu().numpy()  # (N, 5, 6)
        
        extended_data['robot_keypoints_trans'] = pos_in_pelvis  # (N, 5, 3)
        extended_data['robot_keypoints_rot'] = quat_in_pelvis_6d  # (N, 5, 6)
        print(f"[INFO] Robot keypoints SE3 data added to extended dataset: shape {pos_in_pelvis.shape}, {quat_in_pelvis_6d.shape}")
    
    # Save extended dataset
    print(f"Saving extended dataset to {output_npz_datasets_path}...")
    np.savez(output_npz_datasets_path, **extended_data)
    print(f"Extended dataset saved successfully!")


def main():
    """Main function to process multiple datasets."""
    if os.path.isfile(args.npz_dataset_path):
        single_file_extend(
            npz_file_path=args.npz_dataset_path,
            disable_smplx_6d_data=args.disable_smplx_6d_data,
            disable_pico_data=args.disable_pico_data,
            smplx_dataset_path=args.smplx_dataset_path,
            enable_smplx_global_rotation=args.enable_smplx_global_rotation,
            output_npz_datasets_path=args.output_npz_datasets_path,
        )
    elif os.path.isdir(args.npz_dataset_path):
        # Process directory recursively
        npz_root = args.npz_dataset_path
        smplx_root = args.smplx_dataset_path
        output_root = args.output_npz_datasets_path
        
        # Create output root directory if it doesn't exist
        os.makedirs(output_root, exist_ok=True)
        
        print(f"Processing directory tree: {npz_root}")
        print(f"SMPL-X directory: {smplx_root}")
        print(f"Output directory: {output_root}")
        
        # Walk through the NPZ directory tree
        for root, dirs, files in os.walk(npz_root):
            for file in files:
                if file.endswith('.npz'):
                    npz_file_path = os.path.join(root, file)
                    
                    # Construct corresponding paths
                    rel_path = os.path.relpath(npz_file_path, npz_root)
                    smplx_file_path = os.path.join(smplx_root, rel_path) if smplx_root else None
                    output_file_path = os.path.join(output_root, rel_path)
                    
                    # Create output subdirectory if needed
                    output_dir = os.path.dirname(output_file_path)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Check if corresponding SMPLX file exists
                    if smplx_file_path and not os.path.exists(smplx_file_path):
                        print(f"[WARNING] SMPL-X file not found: {smplx_file_path}")
                        if args.disable_smplx_6d_data:
                            print(f"[INFO] Skipping SMPL-X processing for {npz_file_path}")
                            smplx_file_path = None
                        else:
                            print(f"[ERROR] SMPL-X file required but not found. Skipping {npz_file_path}")
                            continue
                    
                    print(f"\nProcessing: {rel_path}")
                    try:
                        single_file_extend(
                            npz_file_path=npz_file_path,
                            disable_smplx_6d_data=args.disable_smplx_6d_data,
                            disable_pico_data=args.disable_pico_data,
                            smplx_dataset_path=smplx_file_path,
                            enable_smplx_global_rotation=args.enable_smplx_global_rotation,
                            output_npz_datasets_path=output_file_path,
                        )
                    except Exception as e:
                        print(f"[ERROR] Failed to process {npz_file_path}: {e}")
                        continue
        
        print(f"\nDirectory processing completed!")


if __name__ == "__main__":
    main()
    
    