import torch

from .mathutils import *

# SMPL-X kinematic tree
PARENTS = [
    -1,   # 0: pelvis (root, no parent)
    0,    # 1: left hip
    0,    # 2: right hip
    0,    # 3: spine1
    1,    # 4: left knee
    2,    # 5: right knee
    3,    # 6: spine2
    4,    # 7: left ankle
    5,    # 8: right ankle
    6,    # 9: spine3
    7,    # 10: left foot
    8,    # 11: right foot
    9,    # 12: neck
    9,    # 13: left collar
    9,    # 14: right collar
    12,   # 15: head
    13,   # 16: left shoulder
    14,   # 17: right shoulder
    16,   # 18: left elbow
    17,   # 19: right elbow
    18,   # 20: left wrist
    19,   # 21: right wrist
]

def axis_angle_to_matrix(axis_angles: torch.Tensor) -> torch.Tensor:
    # compute rotation matrix from axis-angle
    angle = torch.norm(axis_angles, dim=-1, keepdim=False)  # (N,)
    axis = axis_angles / (angle.unsqueeze(-1) + 1e-8)  # (N, 3)
    
    quat = quat_from_angle_axis(angle, axis)  # (N, 4)
    rot_mats = matrix_from_quat(quat)  # (N, 3, 3)
    return rot_mats


def rotation_matrix_to_6d(rotation_matrices: torch.Tensor) -> torch.Tensor:
    # Extract first 2 columns and flatten
    shape = rotation_matrices.shape[:-2]
    rot_6d = rotation_matrices[..., :2].reshape(*shape, 6)
    return rot_6d


def local_smplx_axis_angle_6d(axis_angles: torch.Tensor) -> torch.Tensor:
    N, J, _ = axis_angles.shape
    rot_mats = axis_angle_to_matrix(axis_angles.reshape(-1, 3))  # (N*21, 3, 3)
    rot_6d = rotation_matrix_to_6d(rot_mats)  # (N*21, 6)
    rot_6d = rot_6d.reshape(N, J, 6)  # (N, 21, 6)
    return rot_6d

def global_smplx_axis_angle_to_matrix(axis_angles: torch.Tensor) -> torch.Tensor:
    N, J, _ = axis_angles.shape
    
    # Convert axis-angle to rotation matrices
    rot_mats = axis_angle_to_matrix(axis_angles.reshape(-1, 3))  # (N*21, 3, 3)
    rot_mats = rot_mats.reshape(N, J, 3, 3)  # (N, 21, 3, 3)
    
    # Initialize global rotation matrices with identity for pelvis
    device = rot_mats.device
    global_rots = [torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)]  # List with (N, 3, 3)
    
    # Forward kinematics: compute global rotations
    for i in range(1, 22):  # joints 1~21
        parent_idx = PARENTS[i]
        local_R = rot_mats[:, i - 1]  # (N, 3, 3)
        parent_R = global_rots[parent_idx]  # (N, 3, 3)
        global_R = torch.bmm(parent_R, local_R)  # (N, 3, 3)
        global_rots.append(global_R)
    
    # Stack global rotations
    global_rots = torch.stack(global_rots, dim=1) # (N, 22, 3, 3)
    return global_rots[:, 1:]  # (N, 21, 3, 3)

    
def global_smplx_axis_angle_6d(axis_angles: torch.Tensor) -> torch.Tensor:
    rot_mats = global_smplx_axis_angle_to_matrix(axis_angles)  # (N, 21, 3, 3)
    N, J, _, _ = rot_mats.shape
    rot_mats_flat = rot_mats.reshape(-1, 3, 3)  # (N*21, 3, 3)
    rot_6d = rotation_matrix_to_6d(rot_mats_flat)  # (N*21, 6)
    rot_6d = rot_6d.reshape(N, J, 6)  # (N, 21, 6)
    return rot_6d


