from isaaclab.utils import configclass
import os
from dataclasses import MISSING

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks import EXTEMDED_DATASETS_DIR

from whole_body_tracking.tasks.tracking.sonic_multi_tracking_env_cfg import TrackingEnvCfg as SONIC_MultiTrackingEnvCfg
from whole_body_tracking.tasks.tracking.mdp import SONIC_MultiMotionCommandCfg
from whole_body_tracking.tasks.tracking.sonic_multi_tracking_env_cfg import VELOCITY_RANGE, ObservationsCfgV2, ObservationsCfgV3

#######################################
# SONIC Multi Tracking Env with SMPLX observations
#######################################
class SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX(SONIC_MultiTrackingEnvCfg):
    
    SONIC_FLAG: bool = True
    
    def __post_init__(self):
        super().__post_init__()

        self.rewards.motion_global_anchor_pos.weight = 0.0
        
        # support smplx observations
        self.observations = ObservationsCfgV2()
        
        # Dataset paths (must be pre-processed by extend_datasets.py to include SMPL-X extended keys)
        extended_dataset_path = os.path.join(EXTEMDED_DATASETS_DIR, "lafan1_dataset")
        
        self.commands.motion = SONIC_MultiMotionCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range=VELOCITY_RANGE,
            joint_position_range=(-0.1, 0.1),
            # Dataset configuration (unified API with extended keys from extend_datasets.py)
            dataset_dirs=[extended_dataset_path],
            robot_name="g1",
            splits=["walk_subset"],
        )

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]


class SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX_wKeypoints(SONIC_MultiTrackingEnvCfg):
        
    SONIC_FLAG: bool = True
    SONIC_Keypoints_Export: bool = True
    
    def __post_init__(self):
        super().__post_init__()

        self.rewards.motion_global_anchor_pos.weight = 0.0
        
        # support smplx observations
        self.observations = ObservationsCfgV3()
        
        # Dataset paths (must be pre-processed by extend_datasets.py to include SMPL-X extended keys)
        extended_dataset_path = os.path.join(EXTEMDED_DATASETS_DIR, "lafan1_dataset")
        
        self.commands.motion = SONIC_MultiMotionCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=True,
            pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range=VELOCITY_RANGE,
            joint_position_range=(-0.1, 0.1),
            # Dataset configuration (unified API with extended keys from extend_datasets.py)
            dataset_dirs=[extended_dataset_path],
            robot_name="g1",
            splits=["train"],
        )
        self.observations.policy.projected_gravity = None
        self.observations.critic.projected_gravity = None
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
