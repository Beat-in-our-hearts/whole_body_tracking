from isaaclab.utils import configclass
import os
from dataclasses import MISSING

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks import NPZ_DATASETS_DIR, SMPLX_DATASETS_DIR

from whole_body_tracking.tasks.tracking.sonic_multi_tracking_env_cfg import TrackingEnvCfg as SONIC_MultiTrackingEnvCfg
from whole_body_tracking.tasks.tracking.mdp import SONIC_MultiMotionCommandCfg
from whole_body_tracking.tasks.tracking.sonic_multi_tracking_env_cfg import VELOCITY_RANGE, ObservationsCfgV2

#######################################
# SONIC
#######################################
class SONIC_G1FlatMultiTrackingEnvCfg_wSMPLX(SONIC_MultiTrackingEnvCfg):
    
    SONIC_FLAG: bool = True
    
    def __post_init__(self):
        super().__post_init__()

        self.rewards.motion_global_anchor_pos.weight = 0.0
        
        # support smplx observations
        self.observations = ObservationsCfgV2()
        
        lafan1_dataset_path = os.path.join(NPZ_DATASETS_DIR, "LAFAN1_Retargeting_Dataset")
        smplx_dataset_path = os.path.join(SMPLX_DATASETS_DIR, "lafan1_smplx_datasets")
        
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
            # Dataset configuration
            robot_dataset={lafan1_dataset_path: ["walk_subset", ]},
            smplx_dataset=[smplx_dataset_path,],
            robot_name="g1",
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
