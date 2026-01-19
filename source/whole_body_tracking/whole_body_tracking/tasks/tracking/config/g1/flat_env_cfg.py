from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg, MultiTracking_TrackingEnvCfg, GAEMimic_TrackingEnvCfg
from whole_body_tracking.tasks import REPLAY_DATASETS_DIR, EXTEMDED_DATASETS_DIR
import os

@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    
    task_type: str = "single_motion"
    
    def __post_init__(self):
        super().__post_init__()

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

@configclass
class MultiTracking_G1FlatEnvCfg(MultiTracking_TrackingEnvCfg):
    
    task_type: str = "multi_motion"
    
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        
        # multi motion tracking settings
        self.commands.motion.robot_name = "g1"
        self.commands.motion.dataset_dirs = [os.path.join(EXTEMDED_DATASETS_DIR, "lafan1_dataset"),]
        self.commands.motion.splits = ["walk_subset", ]
        
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
        
@configclass
class GAEMimic_G1FlatEnvCfg(GAEMimic_TrackingEnvCfg):
    
    task_type: str = "gae_mimic"
    
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        
        # gaemimic motion tracking settings
        self.commands.motion.robot_name = "g1"
        self.commands.motion.dataset_dirs = [
            os.path.join(EXTEMDED_DATASETS_DIR, "lafan1_dataset"),
            os.path.join(EXTEMDED_DATASETS_DIR, "100style_dataset"),
            ]
        self.commands.motion.splits = ["train", "train", ]
        
        self.commands.motion.adaptive_uniform_ratio = 0.0
        self.commands.motion.adaptive_cap = 20
        self.commands.motion.adaptive_alpha = 2e-4
        
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
        
@configclass
class Ablation_GAEMimic_G1FlatEnvCfg(GAEMimic_G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.dataset_dirs = [os.path.join(EXTEMDED_DATASETS_DIR, "100style_dataset")]
        self.commands.motion.splits = ["train"]
        self.commands.motion.adaptive_uniform_ratio = 0.8
        
@configclass
class Play_GAEMimic_G1FlatEnvCfg(GAEMimic_G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.dataset_dirs = [
            os.path.join(EXTEMDED_DATASETS_DIR, "lafan1_dataset"),
            # os.path.join(EXTEMDED_DATASETS_DIR, "100style_dataset"),
            # os.path.join(EXTEMDED_DATASETS_DIR, "omomo_dataset"),
        ]
        self.commands.motion.splits = [
            "train",
            # "train",
            # "train",
        ]
        
        
@configclass
class GAEMimic_SingleFinetune_G1FlatEnvCfg(GAEMimic_TrackingEnvCfg):
    task_type: str = "gae_mimic"
    
    def __post_init__(self):
        super().__post_init__()
        
        # specify which modality to finetune
        self.finetune_task("robot")
        
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        
        # gaemimic motion tracking settings
        self.commands.motion.robot_name = "g1"
        self.commands.motion.dataset_dirs = [
            os.path.join(EXTEMDED_DATASETS_DIR, "lafan1_dataset"),
            # os.path.join(EXTEMDED_DATASETS_DIR, "100style_dataset"),
            ]
        self.commands.motion.splits = ["train", ]
        
        self.commands.motion.adaptive_uniform_ratio = 0.0
        self.commands.motion.adaptive_cap = 20
        self.commands.motion.adaptive_alpha = 5e-4
        self.curriculum.adaptive_sampling_ratio.params["delta_ratio"] = 5e-2
        
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
        
    def finetune_task(self, cmd_name):
        if cmd_name == "robot":
            self.observations.policy.human_command = None
            self.observations.policy.keypoints_command = None
            self.observations.critic.human_command = None
            self.observations.critic.keypoints_command = None
        elif cmd_name == "human":
            self.observations.policy.robot_command = None
            self.observations.policy.keypoints_command = None
            self.observations.critic.robot_command = None
            self.observations.critic.keypoints_command = None
        elif cmd_name == "keypoints":
            self.observations.policy.robot_command = None
            self.observations.policy.human_command = None
            self.observations.critic.robot_command = None
            self.observations.critic.human_command = None