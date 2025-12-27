import os
from typing import Literal
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

class Tracking_OnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, 
        train_cfg: dict, 
        task_type: Literal["single_motion", "multi_motion", "gae_mimic"] = None,
        log_dir: str | None = None, 
        device="cpu", 
        registry_name: str = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self.task_type = task_type

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            base_filename = policy_path.split("/")[-2]
            
            if self.task_type in ["single_motion", "multi_motion"]:
                filename = base_filename + f"_{self.task_type}.onnx"
                export_motion_policy_as_onnx(
                    self.env.unwrapped, 
                    self.alg.policy, 
                    task_type=self.task_type,
                    normalizer=self.obs_normalizer, 
                    path=policy_path, 
                    filename=filename,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
                
            elif self.task_type == "gae_mimic":
                # gae_mimic - export robot version
                filename_robot = base_filename + "_robot.onnx"
                export_motion_policy_as_onnx(
                    self.env.unwrapped, 
                    self.alg.policy, 
                    task_type=self.task_type,
                    gaemimic_task="robot",
                    normalizer=self.obs_normalizer, 
                    path=policy_path, 
                    filename=filename_robot,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename_robot)
                wandb.save(policy_path + filename_robot, base_path=os.path.dirname(policy_path))
                
                # gae_mimic - export human version
                filename_human = base_filename + "_human.onnx"
                export_motion_policy_as_onnx(
                    self.env.unwrapped, 
                    self.alg.policy, 
                    task_type=self.task_type,
                    gaemimic_task="human",
                    normalizer=self.obs_normalizer, 
                    path=policy_path, 
                    filename=filename_human,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename_human)
                wandb.save(policy_path + filename_human, base_path=os.path.dirname(policy_path))
                
                # gae_mimic - export keypoints version
                filename_keypoints = base_filename + "_keypoints.onnx"
                export_motion_policy_as_onnx(
                    self.env.unwrapped, 
                    self.alg.policy, 
                    task_type=self.task_type,
                    gaemimic_task="keypoints",
                    normalizer=self.obs_normalizer, 
                    path=policy_path, 
                    filename=filename_keypoints,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename_keypoints)
                wandb.save(policy_path + filename_keypoints, base_path=os.path.dirname(policy_path))
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")
            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None