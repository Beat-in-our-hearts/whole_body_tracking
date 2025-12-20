import os
from typing import Literal
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, 
        train_cfg: dict, 
        type: Literal["single_motion", "multi_motion"],
        log_dir: str | None = None, 
        device="cpu", 
        registry_name: str = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self.type = type

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            base_filename = policy_path.split("/")[-2]
            
            # 1. Export obs_full version - each observation term as separate input
            filename_obs_full = base_filename + "_obs_full.onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, 
                self.alg.policy, 
                type=self.type,
                normalizer=self.obs_normalizer, 
                path=policy_path, 
                filename=filename_obs_full,
                obs_full=True,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename_obs_full)
            wandb.save(policy_path + filename_obs_full, base_path=os.path.dirname(policy_path))
            
            # 2. Export traditional version - single concatenated obs input
            filename = base_filename + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, 
                self.alg.policy, 
                type=self.type,
                normalizer=self.obs_normalizer, 
                path=policy_path, 
                filename=filename,
                obs_full=False,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None


class SONICOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, 
        train_cfg: dict, 
        log_dir: str | None = None, 
        device="cpu", 
        registry_name: str = None,
        enable_smplx_export: bool = True,
        enable_keypoints_export: bool = False,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self.enable_smplx_export = enable_smplx_export
        self.enable_keypoints_export = enable_keypoints_export
    
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            base_filename = policy_path.split("/")[-2]
            
            # 1. Export robot policy
            filename_obs_full = base_filename + "_robot_policy.onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, 
                self.alg.policy, 
                type="sonic_robot",
                normalizer=self.obs_normalizer, 
                path=policy_path, 
                filename=filename_obs_full,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename_obs_full)
            wandb.save(policy_path + filename_obs_full, base_path=os.path.dirname(policy_path))
            
            if self.enable_smplx_export:
                # 2. Export smplx policy
                filename = base_filename + "_smplx_policy.onnx"
                export_motion_policy_as_onnx(
                    self.env.unwrapped, 
                    self.alg.policy, 
                    type="sonic_human",
                    normalizer=self.obs_normalizer, 
                    path=policy_path, 
                    filename=filename,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
                
            # 3. Export smplx keypoints policy
            if self.enable_keypoints_export:
                filename_keypoints = base_filename + "_smplx_keypoints_policy.onnx"
                export_motion_policy_as_onnx(
                    self.env.unwrapped, 
                    self.alg.policy, 
                    type="sonic_keypoints",
                    normalizer=self.obs_normalizer, 
                    path=policy_path, 
                    filename=filename_keypoints,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename_keypoints)
                wandb.save(policy_path + filename_keypoints, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
