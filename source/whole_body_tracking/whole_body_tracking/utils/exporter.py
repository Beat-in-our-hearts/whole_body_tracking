# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import copy
import torch
from typing import Literal
import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand

def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv | None,
    actor_critic: object,
    path: str,
    task_type: Literal["single_motion", "multi_motion", "gae_mimic"] = "multi_motion",
    gaemimic_task: Literal["robot", "human", "keypoints"] = "robot",
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    if task_type == "single_motion":
        policy_exporter = _Onnx_Motion_PolicyExporter(env, actor_critic, normalizer, verbose)
    elif task_type == "multi_motion":
        policy_exporter = _Onnx_MultiMotion_PolicyExporter(env, actor_critic, normalizer, verbose)
    elif task_type == "gae_mimic":
        policy_exporter = _Onnx_GAEMimic_PolicyExporter(actor_critic, env, normalizer, verbose, task=gaemimic_task)
    else:
        raise ValueError(f"Unknown policy export type: {task_type}")
    
    policy_exporter.export(path, filename)


class _Onnx_Motion_PolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]
        
        self.observation_names = env.observation_manager.active_terms["policy"]
        group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"]
        self.observation_dims = [dims[-1] for dims in group_obs_term_dim]
        
    def forward(self, *args):
        # args contains separate observation terms
        obs = torch.cat(args[:-1], dim=-1)
        time_step = args[-1]

        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(obs)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")

        # Create separate dummy inputs for each observation term
        dummy_inputs = []
        for dim in self.observation_dims:
            dummy_inputs.append(torch.zeros(1, dim))
            
        # Add time_step as the last input
        time_step = torch.zeros(1, 1)
        dummy_inputs.append(time_step)
        
        input_names = list(self.observation_names) + ["time_step"]
        
        torch.onnx.export(
            self,
            tuple(dummy_inputs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=input_names,
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )

class _Onnx_MultiMotion_PolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        
        self.observation_names = env.observation_manager.active_terms["policy"]
        group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"]
        self.observation_dims = [dims[-1] for dims in group_obs_term_dim]
        
        if verbose:
            print(f"Observation names: {self.observation_names}")
            print(f"Observation dims: {self.observation_dims}")

    def forward(self, *args):
        obs = torch.cat(args, dim=-1)
        return self.actor(self.normalizer(obs))

    def export(self, path, filename):
        self.to("cpu")
        
        # Create separate dummy inputs for each observation term
        dummy_inputs = []
        for dim in self.observation_dims:
            dummy_inputs.append(torch.zeros(1, dim))
        
        input_names = list(self.observation_names)
        
        torch.onnx.export(
            self,
            tuple(dummy_inputs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=input_names,
            output_names=["actions"],
            dynamic_axes={},
        )


def register_forward(version):
    def decorator(func):
        func._forward_version = version
        return func
    return decorator


def select_forward(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if not hasattr(self, 'task'):
            raise AttributeError("Instance must have 'task' attribute after __init__")

        chosen_method = None
        for name in dir(cls):
            method = getattr(cls, name)
            if callable(method) and hasattr(method, '_forward_version'):
                if method._forward_version == self.task:
                    chosen_method = method
                    break

        if chosen_method is None:
            raise ValueError(f"No forward implementation for task: {self.task}")

        self.forward = chosen_method.__get__(self, cls)

    cls.__init__ = new_init
    return cls

@select_forward
class _Onnx_GAEMimic_PolicyExporter(_OnnxPolicyExporter):
    """Keypoints Policy Exporter for SONIC architecture.
    
    This exporter exports only the keypoints command branch of the Actor_SONIC policy.
    It uses the keypoints_encoder, FSQ quantizer, and action decoder.
    Always exports with separate inputs for each observation term (excluding robot_command).
    """
    def __init__(self, actor_critic, env=None, normalizer=None, verbose=False, task:Literal["robot", "human", "keypoints"] = "robot"):
        super().__init__(actor_critic, normalizer, verbose)
        
        self.task = task
        
        # Extract Actor_SONIC specific dimensions
        assert hasattr(actor_critic.actor, "actor_sk_dim") and \
               hasattr(actor_critic.actor, "actor_sg_dim") and \
               hasattr(actor_critic.actor, "actor_sh_dim"), "Actor does not have keypoints, robot, and smplx dimensions."
        
        self.actor_sg_dim = actor_critic.actor.actor_sg_dim
        self.actor_sh_dim = actor_critic.actor.actor_sh_dim
        self.actor_sk_dim = actor_critic.actor.actor_sk_dim
        
        self.num_actions = actor_critic.actor.num_actions
        
        # Extract observation manager info and filter for human only
        all_obs_names = env.observation_manager.active_terms["policy"]
        group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"]
        all_obs_dims = [dims[-1] for dims in group_obs_term_dim]
        
        # Filter command
        self.robot_observation_names = []
        self.robot_observation_dims = []
        
        for i, name in enumerate(all_obs_names):
            if name not in ["human_command", "keypoints_command"]:
                self.robot_observation_names.append(name)
                self.robot_observation_dims.append(all_obs_dims[i])
                
        self.human_observation_names = []
        self.human_observation_dims = []
        
        for i, name in enumerate(all_obs_names):
            if name not in ["robot_command", "keypoints_command"]:
                self.human_observation_names.append(name)
                self.human_observation_dims.append(all_obs_dims[i])
                
        self.keypoints_observation_names = []
        self.keypoints_observation_dims = []
        
        for i, name in enumerate(all_obs_names):
            if name not in ["robot_command", "human_command"]:
                self.keypoints_observation_names.append(name)
                self.keypoints_observation_dims.append(all_obs_dims[i])

    @register_forward("robot")
    def forward_robot(self, *args):
        obs = torch.cat(args, dim=-1)
        robot_command = obs[:, :self.actor_sg_dim]
        proprioceptive_state = obs[:, self.actor_sg_dim:]
        return self.actor.forward_robot_exporter(robot_command, proprioceptive_state)

    @register_forward("human")
    def forward_human(self, *args):
        obs = torch.cat(args, dim=-1)
        human_command = obs[:, :self.actor_sh_dim]
        proprioceptive_state = obs[:, self.actor_sh_dim:]
        return self.actor.forward_smplx_exporter(human_command, proprioceptive_state)

    @register_forward("keypoints")
    def forward_keypoints(self, *args):
        obs = torch.cat(args, dim=-1)
        keypoints_command = obs[:, :self.actor_sk_dim]
        proprioceptive_state = obs[:, self.actor_sk_dim:]
        return self.actor.forward_keypoints_exporter(keypoints_command, proprioceptive_state)

    def export(self, path, filename):
        self.to("cpu")
        
        if self.task == "robot":
            self.observation_names = self.robot_observation_names
            self.observation_dims = self.robot_observation_dims
        elif self.task == "human":
            self.observation_names = self.human_observation_names
            self.observation_dims = self.human_observation_dims
        elif self.task == "keypoints":
            self.observation_names = self.keypoints_observation_names
            self.observation_dims = self.keypoints_observation_dims
        else:
            raise ValueError(f"Unknown task for GAEMimic exporter: {self.task}")
        
        # Create separate dummy inputs for each observation term (excluding robot_command)
        dummy_inputs = []
        for dim in self.observation_dims:
            dummy_inputs.append(torch.zeros(1, dim))
            
        input_names = list(self.observation_names)
        
        torch.onnx.export(
            self,
            tuple(dummy_inputs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=input_names,
            output_names=["actions"],
            dynamic_axes={},
        )

def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_dims: list[int] = []  # Add observation dimensions

    # Get observation dimensions for each group
    group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"] # list[list[int]]
    observation_dims = [dims[-1] for dims in group_obs_term_dim]
    
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "body_names": env.scene["robot"].data.body_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_dims": observation_dims,  # Add to metadata
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "tracking_body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
