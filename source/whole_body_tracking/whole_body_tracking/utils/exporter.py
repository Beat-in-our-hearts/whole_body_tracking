# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
from typing import Literal
import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    type: Literal["single_motion", "multi_motion"] = "multi_motion",
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
    obs_full: bool = False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if type == "multi_motion":
        policy_exporter = _OnnxMultiMotionPolicyExporter(env, actor_critic, normalizer, verbose, obs_full)
    elif type == "single_motion":
        policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose, obs_full)
    else:
        raise ValueError(f"Unknown policy export type: {type}")
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False, obs_full=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]
        self.obs_full = obs_full
        
        if obs_full:
            self.observation_names = env.observation_manager.active_terms["policy"]
            group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"]
            self.observation_dims = [dims[-1] for dims in group_obs_term_dim]
            
            self.observation_history_lengths: list[int] = []
            if env.observation_manager.cfg.policy.history_length is not None:
                self.observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(self.observation_names)
            else:
                for name in self.observation_names:
                    term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
                    history_length = term_cfg["history_length"]
                    self.observation_history_lengths.append(1 if history_length == 0 else history_length) 


    def forward(self, *args):
        if self.obs_full:
            # args contains separate observation terms
            obs = torch.cat(args[:-1], dim=-1)
            time_step = args[-1]
        else:
            # args contains concatenated obs and time_step
            obs = args[0]
            time_step = args[1]
        
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
        
        if self.obs_full:
            # Create separate dummy inputs for each observation term
            dummy_inputs = []
            for dim, history_len in zip(self.observation_dims, self.observation_history_lengths):
                total_dim = dim * history_len
                dummy_inputs.append(torch.zeros(1, total_dim))
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
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            time_step = torch.zeros(1, 1)
            torch.onnx.export(
                self,
                (obs, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "time_step"],
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
        
class _OnnxMultiMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False, obs_full=False):
        super().__init__(actor_critic, normalizer, verbose)
        self.obs_full = obs_full
        
        if obs_full:
            self.observation_names = env.observation_manager.active_terms["policy"]
            group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"]
            self.observation_dims = [dims[-1] for dims in group_obs_term_dim]
            
            self.observation_history_lengths: list[int] = []
            if env.observation_manager.cfg.policy.history_length is not None:
                self.observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(self.observation_names)
            else:
                for name in self.observation_names:
                    term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
                    history_length = term_cfg["history_length"]
                    self.observation_history_lengths.append(1 if history_length == 0 else history_length)
            
            if verbose:
                print(f"Observation names: {self.observation_names}")
                print(f"Observation dims: {self.observation_dims}")
                print(f"Observation history lengths: {self.observation_history_lengths}")

    def forward(self, *args):
        """
        Forward pass through the ONNX policy exporter.
        Args:
            *args: Either a single concatenated observation tensor, or multiple separate observation tensors
        Returns:
            torch.Tensor: The scaled action tensor.
        NOTE: no action offset here, need 
        """
        if self.obs_full:
            # args contains separate observation terms, concatenate them
            obs = torch.cat(args, dim=-1)
        else:
            # args contains a single concatenated observation tensor
            obs = args[0]
        
        return self.actor(self.normalizer(obs))

    def export(self, path, filename):
        self.to("cpu")
        
        if self.obs_full:
            # Create separate dummy inputs for each observation term
            dummy_inputs = []
            for dim, history_len in zip(self.observation_dims, self.observation_history_lengths):
                total_dim = dim * history_len
                dummy_inputs.append(torch.zeros(1, total_dim))
            
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
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            # pass the inputs as a tuple
            torch.onnx.export(
                self,
                (obs,),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
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
    observation_history_lengths: list[int] = []
    observation_dims: list[int] = []  # Add observation dimensions

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)
    
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
        "observation_history_lengths": observation_history_lengths,
        "observation_dims": observation_dims,  # Add to metadata
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "motion_anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "motion_key_body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
