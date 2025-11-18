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
    type: Literal["single_motion", "multi_motion", "new_multi_motion"] = "multi_motion",
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if type == "multi_motion":
        policy_exporter = _OnnxMultiMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    elif type == "single_motion":
        policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    elif type == "new_multi_motion":
        policy_exporter = _NewOnnxMultiMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    else:
        raise ValueError(f"Unknown policy export type: {type}")
    policy_exporter.export(path, filename)


def export_multi_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
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


    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
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
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)

    def forward(self, x):
        """
        Forward pass through the ONNX policy exporter.
        Args:
            x (torch.Tensor): The input observation tensor.
        Returns:
            torch.Tensor: The scaled action tensor.
        NOTE: no action offset here, need 
        """
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
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

class _NewOnnxMultiMotionPolicyExporter(_OnnxPolicyExporter):
    """
    ONNX exporter with detailed observation inputs.
    
    Each observation term is exported as a separate input for better interpretability.
    
    Attributes:
        observation_names: List of observation term names
        observation_dims: List of dimensions for each observation term
        input_names: List of input names for ONNX model
        output_names: List of output names for ONNX model
    """
    
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        
        # Get observation names and dimensions
        self.observation_names = env.observation_manager.active_terms["policy"]
        group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"]
        self.observation_dims = [dims[-1] for dims in group_obs_term_dim]
        
        # Store observation history lengths
        self.observation_history_lengths: list[int] = []
        if env.observation_manager.cfg.policy.history_length is not None:
            self.observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(self.observation_names)
        else:
            for name in self.observation_names:
                term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
                history_length = term_cfg["history_length"]
                self.observation_history_lengths.append(1 if history_length == 0 else history_length)
        
        # Prepare input and output names
        self.input_names = list(self.observation_names)
        self.output_names = ["actions"]
        
        if verbose:
            print(f"Observation names: {self.observation_names}")
            print(f"Observation dims: {self.observation_dims}")
            print(f"Observation history lengths: {self.observation_history_lengths}")

    def forward(self, *obs_terms):
        """
        Forward pass through the policy network.
        
        Args:
            *obs_terms: Variable number of observation tensors, one for each observation term
            
        Returns:
            torch.Tensor: The action tensor
        """
        # Concatenate all observation terms
        obs = torch.cat(obs_terms, dim=-1)
        
        # Pass through normalizer and actor
        return self.actor(self.normalizer(obs))

    def export(self, path, filename):
        """
        Export the policy to ONNX format with detailed observation inputs.
        
        Args:
            path: Directory path to save the ONNX model
            filename: Name of the ONNX file
        """
        self.to("cpu")
        
        # Create dummy inputs for each observation term
        dummy_inputs = []
        for i, (name, dim, history_len) in enumerate(zip(
            self.observation_names, 
            self.observation_dims, 
            self.observation_history_lengths
        )):
            # Account for history length in dimension
            total_dim = dim * history_len
            dummy_inputs.append(torch.zeros(1, total_dim))
        
        # Export to ONNX
        torch.onnx.export(
            self,
            tuple(dummy_inputs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=self.input_names,
            output_names=self.output_names,
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
