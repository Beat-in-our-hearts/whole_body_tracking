# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--disable_multi_motion", action="store_true", default=False, help="Disable multi-motion training.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file to load.")
parser.add_argument("--pretrain_vae_ckpt", type=str, default=None, help="Path to the pre-trained VAE checkpoint.")
parser.add_argument("--resume_wandb_run_path", type=str, default=None, help="Path to the wandb run to resume from")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import torch.distributed as dist
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs.common import ViewerCfg

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner
from whole_body_tracking.utils.my_on_policy_runner import SONICOnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed
        
    # set viewer configuration
    env_cfg.viewer = ViewerCfg(
        eye = (8.0, 8.0, 8.0),
        lookat = (0.0, 0.0, 0.0),
        env_index = 20,
        origin_type = "env", # "asset_root",
        asset_name = "robot",
    )

    if args_cli.disable_multi_motion:
        assert args_cli.motion_file is not None, "Motion file must be specified when disabling multi-motion."
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    sonic_flag = getattr(env_cfg, "SONIC_FLAG", False)
    if sonic_flag:
        if args_cli.pretrain_vae_ckpt is not None:
            agent_cfg.algorithm.pretrain_vae = True
            print("="*50)
            print(f"[INFO]: Enabled VAE pretraining in SONIC PPO algorithm config.")
            print("="*50)
            
        runner = SONICOnPolicyRunner(
            env, 
            agent_cfg.to_dict(), 
            log_dir=log_dir, 
            device=agent_cfg.device,
            enable_keypoints_export=getattr(env_cfg, "SONIC_Keypoints_Export", False),
        )
    else:
        runner = OnPolicyRunner(
            env, 
            agent_cfg.to_dict(), 
            type="single_motion" if args_cli.disable_multi_motion else "multi_motion",
            log_dir=log_dir, 
            device=agent_cfg.device,
        )
    
    # load pre-trained VAE checkpoint if specified
    if args_cli.pretrain_vae_ckpt is not None:
        print(f"[INFO]: Loading pre-trained VAE checkpoint from: {args_cli.pretrain_vae_ckpt}")
        cpu_vae_ckpt = torch.load(args_cli.pretrain_vae_ckpt, map_location='cpu')
        # NOTE vae just part of the actor, use strict=False to ignore missing keys
        load_result = runner.alg.policy.actor.load_state_dict(cpu_vae_ckpt, strict=False)
        print(f"[INFO]: VAE checkpoint load result: {load_result}")
        # TODO freeze VAE parameters, but `normalizer` has some problems when freezing
        runner.alg.policy.actor.freeze_encoders_and_decoders()
        print(f"[INFO]: Frozen VAE encoder and decoder parameters.")
        
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        if args_cli.resume_wandb_run_path is None:
            # get path to previous checkpoint
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        else:
            # only local_rank 0 process downloads from wandb
            run_path = args_cli.resume_wandb_run_path
            file = ""
            
            if app_launcher.local_rank == 0:
                import wandb

                api = wandb.Api()
                wandb_run = api.run(run_path)
                
                files = [f.name for f in wandb_run.files() if "model" in f.name]
                file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

                if os.path.exists(f"./logs/rsl_rl/temp_resume/{run_path}/{file}"):
                    print(f"[INFO]: Checkpoint already exists locally: ./logs/rsl_rl/temp_resume/{run_path}/{file}")
                else:
                    wandb_file = wandb_run.file(str(file))
                    wandb_file.download(f"./logs/rsl_rl/temp_resume/{run_path}", replace=True)

                print(f"[INFO]: Loading model checkpoint from wandb: {run_path}/{file}")
            
            # synchronize all processes and broadcast filename from rank 0
            if args_cli.distributed:
                file_list = [file]
                dist.broadcast_object_list(file_list, src=0)
                file = file_list[0]
            
            resume_path = f"./logs/rsl_rl/temp_resume/{run_path}/{file}"
                
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
