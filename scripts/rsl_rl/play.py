"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--disable_multi_motion", action="store_true", default=False, help="Disable multi-motion training.")
parser.add_argument("--datasets", type=str, default=None, help="Comma separated list of datasets to use.")
parser.add_argument("--splits", type=str, default=None, help="Splits name to use for datasets.")
parser.add_argument("--wandb_run_path", type=str, default=None, help="Path to the wandb run to load the model from.")
parser.add_argument("--wandb_alg_cfg", action="store_true", default=False, help="Load algorithm config from wandb run.")
parser.add_argument("--export_name", type=str, default=None, help="Name of the export file.")
parser.add_argument("--activate_signals", choices=["robot", "smplx", "keypoints"], default=None, help="Comma separated list of signals to activate.")

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
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.io import load_yaml
            
# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from isaaclab_rl.rsl_rl import export_policy_as_jit


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    # Download weights from wandb (only for multi-motion mode)
    if not args_cli.disable_multi_motion and args_cli.wandb_run_path:
        import wandb

        run_path = args_cli.wandb_run_path

        api = wandb.Api()
        # if "model" in args_cli.wandb_run_path:
        #     run_path = "/".join(args_cli.wandb_run_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_run_path:
            file = args_cli.wandb_run_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from wandb: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"
        
        if args_cli.wandb_alg_cfg:
            # load agent config from wandb run
            config_dict = wandb_run.config
            # convert to dict
            config_dict = dict(config_dict)
            # update agent_cfg
            config_dict["policy_cfg"]["init_noise_std"] = float(config_dict["policy_cfg"]["init_noise_std"])
            agent_cfg.policy.from_dict(config_dict["policy_cfg"])
            
            # agent_cfg = RslRlOnPolicyRunnerCfg.from_dict(config_dict["alg_cfg"])
            print("[INFO]: Loaded agent config from wandb run:")
            print_dict(agent_cfg.to_dict(), nesting=4)

    else:
        # motion_file from log_root_path/params/env.yaml
        # resume_env_cfg_path = os.path.join(log_root_path, "params", "env.yaml")
        # resume_env_cfg_dict = load_yaml(resume_env_cfg_path)
        # motion_file = resume_env_cfg_dict.get("commands", {}).get("motion", {}).get("motion_file", None)

        # env_cfg.commands.motion.motion_file = motion_file
        # print(f"[INFO]: Using motion file from env.yaml: {motion_file}")
            
        if args_cli.motion_file is not None:
            env_cfg.commands.motion.motion_file = args_cli.motion_file
            print(f"[INFO]: Overriding motion file from CLI: {args_cli.motion_file}")
            
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # NOTE: GAE MIMIC activate signals override
    if args_cli.activate_signals is not None and hasattr(agent_cfg.policy, "activate_signals"):
        agent_cfg.policy.activate_signals = args_cli.activate_signals
        print(f"[INFO]: Overriding activate_signals from CLI: {args_cli.activate_signals}")
    
    from isaaclab.envs.common import ViewerCfg
    
    env_cfg.viewer = ViewerCfg(
        eye = (8.0, 8.0, 8.0),
        # eye = (0.0, 0.0, 10.0),
        lookat = (0.0, 0.0, 0.0),
        env_index = 16 if env_cfg.scene.num_envs > 16 else env_cfg.scene.num_envs -1,
        origin_type = "env",
        # origin_type = "asset_root",
        asset_name = "robot",
    )
    env_cfg.terminations.time_out = None
    env_cfg.events.push_robot = None

    if args_cli.datasets is not None and args_cli.splits is not None:
        env_cfg.commands.motion.dataset_dirs = args_cli.datasets.split(",")
        env_cfg.commands.motion.splits = args_cli.splits.split(",")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    elif hasattr(ppo_runner, "obs_normalizer"): # rsl_rl 2.3.3
        normalizer = ppo_runner.obs_normalizer
    else:
        print("[WARN] No normalizer found for the policy network.")
        normalizer = None
        
    print("[INFO] normalizer:", normalizer)
    
    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    
    # TODO fsq code use the einops, not supported in jit export for now
    try:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        print(f"[INFO] Exported JIT policy to: {os.path.join(export_model_dir, 'policy.pt')}")
    except Exception as e:
        print(f"[ERROR] Failed to export JIT policy: {e}")
        
    # Determine export type
    task_type = getattr(env_cfg, "task_type", None)
    print(f"[INFO] Exporting ONNX policy with type: {task_type}")
    if task_type in ["single_motion", "multi_motion"]:
        if args_cli.export_name is None:
            filename = f"{task_type}_policy.onnx"
        else:
            if not args_cli.export_name.endswith(".onnx"):
                filename = f"{args_cli.export_name}.onnx"
            else:
                filename = args_cli.export_name

        export_motion_policy_as_onnx(
            env.unwrapped,
            policy_nn,
            task_type=task_type,
            normalizer=normalizer,
            path=export_model_dir,
            filename=filename,
        )
        attach_onnx_metadata(
            env.unwrapped, 
            args_cli.wandb_run_path if args_cli.wandb_run_path else "none", 
            export_model_dir,
            filename=filename,
        )
        print(f"[INFO] Exported ONNX policy to: {os.path.join(export_model_dir, filename)}")
    
    elif task_type == "gae_mimic":
        if args_cli.export_name is None:
            base_filename = "gae_mimic_policy"
        else:
            if args_cli.export_name.endswith(".onnx"):
                base_filename = args_cli.export_name[:-5]
            else:
                base_filename = args_cli.export_name
        
        filename_robot = base_filename + "_robot.onnx"
        wandb_run_path = args_cli.wandb_run_path if args_cli.wandb_run_path else "none"
        export_motion_policy_as_onnx(
            env.unwrapped, 
            policy_nn, 
            task_type=task_type,
            gaemimic_task="robot",
            normalizer=normalizer, 
            path=export_model_dir, 
            filename=filename_robot,
        )
        attach_onnx_metadata(env.unwrapped, wandb_run_path, path=export_model_dir, filename=filename_robot)
        print(f"[INFO] Exported ONNX policy to: {os.path.join(export_model_dir, filename_robot)}")
        
        # gae_mimic - export human version
        filename_human = base_filename + "_human.onnx"
        export_motion_policy_as_onnx(
            env.unwrapped, 
            policy_nn, 
            task_type=task_type,
            gaemimic_task="human",
            normalizer=normalizer, 
            path=export_model_dir, 
            filename=filename_human,
        )
        attach_onnx_metadata(env.unwrapped, wandb_run_path, path=export_model_dir, filename=filename_human)
        print(f"[INFO] Exported ONNX policy to: {os.path.join(export_model_dir, filename_human)}")
        
        # gae_mimic - export keypoints version
        filename_keypoints = base_filename + "_keypoints.onnx"
        export_motion_policy_as_onnx(
            env.unwrapped, 
            policy_nn, 
            task_type=task_type,
            gaemimic_task="keypoints",
            normalizer=normalizer, 
            path=export_model_dir, 
            filename=filename_keypoints,
        )
        attach_onnx_metadata(env.unwrapped, wandb_run_path, path=export_model_dir, filename=filename_keypoints)
        print(f"[INFO] Exported ONNX policy to: {os.path.join(export_model_dir, filename_keypoints)}")
        
    # reset environment
    try: # isaacsim 4.5
        obs, _ = env.get_observations()
    except: # isaacsim 5.1
        obs = env.get_observations()
    
    timestep = 0
    # simulate environment
    with tqdm.tqdm(total=args_cli.video_length, desc="Playing") as pbar:
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _ = env.step(actions)
            if args_cli.video:
                timestep += 1
                pbar.update(1)
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
