#!/usr/bin/env python3
"""Evaluation script for multi-motion tracking with systematic motion testing.

This script evaluates a trained policy by:
1. Running each motion clip exactly eval_target_attempts times (default: 128)
2. Using dynamic weight-based sampling to balance evaluation progress
3. Tracking success/failure statistics for each motion
4. Disabling episode timeout termination (only failure and motion completion)
5. Saving detailed results to CSV

Usage:
    # Evaluate with default settings (128 attempts per motion)
    python eval.py --task=MultiTracking-Flat-G1-v0 --checkpoint=model.pt
    
    # Custom number of attempts
    python eval.py --task=MultiTracking-Flat-G1-v0 --checkpoint=model.pt --num_repeats=256
    
    # Custom output path
    python eval.py --task=MultiTracking-Flat-G1-v0 --checkpoint=model.pt --output=results.csv
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate multi-motion tracking policy")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_repeats", type=int, default=128, help="Number of evaluation attempts per motion")
parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: eval_results_<timestamp>.csv)")
parser.add_argument("--progress_interval", type=int, default=100, help="Print progress every N steps")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
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

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.mdp.commands import EvalMultiMotionCommand


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate with RSL-RL agent."""
    # Parse RSL-RL configuration
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # ============ Step 1: Print configuration ============
    print(f"\n{'='*80}")
    print(f"Starting Evaluation for {args_cli.task}")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  Task: {args_cli.task}")
    print(f"  Checkpoint: {args_cli.checkpoint if args_cli.checkpoint else 'from log directory'}")
    print(f"  Target attempts per motion: {args_cli.num_repeats}")
    print(f"  Number of environments: {env_cfg.scene.num_envs}")
    print(f"  Headless mode: {args_cli.headless}")
    
    # ============ Step 2: Modify configuration for evaluation ============
    print("\nModifying configuration for evaluation mode...")
    
    # Switch command class to EvalMultiMotionCommand
    env_cfg.commands.motion.class_type = EvalMultiMotionCommand
    env_cfg.commands.motion.eval_target_attempts = args_cli.num_repeats
    
    # Disable episode timeout termination (keep only failure and motion completion)
    if hasattr(env_cfg, 'terminations') and hasattr(env_cfg.terminations, 'time_out'):
        print("  Disabling time_out termination...")
        env_cfg.terminations.time_out = None
    
    print(f"  Command class: EvalMultiMotionCommand")
    print(f"  Eval target attempts: {args_cli.num_repeats}")
    print(f"  Time-out termination: disabled")
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # Get checkpoint path
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # ============ Step 3: Create environment ============
    print("\nCreating environment...")
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    
    # Get command manager reference
    motion_command = env.unwrapped.command_manager._terms["motion"]
    assert isinstance(motion_command, EvalMultiMotionCommand), \
        f"Expected EvalMultiMotionCommand, got {type(motion_command)}"
    
    num_motions = len(motion_command.dataset)
    num_envs = env.unwrapped.num_envs
    
    print(f"\nEnvironment created:")
    print(f"  Number of parallel environments: {num_envs}")
    print(f"  Number of motions: {num_motions}")
    print(f"  Total evaluation attempts: {num_motions * args_cli.num_repeats}")
    
    # ============ Step 4: Load policy ============
    print(f"\nLoading policy from {resume_path}...")
    
    # Load previously trained model
    log_dir = os.path.dirname(resume_path)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # Obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    
    print(f"  Policy loaded successfully")
    
    # ============ Step 5: Run evaluation loop ============
    print(f"\n{'='*80}")
    print("Starting Evaluation Loop")
    print(f"{'='*80}\n")
    
    # Reset environment
    # NOTE: fixbug for isaaclab2.3.0
    obs = env.get_observations()
    
    step_count = 0
    start_time = time.time()
    last_progress_time = start_time
    
    with torch.inference_mode():
        while not motion_command.check_eval_complete():
            # Get action from policy
            actions = policy(obs)
            
            # Step environment
            obs, _, _, _ = env.step(actions)
            
            step_count += 1
            
            # Print progress periodically
            if step_count % args_cli.progress_interval == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                interval_elapsed = current_time - last_progress_time
                steps_per_sec = step_count / elapsed
                interval_steps_per_sec = args_cli.progress_interval / interval_elapsed
                
                # Get current statistics
                total_completed = motion_command.eval_motion_completed_count.sum().item()
                total_successes = motion_command.eval_motion_success_count.sum().item()
                total_failures = motion_command.eval_motion_failure_count.sum().item()
                current_success_rate = total_successes / total_completed if total_completed > 0 else 0.0
                
                print(f"\n[Step {step_count}] Elapsed: {elapsed:.1f}s")
                print(f"  Rate: {steps_per_sec:.1f} steps/s (interval: {interval_steps_per_sec:.1f} steps/s)")
                print(f"  Total completed: {total_completed} (successes: {total_successes}, failures: {total_failures})")
                print(f"  Current success rate: {current_success_rate:.2%}")
                motion_command.print_progress()
                
                # Estimate remaining time
                incomplete_motions = len(motion_command.get_incomplete_motions())
                avg_completed = motion_command.eval_motion_completed_count.float().mean().item()
                remaining_completed = incomplete_motions * (args_cli.num_repeats - avg_completed)
                if total_completed > 0:
                    avg_steps_per_completed = step_count / total_completed
                    estimated_remaining_steps = remaining_completed * avg_steps_per_completed
                    estimated_remaining_time = estimated_remaining_steps / steps_per_sec
                    print(f"  Estimated remaining time: {estimated_remaining_time / 60:.1f} minutes\n")
                
                last_progress_time = current_time
    
    # ============ Step 6: Collect and save results ============
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}\n")
    
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s")
    print(f"Total steps: {step_count}")
    print(f"Average rate: {step_count / elapsed:.1f} steps/s")
    
    # Get detailed results
    results = motion_command.get_eval_results()
    
    # Print summary statistics
    success_rates = [r['success_rate'] for r in results.values()]
    completion_rates = [r['completion_rate'] for r in results.values()]
    avg_success_rate = sum(success_rates) / len(success_rates)
    avg_completion_rate = sum(completion_rates) / len(completion_rates)
    
    total_completed = sum(r['completed'] for r in results.values())
    total_successes = sum(r['successes'] for r in results.values())
    total_failures = sum(r['failures'] for r in results.values())
    
    print(f"\nOverall Statistics:")
    print(f"  Total motions: {len(results)}")
    print(f"  Total completed: {total_completed}")
    print(f"  Total successes: {total_successes}")
    print(f"  Total failures: {total_failures}")
    print(f"  Average success rate: {avg_success_rate:.2%}")
    print(f"  Average completion rate: {avg_completion_rate:.2%}")
    print(f"  Min success rate: {min(success_rates):.2%}")
    print(f"  Max success rate: {max(success_rates):.2%}")
    
    # Find best and worst performing motions
    sorted_by_success = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)
    print(f"\nTop 5 Best Performing Motions:")
    for i, (motion_id, result) in enumerate(sorted_by_success[:5], 1):
        print(f"  {i}. {result['motion_name']}: {result['success_rate']:.2%} ({result['successes']}/{result['completed']})")
    
    print(f"\nTop 5 Worst Performing Motions:")
    for i, (motion_id, result) in enumerate(sorted_by_success[-5:][::-1], 1):
        print(f"  {i}. {result['motion_name']}: {result['success_rate']:.2%} ({result['successes']}/{result['completed']})")
    
    
    # Save to CSV
    if args_cli.output is None:
        # Save to log_dir/eval/ folder
        eval_dir = Path(log_dir) / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = eval_dir / f"eval_results_{timestamp}.csv"
    else:
        output_path = Path(args_cli.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to {output_path}...")
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = [
            'motion_id', 'motion_name', 'quantity',
            'completed', 'successes', 'failures', 
            'success_rate', 'completion_rate', 'motion_length'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for motion_id, result in results.items():
            writer.writerow({
                'motion_id': motion_id,
                'motion_name': result['motion_name'],
                'quantity': result['quantity'],
                'completed': result['completed'],
                'successes': result['successes'],
                'failures': result['failures'],
                'success_rate': f"{result['success_rate']:.4f}",
                'completion_rate': f"{result['completion_rate']:.4f}",
                'motion_length': result['motion_length'],
            })
    
    print(f"Results saved successfully!")
    print(f"  Path: {output_path.absolute()}")
    print(f"  Rows: {len(results)}")
    
    
    # Close environment
    env.close()
    
    print(f"\n{'='*80}")
    print("Evaluation finished!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
