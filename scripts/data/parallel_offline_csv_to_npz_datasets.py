"""This script batch processes motions from CSV files and outputs them to NPZ files

.. code-block:: bash

    # Usage - Batch processing
    python offline_csv_to_npz_datasets.py \
        --input_dir ./csv_motions/ \
        --output_dir ./npz_motions/ \
        --input_fps 30 \
        --output_fps 50 \
        --recursive \
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Batch convert CSV motion files to NPZ format.")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input CSV motion files.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory for output NPZ motion files.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motions.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded. Applied to all files."
    ),
)
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motions.")
parser.add_argument("--recursive", "-r", action="store_true", help="Recursively search for CSV files in subdirectories.")
parser.add_argument("--pattern", type=str, default="*.csv", help="File pattern to match (default: *.csv).")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments for processing (default: 32).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class EnvState:
    """Tracks the state of each parallel environment."""
    def __init__(self, env_id: int):
        self.env_id = env_id
        self.motion_loader = None
        self.current_frame = 0
        self.csv_file = None
        self.npz_file = None
        self.is_active = False
        self.log = None
        
    def start_new_motion(self, csv_file: str, npz_file: str, motion_loader):
        """Initialize a new motion for this environment."""
        self.csv_file = csv_file
        self.npz_file = npz_file
        self.motion_loader = motion_loader
        self.current_frame = 0
        self.is_active = True
        self.log = {
            "fps": [args_cli.output_fps],
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
        }
    
    def is_complete(self) -> bool:
        """Check if current motion is complete."""
        if not self.is_active or self.motion_loader is None:
            return False
        return self.current_frame >= self.motion_loader.output_frames
    
    def reset(self):
        """Reset environment state."""
        self.motion_loader = None
        self.current_frame = 0
        self.csv_file = None
        self.npz_file = None
        self.is_active = False
        self.log = None


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # convert to wxyz
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        tqdm.write(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        tqdm.write(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_parallel_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    joint_names: list[str],
    csv_files: list[Path],
    input_path: Path,
    output_path: Path,
):
    """Runs the simulation loop with parallel environments."""
    num_envs = args_cli.num_envs
    
    # Initialize environment states
    env_states = [EnvState(i) for i in range(num_envs)]
    
    # Task queue
    task_queue = list(csv_files)
    completed_count = 0
    failed_files = []
    total_files = len(csv_files)
    
    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]
    
    # Helper function to assign new task to an environment
    def assign_task(env_state: EnvState) -> bool:
        """Assign a new motion file to an environment. Returns True if task assigned."""
        if not task_queue:
            return False
        
        csv_file = task_queue.pop(0)
        rel_path = csv_file.relative_to(input_path)
        npz_file = output_path / rel_path.with_suffix(".npz")
        
        try:
            motion_loader = MotionLoader(
                motion_file=str(csv_file),
                input_fps=args_cli.input_fps,
                output_fps=args_cli.output_fps,
                device=sim.device,
                frame_range=args_cli.frame_range,
            )
            env_state.start_new_motion(str(csv_file), str(npz_file), motion_loader)
            return True
        except Exception as e:
            print(f"\n[ERROR]: Failed to load {csv_file}: {e}")
            failed_files.append(str(csv_file))
            return False
    
    # Helper function to save completed motion
    def save_motion(env_state: EnvState):
        """Save the logged motion data to NPZ file."""
        try:
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                env_state.log[k] = np.stack(env_state.log[k], axis=0)
            
            save_path = Path(env_state.npz_file).expanduser()
            if save_path.suffix != ".npz":
                save_path = save_path.with_suffix(".npz")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez(save_path, **env_state.log)
            return True
        except Exception as e:
            print(f"\n[ERROR]: Failed to save {env_state.npz_file}: {e}")
            failed_files.append(env_state.csv_file)
            return False
    
    # Initial task assignment
    print(f"[INFO]: Initializing {num_envs} parallel environments...")
    for env_state in env_states:
        if not assign_task(env_state):
            break
    
    # Progress bar - fixed at bottom with position=0
    pbar = tqdm(total=total_files, desc="Converting CSV to NPZ", position=0, leave=True, dynamic_ncols=True)
    
    # Main simulation loop
    while completed_count < total_files:
        # Prepare batched data for all active environments
        active_env_indices = []
        root_states_batch = []
        joint_pos_batch = []
        joint_vel_batch = []
        
        # Collect states from all active environments
        for env_idx, env_state in enumerate(env_states):
            if not env_state.is_active:
                continue
            
            # Get next state for this environment
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ), _ = env_state.motion_loader.get_next_state()
            
            # Prepare root state for this environment
            root_state = robot.data.default_root_state[env_idx:env_idx+1].clone()
            root_state[:, :3] = motion_base_pos
            root_state[:, :2] += scene.env_origins[env_idx:env_idx+1, :2]
            root_state[:, 3:7] = motion_base_rot
            root_state[:, 7:10] = motion_base_lin_vel
            root_state[:, 10:] = motion_base_ang_vel
            
            # Prepare joint state for this environment
            joint_pos = robot.data.default_joint_pos[env_idx:env_idx+1].clone()
            joint_vel = robot.data.default_joint_vel[env_idx:env_idx+1].clone()
            joint_pos[:, robot_joint_indexes] = motion_dof_pos
            joint_vel[:, robot_joint_indexes] = motion_dof_vel
            
            # Collect for batch processing
            active_env_indices.append(env_idx)
            root_states_batch.append(root_state)
            joint_pos_batch.append(joint_pos)
            joint_vel_batch.append(joint_vel)
            
            # Increment frame counter
            env_state.current_frame += 1
        
        # Write batched data to simulation (vectorized operation)
        if active_env_indices:
            active_env_tensor = torch.tensor(active_env_indices, dtype=torch.int32, device=sim.device)
            root_states_tensor = torch.cat(root_states_batch, dim=0)
            joint_pos_tensor = torch.cat(joint_pos_batch, dim=0)
            joint_vel_tensor = torch.cat(joint_vel_batch, dim=0)
            
            robot.write_root_state_to_sim(root_states_tensor, env_ids=active_env_tensor)
            robot.write_joint_state_to_sim(joint_pos_tensor, joint_vel_tensor, env_ids=active_env_tensor)
        
        # Render and update scene
        sim.render()
        scene.update(sim.get_physics_dt())
        
        # Collect data and check for completion
        for env_idx, env_state in enumerate(env_states):
            if not env_state.is_active:
                continue
            
            # Log data for this environment
            env_state.log["joint_pos"].append(robot.data.joint_pos[env_idx, :].cpu().numpy().copy())
            env_state.log["joint_vel"].append(robot.data.joint_vel[env_idx, :].cpu().numpy().copy())
            env_state.log["body_pos_w"].append(robot.data.body_pos_w[env_idx, :].cpu().numpy().copy())
            env_state.log["body_quat_w"].append(robot.data.body_quat_w[env_idx, :].cpu().numpy().copy())
            env_state.log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[env_idx, :].cpu().numpy().copy())
            env_state.log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[env_idx, :].cpu().numpy().copy())
            
            # Check if this environment completed its motion
            if env_state.is_complete():
                # Save the motion
                if save_motion(env_state):
                    completed_count += 1
                    pbar.update(1)
                else:
                    completed_count += 1
                    pbar.update(1)
                
                # Reset and assign new task
                env_state.reset()
                assign_task(env_state)
    
    pbar.close()
    
    # Return statistics
    return completed_count - len(failed_files), failed_files


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene with multiple environments
    scene_cfg = ReplayMotionsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    
    # Define joint names
    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    
    # Find all CSV files
    input_path = Path(args_cli.input_dir).expanduser()
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")
    
    output_path = Path(args_cli.output_dir).expanduser()
    
    # Discover files
    if args_cli.recursive:
        csv_files = sorted(input_path.rglob(args_cli.pattern))
    else:
        csv_files = sorted(input_path.glob(args_cli.pattern))
    
    if not csv_files:
        print(f"[WARNING]: No files matching pattern '{args_cli.pattern}' found in {input_path}")
        return
    
    print(f"[INFO]: Found {len(csv_files)} CSV files to process")
    print(f"[INFO]: Using {args_cli.num_envs} parallel environments")
    
    # Run parallel processing
    success_count, failed_files = run_parallel_simulator(
        sim, scene, joint_names, csv_files, input_path, output_path
    )
    
    # Print summary
    print(f"\n[INFO]: Conversion complete!")
    print(f"[INFO]: Successfully converted: {success_count}/{len(csv_files)} files")
    if failed_files:
        print(f"[WARNING]: Failed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
