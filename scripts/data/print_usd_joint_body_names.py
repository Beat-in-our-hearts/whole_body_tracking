"""Script to print robot joint names and body names in order by launching the simulator."""

import argparse
import yaml
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Print robot joint and body names.")

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

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class PrintNamesSceneCfg(InteractiveSceneCfg):
    """Configuration for printing robot names scene."""

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


def print_robot_names(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Print joint names and body names of the robot."""
    robot = scene["robot"]

    # Get all joint names
    joint_names = robot.joint_names
    num_joints = len(joint_names)

    # Get all body names
    body_names = robot.body_names
    num_bodies = len(body_names)

    print("\n" + "=" * 80)
    print("ROBOT JOINT NAMES (in order)")
    print("=" * 80)
    for idx, joint_name in enumerate(joint_names):
        print(f"{idx:3d}: {joint_name}")

    print("\n" + "=" * 80)
    print("ROBOT BODY NAMES (in order)")
    print("=" * 80)
    for idx, body_name in enumerate(body_names):
        print(f"{idx:3d}: {body_name}")

    print("\n" + "=" * 80)
    print(f"Total Joints: {num_joints}")
    print(f"Total Bodies: {num_bodies}")
    print("=" * 80 + "\n")

    # Save to file
    output_file = Path(__file__).parent / "robot_names.yaml"
    data = {
        "joints": {
            "total": num_joints,
            "names": {idx: joint_name for idx, joint_name in enumerate(joint_names)}
        },
        "bodies": {
            "total": num_bodies,
            "names": {idx: body_name for idx, body_name in enumerate(body_names)}
        }
    }
    
    with open(output_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Results saved to: {output_file}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Design scene
    scene_cfg = PrintNamesSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Print robot names
    print_robot_names(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
