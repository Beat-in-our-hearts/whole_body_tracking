import wandb
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--file", type=str, default="./motions/motion.npz", help="Path to the motion NPZ file to upload")
argparser.add_argument("--output", type=str, default="lafan_kungfu", help="Name of the output motion in wandb")
args_cli = argparser.parse_args()

REGISTRY_NAME = "motions"
COLLECTION_NAME = args_cli.output

run = wandb.init(project="csv_to_npz", name=COLLECTION_NAME)

logged_artifact = run.log_artifact(artifact_or_path=args_cli.file, name=COLLECTION_NAME, type=REGISTRY_NAME)

run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}")
