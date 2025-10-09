import glob
import os
import wandb
import argparse
import shutil

REGISTRY_NAME = "motions"
run = wandb.init(project="csv_to_npz", name="multi_upload_npz")

argparser = argparse.ArgumentParser()
argparser.add_argument("--file_dir", type=str, default="./datasets/g1_contact/", help="Path to the motion NPZ file to upload")
args_cli = argparser.parse_args()

all_npz_file = sorted(glob.glob(os.path.join(args_cli.file_dir, "*.npz")))

for i, npz_path in enumerate(all_npz_file):
    basename = os.path.basename(npz_path).replace(".npz", "")
    shutil.copyfile(npz_path, "/tmp/motion.npz")
    logged_artifact = run.log_artifact(artifact_or_path="/tmp/motion.npz", name=basename, type=REGISTRY_NAME)
    run.link_artifact(artifact=logged_artifact, target_path=f"wandb-registry-{REGISTRY_NAME}/{basename}")
    print(f"[INFO] [{i+1}/{len(all_npz_file)}] Uploaded motion {basename} to W&B")