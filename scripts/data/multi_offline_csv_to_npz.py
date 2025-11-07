"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash
    
    # Batch mode (process all CSV files in a directory)
    python multi_offline_csv_to_npz.py --input_dir ./csv_motions --output_dir ./npz_motions --input_fps 30 --output_fps 50
"""

import argparse
import os
from pathlib import Path
from rich import print

args = argparse.ArgumentParser(description="Convert multiple motion CSV files to NPZ format.")
args.add_argument("--input_dir", type=str, required=True, help="The directory containing input CSV files.")
args.add_argument("--output_dir", type=str, required=True, help="The directory to save output NPZ files.")
args.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
args.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

if __name__ == "__main__":
    args_cli = args.parse_args()

    input_dir = Path(args_cli.input_dir).expanduser()
    output_dir = Path(args_cli.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(input_dir.glob("*.csv")))
    all_csv_num = len(csv_files)
    print(f"[INFO] Found {all_csv_num} CSV files in {input_dir}")

    for i, csv_file in enumerate(csv_files):
        output_file = output_dir / f"{csv_file.stem}.npz"
        command = (
            f"python scripts/data/offline_csv_to_npz.py "
            f"--input_file {csv_file} "
            f"--output_file {output_file} "
            f"--input_fps {args_cli.input_fps} "
            f"--output_fps {args_cli.output_fps} "
            f"--headless"
        )
        print(f"[INFO] Processing {i+1}/{all_csv_num} {csv_file} -> {output_file}")
        os.system(command)