import os
import argparse
import glob
import time

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv_dir", type=str, default="./datasets/g1", help="Path to input CSV directory")
args = argparser.parse_args()

csv_dir = args.csv_dir
all_csv_file = glob.glob(os.path.join(csv_dir, "*.csv"))
all_csv_file.sort()
print(f"Found {len(all_csv_file)} CSV files")

for i, csv_path in enumerate(all_csv_file):
    print(f"Processing {csv_path} ...")
    basename = os.path.basename(csv_path).replace(".csv", "")
    # convert to npz
    # python scripts/csv_to_npz.py --input_file ./datasets/g1/dance1_subject3.csv --input_fps 30 --output_name dance1_subject3 --headless
    os.system(f"python scripts/csv_to_npz.py --input_file {csv_path}  --input_fps 30 --output_name {basename} --headless")
    print(f"[INFO] [i/{len(all_csv_file)}] Saved motion {basename}")
