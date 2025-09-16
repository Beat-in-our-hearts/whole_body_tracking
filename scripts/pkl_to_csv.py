"""
GMR Retargeting - Convert PKL to CSV
GMR PKL file:
pkl keys: ["fps", "root_pos", "root_rot", "dof_pos"]
root_rot format: xyzw
    
CSV file:
motion[:, :3]: motion_base_poss_input
motion[:, 3:7]: motion_base_rots_input (xyzw)
motion[:, 7:]: motion_dof_poss_input
no header
    
Author: Zuxing Lu
Data: 2025/09/16
"""
import os
import pickle
import pandas as pd
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_pkl", type=str, required=True, help="Path to input PKL file")
argparser.add_argument("--output_path", type=str, required=True, help="Path to output CSV file")
args = argparser.parse_args()

    
def pkl_to_csv(pkl_path, csv_path):
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    motion_base_poss_input = data["root_pos"]
    motion_base_rots_input = data["root_rot"]
    motion_dof_poss_input = data["dof_pos"]

    motion = np.concatenate([motion_base_poss_input, motion_base_rots_input, motion_dof_poss_input], axis=1)

    # check dir of csv_path
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(motion)
    csv_file_name = os.path.join(csv_path, os.path.basename(pkl_path).replace('.pkl', '.csv'))
    df.to_csv(csv_file_name, index=False)
    print(f"Converted {pkl_path} to {csv_file_name}")
    
if __name__ == "__main__":
    pkl_to_csv(args.input_pkl, args.output_path)