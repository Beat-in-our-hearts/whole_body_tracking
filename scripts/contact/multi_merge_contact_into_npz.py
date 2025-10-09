import numpy as np
import argparse
import glob
import os
import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_npz_dir", type=str, default="../../outputs/g1_npz_base/", help="Path to input NPZ file")
argparser.add_argument("--input_contact_dir", type=str, default="../../datasets/g1_contact/", help="Path to input contact NPY file")
argparser.add_argument("--output_npz_dir", type=str, default="../../datasets/g1_withcontact/", help="Path to output NPZ file")
args = argparser.parse_args()

all_npz_file = sorted(glob.glob(os.path.join(args.input_npz_dir, "*.npz")))
all_npy_file = sorted(glob.glob(os.path.join(args.input_contact_dir, "*.npy")))
os.makedirs(args.output_npz_dir, exist_ok=True)

for i in tqdm.trange(len(all_npz_file)):
    npz_base_name = os.path.basename(all_npz_file[i]).replace(".npz", "")
    npy_base_name = os.path.basename(all_npy_file[i]).replace(".npy", "")
    assert npz_base_name == npy_base_name, f"File names do not match: {npz_base_name} vs {npy_base_name}"
    
    out_npz_path = os.path.join(args.output_npz_dir, npz_base_name + "_withcontact.npz")

    motion_data = np.load(all_npz_file[i])
    contact_data = np.load(all_npy_file[i])  # [num_frames, 2]
    motion_dict = {}
    for key in motion_data.keys():
        motion_dict[key] = motion_data[key]
    joint_pos = motion_dict['joint_pos']  # [target_frames, num_joints, 3]
    target_frames = joint_pos.shape[0]
    contact_frames = contact_data.shape[0]
    target_contact_data = np.zeros((target_frames, 2))
    for i in range(target_frames):
        src_i = min(int(i * contact_frames / target_frames), contact_frames - 1)
        target_contact_data[i] = contact_data[src_i]

    target_contact_data = target_contact_data.astype(bool)
    motion_dict['contact'] = target_contact_data  # [target_frames, 2]

    np.savez(out_npz_path, **motion_dict)

