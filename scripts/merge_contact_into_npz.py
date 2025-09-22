import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--input_npz", type=str, default="./outputs/armswing_origin.npz", help="Path to input NPZ file")
argparser.add_argument("--input_contact", type=str, default="./outputs/armswing_contact.npy", help="Path to input contact NPY file")
argparser.add_argument("--output_npz", type=str, default="./outputs/armswing_withcontact.npz", help="Path to output NPZ file")
args = argparser.parse_args()

# 加载数据
motion_data = np.load(args.input_npz)
contact_data = np.load(args.input_contact)  # [num_frames, 2]

# 提取所有数据到字典中
motion_dict = {}
for key in motion_data.keys():
    motion_dict[key] = motion_data[key]

# 获取关节位置数据并计算帧数
joint_pos = motion_dict['joint_pos']  # [target_frames, num_joints, 3]
target_frames = joint_pos.shape[0]
contact_frames = contact_data.shape[0]

# 对齐帧数
target_contact_data = np.zeros((target_frames, 2))
for i in range(target_frames):
    src_i = min(int(i * contact_frames / target_frames), contact_frames - 1)
    target_contact_data[i] = contact_data[src_i]

# int to bool
target_contact_data = target_contact_data.astype(bool)

# 将接触数据添加到字典中
motion_dict['contact'] = target_contact_data  # [target_frames, 2]

# 保存为新的 NPZ 文件
np.savez(args.output_npz, **motion_dict)

print("文件保存成功！")
print("包含的键:", list(motion_dict.keys()))