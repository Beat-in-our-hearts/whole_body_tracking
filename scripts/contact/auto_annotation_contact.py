import mujoco
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import cv2
import glob


def cvs2video(xml_path, csv_path, save_dir, temp_image_dir="temp_images"): 
    """
    Convert csv mocap data to video
    Args:
        csv_path (str): path to csv file, example `../../datasets/g1/dance1_subject1.csv`
        save_dir (str): dir to save video, example `./videos`
    """
    # basename, example `dance1_subject1`
    basename = os.path.basename(csv_path).split(".")[0]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_image_dir, basename), exist_ok=True)
    
    csv_data = csv.reader(open(csv_path))
    np_data = np.array(list(csv_data)).astype(np.float32)

    # from xyzw to wxyz
    np_qpos = np_data.copy()
    np_qpos[:, 3:7] = np_data[:, [6, 3, 4, 5]]

    # mujoco
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    
    # render
    images = []
    with mujoco.Renderer(mj_model) as renderer:
        for i in tqdm.trange(np_qpos.shape[0]):
            mj_data.qpos[:] = np_qpos[i]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data)
            image = renderer.render()
            images.append(image)
    print(f"[INFO] Rendered {len(images)} images.")
    
    # save video, tqdm
    for i in tqdm.trange(len(images)):
        cv2.imwrite(os.path.join(temp_image_dir, basename, f"image_{i:04d}.png"), images[i])
    print(f"[INFO] Saved {len(images)} images to {os.path.join(temp_image_dir, basename)}")
    
    os.system(f"ffmpeg -framerate 30 -i {os.path.join(temp_image_dir, basename)}/image_%04d.png -c:v libx264 -pix_fmt yuv420p {os.path.join(save_dir, basename + '.mp4')}")
    print(f"[INFO] Saved video to {os.path.join(save_dir, basename + '.mp4')}")

 
def cvs_recorder_foot(xml_path, csv_path, save_dir, foot_names=["left_ankle_roll_link", "right_ankle_roll_link"]): 
    """
    Convert csv mocap data to video
    Args:
        csv_path (str): path to csv file, example `../../datasets/g1/dance1_subject1.csv`
        save_dir (str): dir to save video, example `./videos`
    """
    # basename, example `dance1_subject1`
    basename = os.path.basename(csv_path).split(".")[0]
    os.makedirs(save_dir, exist_ok=True)
    
    csv_data = csv.reader(open(csv_path))
    np_data = np.array(list(csv_data)).astype(np.float32)

    # from xyzw to wxyz
    np_qpos = np_data.copy()
    np_qpos[:, 3:7] = np_data[:, [6, 3, 4, 5]]

    # mujoco
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    array_foot_z = []
    for i in tqdm.trange(np_qpos.shape[0]):
        mj_data.qpos[:] = np_qpos[i]
        mujoco.mj_forward(mj_model, mj_data)
        # foot_pos
        foot_pos = np.array([mj_data.body(name).xpos for name in foot_names])
        foot_z = foot_pos[:, 2]  # (2,)
        array_foot_z.append(foot_z)
    np_foot_z = np.array(array_foot_z)  # (n_frames, 2)
    
    # plot foot_z
    plt.figure(figsize=(20, 5), dpi=300)
    plt.plot(np_foot_z[:, 0], label=foot_names[0])
    plt.plot(np_foot_z[:, 1], label=foot_names[1])
    plt.ylabel("Foot Height (m)")
    plt.xlim(left=0, right=np_foot_z.shape[0])
    plt.ylim(bottom=0)
    plt.legend()
    if save_dir:
        save_path = os.path.join(save_dir, basename + "_foot_z.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"[INFO] Saved foot z plot to {save_path}")
    plt.close()
    
def csv_auto_annotate_contact(xml_path, csv_path, save_dir, npy_dir, temp_image_dir="temp_images", foot_names=["left_toe_link", "right_toe_link"], height_threshold=0.01, save_video=False):
    """
    Auto annotate contact based on foot height
    Args:
        csv_path (str): path to csv file, example `../../datasets/g1/dance1_subject1.csv`
        save_dir (str): dir to save video, example `./videos`
        height_threshold (float): height threshold to determine contact
    """
    # basename, example `dance1_subject1`
    basename = os.path.basename(csv_path).split(".")[0]
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_image_dir, basename), exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    
    csv_data = csv.reader(open(csv_path))
    np_data = np.array(list(csv_data)).astype(np.float32)

    # from xyzw to wxyz
    np_qpos = np_data.copy()
    np_qpos[:, 3:7] = np_data[:, [6, 3, 4, 5]]

    # mujoco
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    array_foot_z = []
    images = []
    with mujoco.Renderer(mj_model) as renderer:
        for i in tqdm.trange(np_qpos.shape[0]):
            mj_data.qpos[:] = np_qpos[i]
            mujoco.mj_forward(mj_model, mj_data)
            if save_video:
                renderer.update_scene(mj_data)
                image = renderer.render()
                images.append(image)
            # foot_pos
            foot_pos = np.array([mj_data.body(name).xpos for name in foot_names])
            foot_z = foot_pos[:, 2]  # (2,)
            array_foot_z.append(foot_z)
    np_foot_z = np.array(array_foot_z)  # (n_frames, 2)

    # get base height
    first_10_frame_foot_z = np_foot_z[:10, :].mean(axis=0)
    foot_base_height = first_10_frame_foot_z.min()

    # determine contact
    contact_labels = (np_foot_z < foot_base_height+height_threshold).astype(np.int32)  # (n_frames, 2), 1 for contact, 0 for no contact
    np.save(os.path.join(npy_dir, basename + ".npy"), contact_labels)
    print(f"[INFO] Saved contact labels to {os.path.join(npy_dir, basename + '.npy')}")

    # puttext only '[0,1]' format
    if save_video:
        for i in tqdm.trange(len(images)):
            text = f"[{contact_labels[i, 0]}, {contact_labels[i, 1]}]"
            cv2.putText(images[i], text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(temp_image_dir, basename, f"image_{i:04d}.png"), images[i])
        print(f"[INFO] Saved {len(images)} images to {os.path.join(temp_image_dir, basename)}")
        
        os.system(f"ffmpeg -framerate 30 -i {os.path.join(temp_image_dir, basename)}/image_%04d.png -c:v libx264 -pix_fmt yuv420p {os.path.join(save_dir, basename + '_contact.mp4')}")
        print(f"[INFO] Saved video to {os.path.join(save_dir, basename + '.mp4')}")


def test_csv2video():
    csv_dir = "../../datasets/g1"
    all_csv_file = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_csv_file.sort()
    print(f"Found {len(all_csv_file)} CSV files")

    for csv_path in all_csv_file:
        cvs2video(
            xml_path="/home/ac/Desktop/2025/project_3/GMR/assets/unitree_g1/g1_mocap_29dof.xml",
            csv_path=csv_path,
            save_dir="./videos",
            temp_image_dir="./images"
        )

    print("All videos processed!")

def test_cvs_recorder_foot():
    csv_dir = "../../datasets/g1"
    all_csv_file = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_csv_file.sort()
    print(f"Found {len(all_csv_file)} CSV files")

    for csv_path in all_csv_file:
        cvs_recorder_foot(
            xml_path="/home/ac/Desktop/2025/project_3/GMR/assets/unitree_g1/g1_mocap_29dof.xml",
            csv_path=csv_path,
            save_dir="./images/foot_z",
            foot_names=["left_toe_link", "right_toe_link"]
        )

    print("All foot z plots processed!")
    
    
def test_csv_auto_annotate_contact():
    csv_dir = "../../datasets/g1"
    all_csv_file = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_csv_file.sort()
    print(f"Found {len(all_csv_file)} CSV files")
    
    for csv_path in all_csv_file:
        csv_auto_annotate_contact(
            xml_path="/home/ac/Desktop/2025/project_3/GMR/assets/unitree_g1/g1_mocap_29dof.xml",
            csv_path=csv_path,
            npy_dir="../../datasets/g1_contact",
            save_dir="./videos/contact",
            foot_names=["left_toe_link", "right_toe_link"],
            height_threshold=0.01
        )
    print("Auto annotation done!")
    
if __name__ == "__main__":
    # test_csv2video()
    # test_cvs_recorder_foot()
    test_csv_auto_annotate_contact()