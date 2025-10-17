import mujoco
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
import cv2
import glob
from concurrent.futures import ThreadPoolExecutor

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
    with mujoco.Renderer(mj_model, width=1920, height=1080) as renderer:
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
    
    def add_sphere_markers_to_xml(xml_path, foot_names):
        from xml.etree import ElementTree as ET
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        worldbody = root.find('worldbody')
        if worldbody is None:
            worldbody = ET.SubElement(root, 'worldbody')
        
        for idx, foot_name in enumerate(foot_names):
            body = ET.SubElement(worldbody, 'body', {
                'name': f"marker_{foot_name}",
                'mocap': 'true'  # 关键：设置为 mocap body
            })
            geom = ET.SubElement(body, 'geom', {
                'name': f'sphere_marker_{idx}',
                'type': 'sphere',
                'size': '0.05',
                'rgba': '0 0 1 1',  
                'contype': '0',  
                'conaffinity': '0',
                'group': '1' 
            })
            
        xml_dir = os.path.dirname(xml_path)
        temp_xml_path = os.path.join(xml_dir, "temp_with_markers.xml")
        tree.write(temp_xml_path, encoding='utf-8', xml_declaration=True)
        
        return temp_xml_path

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
    temp_xml_path = add_sphere_markers_to_xml(xml_path, foot_names)
    
    try:
        mj_model = mujoco.MjModel.from_xml_path(temp_xml_path)
        mj_data = mujoco.MjData(mj_model)
        
        mocap_body_indices = []
        marker_geom_indices = []
        marker_base_rgba = []
        for idx, foot_name in enumerate(foot_names):
            body_id = mj_model.body(f"marker_{foot_name}").id
            mocap_id = mj_model.body_mocapid[body_id]
            mocap_body_indices.append(mocap_id)

            geom_id = mj_model.geom(f"sphere_marker_{idx}").id
            marker_geom_indices.append(geom_id)
            marker_base_rgba.append(mj_model.geom_rgba[geom_id].copy())

        array_foot_z = []
        array_contact_state = []
        images = []
        
        foot_base_height = 0.0
        contact_flag = np.ones(len(foot_names), dtype=np.int32)
        
        with mujoco.Renderer(mj_model, width=1920, height=1080) as renderer:
            for i in tqdm.trange(np_qpos.shape[0]):
                mj_data.qpos[:] = np_qpos[i]
                mujoco.mj_forward(mj_model, mj_data)
            
                foot_pos = np.array([mj_data.body(name).xpos for name in foot_names])
                if i == 0:
                    foot_base_height = foot_pos[:, 2].min()
                
                low_thr = foot_base_height + height_threshold
                high_thr = foot_base_height + 2 * height_threshold
                contact_state = np.full(len(foot_names), 1, dtype=np.int32)
                contact_state[foot_pos[:, 2] < low_thr] = 0   # 接触
                contact_state[foot_pos[:, 2] > high_thr] = 2  # 悬空
                
                for idx, pos in enumerate(foot_pos):
                    mocap_id = mocap_body_indices[idx]
                    mj_data.mocap_pos[mocap_id] = pos
                    mj_data.mocap_quat[mocap_id] = [1, 0, 0, 0]

                    geom_id = marker_geom_indices[idx]
                    if contact_state[idx] == 0:
                        mj_model.geom_rgba[geom_id] = [1.0, 0.0, 0.0, 1.0]      # 红色
                    elif contact_state[idx] == 1:
                        mj_model.geom_rgba[geom_id] = [1.0, 0.6, 0.0, 1.0]      # 临界橙色
                    else:
                        mj_model.geom_rgba[geom_id] = marker_base_rgba[idx]     # 原色
                
                mujoco.mj_forward(mj_model, mj_data)
                    
                if save_video:     
                    renderer.update_scene(mj_data)
                    image = renderer.render()
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    images.append(image)
                    
                foot_z = foot_pos[:, 2]
                array_foot_z.append(foot_z)
                array_contact_state.append(contact_state.copy())
                
        np_foot_z = np.array(array_foot_z)  # (n_frames, 2)

        # get base height
        first_10_frame_foot_z = np_foot_z[:10, :].mean(axis=0)
        foot_base_height = first_10_frame_foot_z.min()

        # determine contact
        contact_labels = np.array(array_contact_state, dtype=np.int32)
        np.save(os.path.join(npy_dir, basename + ".npy"), contact_labels)
        print(f"[INFO] Saved contact labels to {os.path.join(npy_dir, basename + '.npy')}")

        # puttext only '[0,1]' format
        if save_video:
            for i in tqdm.trange(len(images)):
                text = f"[{contact_labels[i, 0]}, {contact_labels[i, 1]}]"
                cv2.putText(images[i], text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(temp_image_dir, basename, f"image_{i:04d}.png"), images[i])
            print(f"[INFO] Saved {len(images)} images to {os.path.join(temp_image_dir, basename)}")
            
            os.system(f"ffmpeg -framerate 30 -i {os.path.join(temp_image_dir, basename)}/image_%04d.png -c:v libx264 -pix_fmt yuv420p {os.path.join(save_dir, basename + '.mp4')}")
            print(f"[INFO] Saved video to {os.path.join(save_dir, basename + '.mp4')}")
    
    finally:
        # 清理临时文件
        if os.path.exists(temp_xml_path):
            os.remove(temp_xml_path)
            print(f"[INFO] Cleaned up temporary XML file: {temp_xml_path}")

def mj_csv_auto_annotate_contact(xml_path: str, 
                                 csv_path: str, 
                                 npy_dir: str, 
                                 save_dir: str, 
                                 temp_image_dir: str, 
                                 foot_names: list[str], 
                                 thresh: list[float], 
                                 save_video: bool=False, 
                                 video_wh:list[float]=[1920, 1080]):

    def _write_contact_frame(frame_dir: str, idx: int, image, labels) -> None:
        frame = image.copy()
        text = "[" + ", ".join(map(str, labels)) + "]"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(frame_dir, f"image_{idx:04d}.png"), frame)


    def save_contact_frames_simple(images, contact_labels, frame_dir: str) -> None:
        for idx, image in enumerate(images):
            _write_contact_frame(frame_dir, idx, image, contact_labels[idx])

    def _add_contact_geom(xml_path: str, foot_link_names=list[str]) -> str | None:
        from xml.etree import ElementTree as ET
        import tempfile
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        worldbody = root.find('worldbody')
        if worldbody is None:
            worldbody = ET.SubElement(root, 'worldbody')
        
        for foot_link_name in foot_link_names:
            # find body
            body = root.find(f".//body[@name='{foot_link_name}']")
            if body is None:
                print(f"[WARN] Foot link body '{foot_link_name}' not found in XML.")
                continue

            # add geom
            geom = ET.SubElement(body, "geom", {
                'name': f"{foot_link_name}",
                'type': 'mesh',
                'mesh': f"{foot_link_name}",
                'rgba': '0.7 0.7 0.7 1'
            })
            
            # add blue sphere marker geom
            sphere_body = ET.SubElement(worldbody, 'body', {
                'name': f"marker_{foot_link_name}",
                'mocap': 'true'  # 关键：设置为 mocap body
            })

            marker_geom = ET.SubElement(sphere_body, "geom", {
                'name': f'sphere_marker_{foot_link_name}',
                'type': 'sphere',
                'size': '0.05',
                'rgba': '0 0 1 1',
                'contype': '0',
                'conaffinity': '0',
                'group': '1'
            })

        # save to temp file
        xml_dir = os.path.dirname(os.path.abspath(xml_path)) or "."
        fd, temp_path = tempfile.mkstemp(dir=xml_dir, suffix="_contact_geoms.xml")
        os.close(fd)
        tree.write(temp_path, encoding="utf-8", xml_declaration=True)
        return temp_path
    
    # basename, example `dance1_subject1`
    basename = os.path.basename(csv_path).split(".")[0]
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_image_dir, basename), exist_ok=True)
    os.makedirs(os.path.join(temp_image_dir, 'plot_distance'), exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    
    csv_data = csv.reader(open(csv_path))
    np_data = np.array(list(csv_data)).astype(np.float32)

    # from xyzw to wxyz
    np_qpos = np_data.copy()
    np_qpos[:, 3:7] = np_data[:, [6, 3, 4, 5]]
    
    temp_xml_path = _add_contact_geom(xml_path, foot_names)
    
    try:
        mj_model = mujoco.MjModel.from_xml_path(temp_xml_path)
        mj_data = mujoco.MjData(mj_model)
        
        floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_geom_ids = {name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in foot_names}
        marker_geom_ids = {name: mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f'sphere_marker_{name}') for name in foot_names}

        base_foot_dist = {name: 0.0 for name in foot_names}
        distance = {name: [] for name in foot_names}
        contact_info = {name: [] for name in foot_names}
        images = []

        with mujoco.Renderer(mj_model, width=video_wh[0], height=video_wh[1]) as renderer:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            track_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = track_id
            cam.fixedcamid = -1
            cam.distance = 3.0
            cam.azimuth = 120.0
            cam.elevation = -25.0
            cam.lookat = np.array([0.0, 0.0, 0.3])
            
            for i in tqdm.trange(np_qpos.shape[0]):
                mj_data.qpos[:] = np_qpos[i]
                mujoco.mj_forward(mj_model, mj_data)
                
                foot_pos = np.array([mj_data.body(name).xpos for name in foot_names])
                
                # compute distance and contact
                for name, gid in foot_geom_ids.items():
                    fromto = np.zeros(6, dtype=np.float64)
                    dist = mujoco.mj_geomDistance(mj_model, mj_data, floor_geom_id, gid, 1.0, fromto)
                    distance[name].append(dist)
        
                    if i <= 50:
                        base_foot_dist[name] = max(base_foot_dist[name], dist)
                    if dist <= base_foot_dist[name] + thresh[0]:
                        contact_info[name].append(1)
                    elif dist >= base_foot_dist[name] + thresh[1]:
                        contact_info[name].append(0)
                    else: # middle status: 2
                        contact_info[name].append(2)

                # markers
                for name, pos in zip(foot_names, foot_pos):
                    mocap_id = mj_model.body_mocapid[mj_model.body(f"marker_{name}").id]
                    mj_data.mocap_pos[mocap_id] = pos
                    mj_data.mocap_quat[mocap_id] = [1, 0, 0, 0]
                    
                    # contact color, 1: red 0: blue 2: orange
                    geom_id = marker_geom_ids[name]
                    if contact_info[name][-1] == 1:
                        mj_model.geom_rgba[geom_id] = [1.0, 0.0, 0.0, 1.0]      # red
                    elif contact_info[name][-1] == 0:
                        mj_model.geom_rgba[geom_id] = [0.0, 0.0, 1.0, 1.0]      # blue
                    else:
                        mj_model.geom_rgba[geom_id] = [1.0, 0.6, 0.0, 1.0]      # orange

                mujoco.mj_forward(mj_model, mj_data)
                
                if save_video:     
                    renderer.update_scene(mj_data, camera=cam)
                    image = renderer.render()
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    images.append(image)
    
        # save contact info
        contact_labels = np.array([contact_info[name] for name in foot_names])
        contact_labels = contact_labels.T.astype(np.int32)  # (n_frames, n_feet)
        np.save(os.path.join(npy_dir, basename + ".npy"), contact_labels)
        print(f"[INFO] Saved contact labels to {os.path.join(npy_dir, basename + '.npy')}")
        
        # plot distance
        plt.figure(figsize=(40, 5), dpi=300)
        for name, dists in distance.items():
            plt.plot(dists, label=name, linewidth=1)
        plt.legend()
        plt.xlim(0, len(dists))
        plt.ylim(-0.03, 0.05)
        plt.savefig(os.path.join(temp_image_dir, 'plot_distance', f"{basename}.png"), bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
        if save_video:
            frame_dir = os.path.join(temp_image_dir, basename)
            max_workers = min(8, len(images))

            def _save(idx: int) -> None:
                _write_contact_frame(frame_dir, idx, images[idx], contact_labels[idx])

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm.tqdm(executor.map(_save, range(len(images))), total=len(images), desc="Annotating frames", leave=False))
            print(f"[INFO] Saved {len(images)} images to {frame_dir}")
            
            os.system(f"ffmpeg -framerate 30 -i {os.path.join(temp_image_dir, basename)}/image_%04d.png -c:v libx264 -pix_fmt yuv420p {os.path.join(save_dir, basename + '.mp4')}")
            print(f"[INFO] Saved video to {os.path.join(save_dir, basename + '.mp4')}")
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_xml_path):
            os.remove(temp_xml_path)
            print(f"[INFO] Cleaned up temporary XML file: {temp_xml_path}")
    

    

def test_csv2video():
    csv_dir = "../../datasets/g1"
    all_csv_file = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_csv_file.sort()
    print(f"Found {len(all_csv_file)} CSV files")

    for csv_path in all_csv_file:
        cvs2video(
            xml_path="/home/ac/Desktop/2025/project_3/GMR/assets/unitree_g1/g1_mocap_29dof.xml",
            csv_path=csv_path,
            save_dir="./videos/base",
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
            height_threshold=0.01,
            save_video=True,
        )
    print("Auto annotation done!")
    
    
def test_csv_auto_annotate_contact_v2():
    csv_dir = "../../datasets/g1"
    all_csv_file = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_csv_file.sort()
    print(f"Found {len(all_csv_file)} CSV files")
    
    csv_path = all_csv_file[0]
    # for csv_path in all_csv_file:
    csv_auto_annotate_contact(
        xml_path="/home/ac/Desktop/2025/project_3/GMR/assets/unitree_g1/g1_mocap_29dof.xml",
        csv_path=csv_path,
        npy_dir="../../datasets/g1_contact",
        save_dir="./videos/vis_contact",
        foot_names=["left_toe_link", "right_toe_link"],
        height_threshold=0.01,
        save_video=True
    )
    print("Auto annotation done!")
    

def test_mj_csv_auto_annotate_contact():
    csv_dir = "../../datasets/g1"
    all_csv_file = glob.glob(os.path.join(csv_dir, "*.csv"))
    all_csv_file.sort()
    print(f"Found {len(all_csv_file)} CSV files")
    
    for csv_path in all_csv_file:
        mj_csv_auto_annotate_contact(
            xml_path="/home/ac/Desktop/2025/project_3/GMR/assets/unitree_g1/g1_mocap_29dof.xml",
            csv_path=csv_path,
            npy_dir="../../datasets/g1_contact_mj",
            save_dir="./videos/vis/tmp",
            temp_image_dir="./temp_images/mj_vis_contact",
            foot_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            thresh=[0.003, 0.02],
            save_video=True,
            video_wh=[640, 480],
        )
    print("MJ Auto annotation done!")    

    
if __name__ == "__main__":
    # test_csv2video()
    # test_cvs_recorder_foot()
    # test_csv_auto_annotate_contact()
    # test_csv_auto_annotate_contact_v2()
    test_mj_csv_auto_annotate_contact()