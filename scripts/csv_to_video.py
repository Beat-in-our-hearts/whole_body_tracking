#!/usr/bin/env python3
"""
Convert CSV mocap data to video with tracking camera
Usage:
    # Single file
    python csv_to_video_simple.py --xml robot.xml --input data.csv --output ./videos
    
    # Batch process directory
    python csv_to_video_simple.py --xml robot.xml --input ./csv_dir --output ./videos
"""

import mujoco
import csv
import numpy as np
import os
import tqdm
import concurrent.futures
import cv2
import glob
import argparse
from pathlib import Path


def csv2video_tracking(xml_path, csv_path, save_dir, temp_image_dir="/tmp/temp_images", 
                       video_wh=[1280, 720], track_body="pelvis",
                       distance=3.0, azimuth=120.0, elevation=-25.0, lookat=[0.0, 0.0, 0.3]):
    """
    Convert csv mocap data to video with tracking camera
    Args:
        xml_path (str): path to mujoco xml file
        csv_path (str): path to csv file
        save_dir (str): dir to save video
        temp_image_dir (str): temporary directory for images
        video_wh (list): video width and height [width, height]
        track_body (str): body name to track, default "pelvis"
        distance (float): camera distance
        azimuth (float): camera azimuth angle
        elevation (float): camera elevation angle
        lookat (list): camera lookat position [x, y, z]
    """
    # basename, example `dance1_subject1`
    basename = os.path.basename(csv_path).split(".")[0]
    os.makedirs(save_dir, exist_ok=True)
    frame_dir = os.path.join(temp_image_dir, basename)
    os.makedirs(frame_dir, exist_ok=True)
    
    csv_data = csv.reader(open(csv_path))
    np_data = np.array(list(csv_data)).astype(np.float32)

    # from xyzw to wxyz
    np_qpos = np_data.copy()
    np_qpos[:, 3:7] = np_data[:, [6, 3, 4, 5]]

    # mujoco
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    
    # render with tracking camera
    images = []
    with mujoco.Renderer(mj_model, width=video_wh[0], height=video_wh[1]) as renderer:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        track_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, track_body)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = track_id
        cam.fixedcamid = -1
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        cam.lookat = np.array(lookat)
        
        for i in tqdm.trange(np_qpos.shape[0], desc=f"Rendering {basename}"):
            mj_data.qpos[:] = np_qpos[i]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam)
            image = renderer.render()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            images.append(image)
    print(f"[INFO] Rendered {len(images)} images.")
    
    # save images (multithreaded with fallback)
    try:
        paths = [os.path.join(frame_dir, f"image_{i:04d}.png") for i in range(len(images))]
        # Use ThreadPoolExecutor to write images in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # submit all write tasks
            futures = [executor.submit(cv2.imwrite, paths[i], images[i]) for i in range(len(images))]
            # iterate over completed futures with progress bar
            for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Saving frames {basename}"):
                pass

        print(f"[INFO] Saved {len(images)} images to {frame_dir} (multithreaded)")
    except Exception as e:
        # If anything goes wrong with multithreaded saving, fall back to single-threaded
        print(f"[WARN] Multithreaded save failed, falling back to single-threaded save: {e}")
        for i in tqdm.trange(len(images), desc=f"Saving frames {basename}"):
            cv2.imwrite(os.path.join(frame_dir, f"image_{i:04d}.png"), images[i])
        print(f"[INFO] Saved {len(images)} images to {frame_dir}")
    
    # create video
    video_path = os.path.join(save_dir, basename + '.mp4')
    os.system(f"ffmpeg -y -framerate 30 -i {frame_dir}/image_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path} 2>&1 | grep -v 'frame='")
    print(f"[INFO] Saved video to {video_path}")
    
    # cleanup temp images
    for img_file in glob.glob(os.path.join(frame_dir, "*.png")):
        os.remove(img_file)
    os.rmdir(frame_dir)


def main():
    parser = argparse.ArgumentParser(description='Convert CSV mocap data to video with tracking camera')
    parser.add_argument('--xml', type=str, required=True, 
                        help='Path to MuJoCo XML file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file or directory containing CSV files')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output video directory')
    parser.add_argument('--temp', type=str, default='/tmp/temp_images',
                        help='Temporary directory for images (default: /tmp/temp_images)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Video width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Video height (default: 720)')
    parser.add_argument('--track-body', type=str, default='pelvis',
                        help='Body name to track (default: pelvis)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Check if input is a file or directory
    if input_path.is_file():
        if not input_path.suffix == '.csv':
            print(f"[ERROR] Input file must be a CSV file: {input_path}")
            return
        csv_files = [str(input_path)]
    elif input_path.is_dir():
        csv_files = sorted(glob.glob(os.path.join(str(input_path), "**/*.csv"), recursive=True))
        if len(csv_files) == 0:
            print(f"[ERROR] No CSV files found in directory: {input_path}")
            return
    else:
        print(f"[ERROR] Input path does not exist: {input_path}")
        return
    
    print(f"[INFO] Found {len(csv_files)} CSV file(s) to process")
    
    # Process each CSV file
    for csv_path in csv_files:
        print(f"\n[INFO] Processing: {csv_path}")
        try:
            csv2video_tracking(
                xml_path=args.xml,
                csv_path=csv_path,
                save_dir=args.output,
                temp_image_dir=args.temp,
                video_wh=[args.width, args.height],
                track_body=args.track_body,
                distance=3.0,
                azimuth=120.0,
                elevation=-25.0,
                lookat=[0.0, 0.0, 0.3]
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {csv_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n[INFO] All done! Processed {len(csv_files)} file(s)")


if __name__ == "__main__":
    main()
