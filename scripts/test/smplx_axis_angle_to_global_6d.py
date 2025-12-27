"""
Test script to visualize SMPL-X axis-angle rotations converted to 6D representation.
Visualizes 21 joints as RGB 3D arrows in a 3x7 subplot grid and saves as video.

Usage:
    python smplx_axis_angle_to_global_6d.py \
        --smplx_npz_path <path_to_smplx.npz> \
        --output_video <output.mp4> \
        --fps 30 \
        --frame_range 0 100
"""

import os
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for speed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path
import time
from multiprocessing import Pool
import tempfile
import subprocess
from tqdm import tqdm

# Import from my_math
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.my_math import (
    global_smplx_axis_angle_to_matrix
)
import torch

def load_and_convert_smplx(smplx_npz_path):
    """Load SMPL-X NPZ file and convert to local 6D representation."""
    print(f"Loading SMPL-X data from {smplx_npz_path}...")
    smplx_data = np.load(smplx_npz_path, allow_pickle=True)
    
    # Get pose_body: (N, 63) -> reshape to (N, 21, 3)
    pose_body = smplx_data['pose_body'].reshape(-1, 21, 3)
    N = pose_body.shape[0]
    print(f"Loaded {N} frames with 21 joints each")
    
    # Convert to torch tensor
    with torch.no_grad():
        pose_body_torch = torch.from_numpy(pose_body).float()  # (N, 21, 3)
    
        # Convert to 6D representation (local rotations) using my_math
        print("Converting axis-angle to global rotation matrices...")
        global_rot_mats = global_smplx_axis_angle_to_matrix(pose_body_torch)  # (N, 21, 3, 3) - torch tensor
    
    # Convert back to numpy for visualization
    print(f"Conversion complete: {global_rot_mats.shape}")
    return global_rot_mats.numpy(), N

def visualize_frame(frame_idx, rot_mats, ax_list, arrow_style='simple'):
    """
    Visualize a single frame with 21 joints as 3D arrows in 3x7 subplots.
    
    Args:
        frame_idx: Frame index
        rot_mats: Array of shape (21, 3, 3) representing rotation matrices for each joint
        ax_list: List of 21 matplotlib 3D axes
        arrow_style: 'simple' for lines (fast) or 'arrows' for quiver (slow)
    """
    
    # Clear all axes
    for ax in ax_list:
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.tick_params(labelsize=6)
    
    # Visualize each joint
    for joint_idx, ax in enumerate(ax_list):
        rot_mat = rot_mats[joint_idx]  # (3, 3)
        
        # Extract basis vectors (columns of rotation matrix)
        x_axis = rot_mat[:, 0]  # Red
        y_axis = rot_mat[:, 1]  # Green
        z_axis = rot_mat[:, 2]  # Blue
        
        # Origin
        origin = np.array([0, 0, 0])
        
        if arrow_style == 'simple':
            # Fast: use simple lines instead of quiver
            ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 
                   'r-', linewidth=2)
            ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 
                   'g-', linewidth=2)
            ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 
                   'b-', linewidth=2)
        else:
            # Slow: use quiver arrows
            ax.quiver(origin[0], origin[1], origin[2], 
                     x_axis[0], x_axis[1], x_axis[2], 
                     color='r', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(origin[0], origin[1], origin[2], 
                     y_axis[0], y_axis[1], y_axis[2], 
                     color='g', arrow_length_ratio=0.2, linewidth=2)
            ax.quiver(origin[0], origin[1], origin[2], 
                     z_axis[0], z_axis[1], z_axis[2], 
                     color='b', arrow_length_ratio=0.2, linewidth=2)
        
        # Set title with joint index
        ax.set_title(f'J{joint_idx}', fontsize=8)
        ax.grid(True, alpha=0.2)


def render_frame_to_file(args):
    """Render a single frame and save as PNG file."""
    frame_idx, rot_mats, frame_N, dpi, arrow_style, output_dir = args
    
    # Create figure with 3x7 subplots
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    ax_list = []
    for i in range(21):
        ax = fig.add_subplot(3, 7, i + 1, projection='3d')
        ax_list.append(ax)
    
    # Visualize frame
    visualize_frame(frame_idx, rot_mats, ax_list, arrow_style)
    fig.suptitle(f'Frame {frame_idx}/{frame_N}', fontsize=12)
    
    # Save as PNG
    output_path = Path(output_dir) / f"frame_{frame_idx:06d}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def create_video(global_rot_mats, output_video, fps=30, frame_range=None, interval=1, dpi=80, arrow_style='simple', num_workers=4):
    """
    Create video by rendering frames to PNG first, then encoding with FFmpeg.
    
    Args:
        global_rot_mats: Array of shape (N, 21, 3, 3) representing rotation matrices for each joint
        output_video: Output video file path
        fps: Frames per second
        frame_range: Tuple of (start_frame, end_frame) or None for all frames
        interval: Sample every interval-th frame
        dpi: DPI for figure rendering
        arrow_style: 'simple' or 'arrows'
        num_workers: Number of parallel rendering processes
    """
    N = global_rot_mats.shape[0]
    
    if frame_range is None:
        start_frame, end_frame = 0, N
    else:
        start_frame, end_frame = frame_range
        end_frame = min(end_frame, N)
    
    # Get frame indices with interval sampling
    frame_indices = list(range(start_frame, end_frame, interval))
    num_frames = len(frame_indices)
    
    print(f"\n{'='*70}")
    print(f"Creating video: {num_frames} frames")
    print(f"Settings: interval={interval}, dpi={dpi}, arrow_style={arrow_style}, workers={num_workers}")
    print(f"Frame range: {start_frame} to {end_frame}")
    print(f"Output: {output_video}")
    print(f"{'='*70}\n")
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix="frame_")
    print(f"Rendering frames to: {temp_dir}\n")
    
    try:
        # Prepare rendering tasks
        tasks = [(idx, global_rot_mats[idx], N, dpi, arrow_style, temp_dir) 
                 for idx in frame_indices]
        
        # Parallel rendering with progress bar
        print(f"Step 1: Rendering {num_frames} frames with {num_workers} workers...")
        with Pool(num_workers) as pool:
            list(tqdm(pool.imap_unordered(render_frame_to_file, tasks), 
                      total=num_frames, desc="Rendering"))
        
        print(f"\nStep 2: Encoding video with FFmpeg...")
        
        # FFmpeg command to create video from PNG sequence using glob pattern
        frame_pattern = str(Path(temp_dir) / "frame_*.png")
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-framerate', str(fps),
            '-pattern_type', 'glob',  # Use glob pattern for image sequence
            '-i', frame_pattern,
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions for h264
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',  # Quality (18-28, lower=better, default 23)
            str(output_video)
        ]
        
        # Run FFmpeg
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            raise RuntimeError("FFmpeg encoding failed")
        
        print(f"\n✓ Video saved to {output_video}")
        print(f"  Total frames: {num_frames}")
        print(f"  Output fps: {fps}\n")
        
    finally:
        # Clean up temporary directory
        import shutil
        print(f"Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Visualize SMPL-X rotations as global rotation matrices.")
    parser.add_argument("--smplx_npz_path", type=str, required=True, help="Path to SMPL-X NPZ file.")
    parser.add_argument("--output_video", type=str, required=True, help="Output video file path.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video.")
    parser.add_argument("--frame_range", nargs=2, type=int, metavar=("START", "END"),
                       help="Frame range to visualize (START END, both inclusive).")
    parser.add_argument("--interval", type=int, default=5, 
                       help="Sample every interval-th frame (1=all, 5=every 5th, etc). Use larger values to speed up.")
    parser.add_argument("--dpi", type=int, default=80,
                       help="DPI for rendering (lower=faster). Try 60-80 for speed, 120+ for quality.")
    parser.add_argument("--arrow_style", type=str, default='simple', choices=['simple', 'arrows'],
                       help="'simple'=fast lines, 'arrows'=slow quiver arrows")
    parser.add_argument("--num_workers", type=int, default=24,
                       help="Number of parallel workers for rendering frames")
    args = parser.parse_args()
    
    # Load and convert SMPL-X data
    global_rot_mats, N = load_and_convert_smplx(args.smplx_npz_path)
    
    os.makedirs(Path(args.output_video).parent, exist_ok=True)
    # Create video with optimizations
    create_video(
        global_rot_mats, 
        args.output_video, 
        fps=args.fps,
        frame_range=tuple(args.frame_range) if args.frame_range else None,
        interval=args.interval,
        dpi=args.dpi,
        arrow_style=args.arrow_style,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
