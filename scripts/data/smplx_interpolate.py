import numpy as np
import os
import argparse
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


class SMPLX_Interpolator:
    def __init__(self, target_fps: float):
        self._target_fps = target_fps
    
    def load_motion(self, file_path: str):
        data = np.load(file_path, allow_pickle=True)
        _gender = data.get("gender", "neutral")
        _surface_model_type = data.get("surface_model_type", "smplx")
        _mocap_frame_rate = data["mocap_frame_rate"]
        _root_orient = data["root_orient"]
        _trans = data["trans"]
        _poses = data["poses"]
        _betas = data["betas"]
        _pose_body = data["pose_body"]
        return {
            "gender": _gender,
            "surface_model_type": _surface_model_type,
            "mocap_frame_rate": _mocap_frame_rate,
            "root_orient": _root_orient,
            "trans": _trans,
            "pose_body": _pose_body,
            "poses": _poses,
            "betas": _betas,
        }
        
    def save_interpolated_motion(self, motion_data: dict[str, np.ndarray | str | float], save_path: str):
        data = {
            "gender": motion_data["gender"],
            "surface_model_type": motion_data["surface_model_type"],
            "mocap_frame_rate": self._target_fps,
            "betas": motion_data["betas"],
            "root_orient": motion_data["root_orient"],
            "trans": motion_data["trans"],
            "poses": motion_data["poses"],
            'pose_body': motion_data["pose_body"],
            'pose_hand': motion_data["poses"][:, 66:156], # 30*3
            'pose_jaw': motion_data["poses"][:, 156:159], # 3
            'pose_eye': motion_data["poses"][:, 159:165], # 2*3
        }
        # save npz
        np.savez(save_path, **data)
        
    def _interpolate_rotations(self, rotations: np.ndarray, original_fps: float) -> np.ndarray:
        """Interpolate rotations using SLERP.
        
        Args:
            rotations: (N, 3) array of axis-angle rotation vectors.
            original_fps: Original frames per second.
        Returns:
            Interpolated rotations as (M, 3) array of rotation vectors.
        """
        # Convert rotation vectors to quaternions
        n_frames = rotations.shape[0]
        rot_objs = Rotation.from_rotvec(rotations)
        original_times = np.arange(n_frames) / original_fps
        original_times[-1] += 1e-3  # Ensure the last time is included
        total_duration = original_times[-1]
        slerp = Slerp(original_times, rot_objs)
        
        # Compute target times
        target_times = np.arange(0, total_duration + 1e-6, 1.0 / self._target_fps)
        interp_rots = slerp(target_times)
        interp_rotvecs = interp_rots.as_rotvec()
        return interp_rotvecs
    
    def _interpolate_trans(self, trans: np.ndarray, original_fps: float) -> np.ndarray:
        """Interpolate translations using linear interpolation.
        
        Args:
            trans: (N, 3) array of translations.
            original_fps: Original frames per second.
        Returns:
            Interpolated translations as (M, 3) array.
        """

        n_frames = trans.shape[0]
        original_times = np.arange(n_frames) / original_fps
        total_duration = original_times[-1]
        target_times = np.arange(0, total_duration + 1e-6, 1.0 / self._target_fps)
        
        # Vectorized interpolation: interp1d supports (N, D) when axis=0
        interpolator = interp1d(
            original_times,
            trans,
            kind='linear',
            axis=0,
            assume_sorted=True,
            fill_value='extrapolate'
        )
        interp_trans = interpolator(target_times)
        return interp_trans
    
    def process_file(self, input_path: str, output_path: str):
        motion_data = self.load_motion(input_path)
        original_fps = motion_data["mocap_frame_rate"]
        
        # Interpolate
        interp_root_orient = self._interpolate_rotations(motion_data["root_orient"], original_fps).astype(np.float32) # (M, 3)
        interp_trans = self._interpolate_trans(motion_data["trans"], original_fps).astype(np.float32) # (M, 3)
        pose_body = motion_data["pose_body"].reshape(-1, 21, 3)  # (N, 21, 3)
        interp_pose_body_list = []
        for i in range(21):
            interp_body_part = self._interpolate_rotations(pose_body[:, i, :], original_fps).astype(np.float32)  # (M, 3)
            interp_pose_body_list.append(interp_body_part)
        interp_pose_body = np.stack(interp_pose_body_list, axis=1).reshape(-1, 63)  # (M, 63)
        
        # Update motion data
        new_motion_data = {
            "gender": motion_data["gender"],
            "surface_model_type": motion_data["surface_model_type"],
            "mocap_frame_rate": self._target_fps,
            "betas": motion_data["betas"],
            "root_orient": interp_root_orient,
            "trans": interp_trans,
            "pose_body": interp_pose_body,
            "pose_hand": np.zeros((interp_pose_body.shape[0], 90), dtype=np.float32), # Placeholder
            "pose_jaw": np.zeros((interp_pose_body.shape[0], 3), dtype=np.float32), # Placeholder
            "pose_eye": np.zeros((interp_pose_body.shape[0], 6), dtype=np.float32), # Placeholder
            "poses": np.concatenate([interp_root_orient, interp_pose_body,
                                     np.zeros((interp_pose_body.shape[0], 99), dtype=np.float32)], axis=1),
        }
        
        # save
        self.save_interpolated_motion(new_motion_data, output_path)
        
def main():
    parser = argparse.ArgumentParser(description="Interpolate SMPL-X motion data to a target FPS.")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npz file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output .npz file.")
    parser.add_argument("--target_fps", type=float, required=True, help="Target frames per second.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers for processing.")
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, os.path.basename(args.input))
        smplx_interpolator = SMPLX_Interpolator(target_fps=args.target_fps)
        smplx_interpolator.process_file(args.input, output_path)
        
    elif os.path.isdir(args.input):
        # Parallel processing for directory with progress bar
        
        # Recursively find all .npz files in input directory using glob
        input_files = sorted(glob.glob(os.path.join(args.input, '**', '*.npz'), recursive=True))
        
        if not input_files:
            print(f"No .npz files found in {args.input}")
            return
        
        # Create output directory structure
        def get_relative_output_path(input_file: str, input_dir: str, output_dir: str) -> str:
            """Get the output path preserving the directory structure."""
            rel_path = os.path.relpath(input_file, input_dir)
            return os.path.join(output_dir, rel_path)
        
        # Prepare output paths and create directories
        output_files = []
        for input_file in input_files:
            output_file = get_relative_output_path(input_file, args.input, args.output_dir)
            output_dir = os.path.dirname(output_file)
            os.makedirs(output_dir, exist_ok=True)
            output_files.append(output_file)
        
        # Process files in parallel with progress bar
        def process_file_wrapper(input_file: str, output_file: str, target_fps: float):
            """Wrapper for processing a single file with its own interpolator."""
            try:
                interpolator = SMPLX_Interpolator(target_fps=target_fps)
                interpolator.process_file(input_file, output_file)
                return input_file, output_file, None
            except Exception as e:
                return input_file, output_file, str(e)
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_file_wrapper, in_file, out_file, args.target_fps): (in_file, out_file)
                for in_file, out_file in zip(input_files, output_files)
            }
            
            # Display progress bar
            with tqdm(total=len(futures), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    input_file, output_file, error = future.result()
                    if error:
                        pbar.write(f"❌ Error processing {input_file}: {error}")
                    else:
                        rel_output = os.path.relpath(output_file, args.output_dir)
                        pbar.write(f"✓ {rel_output}")
                    pbar.update(1)
                    
                    
if __name__ == "__main__":
    main()