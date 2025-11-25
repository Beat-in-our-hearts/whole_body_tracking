"""Unified motion dataset for pairing robot NPZs with SMPL-X NPZs.

This module provides `Unify_Motion_Dataset` which reads dataset info
from robot dataset `info.yaml` files and pairs each robot NPZ with a
corresponding SMPL-X NPZ found in the provided `smplx_dataset` roots.

Contract:
- `robot_dataset` is a dict mapping dataset_dir -> list of splits. Robot
  dataset layout: `dataset_dir/robot_name/[split]/{motion_name}.npz` and
  must contain `info.yaml` describing splits and motion names.
- `smplx_dataset` is a dict mapping dataset_dir -> list of splits. SMPL-X
  dataset layout: `dataset_dir/[split]/{motion_name}.npz`.

Behavior:
- The authoritative info comes from `info.yaml` files in robot datasets.
- For every motion referenced in those splits, the loader will ensure a
  robot NPZ exists and will search the provided SMPL-X dataset roots for
  a matching SMPL-X NPZ in the same split name. If no SMPL-X NPZ is
  found for a motion, an exception is raised.

The loader uses separate readers for robot and SMPL-X NPZs because their
array keys may differ.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from torch.utils.data import Dataset


class Unify_Motion_Dataset(Dataset):
    """Dataset that pairs robot NPZs with SMPL-X NPZs.

    Args:
        robot_dataset: dict mapping dataset_dir -> list of splits (e.g. {"/path/to/dir": ["train", "val"]})
        smplx_dataset: list of dataset directories (e.g. ["/path/to/smplx"])
        robot_name: robot folder name under robot_dataset directories
    """

    def __init__(
        self,
        robot_dataset: Dict[str, List[str]],
        smplx_dataset: List[str],
        robot_name: str,
    ) -> None:
        super().__init__()
        self.robot_name = robot_name

        if len(robot_dataset) != len(smplx_dataset):
            raise ValueError("robot_dataset and smplx_dataset must have the same number of entries")
        
        # Normalize and resolve dataset paths
        self.robot_roots: list[Path] = [Path(os.path.abspath(k)) for k in robot_dataset.keys()]
        self.smplx_roots: list[Path] = [Path(os.path.abspath(k)) for k in smplx_dataset]
        self.robot_splits: list[List[str]] = list(robot_dataset.values())

        # Storage for paired paths and metadata
        self.robot_npz_paths: List[Path] = []
        self.smplx_npz_paths: List[Path] = []
        self.motion_names: List[str] = []
        self.quantities: List[int] = []
        self.dataset_sources: List[str] = []

        # Build index from robot datasets' info.yaml
        self._build_index()

        print(f"[Unify_Motion_Dataset] Paired {len(self.robot_npz_paths)} clips")

    def _build_index(self) -> None:
        """Read info.yaml from each robot dataset and find corresponding npz files.

        info.yaml in robot_dataset is authoritative. For each motion referenced
        there we require both the robot NPZ and a matching SMPL-X NPZ to exist
        (SMPL-X is searched across all provided smplx roots under the same
        split name). If a SMPL-X NPZ cannot be found, raise FileNotFoundError.
        """
        for i, (robot_root, splits) in enumerate(zip(self.robot_roots, self.robot_splits)):
            # find info.yaml
            info_path: Optional[Path] = None
            for ext in (".yaml", ".yml"):
                candidate = robot_root / f"info{ext}"
                if candidate.exists():
                    info_path = candidate
                    break

            if info_path is None:
                raise FileNotFoundError(f"info.yaml not found in robot dataset: {robot_root}")

            with open(info_path, "r") as f:
                info = yaml.safe_load(f) or {}

            robot_dataset_name = info["dataset"]

            for split in splits:
                split_info = info.get(split, {})
                if not split_info:
                    raise ValueError(f"No '{split}' data in {robot_dataset_name} (checked {info_path})")

                # robot-specific directory
                robot_dir = robot_root / self.robot_name
                if not robot_dir.exists():
                    raise FileNotFoundError(f"Robot directory not found: {robot_dir}")

                for motion_name, quantity in split_info.items():
                    robot_npz = robot_dir / f"{motion_name}.npz"
                    if not robot_npz.exists():
                        raise FileNotFoundError(f"Robot NPZ not found: {robot_npz}")

                    # Find corresponding smplx npz among all smplx roots under same split
                    smplx_npz: Optional[Path] = None
                    smplx_root = self.smplx_roots[i]
                    candidate = smplx_root / f"{motion_name}.npz"
                    if candidate.exists():
                        smplx_npz = candidate
                    
                    if smplx_npz is None:
                        raise FileNotFoundError(
                            f"SMPL-X NPZ for motion '{motion_name}' (split '{split}') not found in any provided smplx_dataset roots"
                        )

                    self.robot_npz_paths.append(robot_npz)
                    self.smplx_npz_paths.append(smplx_npz)
                    self.motion_names.append(motion_name)
                    self.quantities.append(quantity)
                    self.dataset_sources.append(f"{robot_dataset_name}:{split}")

    def __len__(self) -> int:
        return len(self.robot_npz_paths)

    def _load_robot_npz(self, path: Path) -> Dict[str, Any]:
        data = np.load(path)
        # Expect keys similar to Motion_Dataset
        required = [
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Robot NPZ {path} missing keys: {missing}")

        motion = {k: data[k] for k in required}
        fps = int(data["fps"][0]) if "fps" in data else None
        length = motion["joint_pos"].shape[0]
        duration = length / fps if fps else None

        return {
            "motion": motion,
            "fps": fps,
            "length": length,
            "duration": duration,
            "npz_path": str(path),
        }

    def _load_smplx_npz(self, path: Path) -> Dict[str, Any]:
        data = np.load(path)
        # SMPL-X NPZs may use different keys. We'll expose raw motion and try
        # to detect fps/length when possible.
        motion = {k: data[k] for k in data.files}

        fps = None
        if "fps" in data:
            fps = int(data["fps"])
        elif "frame_rate" in data:
            fps = int(data["frame_rate"])
        elif "mocap_frame_rate" in data:
            fps = int(data["mocap_frame_rate"])
        else:
            raise ValueError(f"SMPL-X NPZ {path} missing fps/frame_rate/mocap_frame_rate key")

        # Try to infer length from any array with shape (T, ...)
        length = None
        for arr in motion.values():
            if isinstance(arr, np.ndarray) and arr.ndim >= 1:
                length = arr.shape[0]
                break

        duration = length / fps if (fps and length) else None

        return {
            "motion": motion,
            "fps": fps,
            "length": length,
            "duration": duration,
            "npz_path": str(path),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        robot_path = self.robot_npz_paths[idx]
        smplx_path = self.smplx_npz_paths[idx]

        robot_item = self._load_robot_npz(robot_path)
        smplx_item = self._load_smplx_npz(smplx_path)

        return {
            "motion_name": self.motion_names[idx],
            "quantity": self.quantities[idx],
            "dataset_source": self.dataset_sources[idx],
            "robot": robot_item,
            "smplx": smplx_item,
        }

    def get_motion_info(self) -> List[Dict[str, Any]]:
        info_list: List[Dict[str, Any]] = []
        for i in range(len(self)):
            robot_path = self.robot_npz_paths[i]
            robot_data = np.load(robot_path)
            fps = int(robot_data["fps"][0]) if "fps" in robot_data else None
            length = robot_data["joint_pos"].shape[0]
            info_list.append(
                {
                    "index": i,
                    "motion_name": self.motion_names[i],
                    "robot_npz": str(robot_path),
                    "smplx_npz": str(self.smplx_npz_paths[i]),
                    "quantity": self.quantities[i],
                    "fps": fps,
                    "length": length,
                    "duration": length / fps if fps else None,
                    "dataset_source": self.dataset_sources[i],
                }
            )
        return info_list

    def get_statistics(self) -> Dict[str, Any]:
        total_frames = 0
        total_duration = 0.0
        lengths: List[int] = []

        for i in range(len(self)):
            robot_data = np.load(self.robot_npz_paths[i])
            fps = int(robot_data["fps"][0]) if "fps" in robot_data else None
            length = robot_data["joint_pos"].shape[0]
            duration = length / fps if fps else 0
            total_frames += length
            total_duration += duration
            lengths.append(length)

        return {
            "num_clips": len(self),
            "total_frames": total_frames,
            "total_duration": total_duration,
            "avg_frames_per_clip": total_frames / len(self) if len(self) > 0 else 0,
            "avg_duration_per_clip": total_duration / len(self) if len(self) > 0 else 0,
            "min_frames": min(lengths) if lengths else 0,
            "max_frames": max(lengths) if lengths else 0,
            "quantity_distribution": self._quantity_stats(),
        }

    def _quantity_stats(self) -> Dict[int, int]:
        stats: Dict[int, int] = {}
        for q in self.quantities:
            stats[q] = stats.get(q, 0) + 1
        return stats


if __name__ == "__main__":
    # Small example usage demonstration (not a full test)
    import argparse

    parser = argparse.ArgumentParser("Unify_Motion_Dataset demo")
    parser.add_argument("--robot_dataset", nargs="+", help="robot_dataset_dir:split1,split2 (repeatable)")
    parser.add_argument("--smplx_dataset", nargs="+", help="smplx_dataset_dir:split1,split2 (repeatable)")
    parser.add_argument("--robot_name", default="g1")
    args = parser.parse_args()

    def parse_map(items: Optional[List[str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        if not items:
            return out
        for item in items:
            if ":" not in item:
                raise ValueError("use format path:split1,split2")
            root, splits = item.split(":", 1)
            out[root] = splits.split(",")
        return out

    robot_map = parse_map(args.robot_dataset)
    smplx_map = parse_map(args.smplx_dataset)

    ds = Unify_Motion_Dataset(robot_map, smplx_map, args.robot_name)
    print(ds.get_statistics())
    data_1 = ds[0]
    print(data_1.keys())
    print(data_1["robot"]['motion'].keys())
    print(data_1["smplx"]['motion'].keys())
    print(data_1["robot"]['motion']['joint_pos'].shape)
    print(data_1["smplx"]['motion']['pose_body'].shape)
    print(data_1["smplx"]['motion']['poses'].shape)
    print(data_1["smplx"]['motion']['pose_eye'].shape)
    
    # check length consistency
    for i in range(len(ds)):
        robot_len = ds[i]["robot"]['motion']['joint_pos'].shape[0]
        smplx_len = ds[i]["smplx"]['motion']['pose_body'].shape[0]
        motion_name = ds[i]["motion_name"]
        if robot_len != smplx_len:
            print(f"Length mismatch in motion {motion_name}: robot {robot_len} vs smplx {smplx_len}")