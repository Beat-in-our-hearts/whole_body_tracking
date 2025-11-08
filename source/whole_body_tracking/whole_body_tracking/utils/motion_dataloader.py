"""Motion dataloader for loading and sampling motion data from NPZ files.

This module provides a PyTorch-style dataloader with weighted sampling support
and efficient vectorized batch indexing for multi-motion tracking in 
reinforcement learning environments.
"""

from collections.abc import Sequence
import torch

from whole_body_tracking.utils.motion_dataset import Motion_Dataset

class Motion_Dataloader:
    """Dataloader for sampling motion clips with optional weighted sampling.
    
    Uses efficient concatenation + offset indexing for vectorized batch access.
    All motion sequences are concatenated into single tensors with offset tracking.
    
    Args:
        dataset: Motion_Dataset instance
        body_indexes: Indices of bodies to track in motions
        device: Device to load tensors on
        
    Example:
        >>> dataset = Motion_Dataset(...)
        >>> dataloader = Motion_Dataloader(dataset, body_indexes=[0, 1, 2])
        >>> 
        >>> # Vectorized batch indexing (4096 envs)
        >>> motion_ids = torch.randint(0, 40, (4096,))
        >>> time_steps = torch.randint(0, 100, (4096,))
        >>> joint_pos = dataloader.batch_index(motion_ids, time_steps, 'joint_pos')
        >>> 
        >>> # Uniform sampling
        >>> indices = dataloader.sample(n=10)
        >>> 
        >>> # Weighted sampling
        >>> weights = compute_weights(...)
        >>> indices = dataloader.sample(n=10, weights=weights)
    """
    
    def __init__(
        self,
        dataset: Motion_Dataset,
        body_indexes: Sequence[int],
        device: str = "cuda"
    ):
        """Initialize the dataloader with concatenated sequences.
        
        Args:
            dataset: Motion_Dataset instance
            body_indexes: Body indices to track
            device: Device to load tensors on
        """
        self.dataset = dataset
        self.body_indexes = body_indexes
        self.device = device
        self.num_motions = len(dataset)
        
        print(f"[Motion_Dataloader] Loading and concatenating {self.num_motions} motions...")
        
        # Load all motions and concatenate into single tensors
        self._preload_and_concatenate()
        
        print(f"[Motion_Dataloader] Initialization complete. Total frames: {self.motion_buffer['joint_pos'].shape[0]}")
    
    def _preload_and_concatenate(self):
        """Preload all motions and concatenate into single tensors with offset tracking.
        
        This method loads all motion data upfront and concatenates sequences along
        the time dimension. Each motion's starting position is tracked in offsets.
        
        Memory-efficient: No padding, only raw data storage.
        """
        # Temporary lists for collecting data
        data_lists = {
            'joint_pos': [],
            'joint_vel': [],
            'body_pos_w': [],
            'body_quat_w': [],
            'body_lin_vel_w': [],
            'body_ang_vel_w': [],
        }
        lengths = []
        fps_list = []
        
        # Load all motions
        for i in range(self.num_motions):
            sample = self.dataset[i]
            motion_data = sample["motion"]
            
            # Append to lists
            for key in data_lists.keys():
                data_lists[key].append(
                    torch.tensor(motion_data[key], dtype=torch.float32, device=self.device)
                )
            
            lengths.append(sample["length"])
            fps_list.append(sample["fps"])
        
        # Concatenate all sequences along time dimension (axis 0)
        self.motion_buffer = {
            key: torch.cat(data_list, dim=0)
            for key, data_list in data_lists.items()
        }
        # motion_buffer contains:
        #   'joint_pos': [sum_T, num_joints]
        #   'joint_vel': [sum_T, num_joints]
        #   'body_pos_w': [sum_T, num_bodies, 3]
        #   'body_quat_w': [sum_T, num_bodies, 4]
        #   'body_lin_vel_w': [sum_T, num_bodies, 3]
        #   'body_ang_vel_w': [sum_T, num_bodies, 3]
        
        # Compute offsets for each motion (cumulative sum of lengths)
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)  # [num_motions]
        self.motion_offsets = torch.cat([
            torch.tensor([0], device=self.device),
            torch.cumsum(self.motion_lengths, dim=0)[:-1]
        ], dim=0)  # [num_motions], offsets[i] = starting index of motion i
        
        # Store FPS for each motion
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)
        
        print(f"[Motion_Dataloader] Concatenated tensors:")
        print(f"  joint_pos: {self.motion_buffer['joint_pos'].shape}")
        print(f"  body_pos_w: {self.motion_buffer['body_pos_w'].shape}")
        print(f"  motion_lengths: {self.motion_lengths.shape}, range: [{self.motion_lengths.min()}, {self.motion_lengths.max()}]")
        print(f"  motion_offsets: {self.motion_offsets.shape}")
    
    def batch_index(
        self,
        motion_ids: torch.Tensor,
        time_steps: torch.Tensor,
        data_key: str,
    ) -> torch.Tensor:
        """Vectorized batch indexing for multi-environment motion data.
        
        This is the core high-performance method that replaces loops with
        pure tensor operations for GPU acceleration.
        
        Args:
            motion_ids: [N] tensor, motion index for each environment (0 to num_motions-1)
            time_steps: [N] tensor, current time step for each environment
            data_key: Key of data to retrieve, one of:
                - 'joint_pos': Joint positions
                - 'joint_vel': Joint velocities
                - 'body_pos_w': Body positions (world frame)
                - 'body_quat_w': Body quaternions (world frame)
                - 'body_lin_vel_w': Body linear velocities (world frame)
                - 'body_ang_vel_w': Body angular velocities (world frame)
        
        Returns:
            Tensor of shape [N, ...] containing the requested data for each environment
        
        Example:
            >>> motion_ids = torch.tensor([0, 1, 0, 2], device='cuda')  # 4 envs
            >>> time_steps = torch.tensor([10, 20, 15, 5], device='cuda')
            >>> joint_pos = dataloader.batch_index(motion_ids, time_steps, 'joint_pos')
            >>> # joint_pos.shape = [4, num_joints]
        """
        # Safety: clamp time_steps to valid range for each motion
        max_time_steps = self.motion_lengths[motion_ids] - 1  # [N]
        # Use torch.clamp with tensor min/max (element-wise clamping)
        clamped_time_steps = torch.clamp(time_steps, min=torch.tensor(0, device=time_steps.device))
        clamped_time_steps = torch.minimum(clamped_time_steps, max_time_steps)
        
        # Compute global indices: offsets[motion_id] + time_step
        batch_offsets = self.motion_offsets[motion_ids]      # [N]
        global_indices = batch_offsets + clamped_time_steps  # [N]
        
        # Select data tensor from motion buffer
        if data_key not in self.motion_buffer:
            raise ValueError(
                f"Unknown data_key: {data_key}. "
                f"Available keys: {list(self.motion_buffer.keys())}"
            )
        
        data = self.motion_buffer[data_key]
        
        # Advanced indexing: data[global_indices]
        result = data[global_indices]  # [N, ...]
        
        # Filter body_indexes for body-related data
        if data_key.startswith('body_'):
            result = result[:, self.body_indexes]  # [N, num_tracked_bodies, ...]
        
        return result
    
    def get_motion_length(self, motion_id: int) -> int:
        """Get length of a specific motion.
        
        Args:
            motion_id: Index of motion
            
        Returns:
            Number of frames in the motion
        """
        return self.motion_lengths[motion_id].item()
    
    def get_motion_fps(self, motion_id: int) -> float:
        """Get FPS of a specific motion.
        
        Args:
            motion_id: Index of motion
            
        Returns:
            FPS value
        """
        return self.motion_fps[motion_id].item()
    
    def sample(self, n: int, weights: torch.Tensor | list | None = None) -> torch.Tensor:
        """Sample n motion indices with optional weights.
        
        Args:
            n: Number of motion clips to sample
            weights: Optional [num_motions] tensor or list of sampling weights.
                    If None, uniform sampling is used.
                    Weights will be normalized internally.
        
        Returns:
            motion_indices: Tensor[n], sampled motion indices in dataset
            
        Example:
            # Uniform sampling
            indices = dataloader.sample(10)
            
            # Weighted sampling based on quantity
            weights = [0.85 if q==1 else 0.10 if q==2 else 0.05 
                      for q in dataset.quantities]
            indices = dataloader.sample(10, weights=weights)
            
            # Custom adaptive sampling
            weights = curriculum_weights * difficulty_scores * diversity_penalty
            indices = dataloader.sample(10, weights=weights)
        """
        if weights is None:
            # Uniform sampling
            weights = torch.ones(self.num_motions, device=self.device)
        else:
            # Convert to tensor if needed
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            else:
                weights = weights.to(self.device)
            
            # Validate shape
            if weights.shape[0] != self.num_motions:
                raise ValueError(
                    f"Weights shape mismatch: expected [{self.num_motions}], got {weights.shape}"
                )
        
        # Ensure positive weights
        weights = torch.clamp(weights, min=1e-8)
        
        # Normalize
        weights = weights / weights.sum()
        
        # Sample
        motion_indices = torch.multinomial(weights, n, replacement=True)
        
        return motion_indices
    

if __name__ == "__main__":
    # Example usage and testing
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Test Motion_Dataloader")
    parser.add_argument(
        "--dataset_dirs",
        type=str,
        nargs="+",
        default=["./datasets/npz_datasets/LAFAN1_Retargeting_Dataset"],
        help="Dataset directory paths",
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default="g1",
        help="Robot name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()
    
    # Create dataset
    print("Creating dataset...")
    dataset = Motion_Dataset(
        dataset_dirs=args.dataset_dirs,
        robot_name=args.robot_name,
        split="train",
    )
    
    # Create dataloader
    print("\nCreating dataloader...")
    body_indexes = list(range(10))  # Track first 10 bodies for testing
    dataloader = Motion_Dataloader(
        dataset=dataset,
        body_indexes=body_indexes,
        device=args.device,
    )
    
    # Test uniform sampling
    print("\n=== Test 1: Uniform Sampling ===")
    indices = dataloader.sample(n=5)
    print(f"Sampled indices: {indices}")
    
    # Test weighted sampling
    print("\n=== Test 2: Weighted Sampling ===")
    weights = [0.85 if q == 1 else 0.10 if q == 2 else 0.05 for q in dataset.quantities]
    indices = dataloader.sample(n=20, weights=weights)
    print(f"Sampled indices: {indices}")
    quantities = [dataset.quantities[idx] for idx in indices.tolist()]
    print(f"Sampled quantities: {quantities}")
    
    # Test batch indexing
    print("\n=== Test 3: Batch Indexing ===")
    motion_ids = indices[:3]
    time_steps = torch.tensor([10, 20, 15], device=args.device)
    
    joint_pos = dataloader.batch_index(motion_ids, time_steps, 'joint_pos')
    body_pos_w = dataloader.batch_index(motion_ids, time_steps, 'body_pos_w')
    
    print(f"Batch indexed 3 motions:")
    print(f"  joint_pos shape: {joint_pos.shape}")
    print(f"  body_pos_w shape: {body_pos_w.shape}")
    
    # Test motion info
    print(f"\n=== Test 4: Motion Info ===")
    for i in range(min(3, len(dataset))):
        length = dataloader.get_motion_length(i)
        fps = dataloader.get_motion_fps(i)
        print(f"Motion {i}: length={length} frames, fps={fps}")
    
    print("\n✓ All tests passed!")
    
    
    # Test 4096 env batch indexing
    # Elapsed time for 4096 env batch indexing: 7.983456134796143 ms
    print("\n=== Test 5: 4096 Env Batch Indexing ===")
    num_envs = 4096
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    motion_ids = dataloader.sample(n=num_envs)
    time_steps = torch.randint(0, 100, (num_envs,), device=args.device)
    joint_pos = dataloader.batch_index(motion_ids, time_steps, 'joint_pos')
    joint_vel = dataloader.batch_index(motion_ids, time_steps, 'joint_vel')
    body_pos_w = dataloader.batch_index(motion_ids, time_steps, 'body_pos_w')
    body_quat_w = dataloader.batch_index(motion_ids, time_steps, 'body_quat_w')
    body_lin_vel_w = dataloader.batch_index(motion_ids, time_steps, 'body_lin_vel_w')
    body_ang_vel_w = dataloader.batch_index(motion_ids, time_steps, 'body_ang_vel_w')
    
    end_time.record()
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time)
    print(f"Elapsed time for 4096 env batch indexing: {elapsed_time} ms")
    print(f"  joint_pos shape: {joint_pos.shape}")