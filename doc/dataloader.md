# motion_dataloader

pytorch-based `Dataset` and `DataLoader` for loading motion data from NPZ files.

```python
class Motion_Dataset(torch.utils.data.Dataset):
    pass

class Motion_Dataloader(torch.utils.data.DataLoader):
    pass
```

## features:
- Load motion data from NPZ files.
- Support quantity-based sampling.
- Configurable via YAML files.
- Statistics all motion data frames length.
- Configurable train and val split.


## Dataset Information
- `dataset_dir`: ./datasets/npz_datasets/{dataset_name}/{robot_name}/
- `dataset_info`: ./datasets/npz_datasets/{dataset_name}/info.json
    - `dataset`: dataset name
    - `train`: dict, key: subject name, value: quantity of motion clips (1: best, 2: medium, 3: hard)
    - `val`: dict, key: subject name, value: quantity of motion clips


```json
{
    "dataset": "LAFAN1_Retargeting_Dataset",
    "train": {
        "dance1_subject1": 1, 
        // ... (more train subjects)
    },
    "val": {
    }
}
```

## data file format
- npz file format:
    - Each file contains motion data for a specific subject and action.
    - Keys: ['fps', 'joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w']

## Motion_Dataset

Based on `torch.utils.data.Dataset`.
Only storage the npz file paths and provide data loading interface. 
The data loading is implemented in `__getitem__` function for each npz file.
Why? Because the motion data is usually large, loading all data into memory may cause OOM.    

- `__init__`:
    - `dataset_dirs: list[str]`: dataset directory paths.
    - `robot_name: str`: robot name.
    - `split: str`: train or val split.

- `__len__`: return the number of npz files.

- `__getitem__`: load the motion data from npz file and return a dict:
    - `motion`: tuple of np.ndarray, motion data.
    - `fps`: int, frames per second of the motion data.
    - `length`: int, length of the motion data.
    - `npz_path`: str, path of the npz file.


## Motion_Dataloader

```
MultiMotionCommand.__init__()
  └─> Motion_Dataloader(dataset, body_indexes, device)
        └─> _preload_and_concatenate()
              ├─> 加载所有 NPZ 文件
              ├─> 拼接到 motion_buffer dict
              └─> 计算 motion_offsets

每帧更新:
  _update_command()
    ├─> motion_lengths[motion_indices] 检查是否结束
    ├─> _resample_command(env_ids) 如需重采样
    │     ├─> _compute_sampling_weights() 计算权重
    │     ├─> dataloader.sample(weights) 采样新 motions
    │     ├─> 自适应 bin 采样确定 time_steps
    │     └─> _init_robot_state(env_ids)
    │           └─> dataloader.batch_index() 批量获取数据 × 6 次
    └─> 更新相对姿态

Property 访问:
  joint_pos / body_pos_w / etc.
    └─> _batch_index_motions(data_key)
          └─> dataloader.batch_index(motion_indices, time_steps, data_key)
                ├─> global_indices = offsets[motion_ids] + time_steps
                └─> motion_buffer[data_key][global_indices]

```


Based on `torch.utils.data.DataLoader`, support quantity-based sampling and curriculum learning.
Different from standard DataLoader, not use the batch_size parameter.
It is used to RL training for `4096 x num_gpus` agents, each agent needs one motion clip.

### Core Features

#### 1. Dynamic Batch Size Sampling
Unlike standard DataLoader with fixed batch_size, this dataloader supports variable batch size sampling.
- When `n` agents reset (reset count varies in parallel RL), call `dataloader.sample(n)` to get `n` motion clips.
- Each call to `sample(n)` returns different number of motion clips based on reset count.
- Supports batch sizes from 1 to `num_envs` (e.g., 4096).

#### 2. Curriculum Learning via Quantity-based Adaptive Sampling
Motion clips are rated by quality/difficulty:
- **Quantity 1**: High quality, low difficulty (best for initial training)
- **Quantity 2**: Medium quality or difficulty
- **Quantity 3**: Low quality or high difficulty (for advanced training)

The dataloader implements **two-stage adaptive sampling** with curriculum learning:

**Stage 1: Motion-Level Sampling (Quantity-based Adaptive Sampling)**
First, select which motion to use based on quantity and motion-level difficulty.

**Stage 2: Time-Level Sampling (Bin-based Adaptive Sampling)**
Then, select time step within the chosen motion based on bin failure rates.

**Improved Curriculum Schedule with Conservative Initial Weights:**
```python
# Pseudo-code for curriculum schedule
def get_curriculum_weights(current_step, total_steps, schedule="cosine"):
    """
    Returns sampling weights for each quantity level.
    Uses conservative initial distribution: [0.85, 0.10, 0.05]
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        schedule: "linear", "cosine", or "step"
    
    Returns:
        weights: [w1, w2, w3] for quantity [1, 2, 3]
    """
    progress = current_step / total_steps  # 0.0 to 1.0
    
    if schedule == "cosine":
        # Smooth transition using cosine annealing
        alpha = (1 - math.cos(progress * math.pi)) / 2
    elif schedule == "linear":
        alpha = progress
    elif schedule == "step":
        # Step-wise curriculum (e.g., change every 25% of training)
        alpha = math.floor(progress * 4) / 4
    
    # Conservative initial weights: focus heavily on quantity 1
    # Initial: [0.85, 0.10, 0.05]
    # Final:   [0.50, 0.30, 0.20] (more balanced)
    w1 = 0.85 - alpha * 0.35  # From 0.85 to 0.50
    w2 = 0.10 + alpha * 0.20  # From 0.10 to 0.30
    w3 = 0.05 + alpha * 0.15  # From 0.05 to 0.20
    
    # Normalize (already normalized, but ensure)
    total = w1 + w2 + w3
    return [w1/total, w2/total, w3/total]
```

**Example Training Schedule:**
```
Training Progress:    0%      25%     50%     75%     100%
Quantity 1 weight:   0.85    0.76    0.68    0.59    0.50
Quantity 2 weight:   0.10    0.15    0.20    0.25    0.30
Quantity 3 weight:   0.05    0.09    0.13    0.16    0.20
```

**Why [0.85, 0.10, 0.05] Initial Distribution?**
- **Conservative Start**: 85% on best quality ensures stable initial learning
- **Limited Exploration**: Small exposure to harder data (10% + 5%) prevents early overfitting
- **Adaptive Refinement**: Combined with motion-level adaptive sampling, harder motions within each quantity level get naturally prioritized based on failure rates

#### 3. Motion Reuse Across Parallel Agents
In parallel RL with `4096 x num_gpus` agents, same motion clips may be assigned to multiple agents:
- **Challenge**: Same motion loaded to multiple agents reduces diversity
- **Solution**: Implement motion diversity scoring and smart assignment

**Motion Assignment Strategy:**
```python
# Pseudo-code for motion assignment with diversity
def sample_with_diversity(n, current_assignments, diversity_threshold=0.3):
    """
    Sample n motion clips with diversity constraint.
    
    Args:
        n: Number of motion clips needed
        current_assignments: Current motion indices assigned to all agents [num_envs]
        diversity_threshold: Max ratio of agents that can share same motion
    
    Returns:
        sampled_indices: [n] indices of motion clips
    """
    # Count current motion usage
    motion_usage_count = count_motion_usage(current_assignments)  # [num_motions]
    max_usage = num_envs * diversity_threshold
    
    # Get curriculum weights
    curriculum_weights = get_curriculum_weights(current_step, total_steps)
    
    # Adjust sampling probability based on:
    # 1. Curriculum weights (quantity-based)
    # 2. Current usage (diversity penalty)
    sampling_probs = []
    for i, motion in enumerate(all_motions):
        # Base probability from curriculum
        base_prob = curriculum_weights[motion.quantity - 1]
        
        # Diversity penalty (reduce prob if overused)
        usage_penalty = max(0, 1 - motion_usage_count[i] / max_usage)
        
        sampling_probs[i] = base_prob * usage_penalty
    
    # Normalize and sample
    sampling_probs = normalize(sampling_probs)
    sampled_indices = multinomial_sample(sampling_probs, n, replacement=True)
    
    return sampled_indices
```

#### 4. Simple and Flexible Interface
The dataloader provides a simple weighted sampling interface, giving full control to the caller:

**Core Interface:**
```python
class Motion_Dataloader:
    def __init__(self, dataset, device="cuda"):
        """
        Initialize the dataloader.
        
        Args:
            dataset: Motion_Dataset instance
            device: Device to load tensors on
        """
        self.dataset = dataset
        self.device = device
        self.num_motions = len(dataset)
        self.motion_loaders = {}  # Cache for loaded MotionLoader instances
        
    def sample(self, n, weights=None):
        """
        Sample n motion clips with optional weights.
        
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
        
        # Ensure positive weights
        weights = torch.clamp(weights, min=1e-8)
        
        # Normalize
        weights = weights / weights.sum()
        
        # Sample
        motion_indices = torch.multinomial(weights, n, replacement=True)
        
        return motion_indices
    
    def get_motion_loader(self, motion_idx):
        """
        Get or create MotionLoader for a specific motion.
        Loaders are cached for efficiency.
        
        Args:
            motion_idx: Index of motion in dataset
            
        Returns:
            MotionLoader instance compatible with commands.py::MotionLoader
        """
        if motion_idx not in self.motion_loaders:
            # Load motion data from dataset
            sample = self.dataset[motion_idx]
            
            # Create MotionLoader instance (convert numpy to torch)
            loader = self._create_motion_loader(sample)
            self.motion_loaders[motion_idx] = loader
        
        return self.motion_loaders[motion_idx]
    
    def get_motion_loaders(self, motion_indices):
        """
        Get MotionLoaders for multiple motions.
        
        Args:
            motion_indices: Tensor or list of motion indices
            
        Returns:
            List of MotionLoader instances
        """
        if isinstance(motion_indices, torch.Tensor):
            motion_indices = motion_indices.tolist()
        
        return [self.get_motion_loader(idx) for idx in motion_indices]
    
    def _create_motion_loader(self, sample):
        """
        Create MotionLoader instance from dataset sample.
        Compatible with commands.py::MotionLoader interface.
        """
        # This will be implemented to match MotionLoader in commands.py
        # Convert numpy arrays to torch tensors on device
        pass
```

**Why This Design?**

1. **Flexibility**: Caller decides sampling strategy (curriculum, adaptive, etc.)
2. **Simplicity**: Dataloader only handles weighted sampling, not policy
3. **Composability**: Easy to combine different weighting strategies
4. **Testability**: Each component can be tested independently
5. **No Hidden State**: No internal curriculum/diversity tracking

**Usage Examples:**

```python
# Example 1: Simple quantity-based curriculum
def get_quantity_weights(dataset, progress):
    """Get weights based on motion quantity and training progress."""
    alpha = (1 - math.cos(progress * math.pi)) / 2
    w1 = 0.85 - alpha * 0.35  # 0.85 -> 0.50
    w2 = 0.10 + alpha * 0.20  # 0.10 -> 0.30
    w3 = 0.05 + alpha * 0.15  # 0.05 -> 0.20
    
    weights = []
    for quantity in dataset.quantities:
        if quantity == 1:
            weights.append(w1)
        elif quantity == 2:
            weights.append(w2)
        else:
            weights.append(w3)
    return torch.tensor(weights)

# Sample with curriculum
progress = current_step / total_steps
weights = get_quantity_weights(dataset, progress)
indices = dataloader.sample(n=128, weights=weights)

# Example 2: Adaptive sampling with diversity
def get_adaptive_weights(dataset, bin_failed_counts, agent_motion_indices, 
                        progress, diversity_threshold):
    """Combine curriculum, difficulty, and diversity."""
    # Curriculum weights
    curriculum_weights = get_quantity_weights(dataset, progress)
    
    # Motion difficulty from bin failures
    motion_difficulty = torch.tensor([
        bin_failed_counts[i].mean() for i in range(len(dataset))
    ])
    
    # Diversity penalty
    motion_usage = torch.bincount(agent_motion_indices, minlength=len(dataset))
    max_usage = len(agent_motion_indices) * diversity_threshold
    diversity_penalty = torch.clamp(1.0 - motion_usage / max_usage, min=0.1)
    
    # Combine
    curriculum_alpha = 1.0 - progress * 0.5
    adaptive_alpha = progress * 0.5
    
    weights = (
        curriculum_alpha * curriculum_weights + 
        adaptive_alpha * (motion_difficulty + 0.1)
    ) * diversity_penalty
    
    return weights

# Sample with full strategy
weights = get_adaptive_weights(dataset, bin_failed_counts, 
                               agent_motion_indices, progress, 0.3)
indices = dataloader.sample(n=128, weights=weights)

# Example 3: Pure adaptive (ignore curriculum)
difficulty_weights = torch.tensor([
    bin_failed_counts[i].mean() for i in range(len(dataset))
])
indices = dataloader.sample(n=128, weights=difficulty_weights)
```

### Adaptive Sampling for Multi-Motion Tracking

The current `_adaptive_sampling` in `commands.py` works for single motion file. 
To adapt it for multi-motion tracking with curriculum learning:

#### Step 1: Extend Bin-based Failure Tracking to Multiple Motions

**Current (Single Motion):**
```python
# commands.py::MotionCommand
self.bin_count = motion.time_step_total // timesteps_per_bin + 1
self.bin_failed_count = Tensor[bin_count]  # Track failure rate per time bin
```

**Extended (Multiple Motions):**
```python
# New Multi-Motion Command
self.num_motions = len(dataset)
self.bin_counts = []  # Different motions have different lengths
self.bin_failed_counts = []  # List of tensors, one per motion

for motion_idx in range(self.num_motions):
    motion_length = dataset[motion_idx]['length']
    bin_count = motion_length // timesteps_per_bin + 1
    self.bin_counts.append(bin_count)
    self.bin_failed_counts.append(torch.zeros(bin_count))

# Track which motion is assigned to each agent
self.agent_motion_indices = torch.zeros(num_envs, dtype=torch.long)  # [num_envs]
self.agent_time_steps = torch.zeros(num_envs, dtype=torch.long)      # [num_envs]
```

#### Step 2: Motion-Level Adaptive Sampling

**Two-Stage Adaptive Sampling Strategy:**

The improved sampling strategy uses **hierarchical adaptive sampling**:
1. **First Stage**: Select motion_id based on quantity and motion-level difficulty
2. **Second Stage**: Select bin_id within the chosen motion based on bin failure rates

```python
def _adaptive_motion_sampling(self, env_ids):
    """
    Two-stage adaptive sampling for motion tracking.
    
    Stage 1: Motion-Level (Quantity-based + Motion-level adaptive)
        - Decide which motion to use (motion_id)
        - Combines curriculum learning with motion-level failure rates
    
    Stage 2: Time-Level (Bin-based adaptive)
        - Decide which time step within motion (bin_id)
        - Based on per-bin failure rates
    
    Combines:
    1. Curriculum learning (quantity-based)
    2. Motion-level adaptive sampling (motion difficulty from aggregated failures)
    3. Bin-level adaptive sampling (temporal difficulty within motion)
    4. Diversity constraint (avoid motion reuse)
    """
    n = len(env_ids)
    
    # ============================================================
    # STAGE 1: Motion-Level Sampling
    # ============================================================
    
    # 1. Get curriculum weights for each quantity level [3]
    #    Initial: [0.85, 0.10, 0.05] for quantity [1, 2, 3]
    quantity_curriculum_weights = self.get_curriculum_weights()  # [3]
    
    # 2. Build per-motion curriculum weights based on quantity
    #    Map each motion's quantity to its curriculum weight
    motion_curriculum_weights = torch.zeros(self.num_motions, device=self.device)
    for motion_idx in range(self.num_motions):
        quantity = self.dataset.quantities[motion_idx]  # 1, 2, or 3
        motion_curriculum_weights[motion_idx] = quantity_curriculum_weights[quantity - 1]
    
    # 3. Calculate motion-level difficulty from bin failure rates
    #    Average failure rate across all bins of each motion
    motion_difficulty = torch.zeros(self.num_motions, device=self.device)
    for motion_idx in range(self.num_motions):
        # Aggregate bin failures to get overall motion difficulty
        avg_failure = self.bin_failed_counts[motion_idx].mean()
        motion_difficulty[motion_idx] = avg_failure
    
    # 4. Combine curriculum and motion-level adaptive difficulty
    #    Early training: rely more on curriculum (quantity)
    #    Late training: rely more on adaptive difficulty
    progress = self.current_step / self.total_steps
    curriculum_alpha = 1.0 - progress * 0.5  # 1.0 -> 0.5 (keep curriculum influence)
    adaptive_alpha = progress * 0.5           # 0.0 -> 0.5 (grow adaptive influence)
    
    # Motion sampling probability from curriculum and difficulty
    motion_probs = (
        curriculum_alpha * motion_curriculum_weights + 
        adaptive_alpha * (motion_difficulty + self.cfg.adaptive_uniform_ratio)
    )
    
    # 5. Apply diversity penalty to prevent over-reuse
    motion_usage = torch.bincount(self.agent_motion_indices, minlength=self.num_motions)
    max_usage = self.num_envs * self.diversity_threshold
    diversity_penalty = torch.clamp(1.0 - motion_usage / max_usage, min=0.1)  # Min 0.1 to avoid zero prob
    motion_probs = motion_probs * diversity_penalty
    
    # 6. Normalize and sample motion indices
    motion_probs = motion_probs / motion_probs.sum()
    sampled_motion_indices = torch.multinomial(motion_probs, n, replacement=True)
    
    # ============================================================
    # STAGE 2: Time-Level Sampling (Bin-based)
    # ============================================================
    
    sampled_time_steps = torch.zeros(n, dtype=torch.long, device=self.device)
    
    for i, motion_idx in enumerate(sampled_motion_indices):
        # Get bin-level failure distribution for this specific motion
        bin_failures = self.bin_failed_counts[motion_idx]
        bin_count = self.bin_counts[motion_idx]
        
        # Apply kernel smoothing (from original _adaptive_sampling)
        # This helps prevent over-focusing on single difficult frames
        sampling_probs = bin_failures + self.cfg.adaptive_uniform_ratio / bin_count
        
        # Smooth with 1D convolution kernel
        if self.cfg.adaptive_kernel_size > 1:
            sampling_probs = torch.nn.functional.pad(
                sampling_probs.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            sampling_probs = torch.nn.functional.conv1d(
                sampling_probs, 
                self.kernel.view(1, 1, -1)
            ).view(-1)
        
        sampling_probs = sampling_probs / sampling_probs.sum()
        
        # Sample bin index
        sampled_bin = torch.multinomial(sampling_probs, 1)
        
        # Convert bin to time step with random offset within bin
        motion_length = self.motion_lengths[motion_idx]
        time_step = (
            (sampled_bin.float() + torch.rand(1, device=self.device)) 
            / bin_count * motion_length
        ).long()
        
        sampled_time_steps[i] = torch.clamp(time_step, 0, motion_length - 1)
    
    # Update agent assignments
    self.agent_motion_indices[env_ids] = sampled_motion_indices
    self.agent_time_steps[env_ids] = sampled_time_steps
    
    # Load motion data for sampled motions (via dataloader)
    motion_loaders = self.dataloader.get_loaders(sampled_motion_indices)
    
    # Compute sampling metrics for logging
    self._compute_sampling_metrics(
        motion_probs=motion_probs,
        sampled_motion_indices=sampled_motion_indices,
        quantity_weights=quantity_curriculum_weights
    )
    
    return motion_loaders, sampled_motion_indices, sampled_time_steps

def _compute_sampling_metrics(self, motion_probs, sampled_motion_indices, quantity_weights):
    """Compute metrics for monitoring sampling behavior."""
    # Motion-level entropy
    H_motion = -(motion_probs * (motion_probs + 1e-12).log()).sum()
    H_motion_norm = H_motion / math.log(self.num_motions)
    
    # Quantity distribution of sampled motions
    sampled_quantities = torch.tensor([
        self.dataset.quantities[idx] for idx in sampled_motion_indices
    ], device=self.device)
    
    # Update metrics
    self.metrics["sampling_motion_entropy"][:] = H_motion_norm
    self.metrics["sampling_top1_motion_prob"][:] = motion_probs.max()
    self.metrics["sampling_quantity_1_ratio"][:] = (sampled_quantities == 1).float().mean()
    self.metrics["sampling_quantity_2_ratio"][:] = (sampled_quantities == 2).float().mean()
    self.metrics["sampling_quantity_3_ratio"][:] = (sampled_quantities == 3).float().mean()
    self.metrics["curriculum_weight_q1"][:] = quantity_weights[0]
    self.metrics["curriculum_weight_q2"][:] = quantity_weights[1]
    self.metrics["curriculum_weight_q3"][:] = quantity_weights[2]
```

**Key Improvements:**

1. **Conservative Initial Distribution**: Start with [0.85, 0.10, 0.05] to ensure stable learning
2. **Hierarchical Sampling**: Motion-level first, then time-level
3. **Quantity-aware Curriculum**: Each motion's base probability comes from its quantity rating
4. **Motion-level Adaptive**: Aggregate bin failures to get overall motion difficulty
5. **Bin-level Adaptive**: Within chosen motion, sample difficult time periods
6. **Balanced Influence**: Curriculum weight decays from 1.0 to 0.5 (not 0), keeping quantity awareness throughout training

#### Step 3: Update Failure Statistics Per Motion

**Track failures for specific motion and time bin:**

```python
def _update_failure_statistics(self, env_ids):
    """
    Update bin failure counts when agents terminate.
    """
    terminated = self.env.termination_manager.terminated[env_ids]
    
    if torch.any(terminated):
        # Get current motion and time for terminated agents
        failed_env_ids = env_ids[terminated]
        failed_motion_indices = self.agent_motion_indices[failed_env_ids]
        failed_time_steps = self.agent_time_steps[failed_env_ids]
        
        # For each unique motion, update its failure bins
        unique_motions = torch.unique(failed_motion_indices)
        
        for motion_idx in unique_motions:
            # Get agents that failed on this motion
            motion_mask = failed_motion_indices == motion_idx
            motion_failed_times = failed_time_steps[motion_mask]
            
            # Convert time steps to bin indices
            bin_count = self.bin_counts[motion_idx]
            motion_length = self.motion_lengths[motion_idx]
            bin_indices = (motion_failed_times * bin_count / motion_length).long()
            bin_indices = torch.clamp(bin_indices, 0, bin_count - 1)
            
            # Update failure counts (with EMA)
            current_failures = torch.bincount(bin_indices, minlength=bin_count)
            self.bin_failed_counts[motion_idx] = (
                self.adaptive_alpha * current_failures + 
                (1 - self.adaptive_alpha) * self.bin_failed_counts[motion_idx]
            )
```

#### Step 4: Integration with MotionCommand

**Modified MotionCommand initialization:**

```python
class MultiMotionCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        
        # Initialize dataset and dataloader
        self.dataset = Motion_Dataset(
            dataset_dirs=cfg.dataset_dirs,
            robot_name=cfg.robot_name,
            split="train"
        )
        
        self.dataloader = Motion_Dataloader(
            dataset=self.dataset,
            device=self.device
        )
        
        # Per-agent motion tracking
        self.agent_motion_indices = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.agent_time_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        # Per-motion failure tracking
        self.bin_counts = []
        self.bin_failed_counts = []
        for i in range(len(self.dataset)):
            info = self.dataset.get_motion_info()[i]
            bin_count = info['length'] // timesteps_per_bin + 1
            self.bin_counts.append(bin_count)
            self.bin_failed_counts.append(
                torch.zeros(bin_count, device=self.device)
            )
        
    def _resample_command(self, env_ids):
        # Compute sampling weights (implement your strategy here)
        weights = self._compute_sampling_weights()
        
        # Sample motion indices using dataloader
        motion_indices = self.dataloader.sample(len(env_ids), weights=weights)
        
        # Sample time steps within each motion
        time_steps = self._sample_time_steps(motion_indices)
        
        # Get motion loaders
        motion_loaders = self.dataloader.get_motion_loaders(motion_indices)
        
        # Update agent states based on sampled motions
        for i, env_id in enumerate(env_ids):
            motion_loader = motion_loaders[i]
            time_step = time_steps[i]
            
            # Get motion state at time_step
            root_pos = motion_loader.body_pos_w[time_step, 0]
            root_quat = motion_loader.body_quat_w[time_step, 0]
            joint_pos = motion_loader.joint_pos[time_step]
            joint_vel = motion_loader.joint_vel[time_step]
            
            # Add randomization (from cfg.pose_range, etc.)
            # ...
            
            # Write to simulation
            self.robot.write_joint_state_to_sim(...)
            self.robot.write_root_state_to_sim(...)
```

### Configuration Example

```python
@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    class_type: type = MultiMotionCommand
    
    asset_name: str = "robot"
    
    # Dataset configuration
    dataset_dirs: list[str] = ["./datasets/npz_datasets/LAFAN1_Retargeting_Dataset"]
    robot_name: str = "g1"
    
    # Adaptive sampling parameters
    adaptive_alpha: float = 0.1
    adaptive_lambda: float = 0.8
    adaptive_kernel_size: int = 5
    adaptive_uniform_ratio: float = 0.1
    
    # Tracking bodies
    anchor_body_name: str = "pelvis"
    body_names: list[str] = MISSING
    
    # Randomization ranges
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
```

### Summary

The multi-motion dataloader provides a simple and flexible interface:

1. **Variable Batch Size**: `sample(n, weights)` supports dynamic batch sizes from 1 to num_envs
2. **Flexible Weighting**: Caller provides optional weights for any sampling strategy
3. **No Built-in Policy**: Dataloader doesn't enforce curriculum/adaptive logic - caller decides
4. **Efficient Caching**: MotionLoaders are cached to avoid repeated loading
5. **Compatible Interface**: Works with existing MotionCommand structure

**Simple API:**
```python
# Initialize
dataloader = Motion_Dataloader(dataset, device="cuda")

# Uniform sampling
indices = dataloader.sample(n=128)

# Weighted sampling (any strategy)
weights = compute_your_weights(...)  # [num_motions]
indices = dataloader.sample(n=128, weights=weights)

# Get loaders
loaders = dataloader.get_motion_loaders(indices)
```

**Sampling Strategy Examples:**

1. **Quantity-based Curriculum**: `weights = [0.85, 0.10, 0.05]` mapped to motions
2. **Adaptive Difficulty**: `weights = bin_failed_counts.mean(dim=1)`  
3. **Combined Strategy**: `weights = curriculum * (difficulty + uniform) * diversity_penalty`
4. **Custom Strategy**: Any function that produces `[num_motions]` weights

The caller (e.g., `MultiMotionCommand`) implements the full sampling strategy and just calls `sample(n, weights)`.
This keeps the dataloader simple while allowing maximum flexibility for experimentation.







