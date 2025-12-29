# Multi-Modal Multi-Motion Tracking Foundation Model 
a free-retargeting in deploy framework.

## env setup

1. conda env
```
conda create -n env_mimic python=3.10 -y
conda activate env_mimic 
```

2. isaacsim 4.5 + isaaclab 2.1.1

```
# torch 2.7.0 + torchvision 0.22.0 + torchaudio 2.7.0 for cuda 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# isaacsim 4.5
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# eval isaacsim
isaacsim isaacsim.exp.full.kit --headless

# isaaclab 2.1.1
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 90b79bb2d44feb8d833f260f2bf37da3487180ba
./isaaclab.sh -i

# eval isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
cd ..
```

3. rsl_rl

```
# checkout hash
git clone https://github.com/Renforce-Dynamics/rsl_rl.git
cd rsl_rl
git checkout 7fe2ca2a0007e70e1c41ceed6d9f7e6cb43660c2

# install our own rsl_rl
cd ../Isaaclab
./isaaclab.sh -p -m pip install -e ../rsl_rl

# eval
./isaaclab.sh -p -m pip show rsl-rl-lib
cd ..
```

4. whole_body_tracking

```
# 1. main code
git clone https://github.com/Renforce-Dynamics/whole_body_tracking.git
cd whole_body_tracking
git switch dev.autoencoder
cd ..

# 2. unitree_g1 assets, need proxy
git clone https://huggingface.co/datasets/unitreerobotics/unitree_model

# 3. move `unitree_model` to `source/whole_body_tracking/whole_body_tracking/assets`
# mv unitree_model whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets

# 4. install code
python -m pip install -e source/whole_body_tracking

```

## datasets setup

1. datasets download

```
# https://www.modelscope.cn/docs/datasets/upload
pip install modelscope datasets

# https://www.modelscope.cn/datasets/seulzx/smplx_datasets
modelscope download --dataset seulzx/smplx_datasets --local_dir './tmp_datasets'

tar -xjvf /path/to/tmp_datasets/extended_datasets.tar.bz2 -C ./whole_body_tracking/datasets
```

2. datasets structure

```
whole_body_tracking/datasets/
├── extended_datasets/
│   └── lafan1_dataset/
│       ├── info.yaml  # lafan1 dataset split and npa file path
│       └── g1/
│           └── train/
│               ├── dance*.npz
│               ├── fight*.npz
│               ├── jumps*.npz
│               ├── run*.npz
│               ├── sprint*.npz
│               └── walk*.npz
```


##  train & play

1. mini datasets train

```
python scripts/rsl_rl/train.py --headless --task SONIC-MultiTracking-Flat-G1-TripleAE-Scratch-SMPLX-v0
```

2. multi-gpu train

```
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 scripts/rsl_rl/train.py \
--headless --task SONIC-MultiTracking-Flat-G1-TripleAE-Scratch-SMPLX-v0 \
--logger wandb --log_project_name [your_project] --run_name [your_run] \
--distributed
```

<details>
<summary>train args</summary>

- `--task`: Name of the training task
- `--distributed`: Enable multi-GPU or multi-node distributed training
- `--disable_multi_motion`: Disable multi-motion training mode
- `--motion_file`: Path to a single motion file (used with `--disable_multi_motion`)

</details>



3. play from wandb

```
python scripts/rsl_rl/play.py --headless --num_envs 64 \
--task SONIC-MultiTracking-Flat-G1-TripleAE-Scratch-Robot-v0 \
--wandb_run_path [your_run_path] --export_type [sonic_robot|sonic_human|sonic_keypoints] --export_name [output_onnx_name] \
--video --video_length 1000
```

<details>
<summary>play args</summary>

- `--video`: Enable video recording during play (default: False)
- `--video_length`: Length of recorded video in steps (default: 200)
- `--num_envs`: Number of parallel environments to simulate
- `--task`: Name of the task to evaluate
- `--motion_file`: Path to a motion file for single-motion evaluation
- `--disable_multi_motion`: Disable multi-motion mode
- `--datasets`: Comma-separated list of dataset directories to use
- `--splits`: Comma-separated list of dataset splits (e.g., train, test)
- `--wandb_run_path`: Path to wandb run for loading checkpoint (format: `entity/project/run_id`)
- `--wandb_alg_cfg`: Load algorithm config from wandb run instead of log directory
- `--export_type`: Type of ONNX export - `sonic`, `sonic_robot`, `sonic_human`, `sonic_keypoints` (default: multi_motion)
- `--export_name`: Custom name for exported ONNX model file
</details>

## deploy

