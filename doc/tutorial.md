## pip 

```
pip install rich
```

## data download

```
mkdir npz_datasets
mkdir datasets && cd datasets
mkdir origin && cd origin

git clone https://www.modelscope.cn/datasets/AI-ModelScope/LAFAN1_Retargeting_Dataset

```

## data process

```
python scripts/data/multi_offline_csv_to_npz.py --input_fps 30 --output_fps 50 --input_dir ./datasets/origin/LAFAN1_Retargeting_Dataset/g1/ --output_dir ./datasets/npz_datasets/LAFAN1_Retargeting_Dataset/g1/
```

## 