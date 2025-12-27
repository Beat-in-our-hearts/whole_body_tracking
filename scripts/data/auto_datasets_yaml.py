#!/usr/bin/env python3
"""
Auto-generate YAML configuration file based on dataset directory structure.

Directory structure:
- Level 1: robot_type (e.g., g1, h1)
- Level 2: split (e.g., train, test, val)
- Level 3+: data files (.npz)
"""

import os
import yaml
from pathlib import Path
from collections import defaultdict
import argparse


def scan_dataset_dir(dataset_root):
    """
    Scan dataset directory and extract file structure.
    
    Args:
        dataset_root: Root path of the dataset
        
    Returns:
        dict: Data structure organized by robot_type -> split -> files
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_root}")
    
    data_structure = defaultdict(lambda: defaultdict(list))
    
    # Traverse level 1 directories (robot_type)
    for robot_type_dir in sorted(dataset_root.iterdir()):
        if not robot_type_dir.is_dir():
            continue
            
        robot_type = robot_type_dir.name
        
        # Traverse level 2 directories (split)
        for split_dir in sorted(robot_type_dir.iterdir()):
            if not split_dir.is_dir():
                continue
                
            split_name = split_dir.name
            
            # Recursively collect all data files
            for file_path in sorted(split_dir.rglob("*.npz")):
                # Get path relative to split directory
                relative_path = file_path.relative_to(split_dir)
                # Remove file extension
                file_key = str(relative_path.with_suffix(''))
                data_structure[robot_type][split_name].append(file_key)
    
    return data_structure


def generate_yaml_config(data_structure, dataset_name, output_path=None):
    """
    Generate YAML configuration file based on data structure.
    
    Args:
        data_structure: Data structure dictionary
        dataset_name: Name of the dataset
        output_path: Output path, if None returns dictionary only
        
    Returns:
        dict: YAML configuration dictionary
    """
    config = {"dataset": dataset_name}
    train_files = {}  # Store all motion files (merged from all splits)
    
    # Generate configuration for each robot_type and split
    for robot_type, splits in sorted(data_structure.items()):
        for split_name, files in sorted(splits.items()):
            # Create split node
            if split_name not in config:
                config[split_name] = {}
            
            # Add files with format "split/filename: 1"
            for file_key in files:
                full_key = f"{split_name}/{file_key}"
                config[split_name][full_key] = 1
                
                # Collect all files from all splits
                train_files[full_key] = 1
    
    # Add all motion files to config
    if train_files:
        config["train"] = train_files
    
    # Save to file
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"✓ YAML configuration saved to: {output_path}")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate dataset YAML configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate YAML for specified dataset
  python auto_datasets_yaml.py /path/to/dataset -n MyDataset -o output.yaml
  
  # Auto output to dataset_dir/info.yaml
  python auto_datasets_yaml.py /path/to/dataset -n MyDataset
        """
    )
    
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Dataset root directory path"
    )
    
    parser.add_argument(
        "-n", "--name",
        type=str,
        required=True,
        help="Dataset name"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output YAML file path (default: dataset_dir/info.yaml)"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview configuration only, do not save file"
    )
    
    args = parser.parse_args()
    
    # Scan dataset
    print(f"Scanning dataset: {args.dataset_dir}")
    data_structure = scan_dataset_dir(args.dataset_dir)
    
    # Print scan results
    print("\nScan results:")
    for robot_type, splits in sorted(data_structure.items()):
        print(f"  Robot Type: {robot_type}")
        for split_name, files in sorted(splits.items()):
            print(f"    {split_name}: {len(files)} files")
    
    # Generate configuration
    if args.output is None and not args.preview:
        args.output = os.path.join(args.dataset_dir, "info.yaml")
    
    config = generate_yaml_config(
        data_structure,
        args.name,
        output_path=None if args.preview else args.output
    )
    
    # Preview mode
    if args.preview:
        print("\nGenerated YAML configuration preview:")
        print("-" * 60)
        print(yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True))
        print("-" * 60)
    
    print(f"\n✓ Done! Processed {sum(len(files) for splits in data_structure.values() for files in splits.values())} files in total")


if __name__ == "__main__":
    main()