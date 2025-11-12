"""
GMR Retargeting - Convert PKL to CSV
GMR PKL file:
pkl keys: ["fps", "root_pos", "root_rot", "dof_pos"]
root_rot format: xyzw
    
CSV file:
motion[:, :3]: motion_base_poss_input
motion[:, 3:7]: motion_base_rots_input (xyzw)
motion[:, 7:]: motion_dof_poss_input
no header
    
Author: Zuxing Lu
Data: 2025/09/16
Updated: 2025/11/12 - Added batch processing support
"""
import os
import pickle
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

argparser = argparse.ArgumentParser(description="Convert GMR PKL files to CSV format (supports batch processing)")
argparser.add_argument("--input", type=str, help="Path to input PKL file (for single file conversion)")
argparser.add_argument("--input_dir", type=str, help="Path to input directory containing PKL files (for batch conversion)")
argparser.add_argument("--output", type=str, help="Path to output CSV file (for single file conversion)")
argparser.add_argument("--output_dir", type=str, help="Path to output directory for CSV files (for batch conversion)")
argparser.add_argument("--recursive", "-r", action="store_true", help="Recursively search for PKL files in subdirectories")
args = argparser.parse_args()

    
def pkl_to_csv(pkl_path, csv_path):
    """
    Convert single PKL file to CSV format
    
    Args:
        pkl_path: Path to input PKL file
        csv_path: Path to output CSV file
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        motion_base_poss_input = data["root_pos"]
        motion_base_rots_input = data["root_rot"]
        motion_dof_poss_input = data["dof_pos"]

        motion = np.concatenate([motion_base_poss_input, motion_base_rots_input, motion_dof_poss_input], axis=1)

        # Create directory if needed
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        np.savetxt(csv_path, motion, delimiter=',')
        return True
    except Exception as e:
        print(f"❌ Error processing {pkl_path}: {e}")
        return False


def batch_convert_pkl_to_csv(input_dir, output_dir, recursive=True):
    """
    Batch convert PKL files to CSV format
    
    Args:
        input_dir: Input directory containing PKL files
        output_dir: Output directory for CSV files
        recursive: Whether to search subdirectories recursively
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all PKL files
    if recursive:
        pkl_files = list(input_path.rglob("*.pkl"))
    else:
        pkl_files = list(input_path.glob("*.pkl"))
    
    if not pkl_files:
        print(f"⚠️  No PKL files found in {input_dir}")
        return
    
    print(f"Found {len(pkl_files)} PKL files")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Recursive search: {recursive}")
    print("")
    
    success_count = 0
    failed_files = []
    
    # Process each file
    for pkl_file in tqdm(pkl_files, desc="Converting PKL to CSV"):
        # Compute relative path to preserve directory structure
        rel_path = pkl_file.relative_to(input_path)
        
        # Create output path with same directory structure
        csv_file = output_path / rel_path.with_suffix('.csv')
        
        # Convert
        if pkl_to_csv(str(pkl_file), str(csv_file)):
            success_count += 1
        else:
            failed_files.append(str(pkl_file))
    
    # Print summary
    print("\n" + "="*70)
    print(f"Conversion Summary:")
    print(f"  Total files: {len(pkl_files)}")
    print(f"  Successfully converted: {success_count}")
    print(f"  Failed: {len(failed_files)}")
    print(f"  Output directory: {output_dir}")
    print("="*70)
    
    if failed_files:
        print(f"\n⚠️  {len(failed_files)} files failed to convert:")
        for filename in failed_files[:10]:
            print(f"  {filename}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    

if __name__ == "__main__":
    # Validate arguments
    if args.input and args.output:
        # Single file mode
        print(f"Converting single file: {args.input}")
        if pkl_to_csv(args.input, args.output):
            print(f"✅ Converted {args.input} to {args.output}")
        else:
            print(f"❌ Failed to convert {args.input}")
            
    elif args.input_dir and args.output_dir:
        # Batch mode
        batch_convert_pkl_to_csv(args.input_dir, args.output_dir, args.recursive)
        print("\n✅ Batch conversion completed!")
        
    else:
        print("❌ Error: Invalid arguments")
        print("\nUsage:")
        print("  Single file: --input <pkl_file> --output <csv_file>")
        print("  Batch mode:  --input_dir <input_dir> --output_dir <output_dir> [--recursive]")
        argparser.print_help()