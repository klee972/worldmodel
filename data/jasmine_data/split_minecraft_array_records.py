"""
Split Minecraft ArrayRecord files into train/val/test sets (8:1:1 ratio).

This script moves files into train/, val/, test/ subdirectories.
"""

import random
import shutil
from pathlib import Path

# Configuration
SOURCE_DIR = Path("/home/4bkang/rl/jasmine/data/minecraft_arrayrecords")
OUTPUT_DIR = SOURCE_DIR  # Create train/val/test subdirs in the same directory
SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def main():
    # Get all array_record files
    files = sorted([f for f in SOURCE_DIR.iterdir() if f.suffix == ".array_record"])
    print(f"Found {len(files)} array_record files")

    # Shuffle with fixed seed for reproducibility
    random.seed(SEED)
    random.shuffle(files)

    # Calculate split indices
    n_total = len(files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    # Test gets the remainder to ensure all files are used
    n_test = n_total - n_train - n_val

    print(f"Split sizes: train={n_train}, val={n_val}, test={n_test}")

    # Split files
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    # Create output directories
    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, split_files in splits.items():
        split_dir = OUTPUT_DIR / split_name
        split_dir.mkdir(exist_ok=True)

        print(f"Moving {len(split_files)} files to {split_dir}...")

        for file_path in split_files:
            dest_path = split_dir / file_path.name
            shutil.move(str(file_path), str(dest_path))

    print("Done!")

    # Print summary
    print("\nSummary:")
    for split_name in ["train", "val", "test"]:
        split_dir = OUTPUT_DIR / split_name
        n_files = len(list(split_dir.glob("*.array_record")))
        print(f"  {split_name}: {n_files} files")


if __name__ == "__main__":
    main()

