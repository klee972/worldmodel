"""
Create a 1/10 mini dataset from minecraft_arrayrecords_filtered.

Uses symlinks to avoid duplicating data.
"""

import random
from pathlib import Path

# Configuration
SOURCE_DIR = Path("/home/4bkang/rl/jasmine/data/minecraft_arrayrecords_filtered")
OUTPUT_DIR = Path("/home/4bkang/rl/jasmine/data/minecraft_arrayrecords_filtered_mini")
FRACTION = 0.01
SEED = 42




def main():
    random.seed(SEED)
    
    for split in ["train", "val", "test"]:
        input_dir = SOURCE_DIR / split
        output_dir = OUTPUT_DIR / split
        
        if not input_dir.exists():
            print(f"Skipping {split}: {input_dir} does not exist")
            continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = sorted(input_dir.glob("*.array_record"))
        n_sample = max(1, int(len(files) * FRACTION))
        
        sampled_files = random.sample(files, n_sample)
        
        print(f"{split}: {len(files)} files -> {n_sample} files ({FRACTION*100:.0f}%)")
        
        for file_path in sampled_files:
            link_path = output_dir / file_path.name
            if link_path.exists():
                link_path.unlink()
            link_path.symlink_to(file_path.resolve())
    
    print(f"\nDone! Mini dataset created at: {OUTPUT_DIR}")
    print(f"\nTo use:")
    print(f'  data_dir: str = "data/minecraft_arrayrecords_filtered_mini/train"')
    print(f'  val_data_dir: str = "data/minecraft_arrayrecords_filtered_mini/val"')


if __name__ == "__main__":
    main()

