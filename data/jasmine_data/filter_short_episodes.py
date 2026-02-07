"""
Move long episodes to filtered directory.

Each ArrayRecord file contains exactly 1 episode.
Files with sequence_length >= MIN_SEQ_LEN are moved to the filtered directory.
Short episodes stay in the original location.
"""

import shutil
import pickle
from pathlib import Path
import array_record.python.array_record_module as ar
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Configuration
SOURCE_DIR = Path("/home/4bkang/rl/jasmine/data/minecraft_arrayrecords")
OUTPUT_DIR = Path("/home/4bkang/rl/jasmine/data/minecraft_arrayrecords_filtered")
MIN_SEQ_LEN = 32  # Minimum sequence length to keep
NUM_WORKERS = mp.cpu_count()  # Use all available CPU cores

def check_and_move_file(args: tuple[Path, Path, int]) -> tuple[bool, int]:
    """
    Check if episode is long enough, and move file if so.
    
    Returns:
        (was_moved, seq_len): Whether file was moved and the sequence length.
    """
    input_path, output_path, min_seq_len = args
    
    reader = ar.ArrayRecordReader(str(input_path))
    num_records = reader.num_records()
    
    if num_records == 0:
        reader.close()
        return False, 0
    
    # Read just the first record (1 episode per file)
    records = reader.read(0, 1)
    reader.close()
    
    data = pickle.loads(records[0])
    seq_len = data["sequence_length"]
    
    if seq_len >= min_seq_len:
        shutil.move(str(input_path), str(output_path))
        return True, seq_len
    
    return False, seq_len


def main():
    print(f"Using {NUM_WORKERS} workers")
    
    # Process each split
    for split in ["train", "val", "test"]:
        input_dir = SOURCE_DIR / split
        output_dir = OUTPUT_DIR / split
        
        if not input_dir.exists():
            print(f"Skipping {split}: {input_dir} does not exist")
            continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = sorted(input_dir.glob("*.array_record"))
        print(f"\nProcessing {split}: {len(files)} files")
        
        # Prepare arguments for parallel processing
        args_list = [
            (file_path, output_dir / file_path.name, MIN_SEQ_LEN)
            for file_path in files
        ]
        
        moved = 0
        skipped = 0
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(check_and_move_file, args): args for args in args_list}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=split):
                was_moved, _ = future.result()
                if was_moved:
                    moved += 1
                else:
                    skipped += 1
        
        print(f"  {split}: moved {moved}, skipped {skipped} (short episodes)")

    print(f"\nDone! Long episodes moved to: {OUTPUT_DIR}")
    print(f"\nUpdate your config to use the filtered data:")
    print(f'  data_dir: str = "data/minecraft_arrayrecords_filtered/train"')
    print(f'  val_data_dir: str = "data/minecraft_arrayrecords_filtered/val"')


if __name__ == "__main__":
    main()
