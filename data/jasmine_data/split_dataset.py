"""Split chunked minecraft dataset into train/val/test sets.

Hash-based deterministic splitting - iterates through subdirectories,
assigns each stem to train/val/test based on hash.
No pre-scan needed, single pass through files.
"""

import os
import shutil
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass
import tyro


@dataclass
class Args:
    source_dir: str = "/home/4bkang/rl/jasmine/data/minecraft_chunk64_224p"
    output_dir: str = "/home/4bkang/rl/jasmine/data/minecraft_chunk64_224p_split"
    val_size: int = 1000      # Target val files
    test_size: int = 1000     # Target test files
    total_estimate: int = 3_000_000  # Estimated total mp4 stems
    seed: int = 42
    move: bool = True
    dry_run: bool = False


def get_split_for_stem(stem: str, seed: int, test_ratio: float, val_ratio: float) -> str:
    """Deterministically assign a stem to train/val/test based on hash."""
    h = hashlib.md5((stem + str(seed)).encode()).digest()
    value = int.from_bytes(h[:8], 'big') / (2**64)
    
    if value < test_ratio:
        return 'test'
    elif value < test_ratio + val_ratio:
        return 'val'
    else:
        return 'train'


def split_dataset(args: Args):
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    test_ratio = args.test_size / args.total_estimate
    val_ratio = args.val_size / args.total_estimate
    
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Test ratio: {test_ratio:.6f}, Val ratio: {val_ratio:.6f}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    if not args.dry_run:
        for split in ['train', 'val', 'test']:
            (output_dir / split).mkdir(parents=True, exist_ok=True)
        transfer_fn = shutil.move if args.move else shutil.copy2
    
    # Get subdirectories
    print("Getting subdirectories...", flush=True)
    subdirs = [Path(e.path) for e in os.scandir(source_dir) if e.is_dir()]
    subdirs.sort(key=lambda x: x.name)
    print(f"Found {len(subdirs)} subdirectories\n", flush=True)
    
    counts = {'train': 0, 'val': 0, 'test': 0}
    total_files = 0
    start_time = time.time()
    
    for subdir_idx, subdir in enumerate(subdirs):
        subdir_files = 0
        subdir_start = time.time()
        
        print(f"[{subdir_idx+1}/{len(subdirs)}] Processing {subdir.name}...", flush=True)
        
        try:
            with os.scandir(subdir) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    
                    name = entry.name
                    if name.endswith('.mp4'):
                        stem = name[:-4]
                        ext = '.mp4'
                    elif name.endswith('.jsonl'):
                        stem = name[:-6]
                        ext = '.jsonl'
                    else:
                        continue
                    
                    split = get_split_for_stem(stem, args.seed, test_ratio, val_ratio)
                    counts[split] += 1
                    total_files += 1
                    subdir_files += 1
                    
                    if not args.dry_run:
                        dst = output_dir / split / name
                        transfer_fn(entry.path, dst)
                    
                    if total_files % 50000 == 0:
                        elapsed = time.time() - start_time
                        rate = total_files / elapsed
                        print(f"    {total_files:,} files | {rate:,.0f}/s | t={counts['train']:,} v={counts['val']:,} te={counts['test']:,}", flush=True)
        
        except Exception as e:
            print(f"  Error in {subdir.name}: {e}", flush=True)
        
        subdir_elapsed = time.time() - subdir_start
        print(f"  -> {subdir_files:,} files in {subdir_elapsed:.1f}s", flush=True)
        
        # Remove empty subdir after processing
        if not args.dry_run and args.move:
            try:
                if not any(subdir.iterdir()):
                    subdir.rmdir()
            except:
                pass
    
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Total files: {total_files:,}")
    print(f"  Train: {counts['train']:,}")
    print(f"  Val: {counts['val']:,}")
    print(f"  Test: {counts['test']:,}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were moved.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    split_dataset(args)
