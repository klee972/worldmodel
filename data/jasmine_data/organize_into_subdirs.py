"""Organize files into subdirectories based on filename prefix.

Files are named like: 01.23_something.mp4
This script moves them into subdirectories: 01.23/01.23_something.mp4

Uses find + streaming to handle millions of files without memory issues.
"""

import os
import time
from dataclasses import dataclass
import tyro



@dataclass
class Args:
    source_dir: str = "/home/4bkang/rl/jasmine/data/minecraft_chunk64_224p"
    # source_dir: str = "/home/4bkang/rl/jasmine/data/mc_test_chunk_224p"
    dry_run: bool = False
    extensions: tuple[str, ...] = (".mp4", ".jsonl")
    max_files: int = 1000000  # 0 = no limit


def organize_files(args: Args):
    source_dir = args.source_dir
    
    print(f"Source: {source_dir}")
    print(f"Extensions: {args.extensions}")
    print(f"Prefix: everything before first '_'")
    print(f"Dry run: {args.dry_run}")
    print(f"Max files: {args.max_files if args.max_files > 0 else 'unlimited'}")
    print()
    print("Processing files with scandir (no pre-scan needed)...")
    print()
    
    start_time = time.time()
    processed = 0
    moved = 0
    skipped = 0
    errors = 0
    dirs_created = set()
    
    with os.scandir(source_dir) as it:
        for entry in it:
            try:
                # Skip directories
                if entry.is_dir():
                    continue
                
                name = entry.name
                
                # Check extension
                if not any(name.endswith(ext) for ext in args.extensions):
                    continue
                
                # Extract prefix (everything before first '_')
                if '_' not in name:
                    skipped += 1
                    if skipped <= 5:
                        print(f"  Skip no underscore: {name[:40]}")
                    continue
                
                prefix = name.split('_', 1)[0]
                subdir = os.path.join(source_dir, prefix)
                
                is_new_dir = prefix not in dirs_created
                dirs_created.add(prefix)
                
                if not args.dry_run:
                    if is_new_dir:
                        os.makedirs(subdir, exist_ok=True)
                    
                    dest = os.path.join(subdir, name)
                    os.rename(entry.path, dest)
                
                moved += 1
                processed += 1
                
                # Check limit
                if args.max_files > 0 and processed >= args.max_files:
                    print(f"  Reached max_files limit ({args.max_files})")
                    break
                
                if processed % 50000 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"  {processed:,} files | {rate:,.0f}/s | dirs: {len(dirs_created)}", flush=True)
            
            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"  ERROR: {entry.name}: {e}", flush=True)
    
    elapsed = time.time() - start_time
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"  Processed: {processed:,}")
    print(f"  Moved: {moved:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Errors: {errors:,}")
    print(f"  Subdirectories: {len(dirs_created)}")
    
    if args.dry_run:
        print()
        print("This was a dry run. Run with --no-dry-run to actually move files.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    organize_files(args)

