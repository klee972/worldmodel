"""
Resize MP4 videos to a target resolution.

Takes existing MP4 files (e.g., already chunked) and resizes them.
Keeps H.264 compression - minimal quality loss.
"""

import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm



# Configuration
SOURCE_DIR = Path("/home/4bkang/rl/jasmine/data/mc_test")
OUTPUT_DIR = Path("/home/4bkang/rl/jasmine/data/mc_test_384x224")
RESIZE_W = 384          # Target width
RESIZE_H = 224          # Target height
NUM_WORKERS = mp.cpu_count() // 2
CRF = 18                # Quality (lower = better, 18 = visually lossless)


def resize_single_video(args: tuple) -> dict:
    """
    Resize a single MP4 video.
    """
    video_path, output_path, resize_w, resize_h, crf = args
    
    try:
        # force_original_aspect_ratio=disable: force exact dimensions (may distort)
        # setsar=1:1: reset sample aspect ratio to avoid player confusion
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"scale={resize_w}:{resize_h}:force_original_aspect_ratio=disable,setsar=1:1",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(crf),
            "-an",  # No audio
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "status": "error",
                "reason": result.stderr[:200],
                "path": str(video_path)
            }
        
        return {
            "status": "success",
            "path": str(video_path)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "reason": str(e),
            "path": str(video_path)
        }


def main():
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Resize: {RESIZE_W}x{RESIZE_H}")
    print(f"CRF: {CRF}")
    print(f"Using {NUM_WORKERS} workers")
    print()
    
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg")
        return
    
    # Process each split (train/val) or all files if no splits
    splits = ["train", "val"]
    has_splits = any((SOURCE_DIR / s).exists() for s in splits)
    
    if has_splits:
        dirs_to_process = [(SOURCE_DIR / s, OUTPUT_DIR / s) for s in splits if (SOURCE_DIR / s).exists()]
    else:
        dirs_to_process = [(SOURCE_DIR, OUTPUT_DIR)]
    
    total_stats = {"success": 0, "error": 0}
    
    for src_dir, out_dir in dirs_to_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all MP4 files
        video_files = sorted(src_dir.glob("*.mp4"))
        
        if not video_files:
            print(f"No MP4 files found in {src_dir}")
            continue
        
        print(f"Processing {src_dir.name}: {len(video_files)} videos")
        
        # Prepare arguments
        args_list = [
            (str(vf), out_dir / vf.name, RESIZE_W, RESIZE_H, CRF)
            for vf in video_files
        ]
        
        split_stats = {"success": 0, "error": 0}
        errors = []
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for result in tqdm(
                executor.map(resize_single_video, args_list),
                total=len(args_list),
                desc=src_dir.name
            ):
                split_stats[result["status"]] += 1
                if result["status"] == "error":
                    errors.append(result)
        
        print(f"  {src_dir.name}: {split_stats['success']} success, {split_stats['error']} error")
        
        if errors:
            print(f"  Errors ({len(errors)}):")
            for e in errors[:3]:
                print(f"    {Path(e['path']).name}: {e['reason'][:80]}")
        
        total_stats["success"] += split_stats["success"]
        total_stats["error"] += split_stats["error"]
        print()
    
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total: {total_stats['success']} success, {total_stats['error']} error")
    print()
    print(f"Output saved to: {OUTPUT_DIR}")
    
    # Show size comparison
    if has_splits:
        src_files = list((SOURCE_DIR / "train").glob("*.mp4")) + list((SOURCE_DIR / "val").glob("*.mp4"))
        out_files = list((OUTPUT_DIR / "train").glob("*.mp4")) + list((OUTPUT_DIR / "val").glob("*.mp4"))
    else:
        src_files = list(SOURCE_DIR.glob("*.mp4"))
        out_files = list(OUTPUT_DIR.glob("*.mp4"))
    
    if src_files and out_files:
        src_size = sum(f.stat().st_size for f in src_files) / 1024**2
        out_size = sum(f.stat().st_size for f in out_files) / 1024**2
        print()
        print(f"Original size: {src_size:.1f} MB")
        print(f"Resized size:  {out_size:.1f} MB")
        print(f"Compression:   {src_size/out_size:.1f}x smaller")


if __name__ == "__main__":
    main()

