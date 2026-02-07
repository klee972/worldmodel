"""
Chunk long MP4 videos into shorter MP4 clips with corresponding action files.

Keeps H.264 compression intact - no quality loss, minimal size increase.
Uses ffmpeg for efficient video splitting.

Output per chunk:
  - MP4: 64-frame video clip
  - JSONL: 64 actions (except last chunk: T-1 actions)

This preserves the action connecting chunk i to chunk i+1 within chunk i.
"""

import subprocess
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import numpy as np





# Configuration
SOURCE_DIR = Path("/home/4bkang/rl/jasmine/data/minecraft_videos_10")
OUTPUT_DIR = Path("/home/4bkang/rl/jasmine/data/minecraft_chunk64_224p")
ACTION_DIR = Path("/home/4bkang/rl/jasmine/data/open_ai_minecraft_actions_files")  # Original action JSONL files
# SOURCE_DIR = Path("/home/4bkang/rl/jasmine/data/mc_test")
# OUTPUT_DIR = Path("/home/4bkang/rl/jasmine/data/mc_test_chunk_224p")
# ACTION_DIR = Path("/home/4bkang/rl/jasmine/data/mc_test_action")  # Original action JSONL files
CHUNK_FRAMES = 64       # Frames per chunk
TARGET_FPS = 20.0       # Target FPS (None = keep original)
RESIZE_W = 384          # Target width (None = keep original) 
RESIZE_H = 224          # Target height (None = keep original)
NUM_WORKERS = mp.cpu_count() // 2
SUBSAMPLE_RATIO = 1.0   # Use only this fraction of videos (0.1 = 10% for testing)
INCLUDE_ACTIONS = True  # Whether to also chunk action files
DELETE_AFTER_CHUNK = True  # Delete original files after successful chunking (saves disk space)


def load_action_jsonl(jsonl_path: Path) -> list:
    """Load all actions from a JSONL file."""
    actions = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                actions.append(json.loads(line))
    return actions


def save_action_jsonl(actions: list, output_path: Path):
    """Save actions to a JSONL file."""
    with open(output_path, 'w') as f:
        for action in actions:
            f.write(json.dumps(action) + '\n')


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    info = json.loads(result.stdout)
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    
    # Parse frame rate (can be "20/1" or "20.0")
    fps_str = video_stream.get("r_frame_rate", "20/1")
    if "/" in fps_str:
        num, den = map(float, fps_str.split("/"))
        fps = num / den
    else:
        fps = float(fps_str)
    
    # Get frame count
    nb_frames = video_stream.get("nb_frames")
    if nb_frames:
        num_frames = int(nb_frames)
    else:
        # Fallback: calculate from duration
        duration = float(info["format"]["duration"])
        num_frames = int(duration * fps)
    
    return {
        "num_frames": num_frames,
        "fps": fps,
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "duration": float(info["format"]["duration"]),
    }


def chunk_single_video(args: tuple) -> dict:
    """
    Chunk a single MP4 video into multiple MP4 clips with corresponding action files.
    
    Action storage strategy:
    - Each chunk stores `chunk_frames` actions (e.g., 64 actions for 64 frames)
    - EXCEPT the last chunk which stores `video_length - 1` actions
    - This preserves the action connecting chunk i to chunk i+1 within chunk i
    """
    video_path, output_dir, chunk_frames, target_fps, resize_w, resize_h, action_dir, include_actions, delete_after = args
    
    try:
        info = get_video_info(video_path)
        orig_fps = info["fps"]
        num_frames = info["num_frames"]
        
        # Calculate effective frames if downsampling FPS
        if target_fps and target_fps < orig_fps:
            # We'll re-encode at target FPS
            effective_frames = int(num_frames * target_fps / orig_fps)
            use_fps = target_fps
            fps_ratio = orig_fps / target_fps  # For action subsampling
        else:
            effective_frames = num_frames
            use_fps = orig_fps
            fps_ratio = 1.0
        
        num_chunks = effective_frames // chunk_frames
        
        if num_chunks == 0:
            return {
                "status": "skip",
                "chunks": 0,
                "reason": f"too short ({effective_frames} < {chunk_frames})",
                "path": str(video_path)
            }
        
        base_name = Path(video_path).stem
        chunks_created = 0
        
        # Load action data if needed
        actions = None
        if include_actions and action_dir:
            action_path = Path(action_dir) / f"{base_name}.jsonl"
            if action_path.exists():
                actions = load_action_jsonl(action_path)
            else:
                return {
                    "status": "error",
                    "chunks": 0,
                    "reason": f"action file not found: {action_path.name}",
                    "path": str(video_path)
                }
        
        for i in range(num_chunks):
            # Calculate time range
            start_frame = i * chunk_frames
            start_time = start_frame / use_fps
            duration = chunk_frames / use_fps
            
            output_path = output_dir / f"{base_name}_chunk{i:03d}.mp4"
            
            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-y",  # Overwrite
                "-ss", f"{start_time:.4f}",  # Seek to start
                "-i", str(video_path),
                "-t", f"{duration:.4f}",  # Duration
                "-c:v", "libx264",  # Re-encode with H.264
                "-preset", "fast",  # Fast encoding
                "-crf", "18",  # High quality (lower = better, 18 is visually lossless)
            ]
            

            # Build video filter chain
            vf_filters = []
            if target_fps and target_fps < orig_fps:
                vf_filters.append(f"fps={target_fps}")
            if resize_w and resize_h:
                # force_original_aspect_ratio=disable: force exact dimensions
                # setsar=1:1: reset sample aspect ratio
                vf_filters.append(f"scale={resize_w}:{resize_h}:force_original_aspect_ratio=disable,setsar=1:1")
            
            if vf_filters:
                cmd.extend(["-vf", ",".join(vf_filters)])
            
            cmd.extend([
                "-an",  # No audio
                str(output_path)
            ])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return {
                    "status": "error",
                    "chunks": chunks_created,
                    "reason": f"ffmpeg failed on chunk {i}: {result.stderr[:200]}",
                    "path": str(video_path)
                }
            
            # Save corresponding actions
            if actions is not None:
                # Calculate action indices (accounting for FPS change)
                orig_start_frame = int(start_frame * fps_ratio)
                
                # Determine how many actions to save
                is_last_chunk = (i == num_chunks - 1)
                if is_last_chunk:
                    # Last chunk: save chunk_frames - 1 actions (standard T frames -> T-1 actions)
                    num_actions = chunk_frames - 1
                else:
                    # Non-last chunks: save chunk_frames actions
                    # This includes the action that connects to the next chunk
                    num_actions = chunk_frames
                
                orig_end_frame = orig_start_frame + int(num_actions * fps_ratio)
                
                # Subsample actions if FPS changed
                if fps_ratio > 1.0:
                    chunk_actions = [
                        actions[int(orig_start_frame + j * fps_ratio)]
                        for j in range(num_actions)
                        if int(orig_start_frame + j * fps_ratio) < len(actions)
                    ]
                else:
                    chunk_actions = actions[orig_start_frame:orig_start_frame + num_actions]
                
                # Save action JSONL
                action_output_path = output_dir / f"{base_name}_chunk{i:03d}.jsonl"
                save_action_jsonl(chunk_actions, action_output_path)
            
            chunks_created += 1
        
        # Delete original files after successful chunking
        deleted_files = []
        if delete_after and chunks_created > 0:
            # Delete original video
            video_path_obj = Path(video_path)
            if video_path_obj.exists():
                video_path_obj.unlink()
                deleted_files.append(video_path_obj.name)
            
            # Delete original action file if we were processing actions
            if include_actions and action_dir:
                action_path = Path(action_dir) / f"{base_name}.jsonl"
                if action_path.exists():
                    action_path.unlink()
                    deleted_files.append(action_path.name)
        
        return {
            "status": "success",
            "chunks": chunks_created,
            "orig_frames": num_frames,
            "orig_fps": orig_fps,
            "path": str(video_path),
            "deleted": deleted_files
        }
    
    except Exception as e:
        return {
            "status": "error",
            "chunks": 0,
            "reason": str(e),
            "path": str(video_path)
        }


def main():
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Action dir: {ACTION_DIR}" if INCLUDE_ACTIONS else "Actions: disabled")
    print(f"Chunk length: {CHUNK_FRAMES} frames")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Resize: {RESIZE_W}x{RESIZE_H}" if RESIZE_W and RESIZE_H else "Resize: None (original)")
    print(f"Subsample ratio: {SUBSAMPLE_RATIO}" + (" (TEST MODE)" if SUBSAMPLE_RATIO < 1.0 else ""))
    print(f"Include actions: {INCLUDE_ACTIONS}")
    if INCLUDE_ACTIONS:
        print(f"  - Non-last chunks: {CHUNK_FRAMES} actions (includes connecting action)")
        print(f"  - Last chunk: {CHUNK_FRAMES - 1} actions (standard T-1)")
    print(f"Delete after chunk: {DELETE_AFTER_CHUNK}" + (" ⚠️  ORIGINALS WILL BE DELETED!" if DELETE_AFTER_CHUNK else ""))
    print(f"Using {NUM_WORKERS} workers")
    print()
    
    # Check ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg")
        return
    
    # Scan and match video/action files
    print("Scanning files...")
    video_files = sorted(SOURCE_DIR.glob("*.mp4"))
    video_stems = {f.stem for f in video_files}
    
    if INCLUDE_ACTIONS:
        action_files = sorted(ACTION_DIR.glob("*.jsonl"))
        action_stems = {f.stem for f in action_files}
        matched_stems = video_stems & action_stems
        video_only = video_stems - action_stems
        action_only = action_stems - video_stems
        
        print(f"  MP4 files:    {len(video_files)}")
        print(f"  JSONL files:  {len(action_files)}")
        print(f"  Matched pairs: {len(matched_stems)}")
        if video_only:
            print(f"  ⚠️  Videos without actions: {len(video_only)}")
        if action_only:
            print(f"  ⚠️  Actions without videos: {len(action_only)}")
        
        # Only process matched pairs
        video_files = [f for f in video_files if f.stem in matched_stems]
        print(f"  → Will process: {len(video_files)} matched videos")
    else:
        print(f"  MP4 files: {len(video_files)}")
    print()
    
    if not video_files:
        print("No video files to process!")
        return
    
    # Subsample videos for testing
    if SUBSAMPLE_RATIO < 1.0:
        np.random.seed(42)
        num_samples = max(1, int(len(video_files) * SUBSAMPLE_RATIO))
        sample_indices = np.random.choice(len(video_files), size=num_samples, replace=False)
        video_files = [video_files[i] for i in sorted(sample_indices)]
        print(f"Subsampled to {len(video_files)} videos ({SUBSAMPLE_RATIO*100:.0f}%)")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(video_files)} videos")
    print()
    
    # Prepare arguments
    args_list = [
        (str(f), OUTPUT_DIR, CHUNK_FRAMES, TARGET_FPS, RESIZE_W, RESIZE_H, 
         ACTION_DIR if INCLUDE_ACTIONS else None, INCLUDE_ACTIONS, DELETE_AFTER_CHUNK)
        for f in video_files
    ]
    
    stats = {"success": 0, "skip": 0, "error": 0, "chunks": 0, "deleted": 0}
    errors = []
    
    # Process videos
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for result in tqdm(
            executor.map(chunk_single_video, args_list),
            total=len(args_list),
            desc="Chunking"
        ):
            stats[result["status"]] += 1
            stats["chunks"] += result["chunks"]
            stats["deleted"] += len(result.get("deleted", []))
            if result["status"] == "error":
                errors.append(result)
    
    print()
    print(f"Results: {stats['success']} success, {stats['skip']} skip, {stats['error']} error")
    print(f"Videos: {len(video_files)} → {stats['chunks']} chunks")
    if stats['deleted'] > 0:
        print(f"Original files deleted: {stats['deleted']}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  {Path(e['path']).name}: {e.get('reason', 'unknown')[:100]}")
    
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Output saved to: {OUTPUT_DIR}")
    
    # Show size comparison
    new_mp4_files = list(OUTPUT_DIR.glob("*.mp4"))
    new_jsonl_files = list(OUTPUT_DIR.glob("*.jsonl"))
    if new_mp4_files:
        new_mp4_size = sum(f.stat().st_size for f in new_mp4_files) / 1024**3
        new_jsonl_size = sum(f.stat().st_size for f in new_jsonl_files) / 1024**2  # MB for actions
        print()
        # Only show original size if files still exist (not deleted)
        remaining_orig = [f for f in video_files if f.exists()]
        if remaining_orig:
            orig_size = sum(f.stat().st_size for f in remaining_orig) / 1024**3
            print(f"Remaining original size: {orig_size:.1f} GB ({len(remaining_orig)} files)")
        else:
            print("Original files: all deleted ✓")
        print(f"Chunked video size:  {new_mp4_size:.1f} GB ({len(new_mp4_files)} files)")
        if new_jsonl_files:
            print(f"Action files:        {len(new_jsonl_files)} files ({new_jsonl_size:.1f} MB)")


if __name__ == "__main__":
    main()

