"""
Add merged actions (10fps) to video array_record files.

Reads:
- Video files from minecraft_arrayrecords_filtered (10fps)
- Action files from open_ai_minecraft_actions_files (20fps)

Outputs:
- Video files with merged actions added to minecraft_arrayrecords_with_actions
"""

import pickle
import json
from pathlib import Path
import array_record.python.array_record_module as ar
from array_record.python.array_record_module import ArrayRecordWriter
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


# Configuration
ACTION_DIR = Path("/home/4bkang/rl/jasmine/data/open_ai_minecraft_actions_files")
VIDEO_DIR = Path("/home/4bkang/rl/jasmine/data/minecraft_arrayrecords_filtered_mini")
OUTPUT_DIR = Path("/home/4bkang/rl/jasmine/data/minecraft_arrayrecords_filtered_mini_with_actions")
NUM_WORKERS = mp.cpu_count()


def merge_two_actions(action1: dict, action2: dict) -> dict:
    """
    Merge two consecutive action frames (20fps) into one (10fps).
    
    Strategy:
    - Delta values (dx, dy, dwheel): SUM
    - Absolute positions (x, y, yaw, pitch, xpos, ypos, zpos): LAST
    - Buttons/Keys: UNION (any pressed in either frame)
    - Boolean states: OR
    - Inventory/state: LAST
    - Duration: SUM
    """
    merged = {}
    
    # Mouse
    merged["mouse"] = {
        # Deltas: SUM
        "dx": action1["mouse"]["dx"] + action2["mouse"]["dx"],
        "dy": action1["mouse"]["dy"] + action2["mouse"]["dy"],
        "dwheel": action1["mouse"]["dwheel"] + action2["mouse"]["dwheel"],
        # Absolute positions: LAST
        "x": action2["mouse"]["x"],
        "y": action2["mouse"]["y"],
        # Buttons: UNION
        "buttons": list(set(action1["mouse"]["buttons"]) | set(action2["mouse"]["buttons"])),
        "newButtons": list(set(action1["mouse"]["newButtons"]) | set(action2["mouse"]["newButtons"])),
    }
    
    # Keyboard
    merged["keyboard"] = {
        "keys": list(set(action1["keyboard"]["keys"]) | set(action2["keyboard"]["keys"])),
        "newKeys": list(set(action1["keyboard"]["newKeys"]) | set(action2["keyboard"]["newKeys"])),
        "chars": action1["keyboard"]["chars"] + action2["keyboard"]["chars"],
    }
    
    # Boolean states: OR
    merged["isGuiOpen"] = action1["isGuiOpen"] or action2["isGuiOpen"]
    if "isGuiInventory" in action1:
        merged["isGuiInventory"] = action1.get("isGuiInventory", False) or action2.get("isGuiInventory", False)
    
    # Absolute values: LAST
    merged["yaw"] = action2["yaw"]
    merged["pitch"] = action2["pitch"]
    merged["xpos"] = action2["xpos"]
    merged["ypos"] = action2["ypos"]
    merged["zpos"] = action2["zpos"]
    
    # Metadata: LAST
    merged["tick"] = action2["tick"]
    if "milli" in action2:
        merged["milli"] = action2["milli"]
    if "serverTick" in action2:
        merged["serverTick"] = action2["serverTick"]
    
    # Duration: SUM
    if "serverTickDurationMs" in action1 and "serverTickDurationMs" in action2:
        merged["serverTickDurationMs"] = action1["serverTickDurationMs"] + action2["serverTickDurationMs"]
    
    # State: LAST
    if "inventory" in action2:
        merged["inventory"] = action2["inventory"]
    if "hotbar" in action2:
        merged["hotbar"] = action2["hotbar"]
    if "stats" in action2:
        merged["stats"] = action2["stats"]
    
    return merged


def merge_actions_to_10fps(actions: list[dict]) -> list[dict]:
    """
    Merge 20fps actions to 10fps by combining pairs.
    
    For N action frames (20fps), returns N//2 merged actions (10fps).
    """
    merged_actions = []
    
    # Merge pairs: (0,1), (2,3), (4,5), ...
    for i in range(0, len(actions) - 1, 2):
        if i + 1 < len(actions):
            merged = merge_two_actions(actions[i], actions[i + 1])
            merged_actions.append(merged)
    
    return merged_actions


def load_action_file(path: Path) -> list[dict]:
    """Load all actions from a jsonl file."""
    actions = []
    with open(path, 'r') as f:
        for line in f:
            actions.append(json.loads(line.strip()))
    return actions


def get_episode_id_from_video_file(path: Path) -> str:
    """Extract episode ID from video file path."""
    stem = path.stem  # Remove .array_record
    # Remove _chunk### suffix if present
    if "_chunk" in stem:
        stem = stem.rsplit("_chunk", 1)[0]
    return stem


def get_chunk_index(path: Path) -> int:
    """Extract chunk index from video file path. Returns -1 if no chunk."""
    stem = path.stem
    if "_chunk" in stem:
        chunk_str = stem.rsplit("_chunk", 1)[1]
        return int(chunk_str)
    return -1


def process_single_video(args: tuple) -> dict:
    """
    Process a single video file and add actions.
    
    Returns dict with status info.
    """
    video_path, action_path, output_path, all_merged_actions, chunk_start_idx = args
    
    try:
        # Load video data
        reader = ar.ArrayRecordReader(str(video_path))
        num_records = reader.num_records()
        if num_records == 0:
            reader.close()
            return {"status": "skip", "reason": "empty video", "path": str(video_path)}
        
        records = reader.read(0, 1)
        reader.close()
        
        video_data = pickle.loads(records[0])
        seq_len = video_data["sequence_length"]
        
        # Get corresponding merged actions for this chunk
        # video frame i corresponds to merged action i (for transition to frame i+1)
        # But we store action[i] with frame[i] to represent "action taken at this state"
        # So for seq_len frames, we need seq_len actions (last action can be padding or discarded)
        
        # Calculate action indices for this chunk
        action_start = chunk_start_idx
        action_end = action_start + seq_len
        
        if action_end > len(all_merged_actions):
            # Not enough actions, use what we have
            action_end = len(all_merged_actions)
        
        chunk_actions = all_merged_actions[action_start:action_end]
        
        # Pad if needed (for last frame that has no next state)
        while len(chunk_actions) < seq_len:
            # Duplicate last action as padding
            if chunk_actions:
                chunk_actions.append(chunk_actions[-1])
            else:
                return {"status": "error", "reason": "no actions available", "path": str(video_path)}
        
        # Add actions to video data
        video_data["actions"] = chunk_actions
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = ArrayRecordWriter(str(output_path), "group_size:1")
        writer.write(pickle.dumps(video_data))
        writer.close()
        
        return {
            "status": "success",
            "path": str(video_path),
            "video_frames": seq_len,
            "actions_added": len(chunk_actions)
        }
        
    except Exception as e:
        return {"status": "error", "reason": str(e), "path": str(video_path)}


def find_video_files_by_episode(video_dir: Path) -> dict:
    """
    Find all video files and group by episode ID.
    Returns dict: episode_id -> list of (video_path, chunk_index)
    """
    video_files = defaultdict(list)
    
    for split in ["train", "val", "test"]:
        split_dir = video_dir / split
        if not split_dir.exists():
            continue
        
        for video_path in split_dir.glob("*.array_record"):
            episode_id = get_episode_id_from_video_file(video_path)
            chunk_idx = get_chunk_index(video_path)
            video_files[episode_id].append((video_path, chunk_idx, split))
    
    return video_files


def main():
    print(f"Action directory: {ACTION_DIR}")
    print(f"Video directory: {VIDEO_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using {NUM_WORKERS} workers")
    
    # Find all action files
    print("\nLoading action file list...")
    action_files = {p.stem: p for p in ACTION_DIR.glob("*.jsonl")}
    print(f"Found {len(action_files)} action files")
    
    # Find all video files grouped by episode
    print("\nFinding video files...")
    video_files_by_episode = find_video_files_by_episode(VIDEO_DIR)
    print(f"Found {len(video_files_by_episode)} episodes with video files")
    
    # Find matches
    matched_episodes = []
    for episode_id in video_files_by_episode:
        if episode_id in action_files:
            matched_episodes.append(episode_id)
    
    print(f"Matched {len(matched_episodes)} episodes")
    
    # Process each episode
    stats = {"success": 0, "skip": 0, "error": 0}
    
    for episode_id in tqdm(matched_episodes, desc="Processing episodes"):
        action_path = action_files[episode_id]
        video_chunks = video_files_by_episode[episode_id]
        
        # Load and merge actions for this episode
        try:
            actions_20fps = load_action_file(action_path)
            merged_actions = merge_actions_to_10fps(actions_20fps)
        except Exception as e:
            print(f"Error loading actions for {episode_id}: {e}")
            stats["error"] += 1
            continue
        
        # Sort chunks by index
        video_chunks.sort(key=lambda x: x[1])  # Sort by chunk_index
        
        # Calculate cumulative frame counts for each chunk
        # Need to read each chunk to know its length
        chunk_frame_counts = []
        for video_path, chunk_idx, split in video_chunks:
            try:
                reader = ar.ArrayRecordReader(str(video_path))
                if reader.num_records() > 0:
                    records = reader.read(0, 1)
                    data = pickle.loads(records[0])
                    chunk_frame_counts.append(data["sequence_length"])
                else:
                    chunk_frame_counts.append(0)
                reader.close()
            except:
                chunk_frame_counts.append(0)
        
        # Process each chunk
        action_offset = 0
        for i, (video_path, chunk_idx, split) in enumerate(video_chunks):
            output_path = OUTPUT_DIR / split / video_path.name
            
            result = process_single_video((
                video_path, 
                action_path, 
                output_path, 
                merged_actions,
                action_offset
            ))
            
            if result["status"] == "success":
                stats["success"] += 1
            elif result["status"] == "skip":
                stats["skip"] += 1
            else:
                stats["error"] += 1
                if stats["error"] <= 5:  # Only print first few errors
                    print(f"Error: {result}")
            
            # Update offset for next chunk
            action_offset += chunk_frame_counts[i]
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Success: {stats['success']}")
    print(f"Skipped: {stats['skip']}")
    print(f"Errors: {stats['error']}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

