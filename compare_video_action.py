"""
Compare video (array_record) and action (jsonl) files to understand their relationship.
"""

import pickle
import json
from pathlib import Path
import array_record.python.array_record_module as ar
import numpy as np
from collections import defaultdict


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
    
    For N action frames:
    - Video frame 0 corresponds to initial state (no action needed for it)
    - Video frame i (i >= 1) corresponds to merged(action[2*(i-1)], action[2*(i-1)+1])
    
    If actions has length 2N, we get N-1 merged actions (for N video frames).
    """
    merged_actions = []
    
    # Merge pairs: (0,1), (2,3), (4,5), ...
    for i in range(0, len(actions) - 1, 2):
        if i + 1 < len(actions):
            merged = merge_two_actions(actions[i], actions[i + 1])
            merged_actions.append(merged)
    
    return merged_actions


def get_episode_id_from_action_file(path: Path) -> str:
    """Extract episode ID from action file path."""
    # e.g., 10.0_cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl
    return path.stem  # Remove .jsonl extension


def get_episode_id_from_video_file(path: Path) -> str:
    """Extract episode ID from video file path."""
    # e.g., 10.0_cheeky-cornflower-setter-02e496ce4abb-20220421-092639_chunk000.array_record
    stem = path.stem  # Remove .array_record
    # Remove _chunk### suffix if present
    if "_chunk" in stem:
        stem = stem.rsplit("_chunk", 1)[0]
    return stem


def load_action_file(path: Path) -> list[dict]:
    """Load all actions from a jsonl file."""
    actions = []
    with open(path, 'r') as f:
        for line in f:
            actions.append(json.loads(line.strip()))
    return actions


def load_video_file(path: Path) -> dict:
    """Load video data from array_record file."""
    reader = ar.ArrayRecordReader(str(path))
    num_records = reader.num_records()
    print(f"  Number of records in array_record: {num_records}")
    
    if num_records == 0:
        reader.close()
        return None
    
    records = reader.read(0, num_records)
    reader.close()
    
    # Parse first record (usually only one per file)
    data = pickle.loads(records[0])
    return data


def find_matching_files(action_dir: Path, video_dirs: list[Path]):
    """Find matching video and action files."""
    # Get all action files
    action_files = {get_episode_id_from_action_file(p): p 
                    for p in action_dir.glob("*.jsonl")}
    
    # Get all video files
    video_files = defaultdict(list)  # episode_id -> list of chunk paths
    for video_dir in video_dirs:
        for p in video_dir.glob("*.array_record"):
            episode_id = get_episode_id_from_video_file(p)
            video_files[episode_id].append(p)
    
    # Find matches
    matches = []
    for episode_id, action_path in action_files.items():
        if episode_id in video_files:
            video_paths = sorted(video_files[episode_id])
            matches.append((episode_id, action_path, video_paths))
    
    return matches


def compare_episode(episode_id: str, action_path: Path, video_paths: list[Path], show_merge_example: bool = False):
    """Compare video and action data for a single episode."""
    print(f"\n{'='*80}")
    print(f"Episode ID: {episode_id}")
    print(f"{'='*80}")
    
    # Load actions
    print(f"\nAction file: {action_path}")
    actions = load_action_file(action_path)
    print(f"  Total action frames (20fps): {len(actions)}")
    
    if len(actions) > 0:
        first_action = actions[0]
        last_action = actions[-1]
        print(f"  First action tick: {first_action.get('tick', 'N/A')}")
        print(f"  Last action tick: {last_action.get('tick', 'N/A')}")
        print(f"  First action keys: {list(first_action.keys())}")
    
    # Merge actions to 10fps
    merged_actions = merge_actions_to_10fps(actions)
    print(f"  Merged action frames (10fps): {len(merged_actions)}")
    
    # Load video chunks
    total_video_frames = 0
    for i, video_path in enumerate(video_paths):
        print(f"\nVideo chunk {i}: {video_path.name}")
        video_data = load_video_file(video_path)
        
        if video_data:
            seq_len = video_data.get("sequence_length", "N/A")
            print(f"  Sequence length: {seq_len}")
            total_video_frames += seq_len if isinstance(seq_len, int) else 0
            
            # Check for actions in video data
            if "actions" in video_data:
                vid_actions = video_data["actions"]
                print(f"  Actions in video file: {len(vid_actions)} frames")
            else:
                print(f"  Actions in video file: NOT PRESENT")
            
            # Parse video shape
            if "raw_video" in video_data and isinstance(seq_len, int):
                video_bytes = video_data["raw_video"]
                print(f"  Video bytes: {len(video_bytes)}")
                # Assuming 90x160x3 resolution
                expected_bytes = seq_len * 90 * 160 * 3
                print(f"  Expected bytes (90x160x3): {expected_bytes}")
                if len(video_bytes) == expected_bytes:
                    print(f"  Video shape: ({seq_len}, 90, 160, 3)")
                else:
                    # Try to infer shape
                    bytes_per_frame = len(video_bytes) // seq_len
                    print(f"  Bytes per frame: {bytes_per_frame}")
    
    print(f"\n  TOTAL video frames: {total_video_frames}")
    print(f"  TOTAL action frames (20fps): {len(actions)}")
    print(f"  MERGED action frames (10fps): {len(merged_actions)}")
    
    # Calculate ratio
    if total_video_frames > 0 and len(actions) > 0:
        ratio = len(actions) / total_video_frames
        print(f"  Action/Video ratio: {ratio:.2f}")
        print(f"  (Expected 2.0 if video=10fps, action=20fps)")
        
        # Check alignment
        # Video has N frames, merged actions should have N-1 or N frames
        # (depending on whether we need action for initial frame)
        print(f"\n  Alignment check:")
        print(f"    Video frames: {total_video_frames}")
        print(f"    Merged actions: {len(merged_actions)}")
        print(f"    Expected merged (video_frames): {total_video_frames}")
        
    # Show merge example
    if show_merge_example and len(actions) >= 4:
        print(f"\n  --- MERGE EXAMPLE (tick 0+1 → merged[0]) ---")
        a0 = actions[0]
        a1 = actions[1]
        m0 = merged_actions[0]
        
        print(f"  Action tick 0: dx={a0['mouse']['dx']}, dy={a0['mouse']['dy']}, "
              f"yaw={a0['yaw']:.2f}, pitch={a0['pitch']:.2f}")
        print(f"  Action tick 1: dx={a1['mouse']['dx']}, dy={a1['mouse']['dy']}, "
              f"yaw={a1['yaw']:.2f}, pitch={a1['pitch']:.2f}")
        print(f"  Merged[0]:     dx={m0['mouse']['dx']}, dy={m0['mouse']['dy']}, "
              f"yaw={m0['yaw']:.2f}, pitch={m0['pitch']:.2f}")
        print(f"  Check: dx sum = {a0['mouse']['dx'] + a1['mouse']['dy']}, "
              f"yaw last = {a1['yaw']:.2f}")
        
        # Show another example with movement
        for i in range(2, min(20, len(actions)-1), 2):
            a0 = actions[i]
            a1 = actions[i+1]
            if a0['mouse']['dx'] != 0 or a0['mouse']['dy'] != 0 or \
               a1['mouse']['dx'] != 0 or a1['mouse']['dy'] != 0:
                m_idx = i // 2
                m = merged_actions[m_idx]
                print(f"\n  --- MERGE EXAMPLE (tick {i}+{i+1} → merged[{m_idx}]) ---")
                print(f"  Action tick {i}: dx={a0['mouse']['dx']}, dy={a0['mouse']['dy']}, "
                      f"keys={a0['keyboard']['keys']}")
                print(f"  Action tick {i+1}: dx={a1['mouse']['dx']}, dy={a1['mouse']['dy']}, "
                      f"keys={a1['keyboard']['keys']}")
                print(f"  Merged[{m_idx}]: dx={m['mouse']['dx']}, dy={m['mouse']['dy']}, "
                      f"keys={m['keyboard']['keys']}")
                break
    
    return {
        "episode_id": episode_id,
        "action_frames": len(actions),
        "video_frames": total_video_frames,
        "merged_frames": len(merged_actions),
        "ratio": len(actions) / total_video_frames if total_video_frames > 0 else 0
    }


def main():
    action_dir = Path("/home/4bkang/rl/jasmine/data/open_ai_minecraft_actions_files")
    video_dirs = [
        Path("/home/4bkang/rl/jasmine/data/minecraft_arrayrecords_filtered/train"),
        Path("/home/4bkang/rl/jasmine/data/minecraft_arrayrecords_filtered/val"),
        Path("/home/4bkang/rl/jasmine/data/minecraft_arrayrecords_filtered/test"),
    ]
    
    print("Finding matching video and action files...")
    matches = find_matching_files(action_dir, video_dirs)
    print(f"Found {len(matches)} matching episodes")
    
    if matches:
        # Compare first few episodes
        results = []
        for i, (episode_id, action_path, video_paths) in enumerate(matches[:3]):
            # Show merge example for first episode
            result = compare_episode(episode_id, action_path, video_paths, 
                                     show_merge_example=(i == 0))
            results.append(result)
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        for r in results:
            print(f"{r['episode_id']}: "
                  f"video={r['video_frames']}, action={r['action_frames']}, "
                  f"merged={r['merged_frames']}, ratio={r['ratio']:.2f}")


if __name__ == "__main__":
    main()

