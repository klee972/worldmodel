import jax
import numpy as np
import grain
from typing import Any, Optional
import pickle
import json
import os
from pathlib import Path

# Lazy import decord to avoid import errors if not installed
_decord = None
def _get_decord():
    global _decord
    if _decord is None:
        import decord
        decord.bridge.set_bridge('native')
        _decord = decord
    return _decord


# =============================================================================
# Cursor overlay utilities (adapted from VPT data_loader.py)
# =============================================================================

# Original recording height for MineRL contractor data
MINEREC_ORIGINAL_HEIGHT_PX = 720

# Default cursor image path (relative to this file)
DEFAULT_CURSOR_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    "cursors", "mouse_cursor_white_16x16.png"
)


def composite_images_with_alpha(image: np.ndarray, overlay: np.ndarray, alpha: np.ndarray, x: int, y: int):
    """
    Draw overlay image over base image at location (x, y) using alpha blending.
    
    Modifies image in-place.
    
    Args:
        image: Base image (H, W, C) uint8
        overlay: Overlay image (h, w, C) uint8
        alpha: Alpha channel (h, w, 1) float32 in [0, 1]
        x: X position (left edge of overlay)
        y: Y position (top edge of overlay)
    """
    ch = max(0, min(image.shape[0] - y, overlay.shape[0]))
    cw = max(0, min(image.shape[1] - x, overlay.shape[1]))
    if ch == 0 or cw == 0:
        return
    alpha = alpha[:ch, :cw]
    image[y:y + ch, x:x + cw, :] = (
        image[y:y + ch, x:x + cw, :] * (1 - alpha) + 
        overlay[:ch, :cw, :] * alpha
    ).astype(np.uint8)


def load_cursor_image(cursor_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load cursor image and extract alpha channel.
    
    Returns:
        Tuple of (cursor_rgb, cursor_alpha) where:
        - cursor_rgb: (16, 16, 3) uint8
        - cursor_alpha: (16, 16, 1) float32 in [0, 1]
    """
    import cv2
    cursor_image = cv2.imread(cursor_file, cv2.IMREAD_UNCHANGED)
    if cursor_image is None:
        raise FileNotFoundError(f"Cursor image not found: {cursor_file}")
    # Assume 16x16
    cursor_image = cursor_image[:16, :16, :]
    cursor_alpha = cursor_image[:, :, 3:] / 255.0
    cursor_rgb = cursor_image[:, :, :3]
    # Convert BGR to RGB
    cursor_rgb = cv2.cvtColor(cursor_rgb, cv2.COLOR_BGR2RGB)
    return cursor_rgb, cursor_alpha


def load_action_jsonl(jsonl_path: str) -> list[dict]:
    """Load action data from JSONL file."""
    with open(jsonl_path, 'r') as f:
        return [json.loads(line) for line in f]


class ParsePickle(grain.transforms.Map):
    """
    A Grain Map that deserializes pickled bytes into a dictionary.
    This is done once upfront to avoid redundant pickle.loads() calls.
    """

    def map(self, element: Any) -> dict:
        assert isinstance(element, bytes)
        return pickle.loads(element)


class EpisodeLengthFilter(grain.transforms.Filter):
    """
    A Grain Filter that keeps only episodes with sufficient length.
    """

    def __init__(self, seq_len: int):
        """Initializes the filter with sequence length requirements."""
        self.seq_len = seq_len

    def filter(self, element: dict) -> bool:
        """
        Filters episodes based on length.

        Args:
            element: A dictionary representing one record (already parsed).
                     Expected to contain 'sequence_length' (int)

        Returns:
            True if the episode has sufficient length, False otherwise.
        """
        return element["sequence_length"] >= self.seq_len


class ProcessEpisodeAndSlice(grain.transforms.RandomMap):
    """
    A Grain Transformation that combines parsing, slicing, and normalizing.
    """

    def __init__(self, seq_len: int, image_h: int, image_w: int, image_c: int):
        """Initializes the transformation with processing parameters."""
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.image_c = image_c

    def random_map(self, element: dict, rng: np.random.Generator) -> Any:
        """
        Processes a single raw episode from the data source.

        Args:
            element: A dictionary representing one record (already parsed).
                     Expected to contain 'raw_video' (bytes) and 'sequence_length' (int)
            rng: A per-record random number generator provided by the Grain sampler.

        Returns:
            A processed video sequence as a NumPy array with shape
            (seq_len, height, width, channels) and dtype float32.
        """
        video_shape = (
            element["sequence_length"],
            self.image_h,
            self.image_w,
            self.image_c,
        )
        episode_tensor = np.frombuffer(element["raw_video"], dtype=np.uint8)
        episode_tensor = episode_tensor.reshape(video_shape)

        current_episode_len = episode_tensor.shape[0]
        if current_episode_len < self.seq_len:
            raise ValueError(
                f"Episode length {current_episode_len} is shorter than "
                f"requested sequence length {self.seq_len}. This should "
                f"have been filtered out."
            )

        max_start_idx = current_episode_len - self.seq_len

        start_idx = rng.integers(0, max_start_idx + 1)

        seq = episode_tensor[start_idx : start_idx + self.seq_len]

        data_dict = {"videos": seq}
        if "actions" in element.keys():
            actions_tensor = np.array(element["actions"])
            actions = actions_tensor[start_idx : start_idx + self.seq_len]
            data_dict["actions"] = actions

        return data_dict


def get_dataloader(
    array_record_paths: list[str],
    seq_len: int,
    global_batch_size: int,
    image_h: int,
    image_w: int,
    image_c: int,
    num_workers: int = 1,
    prefetch_buffer_size: int = 1,
    seed: int = 42,
):
    """
    Creates a data loading pipeline using Grain.
    """
    if not array_record_paths:
        raise ValueError("array_record_paths list cannot be empty.")

    num_processes = jax.process_count()

    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"the number of JAX processes {num_processes} for proper sharding."
        )
    per_process_batch_size = global_batch_size // num_processes

    source = grain.sources.ArrayRecordDataSource(array_record_paths)

    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=seed,
    )

    operations = [
        # Parse pickle once upfront to avoid redundant deserialization
        ParsePickle(),
        EpisodeLengthFilter(seq_len=seq_len),
        ProcessEpisodeAndSlice(
            seq_len=seq_len, image_h=image_h, image_w=image_w, image_c=image_c
        ),
        grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
    ]

    read_options = grain.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size,
        num_threads=4,  # Increase read threads for larger images
    )
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=8,  # Increase buffer per worker
        read_options=read_options,
    )

    return dataloader


# =============================================================================
# MP4 Video DataLoader (using decord for H.264 decoding)
# =============================================================================

class MP4VideoDataSource(grain.sources.RandomAccessDataSource):
    """
    A Grain data source that reads MP4 video files directly.
    
    Each record is either:
    - A video file path (str) when action loading is disabled
    - A tuple of (video_path, action_path) when action loading is enabled
    """
    
    def __init__(self, video_paths: list[str], action_paths: Optional[list[str]] = None):
        """
        Args:
            video_paths: List of video file paths.
            action_paths: Optional list of corresponding action JSONL paths.
                         If provided, must be same length as video_paths.
        """
        self._video_paths = video_paths
        self._action_paths = action_paths
        
        if action_paths is not None:
            assert len(video_paths) == len(action_paths), \
                f"video_paths ({len(video_paths)}) and action_paths ({len(action_paths)}) must have same length"
    
    def __len__(self) -> int:
        return len(self._video_paths)
    
    def __getitem__(self, index: int):
        if self._action_paths is not None:
            return (self._video_paths[index], self._action_paths[index])
        return self._video_paths[index]
    
    def __repr__(self) -> str:
        action_info = f", with_actions={self._action_paths is not None}" if self._action_paths else ""
        return f"MP4VideoDataSource(num_videos={len(self._video_paths)}{action_info})"


def scan_video_action_pairs(
    video_dir: str,
    action_dir: Optional[str] = None,
    video_extensions: tuple[str, ...] = (".mp4", ".MP4"),
    action_extension: str = ".jsonl",
) -> tuple[list[str], list[str]]:
    """
    Scan video and action directories to find matching pairs.
    
    Args:
        video_dir: Directory containing video files.
        action_dir: Directory containing action JSONL files. If None, 
                    assumes actions are in the same directory as videos.
        video_extensions: Video file extensions to include.
        action_extension: Action file extension (default: .jsonl).
    
    Returns:
        Tuple of (video_paths, action_paths) containing only matched pairs.
    """
    video_dir = Path(video_dir)
    action_dir = Path(action_dir) if action_dir else video_dir
    
    # Collect all video files
    video_files = {}
    for ext in video_extensions:
        for video_path in video_dir.glob(f"*{ext}"):
            stem = video_path.stem
            video_files[stem] = str(video_path)
    
    # Collect all action files
    action_files = {}
    for action_path in action_dir.glob(f"*{action_extension}"):
        stem = action_path.stem
        action_files[stem] = str(action_path)
    
    # Find common stems
    common_stems = set(video_files.keys()) & set(action_files.keys())
    common_stems = sorted(common_stems)  # Sort for reproducibility
    
    video_paths = [video_files[stem] for stem in common_stems]
    action_paths = [action_files[stem] for stem in common_stems]
    
    return video_paths, action_paths


class LoadAndSliceVideo(grain.transforms.RandomMap):
    """
    Load an MP4 video file, decode frames, and extract a random slice.
    
    Uses decord for efficient H.264 decoding with GPU acceleration if available.
    Optionally overlays cursor on GUI frames and loads action data from JSONL files.
    
    Based on VPT data_loader.py with additional features:
    - Attack button stuck fix
    - Hotbar scrollwheel tracking
    - Cursor overlay for GUI frames
    """
    
    def __init__(
        self, 
        seq_len: int, 
        image_h: Optional[int] = None,
        image_w: Optional[int] = None,
        target_fps: Optional[float] = None,
        min_video_length: int = 32,
        add_cursor: bool = False,
        cursor_file: Optional[str] = None,
        load_actions: bool = False,
        action_mapper: Optional[Any] = None,
        action_format: str = "flat",  # "flat", "hierarchical", or "raw"
        fix_attack_stuck: bool = True,
        track_hotbar: bool = True,
    ):
        """
        Args:
            seq_len: Number of frames to extract per sample.
            image_h: Target height (None = use original).
            image_w: Target width (None = use original).
            target_fps: Target FPS for sampling (None = use all frames).
            min_video_length: Minimum video length in frames.
            add_cursor: Whether to overlay cursor on GUI frames.
            cursor_file: Path to cursor PNG image (uses default if None).
            load_actions: Whether to load and return action data.
            action_mapper: Optional CameraHierarchicalActionMapping instance for 
                          converting raw actions to discrete indices. Required if
                          action_format is "flat" or "hierarchical".
            action_format: Format for returned actions:
                          - "flat": (T-1,) single discrete index
                          - "hierarchical": {"buttons": (T-1,), "camera": (T-1,)} 
                          - "raw": list of raw action dicts
            fix_attack_stuck: Fix attack button being stuck from recording start (VPT fix).
            track_hotbar: Track hotbar changes from scrollwheel (VPT fix).
        """
        self.seq_len = seq_len
        self.image_h = image_h
        self.image_w = image_w
        self.target_fps = target_fps
        self.min_video_length = min_video_length
        self.add_cursor = add_cursor
        self.load_actions = load_actions
        self.action_mapper = action_mapper
        self.action_format = action_format
        self.fix_attack_stuck = fix_attack_stuck
        self.track_hotbar = track_hotbar
        
        # Load cursor image if needed
        self.cursor_rgb = None
        self.cursor_alpha = None
        if add_cursor:
            cursor_path = cursor_file or DEFAULT_CURSOR_FILE
            if os.path.exists(cursor_path):
                self.cursor_rgb, self.cursor_alpha = load_cursor_image(cursor_path)
            else:
                print(f"Warning: Cursor file not found at {cursor_path}, disabling cursor overlay")
                self.add_cursor = False
    
    
    def _preprocess_actions(self, action_data: list[dict]) -> list[dict]:
        """
        Preprocess action data with VPT fixes.
        
        Applies:
        - Attack stuck fix: Remove stuck attack button from recording start
        - Hotbar tracking: Add hotbar.N actions when selection changes via scrollwheel
        
        Args:
            action_data: Raw action data from JSONL
        
        Returns:
            Preprocessed action data
        """
        if not action_data:
            return action_data
        
        processed = []
        attack_is_stuck = False
        last_hotbar = 0
        
        for i, step_data in enumerate(action_data):
            step = step_data.copy()
            
            # Fix attack stuck (VPT workaround)
            if self.fix_attack_stuck:
                if i == 0:
                    # Check if attack will be stuck down
                    if step.get("mouse", {}).get("newButtons") == [0]:
                        attack_is_stuck = True
                elif attack_is_stuck:
                    # Check if we press attack down, then it might not be stuck
                    if 0 in step.get("mouse", {}).get("newButtons", []):
                        attack_is_stuck = False
                
                # If still stuck, remove the action
                if attack_is_stuck:
                    mouse = step.get("mouse", {})
                    if "buttons" in mouse:
                        mouse["buttons"] = [b for b in mouse["buttons"] if b != 0]
                        step["mouse"] = mouse
            
            # Track hotbar changes from scrollwheel (VPT workaround)
            if self.track_hotbar:
                current_hotbar = step.get("hotbar", 0)
                if current_hotbar != last_hotbar:
                    # Add synthetic hotbar action
                    if "hotbar_action" not in step:
                        step["hotbar_action"] = current_hotbar + 1  # 1-indexed
                    last_hotbar = current_hotbar
            
            processed.append(step)
        
        return processed
    
    def _overlay_cursor(
        self, 
        frames: np.ndarray, 
        action_data: list[dict], 
        frame_indices: np.ndarray,
        original_height: int,
    ) -> np.ndarray:
        """
        Overlay cursor on frames where GUI is open.
        
        Args:
            frames: Video frames (T, H, W, C)
            action_data: List of action dicts from JSONL
            frame_indices: Indices of frames extracted from video
            original_height: Original video height for scaling
        
        Returns:
            Frames with cursor overlay applied
        """
        if self.cursor_rgb is None:
            return frames
        
        current_height = frames.shape[1]
        scale_factor = current_height / original_height
        
        for i, frame_idx in enumerate(frame_indices):
            if frame_idx >= len(action_data):
                continue
            
            step_data = action_data[frame_idx]
            
            if step_data.get("isGuiOpen", False):
                mouse = step_data.get("mouse", {})
                cursor_x = int(mouse.get("x", 0) * scale_factor)
                cursor_y = int(mouse.get("y", 0) * scale_factor)
                
                # Scale cursor if frame was resized
                if self.image_h is not None and self.image_w is not None:
                    cursor_scale = current_height / original_height
                else:
                    cursor_scale = 1.0
                
                # Apply cursor overlay
                composite_images_with_alpha(
                    frames[i], 
                    self.cursor_rgb, 
                    self.cursor_alpha, 
                    cursor_x, 
                    cursor_y
                )
        
        return frames
    
    def _extract_actions(
        self, 
        action_data: list[dict], 
        frame_indices: np.ndarray
    ):
        """
        Extract actions for the selected frames.
        
        For T frames, we return T-1 actions (action[i] is the action taken 
        after observing frame[i] to get to frame[i+1]).
        
        Args:
            action_data: Preprocessed action data
            frame_indices: Indices of frames extracted
        
        Returns:
            Depending on action_format:
            - "flat": (T-1,) int32 array of discrete indices
            - "hierarchical": {"buttons": (T-1,), "camera": (T-1,)} 
            - "raw": list of T-1 raw action dicts
            Returns None if action data is insufficient.
        """
        if not action_data:
            return None
        
        # Get actions for frames [0, T-2] (corresponding to transitions 0->1, 1->2, ..., T-2->T-1)
        action_indices = frame_indices[:-1]  # T-1 actions
        
        # Check if all indices are valid
        if any(idx >= len(action_data) for idx in action_indices):
            return None
        
        # Extract raw actions for these frames
        raw_actions = [action_data[idx] for idx in action_indices]
        
        if self.action_format == "raw" or self.action_mapper is None:
            # Return raw action dicts
            return raw_actions
        
        try:
            if self.action_format == "flat":
                # Convert to flat discrete indices
                return self.action_mapper.raw_batch_to_discrete_indices(raw_actions)  # (T-1,)
            elif self.action_format == "hierarchical":
                # Convert to hierarchical format
                hierarchical = self.action_mapper.raw_batch_to_hierarchical(raw_actions)
                # Squeeze the last dimension: (T-1, 1) -> (T-1,)
                return {
                    "buttons": hierarchical["buttons"].squeeze(-1),  # (T-1,)
                    "camera": hierarchical["camera"].squeeze(-1),    # (T-1,)
                }
            else:
                raise ValueError(f"Unknown action_format: {self.action_format}")
        except Exception as e:
            print(f"Warning: Failed to convert actions: {e}")
            return None
    
    def random_map(self, record, rng: np.random.Generator) -> Optional[dict]:
        """
        Load and slice a video file, optionally with actions.
        
        Args:
            record: Either a video path (str) or tuple of (video_path, action_path).
            rng: Random number generator.
        
        Returns:
            Dict with:
            - 'videos': (seq_len, H, W, C) uint8 array
            - 'actions': (seq_len-1,) int32 array if load_actions=True and action_mapper provided,
                         or list of seq_len-1 raw action dicts if no mapper
            Returns None if video is too short or required data is missing.
        """
        # Handle both (video_path, action_path) tuple and video_path string
        if isinstance(record, tuple):
            video_path, action_path = record
        else:
            video_path = record
            action_path = None
        
        decord = _get_decord()
        
        try:
            # Use CPU context (GPU requires CUDA build of decord)
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            original_height = vr[0].shape[0]  # Get original height

            # Load action data if needed (for cursor overlay or action loading)
            action_data = None
            if self.add_cursor or self.load_actions:
                if action_path:
                    try:
                        action_data = load_action_jsonl(action_path)
                        action_data = self._preprocess_actions(action_data)
                    except Exception as e:
                        print(f"Warning: Failed to load actions from {action_path}: {e}")
                        if self.load_actions:
                            return None
                elif self.load_actions:
                    # Action path not provided but required
                    print(f"Warning: No action path for {video_path}")
                    return None
            
            # Calculate frame indices based on target FPS
            if self.target_fps is not None and self.target_fps < video_fps:
                # Subsample frames to match target FPS
                frame_step = video_fps / self.target_fps
                available_indices = np.arange(0, total_frames, frame_step).astype(int)
            else:
                available_indices = np.arange(total_frames)
            
            num_available = len(available_indices)
            
            if num_available < self.seq_len:
                # Video too short, skip
                return None
            
            # Random start position
            max_start = num_available - self.seq_len
            start_idx = rng.integers(0, max_start + 1)
            frame_indices = available_indices[start_idx:start_idx + self.seq_len]
            
            # Batch decode frames
            frames = vr.get_batch(frame_indices.tolist()).asnumpy()  # (T, H, W, C)
            
            # Overlay cursor before resizing (uses original coordinates)
            if self.add_cursor and action_data is not None:
                frames = self._overlay_cursor(frames, action_data, frame_indices, original_height)
            
            # Resize if needed
            if self.image_h is not None and self.image_w is not None:
                if frames.shape[1] != self.image_h or frames.shape[2] != self.image_w:
                    import cv2
                    resized = np.empty(
                        (self.seq_len, self.image_h, self.image_w, frames.shape[3]), 
                        dtype=np.uint8
                    )
                    for i, frame in enumerate(frames):
                        resized[i] = cv2.resize(
                            frame, 
                            (self.image_w, self.image_h), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    frames = resized
            
            result = {"videos": frames}
            
            # Extract actions if requested
            if self.load_actions and action_data is not None:
                actions = self._extract_actions(action_data, frame_indices)
                if actions is None:
                    # Failed to extract actions, skip this video
                    return None
                result["actions"] = actions
            
            return result
        
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            return None


class FilterNone(grain.transforms.Filter):
    """Filter out None values (failed video loads)."""
    
    def filter(self, element: Any) -> bool:
        return element is not None


def get_video_dataloader(
    video_dir: str,
    seq_len: int,
    global_batch_size: int,
    image_h: Optional[int] = None,
    image_w: Optional[int] = None,
    image_c: Optional[int] = None,
    target_fps: Optional[float] = None,
    num_workers: int = 4,
    prefetch_buffer_size: int = 2,
    seed: int = 42,
    video_extensions: tuple[str, ...] = (".mp4", ".MP4"),
    add_cursor: bool = False,
    cursor_file: Optional[str] = None,
    action_dir: Optional[str] = None,
    load_actions: bool = False,
    action_mapper: Optional[Any] = None,
    action_format: str = "flat",
    fix_attack_stuck: bool = True,
    track_hotbar: bool = True,
):
    """
    Creates a data loading pipeline for MP4 video files with optional actions.
    
    Args:
        video_dir: Directory containing MP4 files.
        seq_len: Number of frames per sample.
        global_batch_size: Total batch size across all processes.
        image_h: Target image height (None = original).
        image_w: Target image width (None = original).
        target_fps: Target FPS for frame sampling (None = use all frames).
        num_workers: Number of worker processes for data loading.
        prefetch_buffer_size: Prefetch buffer size.
        seed: Random seed.
        video_extensions: File extensions to include.
        add_cursor: Whether to overlay cursor on GUI frames (requires action JSONL files).
        cursor_file: Path to cursor PNG image with alpha channel (uses default if None).
        action_dir: Directory containing JSONL action files. If None, assumes 
                    JSONL files are in the same directory as videos.
        load_actions: Whether to load and return action data from JSONL files.
        action_mapper: Optional CameraHierarchicalActionMapping instance for converting
                       raw actions to discrete indices. Required if action_format is 
                       "flat" or "hierarchical".
        action_format: Format for returned actions:
                       - "flat": (B, T-1) single discrete index tensor
                       - "hierarchical": {"buttons": (B, T-1), "camera": (B, T-1)} dict
                       - "raw": list of raw action dicts
        fix_attack_stuck: Apply VPT fix for stuck attack button at recording start.
        track_hotbar: Apply VPT fix for hotbar changes from scrollwheel.
    
    Returns:
        A Grain DataLoader yielding batches with:
        - 'videos': (B, T, H, W, C) uint8 array
        - 'actions': Depends on action_format:
            - "flat": (B, T-1) int32 array
            - "hierarchical": {"buttons": (B, T-1), "camera": (B, T-1)} 
            - "raw": list of raw action dicts
    """
    video_dir_path = Path(video_dir)
    
    # If actions are needed (for loading or cursor overlay), scan for matched pairs
    action_paths = None
    if load_actions or add_cursor:
        video_paths, action_paths = scan_video_action_pairs(
            video_dir=video_dir,
            action_dir=action_dir,
            video_extensions=video_extensions,
        )
        
        if not video_paths:
            raise ValueError(
                f"No matching video-action pairs found.\n"
                f"Video dir: {video_dir}\n"
                f"Action dir: {action_dir or video_dir}"
            )
        
        print(f"Found {len(video_paths)} matched video-action pairs")
        if add_cursor:
            print(f"Cursor overlay enabled")
        if load_actions:
            print(f"Action loading enabled (format: {action_format})")
            if action_mapper:
                print(f"Using action mapper: {type(action_mapper).__name__}")
    else:
        # No actions needed, just scan videos
        video_paths = []
        for ext in video_extensions:
            video_paths.extend([str(p) for p in video_dir_path.glob(f"*{ext}")])
        video_paths = sorted(video_paths)
        
        if not video_paths:
            raise ValueError(f"No video files found in {video_dir}")
        
        print(f"Found {len(video_paths)} video files in {video_dir}")
    
    num_processes = jax.process_count()
    if global_batch_size % num_processes != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"the number of JAX processes {num_processes}."
        )
    per_process_batch_size = global_batch_size // num_processes
    
    source = MP4VideoDataSource(video_paths, action_paths)
    
    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        shard_options=grain.sharding.ShardByJaxProcess(drop_remainder=True),
        shuffle=True,
        num_epochs=None,
        seed=seed,
    )
    
    operations = [
        LoadAndSliceVideo(
            seq_len=seq_len,
            image_h=image_h,
            image_w=image_w,
            target_fps=target_fps,
            add_cursor=add_cursor,
            cursor_file=cursor_file,
            load_actions=load_actions,
            action_mapper=action_mapper,
            action_format=action_format,
            fix_attack_stuck=fix_attack_stuck,
            track_hotbar=track_hotbar,
        ),
        FilterNone(),
        grain.transforms.Batch(batch_size=per_process_batch_size, drop_remainder=True),
    ]
    
    read_options = grain.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size,
        num_threads=2,
    )
    
    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=operations,
        worker_count=num_workers,
        worker_buffer_size=4,
        read_options=read_options,
    )
    
    return dataloader


if __name__ == "__main__":
    from jasmine.models.dreamer4_models import CameraHierarchicalActionMapping

    # Action mapper 생성
    action_mapper = CameraHierarchicalActionMapping(
        n_camera_bins=11,
        camera_maxval=10,
        camera_binsize=2,
    )

    # Video + Action 함께 로딩
    dataloader = get_video_dataloader(
        video_dir="/home/4bkang/rl/jasmine/data/mc_test_chucked/val",
        seq_len=32,
        global_batch_size=8,
        image_h=224,
        image_w=384,
        load_actions=False,                    # action 로딩 비활성화
        add_cursor=False,                      # cursor 오버레이 비활성화
        # action 사용 시 아래 옵션 활성화:
        # load_actions=True,
        # action_dir="data/open_ai_minecraft_actions_files/",
        # action_mapper=action_mapper,
        # action_format="hierarchical",
        # add_cursor=True,
    )

    for batch in dataloader:
        import pdb; pdb.set_trace()
        videos = batch["videos"]   # (B, T, H, W, C) uint8
        actions = batch["actions"] # (B, T-1) int32
        import pdb; pdb.set_trace()
        break
        
        