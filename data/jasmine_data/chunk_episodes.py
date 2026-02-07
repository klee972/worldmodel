"""Chunk episodes into fixed-length sequences for training.

This module handles chunking of video episodes where for T video frames,
we have T-1 actions (action[i] transitions from frame[i] to frame[i+1]).

When chunking:
- Each chunk contains `chunk_size` frames
- Each chunk contains `chunk_size - 1` actions
- The last chunk may be shorter if the episode doesn't divide evenly
"""

from typing import Dict, Any
import numpy as np


def chunk_episode(
    episode: Dict[str, np.ndarray],
    chunk_size: int,
    min_chunk_size: int = 2,
) -> list[Dict[str, np.ndarray]]:
    """Chunk an episode into fixed-length sequences.
    
    Args:
        episode: Dictionary with 'videos' (T, H, W, C) and optionally
                 'actions' (T-1,) or other arrays
        chunk_size: Target number of frames per chunk
        min_chunk_size: Minimum chunk size to keep
        
    Returns:
        List of chunked episodes
    """
    videos = episode["videos"]
    T = videos.shape[0]
    
    # Calculate number of full chunks
    num_chunks = (T + chunk_size - 1) // chunk_size
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, T)
        
        # Skip chunks that are too small
        if end - start < min_chunk_size:
            continue
        
        chunk = {}
        
        # Chunk videos
        chunk["videos"] = videos[start:end]
        
        # Chunk other arrays
        for key, value in episode.items():
            if key == "videos":
                continue
            
            if isinstance(value, np.ndarray):
                # For actions: T-1 actions for T frames
                if key in ["actions", "button_actions", "camera_actions"]:
                    # Actions are T-1 length, map to frame indices
                    action_start = start
                    action_end = end - 1  # T-1 actions for T frames
                    if action_end > action_start:
                        chunk[key] = value[action_start:action_end]
                    else:
                        # Single frame, no actions
                        chunk[key] = value[:0]  # Empty with same dtype
                else:
                    # Other arrays are assumed to be frame-aligned
                    chunk[key] = value[start:end]
            else:
                # Non-array values are copied as-is
                chunk[key] = value
        
        chunks.append(chunk)
    
    return chunks


def merge_chunks(
    chunks: list[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Merge multiple chunks back into a single episode.
    
    This is the inverse of chunk_episode, useful for evaluation.
    
    Args:
        chunks: List of chunked episodes
        
    Returns:
        Merged episode dictionary
    """
    if not chunks:
        return {}
    
    merged = {}
    
    # Get all keys from first chunk
    keys = list(chunks[0].keys())
    
    for key in keys:
        values = [chunk[key] for chunk in chunks]
        
        if isinstance(values[0], np.ndarray):
            merged[key] = np.concatenate(values, axis=0)
        else:
            # Non-arrays: just take from first chunk
            merged[key] = values[0]
    
    return merged


def validate_episode_shape(
    episode: Dict[str, np.ndarray],
) -> bool:
    """Validate that episode has correct shapes.
    
    For T frames, expects T-1 actions.
    
    Args:
        episode: Episode dictionary
        
    Returns:
        True if valid
    """
    if "videos" not in episode:
        return False
    
    T = episode["videos"].shape[0]
    
    for key in ["actions", "button_actions", "camera_actions"]:
        if key in episode:
            if episode[key].shape[0] != T - 1:
                print(f"Invalid {key} shape: expected {T-1}, got {episode[key].shape[0]}")
                return False
    
    return True


