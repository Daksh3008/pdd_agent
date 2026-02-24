# pdd_no_audio/frame_extraction/smart_sampler.py

"""
Smart frame selection from detected scene changes.
Scales frame count with video duration — longer videos get more frames.
"""

import os
import cv2
import numpy as np
from typing import List, Dict

from pdd_no_audio.config import frame_config


def compute_target_frames(video_duration_seconds: float, max_frames_override: int = None) -> int:
    """
    Compute how many frames to keep based on video duration.

    Short videos (< 10 min): use max_key_frames default (40)
    Longer videos: scale at frames_per_minute rate
    Cap at absolute_max_frames

    Examples:
        5 min  → max(40, 5*4)  = 40 frames
        15 min → max(40, 15*4) = 60 frames
        30 min → max(40, 30*4) = 120 frames
        50 min → max(40, 50*4) = 200 frames
    """
    if max_frames_override:
        base = max_frames_override
    else:
        base = frame_config.max_key_frames

    duration_minutes = video_duration_seconds / 60.0
    scaled = int(duration_minutes * frame_config.frames_per_minute)

    target = max(base, scaled)
    target = min(target, frame_config.absolute_max_frames)

    return target


def select_key_frames(
    scene_changes: List[Dict],
    output_dir: str,
    max_frames: int = None,
    video_path: str = None,
    video_duration: float = None
) -> List[Dict]:
    """
    Select and save key frames from detected scene changes.
    Scales frame count with video duration for longer videos.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute target frame count based on video duration
    if video_duration is None and video_path:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total / fps if fps > 0 else 0
            cap.release()

    target_frames = compute_target_frames(
        video_duration or 0, max_frames
    )

    if not scene_changes:
        print(f"    [Sampler] No scene changes provided")
        if video_path:
            return _extract_evenly_spaced(video_path, output_dir, target_frames)
        return []

    print(
        f"    [Sampler] {len(scene_changes)} scene changes detected, "
        f"target: {target_frames} frames "
        f"(video: {(video_duration or 0)/60:.1f} min)"
    )

    selected = scene_changes

    # If too many, subsample evenly but keep more than before
    if len(selected) > target_frames:
        step = len(selected) / target_frames
        indices = [int(i * step) for i in range(target_frames)]
        # Always include first and last
        if 0 not in indices:
            indices[0] = 0
        if len(selected) - 1 not in indices:
            indices[-1] = len(selected) - 1
        # Remove duplicates and sort
        indices = sorted(set(indices))
        selected = [selected[i] for i in indices]
        print(f"    [Sampler] Subsampled to {len(selected)} frames")
    else:
        print(f"    [Sampler] Keeping all {len(selected)} detected scenes")

    # Save frames to disk
    key_frames = []
    for i, scene in enumerate(selected):
        frame = scene.get("frame")
        if frame is None:
            continue

        timestamp = scene.get("timestamp", 0.0)
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        filename = f"frame_{i:03d}_{minutes}m{seconds:02d}s.jpg"
        frame_path = os.path.join(output_dir, filename)

        cv2.imwrite(frame_path, frame)

        key_frames.append({
            "path": frame_path,
            "timestamp": timestamp,
            "frame_index": scene.get("frame_index", i),
            "ssim_score": scene.get("ssim_score", 0.0),
            "order": i
        })

    # If still too few frames, add evenly spaced
    min_needed = max(10, target_frames // 4)
    if len(key_frames) < min_needed and video_path:
        print(
            f"    [Sampler] Only {len(key_frames)} frames, "
            f"adding evenly spaced to reach {min_needed}..."
        )
        extra = _extract_evenly_spaced(
            video_path, output_dir,
            num_frames=min_needed - len(key_frames),
            existing_timestamps=[kf["timestamp"] for kf in key_frames]
        )
        key_frames.extend(extra)
        key_frames.sort(key=lambda x: x["timestamp"])
        for i, kf in enumerate(key_frames):
            kf["order"] = i

    duration_str = f"{(video_duration or 0)/60:.1f}min"
    rate = len(key_frames) / ((video_duration or 1) / 60)
    print(
        f"    [Sampler] Final: {len(key_frames)} key frames "
        f"({rate:.1f} frames/min for {duration_str})"
    )
    return key_frames


def _extract_evenly_spaced(
    video_path: str,
    output_dir: str,
    num_frames: int = 15,
    existing_timestamps: List[float] = None
) -> List[Dict]:
    """Extract evenly spaced frames as fallback."""
    existing_timestamps = existing_timestamps or []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration <= 0:
        cap.release()
        return []

    start = duration * 0.02
    end = duration * 0.98
    interval = (end - start) / (num_frames + 1)

    frames = []
    for i in range(num_frames):
        timestamp = start + interval * (i + 1)

        if any(abs(timestamp - et) < 2.0 for et in existing_timestamps):
            continue

        frame_idx = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            filename = f"frame_even_{i:03d}_{minutes}m{seconds:02d}s.jpg"
            frame_path = os.path.join(output_dir, filename)
            cv2.imwrite(frame_path, frame)

            frames.append({
                "path": frame_path,
                "timestamp": timestamp,
                "frame_index": frame_idx,
                "ssim_score": -1.0,
                "order": -1
            })

    cap.release()
    print(f"    [Sampler] Added {len(frames)} evenly-spaced fallback frames")
    return frames