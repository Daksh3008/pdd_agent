# pdd_no_audio/frame_extraction/scene_detector.py

"""
SSIM-based scene change detection.
Reads video frames at intervals and detects significant visual changes.
"""

import os
import cv2
import numpy as np
from typing import List, Dict

from pdd_no_audio.config import frame_config


def compute_ssim_gray(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute SSIM between two grayscale frames."""
    if frame1 is None or frame2 is None:
        return 0.0

    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    frame1 = frame1.astype(np.float64)
    frame2 = frame2.astype(np.float64)

    mu1 = cv2.GaussianBlur(frame1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(frame2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(frame1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(frame2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(frame1 * frame2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def compute_histogram_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute histogram correlation between two frames."""
    if frame1 is None or frame2 is None:
        return 0.0

    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(0.0, correlation)


def get_video_info(video_path: str) -> Dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 0, "frames": 0, "duration": 0, "width": 0, "height": 0}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0

    return {
        "fps": fps,
        "frames": total_frames,
        "duration": duration,
        "width": width,
        "height": height
    }


def detect_scene_changes(
    video_path: str,
    ssim_threshold: float = None,
    min_gap_seconds: float = None,
    sample_interval: float = None
) -> List[Dict]:
    """Detect scene changes in video using SSIM comparison."""
    ssim_threshold = ssim_threshold or frame_config.ssim_threshold
    min_gap = min_gap_seconds or frame_config.min_frame_gap_seconds
    sample_interval = sample_interval or frame_config.sample_interval_seconds

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [SceneDetect] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"    [SceneDetect] Video: {duration:.0f}s, {fps:.1f}fps, {total_frames} frames")
    print(f"    [SceneDetect] SSIM threshold: {ssim_threshold}, sample interval: {sample_interval}s")

    frame_skip = max(1, int(fps * sample_interval))
    scene_changes = []
    prev_gray = None
    last_scene_time = -999.0
    frames_checked = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        scene_changes.append({
            "timestamp": 0.0,
            "frame_index": 0,
            "ssim_score": 0.0,
            "frame": first_frame.copy()
        })
        prev_gray = first_gray
        last_scene_time = 0.0

    frame_idx = frame_skip
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp = frame_idx / fps

        ssim = compute_ssim_gray(prev_gray, current_gray)
        frames_checked += 1

        if ssim < ssim_threshold and (timestamp - last_scene_time) >= min_gap:
            hist_diff = compute_histogram_diff(prev_gray, current_gray)

            if hist_diff < 0.95:
                scene_changes.append({
                    "timestamp": timestamp,
                    "frame_index": frame_idx,
                    "ssim_score": ssim,
                    "frame": frame.copy()
                })
                last_scene_time = timestamp

        prev_gray = current_gray
        frame_idx += frame_skip

        if frames_checked % 100 == 0:
            print(f"    [SceneDetect] Checked {frames_checked} samples, found {len(scene_changes)} scenes...")

    cap.release()

    print(f"    [SceneDetect] Checked {frames_checked} samples, found {len(scene_changes)} scene changes")
    return scene_changes