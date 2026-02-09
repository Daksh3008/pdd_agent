# src/frame_extractor.py

"""
Video frame extraction using OpenCV.
Extracts frames at specific timestamps.
"""

import os
import re
from typing import List, Tuple, Dict, Optional
import cv2
from config import ACTION_KEYWORDS


def extract_timestamps_from_transcript(
    transcript_path: str,
    keywords: Dict[str, List[str]] = None
) -> Tuple[List[float], Dict[float, str]]:
    """
    Extract timestamps where action keywords are mentioned.
    
    Args:
        transcript_path: Path to transcript file.
        keywords: Dictionary of action keywords to search for.
        
    Returns:
        Tuple of (list of timestamps, dict mapping timestamp to text).
    """
    keywords = keywords or ACTION_KEYWORDS
    
    if not transcript_path or not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found: {transcript_path}")
        return [], {}
    
    timestamps = []
    transcript_dict = {}
    
    # Flatten all keywords to lowercase set
    all_keywords = set(
        word.lower()
        for keyword_list in keywords.values()
        for word in keyword_list
    )
    
    # Pattern: [start - end] text
    pattern = re.compile(r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(.*)")
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                start_time = float(match.group(1))
                text = match.group(3).strip()
                transcript_dict[start_time] = text
                
                # Check if any keyword is in the text
                text_lower = text.lower()
                if any(kw in text_lower for kw in all_keywords):
                    timestamps.append(start_time)
    
    print(f"Found {len(timestamps)} action timestamps")
    return timestamps, transcript_dict


def extract_frame(
    video_path: str,
    timestamp: float,
    output_dir: str
) -> Optional[str]:
    """
    Extract a single frame from video at specific timestamp.
    
    Args:
        video_path: Path to video file.
        timestamp: Time in seconds to extract frame.
        output_dir: Directory to save extracted frame.
        
    Returns:
        Path to saved frame image, or None if failed.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Invalid video FPS")
        cap.release()
        return None
    
    # Seek to timestamp (in milliseconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print(f"Failed to extract frame at {timestamp:.2f}s")
        return None
    
    # Save frame
    os.makedirs(output_dir, exist_ok=True)
    frame_number = int(timestamp * fps)
    frame_path = os.path.join(output_dir, f"frame_{frame_number}_{timestamp:.2f}.jpg")
    
    cv2.imwrite(frame_path, frame)
    print(f"Frame saved: {frame_path}")
    return frame_path


def extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[float],
    output_dir: str
) -> List[str]:
    """
    Extract multiple frames at given timestamps.
    
    Args:
        video_path: Path to video file.
        timestamps: List of timestamps in seconds.
        output_dir: Directory to save frames.
        
    Returns:
        List of paths to saved frame images.
    """
    frame_paths = []
    
    for timestamp in timestamps:
        frame_path = extract_frame(video_path, timestamp, output_dir)
        if frame_path:
            frame_paths.append(frame_path)
    
    print(f"Extracted {len(frame_paths)} frames")
    return frame_paths


def extract_frames_with_transcripts(
    video_path: str,
    transcript_path: str,
    output_dir: str,
    keywords: Dict[str, List[str]] = None
) -> List[Tuple[str, str]]:
    """
    Extract frames at action timestamps with their transcript text.
    
    Args:
        video_path: Path to video file.
        transcript_path: Path to transcript file.
        output_dir: Directory to save frames.
        keywords: Action keywords to search for.
        
    Returns:
        List of (frame_path, transcript_text) tuples.
    """
    # Get timestamps and text
    timestamps, transcript_dict = extract_timestamps_from_transcript(
        transcript_path, keywords
    )
    
    # Extract frames
    frame_transcript_pairs = []
    
    for timestamp in timestamps:
        frame_path = extract_frame(video_path, timestamp, output_dir)
        if frame_path:
            text = transcript_dict.get(timestamp, "No transcript available.")
            frame_transcript_pairs.append((frame_path, text))
    
    return frame_transcript_pairs


if __name__ == "__main__":
    # Test frame extraction
    test_video = input("Enter video path: ").strip()
    test_transcript = input("Enter transcript path: ").strip()
    
    if os.path.exists(test_video) and os.path.exists(test_transcript):
        results = extract_frames_with_transcripts(
            test_video,
            test_transcript,
            "./test_frames"
        )
        print(f"Extracted {len(results)} frame-transcript pairs")
    else:
        print("Files not found.")