"""
Frame Extraction from Video Segments

This module extracts representative frames from detected motion segments.
"""

import cv2
from PIL import Image
from typing import List, Tuple


def extract_center_frames(video_path: str, segments: List[Tuple[int, int]]):
    """
    Extract center frames from each detected motion segment.
    
    Args:
        video_path: Path to the video file
        segments: List of (start_frame, end_frame) tuples
        
    Returns:
        List of (frame_index, PIL.Image) tuples
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for start, end in segments:
        mid = int((start + end) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append((mid, image))
    cap.release()
    return frames


class FrameExtractor:
    """Class-based wrapper for frame extraction functionality."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        
    def extract_center_frames(self, segments: List[Tuple[int, int]]):
        """Extract center frames from motion segments."""
        return extract_center_frames(self.video_path, segments)


