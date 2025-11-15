"""
Motion Detection using Frame Difference Analysis

This module implements motion detection algorithms to identify
dynamic segments in videos.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def analyze_motion(cap, pixel_change_threshold=25):
    """
    Analyze motion in a video and suggest optimal thresholds.
    
    Args:
        cap: OpenCV VideoCapture object
        pixel_change_threshold: Threshold for detecting pixel changes between frames
        
    Returns:
        numpy array of motion values (number of changed pixels per frame)
    """
    if not cap.isOpened():
        raise IOError("Cannot open video/file")

    ret, prev = cap.read()
    if not ret:
        raise IOError("Cannot read the first frame from the video source")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        delta = cv2.absdiff(prev_gray, gray)
        motion = np.sum(delta > pixel_change_threshold)
        motions.append(motion)

        prev_gray = gray

    cap.release()

    motions = np.array(motions)

    print(f"Min motion: {motions.min()}")
    print(f"Max motion: {motions.max()}")
    print(f"Median motion: {np.median(motions)}")
    print(f"25th percentile: {np.percentile(motions, 25)}")
    print(f"75th percentile: {np.percentile(motions, 75)}")

    suggested_threshold = np.percentile(motions, 75)
    print(f"\nSuggested motion_pixel_threshold: {int(suggested_threshold)} (75th percentile)")

    plt.figure(figsize=(14, 5))
    plt.plot(motions, label='Motion (changed pixels)', linewidth=1.5, alpha=0.7)
    plt.axhline(suggested_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Suggested threshold ({int(suggested_threshold)})')
    plt.title('Motion Detection: Pixel Changes per Frame', fontsize=14, fontweight='bold')
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Motion (pixels changed)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(motions, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(suggested_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Suggested threshold ({int(suggested_threshold)})')
    plt.title('Distribution of Motion Values', fontsize=14, fontweight='bold')
    plt.xlabel('Motion (pixels changed)', fontsize=12)
    plt.ylabel('Number of Frames', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    return motions


def detect_motion(cap, pixel_change_threshold=25, motion_pixel_threshold=50000):
    """
    Detect motion segments in a video.
    
    Args:
        cap: OpenCV VideoCapture object
        pixel_change_threshold: Threshold for detecting pixel changes between frames
        motion_pixel_threshold: Minimum number of changed pixels to consider motion
        
    Returns:
        List of tuples (start_frame, end_frame) for each motion segment
    """
    if not cap.isOpened():
        raise IOError("Cannot open video/file")

    ret, prev = cap.read()
    if not ret:
        raise IOError("Cannot read the first frame from the video source")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_segments = []
    start_frame = None
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            if start_frame is not None:
                motion_segments.append((start_frame, frame_idx - 1))
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        delta = cv2.absdiff(prev_gray, gray)
        motion = np.sum(delta > pixel_change_threshold)

        if motion > motion_pixel_threshold:
            if start_frame is None:
                start_frame = frame_idx
        else:
            if start_frame is not None:
                motion_segments.append((start_frame, frame_idx))
                start_frame = None

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return motion_segments


class MotionDetector:
    """Class-based wrapper for motion detection functionality."""
    
    def __init__(self, pixel_change_threshold=25):
        self.pixel_change_threshold = pixel_change_threshold
        self.motions = None
        self.suggested_threshold = None
        
    def analyze(self, video_path: str):
        """Analyze motion in a video file."""
        cap = cv2.VideoCapture(video_path)
        self.motions = analyze_motion(cap, self.pixel_change_threshold)
        self.suggested_threshold = int(np.percentile(self.motions, 75))
        return self.motions
    
    def detect_segments(self, video_path: str, motion_pixel_threshold: int = None):
        """Detect motion segments in a video file."""
        if motion_pixel_threshold is None:
            if self.suggested_threshold is None:
                cap = cv2.VideoCapture(video_path)
                motions = analyze_motion(cap, self.pixel_change_threshold)
                motion_pixel_threshold = int(np.percentile(motions, 75))
            else:
                motion_pixel_threshold = self.suggested_threshold
        
        cap = cv2.VideoCapture(video_path)
        return detect_motion(cap, self.pixel_change_threshold, motion_pixel_threshold)

