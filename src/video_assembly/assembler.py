"""
Video Assembly and Montage Creation

This module assembles selected video segments into a final montage video.
"""

from moviepy.editor import VideoFileClip, concatenate_videoclips
from typing import List, Tuple, Optional


def assemble_montage(video_path: str,
                    selected_segments: List[Tuple[int, int]],
                    fps: float,
                    output_path: str = "final_montage.mp4",
                    audio: bool = True,
                    min_segment_duration: float = 0.1) -> str:
    """
    Assemble selected segments into a final montage video.
    
    Args:
        video_path: Path to the original video file
        selected_segments: List of (start_frame, end_frame) tuples
        fps: Video frames per second
        output_path: Path for the output montage video
        audio: Whether to preserve audio in the montage
        min_segment_duration: Minimum duration in seconds for a segment (default: 0.1s)
        
    Returns:
        Path to the created montage video
    """
    if not selected_segments:
        raise ValueError("No segments provided for montage assembly")
    
    video = VideoFileClip(video_path)
    video_duration = video.duration
    
    # Sort segments by start time
    sorted_segments = sorted(selected_segments, key=lambda x: x[0])
    
    clips = []
    for start_frame, end_frame in sorted_segments:
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # Ensure times are within video bounds
        start_time = max(0, min(start_time, video_duration))
        end_time = max(start_time + min_segment_duration, min(end_time, video_duration))
        
        # Skip if segment is too short or invalid
        if end_time - start_time < min_segment_duration:
            continue
        
        try:
            clip = video.subclip(start_time, end_time)
            # Ensure clip has video (not just audio)
            if clip.duration > 0:
                clips.append(clip)
        except Exception as e:
            print(f"Warning: Could not create clip from segment {start_frame}-{end_frame}: {e}")
            continue
    
    if not clips:
        video.close()
        raise ValueError("No valid clips could be created from segments")
    
    print(f"   Assembling {len(clips)} video clips...")
    
    # Concatenate clips with method='compose' to handle different sizes
    final_clip = concatenate_videoclips(clips, method='compose')
    
    # Write video file with proper settings
    final_clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac' if audio else None,
        fps=fps,
        preset='medium',
        logger=None  # Suppress verbose output
    )
    
    # Clean up
    final_clip.close()
    for clip in clips:
        clip.close()
    video.close()
    
    return output_path


class VideoAssembler:
    """Class-based wrapper for video assembly functionality."""
    
    def __init__(self, video_path: str, fps: float):
        self.video_path = video_path
        self.fps = fps
    
    def assemble(self, selected_segments: List[Tuple[int, int]],
                output_path: str = "final_montage.mp4",
                audio: bool = True) -> str:
        """Assemble segments into a montage."""
        return assemble_montage(
            self.video_path, selected_segments, self.fps, output_path, audio
        )

