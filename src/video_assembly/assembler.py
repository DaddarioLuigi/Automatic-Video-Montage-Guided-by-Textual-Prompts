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
                    audio: bool = True) -> str:
    """
    Assemble selected segments into a final montage video.
    
    Args:
        video_path: Path to the original video file
        selected_segments: List of (start_frame, end_frame) tuples
        fps: Video frames per second
        output_path: Path for the output montage video
        audio: Whether to preserve audio in the montage
        
    Returns:
        Path to the created montage video
    """
    if not selected_segments:
        raise ValueError("No segments provided for montage assembly")
    
    video = VideoFileClip(video_path)
    clips = [
        video.subclip(start / fps, end / fps)
        for start, end in selected_segments
    ]
    
    if not clips:
        raise ValueError("No valid clips could be created from segments")
    
    final_clip = concatenate_videoclips(clips)
    
    final_clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac' if audio else None
    )
    
    final_clip.close()
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

