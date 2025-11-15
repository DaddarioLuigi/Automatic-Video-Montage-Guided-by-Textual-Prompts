"""
Configuration module for the video montage pipeline.
"""

from .config import (
    MOTION_DETECTION,
    CAPTION_GENERATION,
    PROMPT_PROCESSING,
    CLIP_MATCHING,
    ANALYSIS,
    VIDEO_ASSEMBLY,
    DEFAULT_CONFIG
)

__all__ = [
    'MOTION_DETECTION',
    'CAPTION_GENERATION',
    'PROMPT_PROCESSING',
    'CLIP_MATCHING',
    'ANALYSIS',
    'VIDEO_ASSEMBLY',
    'DEFAULT_CONFIG'
]


