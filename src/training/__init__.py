"""
Training Module

This module provides functionality for fine-tuning BLIP, CLIP,
and other components of the video montage pipeline.
"""

from .blip_finetuning import BlipFineTuner, fine_tune_blip
from .clip_finetuning import ClipFineTuner, fine_tune_clip
from .dataset import VideoFrameDataset, create_finetuning_dataset

__all__ = [
    'BlipFineTuner',
    'fine_tune_blip',
    'ClipFineTuner',
    'fine_tune_clip',
    'VideoFrameDataset',
    'create_finetuning_dataset'
]


