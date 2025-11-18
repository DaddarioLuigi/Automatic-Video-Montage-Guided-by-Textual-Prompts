"""
Caption Generation Module

This module provides functionality to generate textual descriptions
of video frames using BLIP image captioning models.
"""

from .generator import CaptionGenerator, generate_captions

__all__ = ['CaptionGenerator', 'generate_captions']



