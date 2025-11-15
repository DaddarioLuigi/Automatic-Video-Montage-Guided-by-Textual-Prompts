"""
CLIP Matching Module

This module provides functionality to compute semantic similarity
between prompts and captions using CLIP.
"""

from .matcher import CLIPMatcher, compute_similarities, select_segments

__all__ = ['CLIPMatcher', 'compute_similarities', 'select_segments']


