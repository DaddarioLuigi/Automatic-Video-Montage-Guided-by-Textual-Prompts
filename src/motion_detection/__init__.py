"""
Motion Detection Module

This module provides functionality to detect motion segments in videos
using frame difference analysis.
"""

from .detector import MotionDetector, analyze_motion, detect_motion

__all__ = ['MotionDetector', 'analyze_motion', 'detect_motion']



