"""
Baseline Methods for Video Montage Selection

This module implements baseline methods for comparison:
1. Random selection
2. Uniform temporal sampling
3. First N segments
4. Motion-based selection (without semantic matching)
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Dict


def random_selection(motion_segments: List[Tuple[int, int]],
                    n_segments: int,
                    seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Randomly select N segments from motion segments.
    
    Args:
        motion_segments: List of all motion segments
        n_segments: Number of segments to select
        seed: Random seed for reproducibility
        
    Returns:
        List of randomly selected segments
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if n_segments >= len(motion_segments):
        return motion_segments.copy()
    
    return random.sample(motion_segments, n_segments)


def uniform_selection(motion_segments: List[Tuple[int, int]],
                     n_segments: int) -> List[Tuple[int, int]]:
    """
    Uniformly sample segments across the video timeline.
    
    Args:
        motion_segments: List of all motion segments
        n_segments: Number of segments to select
        
    Returns:
        List of uniformly sampled segments
    """
    if n_segments >= len(motion_segments):
        return motion_segments.copy()
    
    sorted_segments = sorted(motion_segments, key=lambda x: x[0])
    indices = np.linspace(0, len(sorted_segments) - 1, n_segments, dtype=int)
    return [sorted_segments[i] for i in indices]


def first_n_selection(motion_segments: List[Tuple[int, int]],
                      n_segments: int) -> List[Tuple[int, int]]:
    """
    Select first N segments by temporal order.
    
    Args:
        motion_segments: List of all motion segments
        n_segments: Number of segments to select
        
    Returns:
        List of first N segments
    """
    sorted_segments = sorted(motion_segments, key=lambda x: x[0])
    return sorted_segments[:n_segments]


def motion_intensity_selection(motion_segments: List[Tuple[int, int]],
                              motions: np.ndarray,
                              n_segments: int) -> List[Tuple[int, int]]:
    """
    Select segments based on motion intensity (without semantic matching).
    
    Args:
        motion_segments: List of all motion segments
        motions: Array of motion values per frame
        n_segments: Number of segments to select
        
    Returns:
        List of segments with highest motion intensity
    """
    segment_intensities = []
    for start, end in motion_segments:
        segment_motions = motions[start:end+1] if end < len(motions) else motions[start:]
        avg_intensity = np.mean(segment_motions) if len(segment_motions) > 0 else 0
        segment_intensities.append((avg_intensity, (start, end)))
    
    segment_intensities.sort(reverse=True, key=lambda x: x[0])
    return [seg for _, seg in segment_intensities[:n_segments]]


class BaselineMethods:
    """Class for running baseline methods."""
    
    def __init__(self, fps: float, seed: Optional[int] = 42):
        self.fps = fps
        self.seed = seed
        self.baseline_results = {}
    
    def run_random_baseline(self, motion_segments: List[Tuple[int, int]],
                           n_segments: int) -> List[Tuple[int, int]]:
        """Run random selection baseline."""
        return random_selection(motion_segments, n_segments, self.seed)
    
    def run_uniform_baseline(self, motion_segments: List[Tuple[int, int]],
                            n_segments: int) -> List[Tuple[int, int]]:
        """Run uniform sampling baseline."""
        return uniform_selection(motion_segments, n_segments)
    
    def run_first_n_baseline(self, motion_segments: List[Tuple[int, int]],
                            n_segments: int) -> List[Tuple[int, int]]:
        """Run first N segments baseline."""
        return first_n_selection(motion_segments, n_segments)
    
    def run_motion_intensity_baseline(self, motion_segments: List[Tuple[int, int]],
                                     motions: np.ndarray,
                                     n_segments: int) -> List[Tuple[int, int]]:
        """Run motion intensity baseline."""
        return motion_intensity_selection(motion_segments, motions, n_segments)
    
    def run_all_baselines(self, motion_segments: List[Tuple[int, int]],
                         n_segments: int,
                         motions: Optional[np.ndarray] = None) -> Dict:
        """
        Run all baseline methods.
        
        Returns:
            Dictionary with baseline results
        """
        results = {
            'random': self.run_random_baseline(motion_segments, n_segments),
            'uniform': self.run_uniform_baseline(motion_segments, n_segments),
            'first_n': self.run_first_n_baseline(motion_segments, n_segments)
        }
        
        if motions is not None:
            results['motion_intensity'] = self.run_motion_intensity_baseline(
                motion_segments, motions, n_segments
            )
        
        self.baseline_results = results
        return results

