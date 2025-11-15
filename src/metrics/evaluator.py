"""
Evaluation Metrics for Video Montage

This module computes various metrics to evaluate the quality
of the generated video montage and the pipeline performance.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter


def compute_precision_recall(similarities: List[Tuple[int, float, str]],
                             selected_segments: List[Tuple[int, int]],
                             motion_segments: List[Tuple[int, int]],
                             threshold: float) -> Dict:
    """
    Compute precision and recall metrics.
    
    Args:
        similarities: List of (frame_index, score, caption) tuples
        selected_segments: List of selected segments
        motion_segments: List of all motion segments
        threshold: Similarity threshold used for selection
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    frame_to_score = {idx: score for idx, score, _ in similarities}
    
    tp = len(selected_segments)
    fp = 0
    fn = 0
    
    for start, end in motion_segments:
        mid_frame = (start + end) // 2
        score = None
        
        for frame_idx, sim_score, _ in similarities:
            if start <= frame_idx <= end:
                score = sim_score
                break
        
        if score is not None:
            is_selected = (start, end) in selected_segments
            if score > threshold and not is_selected:
                fp += 1
            elif score <= threshold and is_selected:
                fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def compute_coverage_metrics(selected_segments: List[Tuple[int, int]],
                             motion_segments: List[Tuple[int, int]],
                             fps: float) -> Dict:
    """
    Compute coverage metrics (how much of the original video is covered).
    
    Args:
        selected_segments: List of selected segments
        motion_segments: List of all motion segments
        fps: Frames per second
        
    Returns:
        Dictionary with coverage metrics
    """
    if not selected_segments or not motion_segments:
        return {
            'coverage_ratio': 0.0,
            'temporal_coverage': 0.0,
            'segment_coverage_ratio': 0.0
        }
    
    selected_duration = sum((e - s) / fps for s, e in selected_segments)
    total_motion_duration = sum((e - s) / fps for s, e in motion_segments)
    video_start = min(s for s, _ in motion_segments)
    video_end = max(e for _, e in motion_segments)
    total_video_duration = (video_end - video_start) / fps
    
    coverage_ratio = selected_duration / total_motion_duration if total_motion_duration > 0 else 0.0
    temporal_coverage = selected_duration / total_video_duration if total_video_duration > 0 else 0.0
    segment_coverage_ratio = len(selected_segments) / len(motion_segments) if len(motion_segments) > 0 else 0.0
    
    return {
        'coverage_ratio': coverage_ratio,
        'temporal_coverage': temporal_coverage,
        'segment_coverage_ratio': segment_coverage_ratio,
        'selected_duration': selected_duration,
        'total_motion_duration': total_motion_duration,
        'total_video_duration': total_video_duration
    }


def compute_diversity_metrics(captions: List[Tuple[int, str]],
                              selected_indices: List[int]) -> Dict:
    """
    Compute diversity metrics for selected segments.
    
    Args:
        captions: List of all (frame_index, caption) tuples
        selected_indices: List of frame indices that were selected
        
    Returns:
        Dictionary with diversity metrics
    """
    if not selected_indices:
        return {
            'vocabulary_size': 0,
            'unique_words_ratio': 0.0,
            'caption_diversity': 0.0
        }
    
        selected_captions = [cap for idx, cap in captions if idx in selected_indices]
    all_words = []
    for caption in selected_captions:
        words = caption.lower().split()
        all_words.extend(words)
    
    if not all_words:
        return {
            'vocabulary_size': 0,
            'unique_words_ratio': 0.0,
            'caption_diversity': 0.0
        }
    
    unique_words = set(all_words)
    vocabulary_size = len(unique_words)
    unique_words_ratio = vocabulary_size / len(all_words) if len(all_words) > 0 else 0.0
    
    word_freq = Counter(all_words)
    total_words = len(all_words)
    entropy = -sum((freq / total_words) * np.log2(freq / total_words) 
                   for freq in word_freq.values() if freq > 0)
    max_entropy = np.log2(vocabulary_size) if vocabulary_size > 0 else 0
    caption_diversity = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return {
        'vocabulary_size': vocabulary_size,
        'unique_words_ratio': unique_words_ratio,
        'caption_diversity': caption_diversity,
        'total_words': len(all_words),
        'entropy': entropy
    }


def compute_temporal_coherence(selected_segments: List[Tuple[int, int]],
                               fps: float) -> Dict:
    """
    Compute temporal coherence metrics (how well segments flow together).
    
    Args:
        selected_segments: List of selected segments (start_frame, end_frame)
        fps: Frames per second
        
    Returns:
        Dictionary with temporal coherence metrics
    """
    if len(selected_segments) < 2:
        return {
            'avg_gap_duration': 0.0,
            'total_gaps': 0,
            'temporal_coherence_score': 1.0
        }
    
    sorted_segments = sorted(selected_segments, key=lambda x: x[0])
    gaps = []
    for i in range(len(sorted_segments) - 1):
        current_end = sorted_segments[i][1]
        next_start = sorted_segments[i + 1][0]
        gap = (next_start - current_end) / fps
        gaps.append(gap)
    
    avg_gap_duration = np.mean(gaps) if gaps else 0.0
    total_gaps = len(gaps)
    avg_segment_duration = np.mean([(e - s) / fps for s, e in sorted_segments])
    coherence_score = 1.0 / (1.0 + avg_gap_duration / avg_segment_duration) if avg_segment_duration > 0 else 0.0
    
    return {
        'avg_gap_duration': avg_gap_duration,
        'total_gaps': total_gaps,
        'temporal_coherence_score': coherence_score,
        'min_gap': np.min(gaps) if gaps else 0.0,
        'max_gap': np.max(gaps) if gaps else 0.0
    }


def compute_all_metrics(similarities: List[Tuple[int, float, str]],
                        selected_segments: List[Tuple[int, int]],
                        motion_segments: List[Tuple[int, int]],
                        captions: List[Tuple[int, str]],
                        fps: float,
                        threshold: float) -> Dict:
    """
    Compute all evaluation metrics.
    
    Args:
        similarities: List of similarity scores
        selected_segments: Selected segments for montage
        motion_segments: All detected motion segments
        captions: All generated captions
        fps: Video frames per second
        threshold: Similarity threshold used
        
    Returns:
        Dictionary with all computed metrics
    """
    selected_frames = set()
    for start, end in selected_segments:
        for frame_idx, _, _ in similarities:
            if start <= frame_idx <= end:
                selected_frames.add(frame_idx)
    
    precision_recall = compute_precision_recall(
        similarities, selected_segments, motion_segments, threshold
    )
    coverage = compute_coverage_metrics(selected_segments, motion_segments, fps)
    diversity = compute_diversity_metrics(captions, list(selected_frames))
    coherence = compute_temporal_coherence(selected_segments, fps)
    
    return {
        **precision_recall,
        **coverage,
        **diversity,
        **coherence
    }


class MetricsEvaluator:
    """Class for evaluating video montage quality."""
    
    def __init__(self, fps: float):
        self.fps = fps
        self.metrics = None
    
    def evaluate(self, similarities: List[Tuple[int, float, str]],
                 selected_segments: List[Tuple[int, int]],
                 motion_segments: List[Tuple[int, int]],
                 captions: List[Tuple[int, str]],
                 threshold: float) -> Dict:
        """Compute all evaluation metrics."""
        self.metrics = compute_all_metrics(
            similarities, selected_segments, motion_segments,
            captions, self.fps, threshold
        )
        return self.metrics
    
    def print_summary(self):
        """Print metrics summary."""
        if self.metrics is None:
            print("No metrics available. Run evaluate() first.")
            return
        
        print("=" * 60)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 60)
        
        print(f"\nClassification Metrics:")
        print(f"   Precision: {self.metrics['precision']:.3f}")
        print(f"   Recall: {self.metrics['recall']:.3f}")
        print(f"   F1 Score: {self.metrics['f1_score']:.3f}")
        
        print(f"\nCoverage Metrics:")
        print(f"   Coverage Ratio: {self.metrics['coverage_ratio']:.3f}")
        print(f"   Temporal Coverage: {self.metrics['temporal_coverage']:.3f}")
        print(f"   Segment Coverage: {self.metrics['segment_coverage_ratio']:.3f}")
        
        print(f"\nDiversity Metrics:")
        print(f"   Vocabulary Size: {self.metrics['vocabulary_size']}")
        print(f"   Unique Words Ratio: {self.metrics['unique_words_ratio']:.3f}")
        print(f"   Caption Diversity: {self.metrics['caption_diversity']:.3f}")
        
        print(f"\nTemporal Coherence:")
        print(f"   Avg Gap Duration: {self.metrics['avg_gap_duration']:.2f}s")
        print(f"   Coherence Score: {self.metrics['temporal_coherence_score']:.3f}")
        
        print("\n" + "=" * 60)

