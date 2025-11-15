"""
Pipeline Analysis and Experiments

This module provides detailed quantitative analysis and threshold
sensitivity experiments for the video montage pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


def analyze_pipeline(motion_segments: List[Tuple[int, int]],
                    frames: List[Tuple[int, any]],
                    captions: List[Tuple[int, str]],
                    similarities: List[Tuple[int, float, str]],
                    selected_segments: List[Tuple[int, int]],
                    fps: float) -> Dict:
    """
    Analyze the complete pipeline performance.
    
    Args:
        motion_segments: List of detected motion segments
        frames: List of extracted frames
        captions: List of generated captions
        similarities: List of similarity scores
        selected_segments: List of selected segments for montage
        fps: Video frames per second
        
    Returns:
        Dictionary with analysis results
    """
    scores = [score for _, score, _ in similarities]
    caption_lengths = [len(caption.split()) for _, _, caption in similarities]
    
    if selected_segments:
        segment_lengths = [(e - s) / fps for s, e in selected_segments]
        total_duration = sum(segment_lengths)
    else:
        segment_lengths = []
        total_duration = 0.0
    
    if motion_segments:
        total_motion_duration = (motion_segments[-1][1] - motion_segments[0][0]) / fps
        compression_ratio = total_duration / total_motion_duration if total_motion_duration > 0 else 0
    else:
        compression_ratio = 0
    
    analysis = {
        'motion_segments_count': len(motion_segments),
        'frames_extracted': len(frames),
        'captions_generated': len(captions),
        'similarity_stats': {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'above_threshold': sum(1 for s in scores if s > 0.25)
        },
        'caption_stats': {
            'avg_length': np.mean(caption_lengths),
            'median_length': np.median(caption_lengths),
            'min_length': np.min(caption_lengths),
            'max_length': np.max(caption_lengths)
        },
        'selection_stats': {
            'selected_count': len(selected_segments),
            'total_duration': total_duration,
            'avg_segment_duration': np.mean(segment_lengths) if segment_lengths else 0,
            'min_segment_duration': np.min(segment_lengths) if segment_lengths else 0,
            'max_segment_duration': np.max(segment_lengths) if segment_lengths else 0,
            'compression_ratio': compression_ratio
        }
    }
    
    return analysis


def threshold_sensitivity_analysis(similarities: List[Tuple[int, float, str]],
                                   motion_segments: List[Tuple[int, int]],
                                   thresholds: List[float] = None) -> Dict:
    """
    Analyze how different thresholds affect segment selection.
    
    Args:
        similarities: List of (frame_index, score, caption) tuples
        motion_segments: List of motion segments
        thresholds: List of threshold values to test
        
    Returns:
        Dictionary with threshold analysis results
    """
    if thresholds is None:
        thresholds = list(np.arange(0.15, 0.45, 0.05))
    
    frame_to_segment = {}
    for seg_idx, (start, end) in enumerate(motion_segments):
        for frame_idx, _, _ in similarities:
            if start <= frame_idx <= end:
                frame_to_segment[frame_idx] = (start, end)
    
    num_selected = []
    avg_scores = []
    
    for thresh in thresholds:
        selected_set = set()
        selected_scores = []
        
        for frame_idx, score, _ in similarities:
            if score >= thresh and frame_idx in frame_to_segment:
                segment = frame_to_segment[frame_idx]
                if segment not in selected_set:
                    selected_set.add(segment)
                    selected_scores.append(score)
        
        num_selected.append(len(selected_set))
        avg_scores.append(np.mean(selected_scores) if selected_scores else 0)
    
    return {
        'thresholds': thresholds,
        'num_selected': num_selected,
        'avg_scores': avg_scores
    }


def plot_analysis_results(similarities: List[Tuple[int, float, str]],
                          selected_segments: List[Tuple[int, int]], fps: float):
    """Create visualization plots for pipeline analysis."""
    scores = [score for _, score, _ in similarities]
    
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    plt.hist(scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(0.25, color='red', linestyle='--', linewidth=2, label='Threshold (0.25)')
    plt.xlabel('CLIP Similarity Score', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Distribution of Similarity Scores', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.subplot(2, 2, 2)
    sorted_scores = sorted(scores, reverse=True)
    plt.plot(sorted_scores, marker='o', markersize=3, linewidth=1.5, alpha=0.7)
    plt.axhline(0.25, color='red', linestyle='--', linewidth=2, label='Threshold (0.25)')
    plt.xlabel('Segment Rank', fontsize=11)
    plt.ylabel('CLIP Similarity Score', fontsize=11)
    plt.title('Similarity Scores (Sorted)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if selected_segments:
        plt.subplot(2, 2, 3)
        segment_starts = [s / fps for s, _ in selected_segments]
        segment_lengths = [(e - s) / fps for s, e in selected_segments]
        
        plt.barh(range(len(selected_segments)), segment_starts, 
                height=0.6, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Time (seconds)', fontsize=11)
        plt.ylabel('Segment Index', fontsize=11)
        plt.title('Temporal Distribution of Selected Segments', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.subplot(2, 2, 4)
        plt.bar(range(len(selected_segments)), segment_lengths, 
               color='coral', alpha=0.7, edgecolor='black')
        plt.xlabel('Segment Index', fontsize=11)
        plt.ylabel('Duration (seconds)', fontsize=11)
        plt.title('Duration of Selected Segments', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


class PipelineAnalyzer:
    """Class for analyzing pipeline performance."""
    
    def __init__(self, fps: float):
        self.fps = fps
        self.analysis_results = None
    
    def analyze(self, motion_segments: List[Tuple[int, int]],
                frames: List[Tuple[int, any]],
                captions: List[Tuple[int, str]],
                similarities: List[Tuple[int, float, str]],
                selected_segments: List[Tuple[int, int]]) -> Dict:
        """Perform complete pipeline analysis."""
        self.analysis_results = analyze_pipeline(
            motion_segments, frames, captions, similarities,
            selected_segments, self.fps
        )
        return self.analysis_results
    
    def print_summary(self):
        """Print analysis summary."""
        if self.analysis_results is None:
            print("No analysis results available. Run analyze() first.")
            return
        
        print("=" * 60)
        print("PIPELINE PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print(f"\n1. Motion Detection:")
        print(f"   - Total motion segments detected: {self.analysis_results['motion_segments_count']}")
        print(f"   - Representative frames extracted: {self.analysis_results['frames_extracted']}")
        
        print(f"\n2. Caption Generation:")
        print(f"   - Captions generated: {self.analysis_results['captions_generated']}")
        print(f"   - Average caption length: {self.analysis_results['caption_stats']['avg_length']:.1f} words")
        
        print(f"\n3. Semantic Filtering & Matching:")
        stats = self.analysis_results['similarity_stats']
        print(f"   - Total similarity scores computed: {len(self.analysis_results['similarity_stats'])}")
        print(f"   - Average similarity score: {stats['mean']:.3f}")
        print(f"   - Segments passing threshold: {stats['above_threshold']}")
        
        sel_stats = self.analysis_results['selection_stats']
        if sel_stats['selected_count'] > 0:
            print(f"\n4. Final Montage:")
            print(f"   - Selected segments: {sel_stats['selected_count']}")
            print(f"   - Total duration: {sel_stats['total_duration']:.2f} seconds")
            print(f"   - Compression ratio: {sel_stats['compression_ratio']:.2%}")
        
        print("\n" + "=" * 60)

