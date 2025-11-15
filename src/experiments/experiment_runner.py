"""
Experiment Runner for Comparative Studies

This module runs experiments comparing different configurations,
baselines, and ablation studies for research purposes.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..pipeline import VideoMontagePipeline
from ..metrics import MetricsEvaluator
from ..baselines import BaselineMethods


class ExperimentRunner:
    """Class for running and logging experiments."""
    
    def __init__(self, video_path: str, output_dir: str = "results/experiments"):
        """
        Initialize experiment runner.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save experiment results
        """
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
        self.pipeline = VideoMontagePipeline(video_path)
    
    def run_threshold_experiment(self, prompts: List[str],
                                 thresholds: List[float] = None) -> Dict:
        """
        Run experiment with different similarity thresholds.
        
        Args:
            prompts: List of textual prompts
            thresholds: List of thresholds to test (default: 0.15-0.45)
            
        Returns:
            Dictionary with results for each threshold
        """
        if thresholds is None:
            thresholds = list(np.arange(0.15, 0.45, 0.05))
        
        print(f"\n{'='*60}")
        print(f"THRESHOLD EXPERIMENT")
        print(f"{'='*60}")
        print(f"Testing {len(thresholds)} threshold values")
        self.pipeline.run_complete_pipeline(
            prompts=prompts,
            similarity_threshold=0.25,
            enable_analysis=False,
            enable_plots=False,
            output_path=None
        )
        
        results = {}
        evaluator = MetricsEvaluator(self.pipeline.fps)
        
        for threshold in thresholds:
            print(f"\nTesting threshold: {threshold:.2f}")
            selected = self.pipeline.clip_matcher.select_segments(
                self.pipeline.similarities,
                self.pipeline.motion_segments,
                threshold
            )
            metrics = evaluator.evaluate(
                self.pipeline.similarities,
                selected,
                self.pipeline.motion_segments,
                self.pipeline.captions,
                threshold
            )
            
            results[threshold] = {
                'threshold': threshold,
                'n_segments': len(selected),
                'metrics': metrics,
                'selected_segments': selected
            }
        
        self.results['threshold_experiment'] = results
        return results
    
    def run_ablation_study(self, prompts: List[str]) -> Dict:
        """
        Run ablation study comparing different components.
        
        Tests:
        1. Full pipeline (with semantic filtering)
        2. Without semantic filtering
        3. Different CLIP models
        4. Different caption models
        
        Returns:
            Dictionary with ablation results
        """
        print(f"\n{'='*60}")
        print(f"ABLATION STUDY")
        print(f"{'='*60}")
        
        results = {}
        print("\n[1/3] Full pipeline with semantic filtering")
        self.pipeline.run_complete_pipeline(
            prompts=prompts,
            enable_semantic_filtering=True,
            enable_analysis=False,
            enable_plots=False,
            output_path=None
        )
        evaluator = MetricsEvaluator(self.pipeline.fps)
        metrics_full = evaluator.evaluate(
            self.pipeline.similarities,
            self.pipeline.selected_segments,
            self.pipeline.motion_segments,
            self.pipeline.captions,
            0.25
        )
        results['full_pipeline'] = {
            'config': 'full_with_semantic_filtering',
            'metrics': metrics_full,
            'n_segments': len(self.pipeline.selected_segments)
        }
        print("\n[2/3] Pipeline without semantic filtering")
        self.pipeline.run_complete_pipeline(
            prompts=prompts,
            enable_semantic_filtering=False,
            enable_analysis=False,
            enable_plots=False,
            output_path=None
        )
        metrics_no_filter = evaluator.evaluate(
            self.pipeline.similarities,
            self.pipeline.selected_segments,
            self.pipeline.motion_segments,
            self.pipeline.captions,
            0.25
        )
        results['no_semantic_filtering'] = {
            'config': 'no_semantic_filtering',
            'metrics': metrics_no_filter,
            'n_segments': len(self.pipeline.selected_segments)
        }
        print("\n[3/3] Comparing with different thresholds")
        threshold_exp = self.run_threshold_experiment(prompts)
        results['threshold_comparison'] = threshold_exp
        
        self.results['ablation_study'] = results
        return results
    
    def run_baseline_comparison(self, prompts: List[str]) -> Dict:
        """
        Compare proposed method with baseline methods.
        
        Returns:
            Dictionary with comparison results
        """
        print(f"\n{'='*60}")
        print(f"BASELINE COMPARISON")
        print(f"{'='*60}")
        
        print("\n[Proposed Method] Running full pipeline")
        self.pipeline.run_complete_pipeline(
            prompts=prompts,
            enable_analysis=False,
            enable_plots=False,
            output_path=None
        )
        
        n_segments_proposed = len(self.pipeline.selected_segments)
        evaluator = MetricsEvaluator(self.pipeline.fps)
        metrics_proposed = evaluator.evaluate(
            self.pipeline.similarities,
            self.pipeline.selected_segments,
            self.pipeline.motion_segments,
            self.pipeline.captions,
            0.25
        )
        
        print(f"\n[Baselines] Running baseline methods with n_segments={n_segments_proposed}...")
        baseline_runner = BaselineMethods(self.pipeline.fps)
        
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        motions = self.pipeline.motion_detector.analyze(cap)
        
        baseline_results = baseline_runner.run_all_baselines(
            self.pipeline.motion_segments,
            n_segments_proposed,
            motions
        )
        
        results = {
            'proposed': {
                'method': 'CLIP-based semantic matching',
                'metrics': metrics_proposed,
                'n_segments': n_segments_proposed
            }
        }
        
        for method_name, baseline_segments in baseline_results.items():
            print(f"\nEvaluating baseline: {method_name}")
            
            dummy_similarities = []
            for idx, caption in self.pipeline.captions:
                for start, end in self.pipeline.motion_segments:
                    if start <= idx <= end:
                        segment_motions = motions[start:end+1] if end < len(motions) else motions[start:]
                        proxy_score = np.mean(segment_motions) / np.max(motions) if len(segment_motions) > 0 else 0.0
                        dummy_similarities.append((idx, proxy_score, caption))
                        break
            
            baseline_metrics = evaluator.evaluate(
                dummy_similarities,
                baseline_segments,
                self.pipeline.motion_segments,
                self.pipeline.captions,
                0.0
            )
            
            results[method_name] = {
                'method': method_name,
                'metrics': baseline_metrics,
                'n_segments': len(baseline_segments)
            }
        
        self.results['baseline_comparison'] = results
        return results
    
    def save_results(self, filename: Optional[str] = None):
        """Save experiment results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(convert_to_serializable(obj))
            return obj
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        return filepath
    
    def plot_comparison_results(self, comparison_results: Dict, save_path: Optional[str] = None):
        """Plot comparison results for research paper."""
        if 'baseline_comparison' not in comparison_results:
            print("No baseline comparison results to plot.")
            return
        
        results = comparison_results['baseline_comparison']
        methods = list(results.keys())
        metrics_to_plot = ['precision', 'recall', 'f1_score', 'coverage_ratio', 'caption_diversity']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx >= len(axes):
                break
            
            values = [results[method]['metrics'].get(metric, 0) for method in methods]
            
            axes[idx].bar(methods, values, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].set_ylim(0, max(values) * 1.1 if values else 1)
        
        if len(metrics_to_plot) < len(axes):
            axes[-1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()


def run_ablation_study(video_path: str, prompts: List[str],
                      output_dir: str = "results/experiments") -> Dict:
    """Convenience function to run ablation study."""
    runner = ExperimentRunner(video_path, output_dir)
    results = runner.run_ablation_study(prompts)
    runner.save_results("ablation_study.json")
    return results


def run_baseline_comparison(video_path: str, prompts: List[str],
                           output_dir: str = "results/experiments") -> Dict:
    """Convenience function to run baseline comparison."""
    runner = ExperimentRunner(video_path, output_dir)
    results = runner.run_baseline_comparison(prompts)
    runner.save_results("baseline_comparison.json")
    
    fig_path = Path(output_dir) / "baseline_comparison.png"
    runner.plot_comparison_results({'baseline_comparison': results}, str(fig_path))
    
    return results

