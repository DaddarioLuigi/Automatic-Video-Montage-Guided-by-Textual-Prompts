#!/usr/bin/env python3
"""
Main script for running all experiments for research paper.

This script runs:
1. Baseline comparison
2. Ablation study
3. Threshold sensitivity analysis
4. Generates all visualizations and tables
"""

import argparse
import sys
from pathlib import Path

from src.experiments import (
    ExperimentRunner, 
    run_baseline_comparison, 
    run_ablation_study,
    plot_threshold_sensitivity,
    plot_ablation_results,
    plot_metric_comparison,
    create_results_table
)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for video montage research"
    )
    parser.add_argument(
        '--video_path',
        type=str,
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        nargs='+',
        required=True,
        help='Textual prompts describing desired scenes'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/experiments',
        help='Directory to save experiment results (default: results/experiments)'
    )
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        choices=['all', 'baseline', 'ablation', 'threshold'],
        default=['all'],
        help='Which experiments to run (default: all)'
    )
    parser.add_argument(
        '--generate_plots',
        action='store_true',
        default=True,
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--generate_latex',
        action='store_true',
        default=True,
        help='Generate LaTeX tables'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("VIDEO MONTAGE RESEARCH EXPERIMENTS")
    print("=" * 60)
    print(f"\nVideo: {args.video_path}")
    print(f"Prompts: {args.prompts}")
    print(f"Output directory: {output_dir}\n")
    
    runner = ExperimentRunner(args.video_path, str(output_dir))
    all_results = {}
    
    # Run baseline comparison
    if 'all' in args.experiments or 'baseline' in args.experiments:
        print("\n" + "=" * 60)
        print("RUNNING BASELINE COMPARISON")
        print("=" * 60)
        baseline_results = runner.run_baseline_comparison(args.prompts)
        all_results['baseline_comparison'] = baseline_results
        
        if args.generate_plots:
            plot_path = figures_dir / 'baseline_comparison.png'
            runner.plot_comparison_results(
                {'baseline_comparison': baseline_results},
                str(plot_path)
            )
        
        if args.generate_latex:
            latex_path = output_dir / 'baseline_comparison_table.tex'
            create_results_table(
                {'baseline_comparison': baseline_results},
                str(latex_path)
            )
    
    # Run ablation study
    if 'all' in args.experiments or 'ablation' in args.experiments:
        print("\n" + "=" * 60)
        print("RUNNING ABLATION STUDY")
        print("=" * 60)
        ablation_results = runner.run_ablation_study(args.prompts)
        all_results['ablation_study'] = ablation_results
        
        if args.generate_plots:
            plot_path = figures_dir / 'ablation_results.png'
            plot_ablation_results(ablation_results, str(plot_path))
    
    # Run threshold sensitivity
    if 'all' in args.experiments or 'threshold' in args.experiments:
        print("\n" + "=" * 60)
        print("RUNNING THRESHOLD SENSITIVITY ANALYSIS")
        print("=" * 60)
        threshold_results = runner.run_threshold_experiment(args.prompts)
        all_results['threshold_experiment'] = threshold_results
        
        if args.generate_plots:
            plot_path = figures_dir / 'threshold_sensitivity.png'
            plot_threshold_sensitivity(threshold_results, str(plot_path))
    
    # Generate comprehensive comparison
    if args.generate_plots and 'baseline_comparison' in all_results:
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE METRICS COMPARISON")
        print("=" * 60)
        plot_path = figures_dir / 'metrics_comparison_heatmap.png'
        plot_metric_comparison(all_results, str(plot_path))
    
    # Save all results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    runner.results = all_results
    runner.save_results('all_experiments.json')
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Figures saved to: {figures_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review results in: {output_dir}/all_experiments.json")
    print(f"  2. Use generated plots for your paper")
    print(f"  3. Use LaTeX table for paper: {output_dir}/baseline_comparison_table.tex")


if __name__ == '__main__':
    main()

