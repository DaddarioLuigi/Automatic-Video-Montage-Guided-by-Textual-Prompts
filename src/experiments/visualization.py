"""
Visualization Utilities for Research Paper

This module provides publication-quality visualizations for research results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def plot_threshold_sensitivity(threshold_results: Dict, save_path: Optional[str] = None):
    """
    Plot threshold sensitivity analysis.
    
    Args:
        threshold_results: Dictionary with threshold experiment results
        save_path: Path to save figure
    """
    thresholds = sorted(threshold_results.keys())
    n_segments = [threshold_results[t]['n_segments'] for t in thresholds]
    f1_scores = [threshold_results[t]['metrics']['f1_score'] for t in thresholds]
    precisions = [threshold_results[t]['metrics']['precision'] for t in thresholds]
    recalls = [threshold_results[t]['metrics']['recall'] for t in thresholds]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(thresholds, n_segments, marker='o', linewidth=2, markersize=8, label='Selected segments')
    ax1.set_xlabel('Similarity Threshold', fontsize=12)
    ax1.set_ylabel('Number of Selected Segments', fontsize=12)
    ax1.set_title('Threshold Sensitivity: Segment Count', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(thresholds, precisions, marker='s', linewidth=2, markersize=6, label='Precision')
    ax2.plot(thresholds, recalls, marker='^', linewidth=2, markersize=6, label='Recall')
    ax2.plot(thresholds, f1_scores, marker='o', linewidth=2, markersize=6, label='F1 Score')
    ax2.set_xlabel('Similarity Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Threshold Sensitivity: Classification Metrics', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Threshold sensitivity plot saved to: {save_path}")
    
    plt.show()


def plot_ablation_results(ablation_results: Dict, save_path: Optional[str] = None):
    """
    Plot ablation study results.
    
    Args:
        ablation_results: Dictionary with ablation study results
        save_path: Path to save figure
    """
    configs = []
    f1_scores = []
    precisions = []
    recalls = []
    coverages = []
    
    if 'full_pipeline' in ablation_results:
        configs.append('Full Pipeline')
        metrics = ablation_results['full_pipeline']['metrics']
        f1_scores.append(metrics['f1_score'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        coverages.append(metrics['coverage_ratio'])
    
    if 'no_semantic_filtering' in ablation_results:
        configs.append('No Semantic\nFiltering')
        metrics = ablation_results['no_semantic_filtering']['metrics']
        f1_scores.append(metrics['f1_score'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        coverages.append(metrics['coverage_ratio'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_data = [
        ('F1 Score', f1_scores),
        ('Precision', precisions),
        ('Recall', recalls),
        ('Coverage Ratio', coverages)
    ]
    
    for idx, (title, data) in enumerate(metrics_data):
        ax = axes[idx // 2, idx % 2]
        ax.bar(configs, data, alpha=0.7, edgecolor='black', color=['steelblue', 'coral'])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Ablation results plot saved to: {save_path}")
    
    plt.show()


def plot_metric_comparison(all_results: Dict, save_path: Optional[str] = None):
    """
    Create comprehensive comparison plot of all metrics.
    
    Args:
        all_results: Dictionary with all experiment results
        save_path: Path to save figure
    """
    if 'baseline_comparison' not in all_results:
        print("No baseline comparison data available.")
        return
    
    comparison = all_results['baseline_comparison']
    methods = list(comparison.keys())
    metrics = ['precision', 'recall', 'f1_score', 'coverage_ratio', 
                'temporal_coverage', 'caption_diversity', 'temporal_coherence_score']
    data = []
    for method in methods:
        method_metrics = []
        for metric in metrics:
            value = comparison[method]['metrics'].get(metric, 0)
            method_metrics.append(value)
        data.append(method_metrics)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(data, 
                xticklabels=[m.replace('_', ' ').title() for m in metrics],
                yticklabels=methods,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Score'},
                ax=ax)
    
    ax.set_title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metric comparison plot saved to: {save_path}")
    
    plt.show()


def create_results_table(comparison_results: Dict, save_path: Optional[str] = None) -> str:
    """
    Create LaTeX table from results for research paper.
    
    Args:
        comparison_results: Dictionary with comparison results
        save_path: Path to save LaTeX table
        
    Returns:
        LaTeX table string
    """
    if 'baseline_comparison' not in comparison_results:
        return ""
    
    results = comparison_results['baseline_comparison']
    methods = list(results.keys())
    
    metrics = [
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('f1_score', 'F1 Score'),
        ('coverage_ratio', 'Coverage'),
        ('caption_diversity', 'Diversity'),
        ('temporal_coherence_score', 'Coherence')
    ]
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Comparison of proposed method with baselines}\n"
    latex += "\\label{tab:baseline_comparison}\n"
    latex += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
    latex += "\\toprule\n"
    latex += "Method & " + " & ".join([label for _, label in metrics]) + " \\\\\n"
    latex += "\\midrule\n"
    
    for method in methods:
        method_name = method.replace('_', ' ').title()
        if method == 'proposed':
            method_name = "\\textbf{Proposed}"
        
        values = []
        for metric, _ in metrics:
            value = results[method]['metrics'].get(metric, 0)
            values.append(f"{value:.3f}")
        
        latex += f"{method_name} & " + " & ".join(values) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to: {save_path}")
    
    return latex

