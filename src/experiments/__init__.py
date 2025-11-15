"""
Experiments Module

This module provides functionality for running comparative experiments
and logging results for research purposes.
"""

from .experiment_runner import ExperimentRunner, run_ablation_study, run_baseline_comparison
from .visualization import plot_threshold_sensitivity, plot_ablation_results, plot_metric_comparison, create_results_table

__all__ = ['ExperimentRunner', 'run_ablation_study', 'run_baseline_comparison',
           'plot_threshold_sensitivity', 'plot_ablation_results', 
           'plot_metric_comparison', 'create_results_table']

