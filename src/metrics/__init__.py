"""
Metrics Module

This module provides evaluation metrics for video montage quality
and pipeline performance.
"""

from .evaluator import MetricsEvaluator, compute_all_metrics

__all__ = ['MetricsEvaluator', 'compute_all_metrics']



