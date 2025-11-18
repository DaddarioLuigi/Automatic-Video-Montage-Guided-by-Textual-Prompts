"""
Baseline Methods Module

This module provides baseline methods for comparison with the
proposed video montage approach.
"""

from .baselines import BaselineMethods, random_selection, uniform_selection, first_n_selection

__all__ = ['BaselineMethods', 'random_selection', 'uniform_selection', 'first_n_selection']



