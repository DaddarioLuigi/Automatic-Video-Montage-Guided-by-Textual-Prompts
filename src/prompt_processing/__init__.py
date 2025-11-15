"""
Prompt Processing Module

This module provides functionality for parsing user prompts and
semantic filtering of captions based on rules.

Now includes:
- Advanced linguistic parsing with spaCy
- Semantic filtering with embeddings
- Text generation and summarization
"""

from .parser import PromptParser, parse_prompts, filter_captions

# Import advanced modules (optional)
try:
    from .advanced_parsing import AdvancedLinguisticParser
    from .semantic_filtering import SemanticFilter, create_semantic_filter
    from .text_generation import TextGenerator, create_text_generator
    __all__ = [
        'PromptParser', 
        'parse_prompts', 
        'filter_captions',
        'AdvancedLinguisticParser',
        'SemanticFilter',
        'create_semantic_filter',
        'TextGenerator',
        'create_text_generator'
    ]
except ImportError:
    __all__ = ['PromptParser', 'parse_prompts', 'filter_captions']


