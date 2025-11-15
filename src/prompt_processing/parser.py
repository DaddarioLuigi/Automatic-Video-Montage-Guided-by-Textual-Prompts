"""
Prompt Parsing and Semantic Filtering

This module parses user-provided natural language prompts and
filters captions based on semantic rules.
"""

import re
from typing import List, Tuple, Dict, Set
from collections import Counter


def parse_prompts(prompts: List[str]) -> Dict:
    """
    Parse user prompts to extract semantic information.
    
    Args:
        prompts: List of textual prompts
        
    Returns:
        Dictionary with parsed prompt information including:
        - keywords: Set of important keywords
        - verbs: Set of action verbs
        - nouns: Set of object nouns
        - original_prompts: Original prompt list
    """
    parsed = {
        'keywords': set(),
        'verbs': set(),
        'nouns': set(),
        'original_prompts': prompts
    }
    
    action_verbs = {
        'adding', 'placing', 'putting', 'removing', 'closing', 'opening',
        'cutting', 'cooking', 'preparing', 'serving', 'plating', 'mixing',
        'pouring', 'holding', 'moving', 'cleaning', 'standing', 'sitting'
    }
    
    for prompt in prompts:
        words = re.findall(r'\b\w+\b', prompt.lower())
        
        for word in words:
            if word in action_verbs:
                parsed['verbs'].add(word)
            else:
                if len(word) > 3:
                    parsed['nouns'].add(word)
        
        parsed['keywords'].update(words)
    
    return parsed


def filter_captions(captions: List[Tuple[int, str]], 
                   parsed_prompts: Dict,
                   min_keyword_matches: int = 1) -> List[Tuple[int, str]]:
    """
    Filter captions based on semantic rules derived from prompts.
    
    Args:
        captions: List of (frame_index, caption) tuples
        parsed_prompts: Dictionary from parse_prompts()
        min_keyword_matches: Minimum number of keywords that must match
        
    Returns:
        Filtered list of (frame_index, caption) tuples
    """
    keywords = parsed_prompts['keywords']
    filtered = []
    
    for idx, caption in captions:
        caption_lower = caption.lower()
        caption_words = set(re.findall(r'\b\w+\b', caption_lower))
        matches = len(caption_words & keywords)
        
        if matches >= min_keyword_matches:
            filtered.append((idx, caption))
    
    return filtered


class PromptParser:
    """Class for parsing and processing user prompts."""
    
    def __init__(self):
        self.parsed_prompts = None
        self.prompts = []
    
    def parse(self, prompts: List[str]) -> Dict:
        """Parse user prompts."""
        self.prompts = prompts
        self.parsed_prompts = parse_prompts(prompts)
        return self.parsed_prompts
    
    def filter_captions(self, captions: List[Tuple[int, str]], 
                       min_keyword_matches: int = 1) -> List[Tuple[int, str]]:
        """Filter captions based on parsed prompts."""
        if self.parsed_prompts is None:
            raise ValueError("Must parse prompts first using parse() method")
        
        return filter_captions(captions, self.parsed_prompts, min_keyword_matches)
    
    def get_keywords(self) -> Set[str]:
        """Get extracted keywords from prompts."""
        if self.parsed_prompts is None:
            return set()
        return self.parsed_prompts['keywords']
    
    def get_verbs(self) -> Set[str]:
        """Get extracted action verbs from prompts."""
        if self.parsed_prompts is None:
            return set()
        return self.parsed_prompts['verbs']

