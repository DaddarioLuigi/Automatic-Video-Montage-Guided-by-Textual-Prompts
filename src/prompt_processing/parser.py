"""
Prompt Parsing and Semantic Filtering

This module parses user-provided natural language prompts and
filters captions based on semantic rules.

Now includes advanced NLP capabilities:
- spaCy-based linguistic parsing
- Semantic filtering with embeddings
- Text generation and summarization
"""

import re
from typing import List, Tuple, Dict, Set, Optional

# Import advanced modules (optional, with fallback)
try:
    from .advanced_parsing import AdvancedLinguisticParser
    HAS_ADVANCED_PARSING = True
except ImportError:
    HAS_ADVANCED_PARSING = False
    AdvancedLinguisticParser = None

try:
    from .semantic_filtering import SemanticFilter
    HAS_SEMANTIC_FILTERING = True
except ImportError:
    HAS_SEMANTIC_FILTERING = False
    SemanticFilter = None

try:
    from .text_generation import TextGenerator
    HAS_TEXT_GENERATION = True
except ImportError:
    HAS_TEXT_GENERATION = False
    TextGenerator = None


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
    """
    Class for parsing and processing user prompts.
    
    Supports both basic (regex-based) and advanced (spaCy-based) parsing.
    """
    
    def __init__(self, 
                 use_advanced_parsing: bool = True,
                 use_semantic_filtering: bool = True,
                 spacy_model: str = "en_core_web_sm",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize prompt parser.
        
        Args:
            use_advanced_parsing: Use spaCy for advanced parsing (default: True)
            use_semantic_filtering: Use embeddings for semantic filtering (default: True)
            spacy_model: spaCy model name
            embedding_model: Sentence transformer model name
        """
        self.parsed_prompts = None
        self.prompts = []
        self.use_advanced_parsing = use_advanced_parsing and HAS_ADVANCED_PARSING
        self.use_semantic_filtering = use_semantic_filtering and HAS_SEMANTIC_FILTERING
        
        # Initialize advanced components if available
        self.advanced_parser = None
        if self.use_advanced_parsing and AdvancedLinguisticParser:
            try:
                self.advanced_parser = AdvancedLinguisticParser(model_name=spacy_model)
            except Exception as e:
                print(f"Warning: Could not initialize advanced parser: {e}")
                print("Falling back to basic parsing")
                self.use_advanced_parsing = False
        
        self.semantic_filter = None
        if self.use_semantic_filtering and SemanticFilter:
            try:
                self.semantic_filter = SemanticFilter(model_name=embedding_model)
            except Exception as e:
                print(f"Warning: Could not initialize semantic filter: {e}")
                print("Falling back to keyword-based filtering")
                self.use_semantic_filtering = False
    
    def parse(self, prompts: List[str], use_advanced: Optional[bool] = None) -> Dict:
        """
        Parse user prompts.
        
        Args:
            prompts: List of textual prompts
            use_advanced: Override default advanced parsing setting
            
        Returns:
            Dictionary with parsed prompt information
        """
        self.prompts = prompts
        
        use_advanced = use_advanced if use_advanced is not None else self.use_advanced_parsing
        
        if use_advanced and self.advanced_parser:
            # Use advanced spaCy-based parsing
            self.parsed_prompts = self.advanced_parser.parse_prompts(prompts)
            
            # Also encode prompts for semantic filtering
            if self.semantic_filter:
                self.semantic_filter.encode_prompts(prompts)
            
            # Convert to format compatible with old interface
            aggregated = self.parsed_prompts.get('aggregated', {})
            self.parsed_prompts['keywords'] = set(aggregated.get('keywords', []))
            self.parsed_prompts['verbs'] = set(aggregated.get('verbs', []))
            self.parsed_prompts['nouns'] = set(aggregated.get('nouns', []))
        else:
            # Use basic regex-based parsing (fallback)
            self.parsed_prompts = parse_prompts(prompts)
        
        return self.parsed_prompts
    
    def filter_captions(self, 
                       captions: List[Tuple[int, str]], 
                       min_keyword_matches: int = 1,
                       semantic_threshold: float = 0.3,
                       use_semantic: Optional[bool] = None) -> List[Tuple[int, str]]:
        """
        Filter captions based on parsed prompts.
        
        Args:
            captions: List of (frame_index, caption) tuples
            min_keyword_matches: Minimum keyword matches (for basic filtering)
            semantic_threshold: Semantic similarity threshold (for advanced filtering)
            use_semantic: Override default semantic filtering setting
            
        Returns:
            Filtered list of (frame_index, caption) tuples
        """
        if self.parsed_prompts is None:
            raise ValueError("Must parse prompts first using parse() method")
        
        use_semantic = use_semantic if use_semantic is not None else self.use_semantic_filtering
        
        if use_semantic and self.semantic_filter and self.semantic_filter.prompt_embeddings is not None:
            # Use semantic filtering with embeddings
            semantic_matches = self.semantic_filter.filter_captions_semantic(
                captions,
                similarity_threshold=semantic_threshold
            )
            # Return in same format as basic filtering
            return [(idx, cap) for idx, cap, _ in semantic_matches]
        else:
            # Use basic keyword-based filtering (fallback)
            return filter_captions(captions, self.parsed_prompts, min_keyword_matches)
    
    def get_keywords(self) -> Set[str]:
        """Get extracted keywords from prompts."""
        if self.parsed_prompts is None:
            return set()
        if isinstance(self.parsed_prompts.get('keywords'), set):
            return self.parsed_prompts['keywords']
        return set(self.parsed_prompts.get('keywords', []))
    
    def get_verbs(self) -> Set[str]:
        """Get extracted action verbs from prompts."""
        if self.parsed_prompts is None:
            return set()
        if isinstance(self.parsed_prompts.get('verbs'), set):
            return self.parsed_prompts['verbs']
        return set(self.parsed_prompts.get('verbs', []))
    
    def get_entities(self) -> List[Dict]:
        """Get named entities from prompts (advanced parsing only)."""
        if not self.use_advanced_parsing or not self.advanced_parser:
            return []
        
        if self.parsed_prompts is None:
            return []
        
        # Extract entities from all parsed prompts
        entities = []
        parsed_list = self.parsed_prompts.get('parsed_prompts', [])
        for parsed in parsed_list:
            entities.extend(parsed.get('entities', []))
        
        return entities
    
    def get_semantic_roles(self) -> List[Dict]:
        """Get semantic roles (subject-verb-object) from prompts (advanced parsing only)."""
        if not self.use_advanced_parsing or not self.advanced_parser:
            return []
        
        if self.parsed_prompts is None:
            return []
        
        return self.parsed_prompts.get('aggregated', {}).get('semantic_roles', [])

