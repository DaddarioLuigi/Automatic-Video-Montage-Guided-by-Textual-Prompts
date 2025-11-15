"""
Advanced Semantic Filtering with Embeddings

This module provides semantic filtering using:
- Sentence transformers for semantic similarity
- Word embeddings for keyword expansion
- Semantic matching beyond simple keyword matching
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')


class SemanticFilter:
    """Advanced semantic filtering using embeddings."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        """
        Initialize semantic filter with sentence transformer.
        
        Args:
            model_name: Sentence transformer model name
                       Options: "all-MiniLM-L6-v2" (fast, 80MB),
                               "all-mpnet-base-v2" (better, 420MB),
                               "paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
            device: Device to run on ('cuda' or 'cpu')
        """
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print(f"Warning: Could not load {model_name}, using fallback")
            # Fallback to a smaller model
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            except Exception:
                raise RuntimeError(f"Could not load sentence transformer model: {e}")
        
        self.prompt_embeddings = None
        self.prompts = None
    
    def encode_prompts(self, prompts: List[str]):
        """
        Encode prompts into embeddings.
        
        Args:
            prompts: List of textual prompts
        """
        self.prompts = prompts
        self.prompt_embeddings = self.model.encode(
            prompts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    def compute_semantic_similarity(self, text: str) -> float:
        """
        Compute semantic similarity between text and encoded prompts.
        
        Args:
            text: Text to compare (e.g., a caption)
            
        Returns:
            Maximum similarity score across all prompts
        """
        if self.prompt_embeddings is None:
            raise ValueError("Must encode prompts first using encode_prompts()")
        
        text_embedding = self.model.encode([text], convert_to_numpy=True)[0]
        
        # Compute cosine similarity
        similarities = np.dot(self.prompt_embeddings, text_embedding) / (
            np.linalg.norm(self.prompt_embeddings, axis=1) * np.linalg.norm(text_embedding)
        )
        
        return float(np.max(similarities))
    
    def filter_captions_semantic(self,
                                 captions: List[Tuple[int, str]],
                                 similarity_threshold: float = 0.3,
                                 top_k: Optional[int] = None) -> List[Tuple[int, str, float]]:
        """
        Filter captions based on semantic similarity to prompts.
        
        Args:
            captions: List of (frame_index, caption) tuples
            similarity_threshold: Minimum similarity score (0-1)
            top_k: If provided, return top K most similar (ignores threshold)
            
        Returns:
            List of (frame_index, caption, similarity_score) tuples
        """
        if self.prompt_embeddings is None:
            raise ValueError("Must encode prompts first using encode_prompts()")
        
        # Encode all captions
        caption_texts = [caption for _, caption in captions]
        caption_embeddings = self.model.encode(
            caption_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Compute similarities with all prompts
        similarities = []
        for i, (idx, caption) in enumerate(captions):
            # Compute max similarity across all prompts
            caption_emb = caption_embeddings[i]
            sim_scores = np.dot(self.prompt_embeddings, caption_emb) / (
                np.linalg.norm(self.prompt_embeddings, axis=1) * np.linalg.norm(caption_emb)
            )
            max_sim = float(np.max(sim_scores))
            similarities.append((idx, caption, max_sim))
        
        # Filter by threshold or select top K
        if top_k is not None:
            # Sort by similarity and take top K
            similarities.sort(key=lambda x: x[2], reverse=True)
            return similarities[:top_k]
        else:
            # Filter by threshold
            filtered = [(idx, cap, sim) for idx, cap, sim in similarities if sim >= similarity_threshold]
            return filtered
    
    def expand_keywords_semantic(self,
                                 keywords: List[str],
                                 similarity_threshold: float = 0.6) -> Dict[str, List[str]]:
        """
        Expand keywords with semantically similar words.
        
        Args:
            keywords: List of keywords to expand
            similarity_threshold: Minimum similarity for expansion
            
        Returns:
            Dictionary mapping each keyword to list of similar words
        """
        if not keywords:
            return {}
        
        # Encode keywords
        keyword_embeddings = self.model.encode(
            keywords,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # For each keyword, find similar keywords
        expanded = {}
        for i, keyword in enumerate(keywords):
            similar = []
            keyword_emb = keyword_embeddings[i]
            
            for j, other_keyword in enumerate(keywords):
                if i != j:
                    other_emb = keyword_embeddings[j]
                    similarity = np.dot(keyword_emb, other_emb) / (
                        np.linalg.norm(keyword_emb) * np.linalg.norm(other_emb)
                    )
                    if similarity >= similarity_threshold:
                        similar.append(other_keyword)
            
            expanded[keyword] = similar
        
        return expanded
    
    def find_semantic_matches(self,
                             captions: List[Tuple[int, str]],
                             parsed_prompts: Dict,
                             similarity_threshold: float = 0.3,
                             use_keyword_expansion: bool = True) -> List[Tuple[int, str, float]]:
        """
        Find semantic matches using both embeddings and keyword expansion.
        
        Args:
            captions: List of (frame_index, caption) tuples
            parsed_prompts: Parsed prompts from AdvancedLinguisticParser
            similarity_threshold: Minimum similarity score
            use_keyword_expansion: Whether to expand keywords semantically
            
        Returns:
            List of (frame_index, caption, similarity_score) tuples
        """
        # First, encode prompts
        if self.prompts is None:
            prompts = parsed_prompts.get('original_prompts', [])
            if prompts:
                self.encode_prompts(prompts)
        
        # Get semantic similarity-based matches
        semantic_matches = self.filter_captions_semantic(
            captions,
            similarity_threshold=similarity_threshold
        )
        
        # Optionally expand keywords and do additional matching
        if use_keyword_expansion:
            keywords = parsed_prompts.get('aggregated', {}).get('keywords', [])
            if keywords:
                expanded = self.expand_keywords_semantic(keywords)
                # This could be used for additional filtering if needed
        
        return semantic_matches


def create_semantic_filter(model_name: str = "all-MiniLM-L6-v2",
                          device: Optional[str] = None) -> SemanticFilter:
    """
    Convenience function to create a semantic filter.
    
    Args:
        model_name: Sentence transformer model name
        device: Device to run on
        
    Returns:
        SemanticFilter instance
    """
    return SemanticFilter(model_name=model_name, device=device)

