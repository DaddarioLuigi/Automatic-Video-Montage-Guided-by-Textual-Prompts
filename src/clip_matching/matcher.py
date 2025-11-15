"""
CLIP-based Semantic Matching

This module computes semantic similarity between user prompts and
generated captions using CLIP (Contrastive Language-Image Pre-training).
"""

import torch
import clip
from typing import List, Tuple, Optional


def compute_similarities(prompts: List[str],
                         captions: List[Tuple[int, str]],
                         model_name: str = "ViT-B/32",
                         device: Optional[str] = None) -> List[Tuple[int, float, str]]:
    """
    Compute CLIP similarity scores between prompts and captions.
    
    Args:
        prompts: List of textual prompts
        captions: List of (frame_index, caption) tuples
        model_name: CLIP model name (default: "ViT-B/32")
        device: Device to run inference on
        
    Returns:
        List of (frame_index, max_similarity_score, caption) tuples
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model, _ = clip.load(model_name, device=device)
    
    prompt_embeddings = [
        clip_model.encode_text(clip.tokenize(p).to(device)) 
        for p in prompts
    ]
    
    similarities = []
    for idx, caption in captions:
        text_embed = clip_model.encode_text(clip.tokenize(caption).to(device))
        sim_scores = [
            torch.cosine_similarity(text_embed, p, dim=-1).item() 
            for p in prompt_embeddings
        ]
        max_sim = max(sim_scores)
        similarities.append((idx, max_sim, caption))
    
    return similarities


def select_segments(similarities: List[Tuple[int, float, str]],
                   motion_segments: List[Tuple[int, int]],
                   threshold: float = 0.25) -> List[Tuple[int, int]]:
    """
    Select segments based on similarity scores above threshold.
    
    Args:
        similarities: List of (frame_index, similarity_score, caption) tuples
        motion_segments: List of (start_frame, end_frame) tuples
        threshold: Minimum similarity score to include segment
        
    Returns:
        List of (start_frame, end_frame) tuples for selected segments
    """
    frame_to_segment = {}
    for seg_idx, (start, end) in enumerate(motion_segments):
        for frame_idx, _, _ in similarities:
            if start <= frame_idx <= end:
                frame_to_segment[frame_idx] = (start, end)
    
    selected_segments = []
    selected_segment_set = set()
    
    for frame_idx, score, _ in similarities:
        if score > threshold and frame_idx in frame_to_segment:
            segment = frame_to_segment[frame_idx]
            if segment not in selected_segment_set:
                selected_segments.append(segment)
                selected_segment_set.add(segment)
    
    return selected_segments


class CLIPMatcher:
    """Class-based wrapper for CLIP matching functionality."""
    
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.clip_model, self.clip_preprocess = clip.load(model_name, device=device)
        self.prompt_embeddings = None
    
    def encode_prompts(self, prompts: List[str]):
        """Pre-encode prompts for faster similarity computation."""
        self.prompt_embeddings = [
            self.clip_model.encode_text(clip.tokenize(p).to(self.device))
            for p in prompts
        ]
    
    def compute_similarities(self, captions: List[Tuple[int, str]]) -> List[Tuple[int, float, str]]:
        """Compute similarities between encoded prompts and captions."""
        if self.prompt_embeddings is None:
            raise ValueError("Must encode prompts first using encode_prompts()")
        
        similarities = []
        for idx, caption in captions:
            text_embed = self.clip_model.encode_text(clip.tokenize(caption).to(self.device))
            sim_scores = [
                torch.cosine_similarity(text_embed, p, dim=-1).item()
                for p in self.prompt_embeddings
            ]
            max_sim = max(sim_scores)
            similarities.append((idx, max_sim, caption))
        
        return similarities
    
    def select_segments(self, similarities: List[Tuple[int, float, str]],
                       motion_segments: List[Tuple[int, int]],
                       threshold: float = 0.25) -> List[Tuple[int, int]]:
        """Select segments based on similarity scores."""
        return select_segments(similarities, motion_segments, threshold)

