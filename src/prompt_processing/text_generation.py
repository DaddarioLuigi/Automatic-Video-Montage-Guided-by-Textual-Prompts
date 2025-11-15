"""
Text Generation and Summarization

This module provides:
- Text summarization of captions
- Paraphrasing of prompts
- Alternative description generation
"""

from typing import List, Tuple, Dict, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import warnings

warnings.filterwarnings('ignore')


class TextGenerator:
    """Text generation and summarization utilities."""
    
    def __init__(self, 
                 summarization_model: str = "facebook/bart-large-cnn",
                 device: Optional[str] = None):
        """
        Initialize text generator.
        
        Args:
            summarization_model: Model for summarization
                                Options: "facebook/bart-large-cnn" (good quality),
                                        "google/pegasus-xsum" (abstractive),
                                        "t5-small" (fast)
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.summarization_model_name = summarization_model
        
        try:
            self.summarizer = pipeline(
                "summarization",
                model=summarization_model,
                device=0 if device == "cuda" else -1,
                tokenizer=summarization_model
            )
        except Exception as e:
            print(f"Warning: Could not load {summarization_model}, using fallback")
            try:
                # Fallback to a smaller model
                self.summarizer = pipeline(
                    "summarization",
                    model="t5-small",
                    device=0 if device == "cuda" else -1
                )
            except Exception:
                raise RuntimeError(f"Could not load summarization model: {e}")
    
    def summarize_captions(self,
                          captions: List[Tuple[int, str]],
                          max_length: int = 50,
                          min_length: int = 10) -> List[Tuple[int, str]]:
        """
        Summarize a list of captions.
        
        Args:
            captions: List of (frame_index, caption) tuples
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            List of (frame_index, summarized_caption) tuples
        """
        summarized = []
        
        for idx, caption in captions:
            try:
                # Summarize individual caption
                summary = self.summarizer(
                    caption,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summarized_text = summary[0]['summary_text']
                summarized.append((idx, summarized_text))
            except Exception as e:
                # If summarization fails, use original caption
                print(f"Warning: Could not summarize caption {idx}: {e}")
                summarized.append((idx, caption))
        
        return summarized
    
    def summarize_caption_collection(self,
                                    captions: List[Tuple[int, str]],
                                    max_length: int = 100,
                                    min_length: int = 30) -> str:
        """
        Create a single summary of all captions combined.
        
        Args:
            captions: List of (frame_index, caption) tuples
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Single summary text
        """
        # Combine all captions
        combined_text = " ".join([caption for _, caption in captions])
        
        if len(combined_text.split()) < min_length:
            # If too short, just return first few captions
            return " ".join([caption for _, caption in captions[:3]])
        
        try:
            summary = self.summarizer(
                combined_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Warning: Could not summarize caption collection: {e}")
            # Return truncated version
            words = combined_text.split()
            return " ".join(words[:max_length])
    
    def generate_alternative_descriptions(self,
                                        caption: str,
                                        num_alternatives: int = 3) -> List[str]:
        """
        Generate alternative descriptions of a caption.
        
        This uses paraphrasing by summarizing with different parameters.
        
        Args:
            caption: Original caption
            num_alternatives: Number of alternative descriptions to generate
            
        Returns:
            List of alternative descriptions
        """
        alternatives = []
        
        # Generate variations by using different summary lengths
        length_variations = [
            (len(caption.split()) // 2, len(caption.split())),
            (len(caption.split()) // 3, len(caption.split()) * 2 // 3),
            (len(caption.split()) // 4, len(caption.split()) // 2)
        ]
        
        for min_len, max_len in length_variations[:num_alternatives]:
            try:
                summary = self.summarizer(
                    caption,
                    max_length=max(max_len, 20),
                    min_length=min(min_len, 10),
                    do_sample=True,
                    temperature=0.7
                )
                alt_text = summary[0]['summary_text']
                if alt_text not in alternatives:
                    alternatives.append(alt_text)
            except Exception:
                continue
        
        # If we don't have enough alternatives, add the original
        while len(alternatives) < num_alternatives:
            alternatives.append(caption)
        
        return alternatives[:num_alternatives]
    
    def expand_prompt(self,
                     prompt: str,
                     context: Optional[str] = None) -> str:
        """
        Expand a prompt with more descriptive text.
        
        Args:
            prompt: Original prompt
            context: Optional context to include
            
        Returns:
            Expanded prompt
        """
        if context:
            expanded_text = f"{context}. {prompt}"
        else:
            expanded_text = prompt
        
        # Use summarization in reverse - generate longer version
        # This is a simple approach; for better results, use a text generation model
        try:
            # Try to generate a longer version by using summarization with longer target
            words = prompt.split()
            target_length = len(words) * 2
            
            # This is a workaround - in practice, you'd use a text generation model
            # For now, we'll just return an expanded version manually
            expanded = f"a scene showing {prompt}"
            return expanded
        except Exception:
            return prompt
    
    def create_prompt_variations(self,
                                prompt: str,
                                num_variations: int = 3) -> List[str]:
        """
        Create variations of a prompt for better matching.
        
        Args:
            prompt: Original prompt
            num_variations: Number of variations to create
            
        Returns:
            List of prompt variations
        """
        variations = [prompt]  # Include original
        
        # Generate variations using summarization with different parameters
        words = prompt.split()
        
        # Variation 1: More concise
        if len(words) > 5:
            try:
                concise = self.summarizer(
                    prompt,
                    max_length=len(words) // 2,
                    min_length=3,
                    do_sample=False
                )
                variations.append(concise[0]['summary_text'])
            except Exception:
                pass
        
        # Variation 2: Expanded
        expanded = f"a video scene of {prompt}"
        if expanded not in variations:
            variations.append(expanded)
        
        # Variation 3: Action-focused
        if "ing" in prompt or any(word.endswith("ing") for word in words):
            action_focused = prompt
        else:
            # Try to make it action-focused
            action_focused = f"someone {prompt}"
        
        if action_focused not in variations:
            variations.append(action_focused)
        
        return variations[:num_variations]


def create_text_generator(summarization_model: str = "facebook/bart-large-cnn",
                         device: Optional[str] = None) -> TextGenerator:
    """
    Convenience function to create a text generator.
    
    Args:
        summarization_model: Model for summarization
        device: Device to run on
        
    Returns:
        TextGenerator instance
    """
    return TextGenerator(summarization_model=summarization_model, device=device)

