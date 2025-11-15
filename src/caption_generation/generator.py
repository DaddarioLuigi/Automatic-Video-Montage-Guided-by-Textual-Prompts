"""
Caption Generation using BLIP

This module generates textual descriptions for video frames using
BLIP (Bootstrapping Language-Image Pre-training) models.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
from typing import List, Tuple, Optional
from tqdm import tqdm


def generate_captions(frames: List[Tuple[int, Image.Image]], 
                     model_name: str = "Salesforce/blip-image-captioning-base",
                     device: Optional[str] = None,
                     context_prompt: Optional[str] = None):
    """
    Generate captions for extracted frames using BLIP.
    
    Args:
        frames: List of (frame_index, PIL.Image) tuples
        model_name: Name of the BLIP model to use
        device: Device to run inference on ('cuda' or 'cpu')
        context_prompt: Optional prompt to provide context for captioning
        
    Returns:
        List of (frame_index, caption_string) tuples
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    captions = []
    for idx, image in tqdm(frames, desc="Generating captions"):
        if context_prompt:
            inputs = processor(images=image, text=context_prompt, return_tensors="pt").to(device)
        else:
            inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.inference_mode():
            out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append((idx, caption))
    
    print(f"\nGenerated {len(captions)} captions")
    return captions


class CaptionGenerator:
    """Class-based wrapper for caption generation functionality."""
    
    def __init__(self, 
                 model_name: str = "Salesforce/blip-image-captioning-base",
                 finetuned_model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 context_prompt: Optional[str] = None):
        """
        Initialize caption generator.
        
        Args:
            model_name: Base BLIP model name (used if finetuned_model_path is None)
            finetuned_model_path: Path to fine-tuned model (optional)
            device: Device to run inference on
            context_prompt: Optional context prompt for captioning
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.finetuned_model_path = finetuned_model_path
        self.context_prompt = context_prompt
        
        model_path = finetuned_model_path if finetuned_model_path else model_name
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
    
    def generate(self, frames: List[Tuple[int, Image.Image]]):
        """Generate captions for frames."""
        return generate_captions(frames, self.model_name, self.device, self.context_prompt)
    
    def generate_with_custom_prompt(self, frames: List[Tuple[int, Image.Image]], 
                                    prompt: str):
        """Generate captions with a custom context prompt."""
        captions = []
        for idx, image in tqdm(frames, desc="Generating captions"):
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            captions.append((idx, caption))
        return captions

