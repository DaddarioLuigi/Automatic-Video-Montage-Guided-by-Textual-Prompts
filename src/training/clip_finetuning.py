"""
Fine-Tuning CLIP for Video-Semantic Matching

This module provides functionality to fine-tune CLIP on video frame-caption
pairs to improve semantic matching accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from typing import Optional, Dict, List
from pathlib import Path
import json

from .dataset import VideoFrameDataset


class ClipFineTuner:
    """Class for fine-tuning CLIP models."""
    
    def __init__(self,
                 base_model: str = "ViT-B/32",
                 device: Optional[str] = None):
        """
        Initialize CLIP fine-tuner.
        
        Args:
            base_model: Base CLIP model to fine-tune
            device: Device to use for training
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.base_model = base_model
        self.model = None
        self.preprocess = None
    
    def load_model(self):
        """Load CLIP model."""
        self.model, self.preprocess = clip.load(self.base_model, device=self.device)
        return self.model, self.preprocess
    
    def contrastive_loss(self, image_features, text_features, temperature=0.07):
        """
        Compute contrastive loss for CLIP fine-tuning.
        
        Args:
            image_features: Image embeddings
            text_features: Text embeddings
            temperature: Temperature parameter
            
        Returns:
            Contrastive loss
        """
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits = torch.matmul(image_features, text_features.t()) / temperature
        
        labels = torch.arange(len(image_features)).to(logits.device)
        
        loss_img = F.cross_entropy(logits, labels)
        loss_txt = F.cross_entropy(logits.t(), labels)
        
        return (loss_img + loss_txt) / 2
    
    def prepare_dataset(self, dataset_path: str, batch_size: int = 32):
        """
        Prepare dataset for training.
        
        Args:
            dataset_path: Path to dataset JSON file
            batch_size: Batch size for training
            
        Returns:
            DataLoader for training
        """
        dataset = VideoFrameDataset(dataset_path, transform=self.preprocess)
        
        def collate_fn(batch):
            images = torch.stack([item['image'] for item in batch]).to(self.device)
            texts = [item['caption'] for item in batch]
            return images, texts
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
    
    def fine_tune(self,
                  train_dataset_path: str,
                  val_dataset_path: Optional[str] = None,
                  output_dir: str = "models/finetuned_clip",
                  num_epochs: int = 5,
                  learning_rate: float = 1e-5,
                  batch_size: int = 32,
                  temperature: float = 0.07,
                  save_interval: int = 1000):
        """
        Fine-tune CLIP model using contrastive learning.
        
        Args:
            train_dataset_path: Path to training dataset JSON
            val_dataset_path: Optional path to validation dataset JSON
            output_dir: Directory to save fine-tuned model
            num_epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            batch_size: Training batch size
            temperature: Temperature for contrastive loss
            save_interval: Save checkpoint every N steps
            
        Returns:
            Path to saved fine-tuned model
        """
        model, preprocess = self.load_model()
        train_loader = self.prepare_dataset(train_dataset_path, batch_size)
        val_loader = None
        if val_dataset_path:
            val_loader = self.prepare_dataset(val_dataset_path, batch_size)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        model.train()
        global_step = 0
        
        print(f"Starting fine-tuning on {len(train_loader.dataset)} samples...")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (images, texts) in enumerate(train_loader):
                optimizer.zero_grad()
                
                image_features = model.encode_image(images)
                text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
                text_features = model.encode_text(text_tokens)
                
                loss = self.contrastive_loss(image_features, text_features, temperature)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % 100 == 0:
                    print(f"Step {global_step}: Loss = {loss.item():.4f}")
                
                if global_step % save_interval == 0:
                    self._save_checkpoint(model, output_dir, global_step)
                
                if val_loader and global_step % 500 == 0:
                    val_loss = self._evaluate(model, val_loader, temperature)
                    print(f"Validation Loss: {val_loss:.4f}")
            
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}: Avg Loss = {avg_loss:.4f}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path / "pytorch_model.bin")
        
        config = {
            'base_model': self.base_model,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'temperature': temperature
        }
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Fine-tuned model saved to: {output_path}")
        
        return str(output_path)
    
    def _save_checkpoint(self, model, output_dir, step):
        """Save model checkpoint."""
        checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")
    
    def _evaluate(self, model, val_loader, temperature):
        """Evaluate on validation set."""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, texts in val_loader:
                image_features = model.encode_image(images)
                text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
                text_features = model.encode_text(text_tokens)
                
                loss = self.contrastive_loss(image_features, text_features, temperature)
                total_loss += loss.item()
        
        model.train()
        return total_loss / len(val_loader)


def fine_tune_clip(train_dataset_path: str,
                   val_dataset_path: Optional[str] = None,
                   output_dir: str = "models/finetuned_clip",
                   **kwargs) -> str:
    """
    Convenience function to fine-tune CLIP.
    
    Args:
        train_dataset_path: Path to training dataset
        val_dataset_path: Optional path to validation dataset
        output_dir: Output directory for fine-tuned model
        **kwargs: Additional training arguments
        
    Returns:
        Path to fine-tuned model
    """
    tuner = ClipFineTuner()
    return tuner.fine_tune(
        train_dataset_path,
        val_dataset_path,
        output_dir,
        **kwargs
    )

