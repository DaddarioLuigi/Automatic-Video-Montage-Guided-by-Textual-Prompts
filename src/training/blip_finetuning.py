"""
Fine-Tuning BLIP for Domain-Specific Captioning

This module provides functionality to fine-tune BLIP on domain-specific
video frames to improve caption quality.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    TrainingArguments,
)
from typing import Optional
from pathlib import Path

from .dataset import VideoFrameDataset


class BlipFineTuner:
    """Class for fine-tuning BLIP models."""
    
    def __init__(self,
                 base_model: str = "Salesforce/blip-image-captioning-base",
                 device: Optional[str] = None):
        """
        Initialize BLIP fine-tuner.
        
        Args:
            base_model: Base BLIP model to fine-tune
            device: Device to use for training
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.base_model = base_model
        self.processor = BlipProcessor.from_pretrained(base_model)
        self.model = None
        self.config = None
    
    def load_model(self):
        """Load BLIP model and processor."""
        self.model = BlipForConditionalGeneration.from_pretrained(self.base_model)
        self.model.to(self.device)
        return self.model, self.processor
    
    def prepare_dataset(self, dataset_path: str, batch_size: int = 4):
        """
        Prepare dataset for training.
        
        Args:
            dataset_path: Path to dataset JSON file
            batch_size: Batch size for training
            
        Returns:
            DataLoader for training
        """
        dataset = VideoFrameDataset(dataset_path)
        
        def collate_fn(batch):
            images = [item['image'] for item in batch]
            captions = [item['caption'] for item in batch]
            
            inputs = self.processor(
                images=images,
                text=captions,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
    
    def fine_tune(self,
                  train_dataset_path: str,
                  val_dataset_path: Optional[str] = None,
                  output_dir: str = "models/finetuned_blip",
                  num_epochs: int = 3,
                  learning_rate: float = 5e-5,
                  batch_size: int = 4,
                  save_steps: int = 500,
                  eval_steps: Optional[int] = None):
        """
        Fine-tune BLIP model.
        
        Args:
            train_dataset_path: Path to training dataset JSON
            val_dataset_path: Optional path to validation dataset JSON
            output_dir: Directory to save fine-tuned model
            num_epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            batch_size: Training batch size
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps (if validation set provided)
            
        Returns:
            Path to saved fine-tuned model
        """
        model, processor = self.load_model()
        train_loader = self.prepare_dataset(train_dataset_path, batch_size)
        val_loader = None
        if val_dataset_path:
            val_loader = self.prepare_dataset(val_dataset_path, batch_size)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=500,
            logging_steps=100,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps" if val_loader else "no",
            save_total_limit=3,
            load_best_model_at_end=True if val_loader else False,
            metric_for_best_model="loss",
            greater_is_better=False,
            push_to_hub=False,
        )
        
        trainer = BlipTrainer(
            model=model,
            args=training_args,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            processor=processor,
            device=self.device
        )
        
        print(f"Starting fine-tuning on {len(train_loader.dataset)} samples...")
        trainer.train()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        processor.save_pretrained(output_path)
        
        print(f"Fine-tuned model saved to: {output_path}")
        
        return str(output_path)


class BlipTrainer:
    """Custom trainer for BLIP fine-tuning."""
    
    def __init__(self, model, args, train_dataloader, eval_dataloader,
                 processor, device):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.processor = processor
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate
        )
        
        # Learning rate scheduler
        num_training_steps = len(train_dataloader) * args.num_train_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps
        )
    
    def train(self):
        """Training loop."""
        self.model.train()
        
        global_step = 0
        for epoch in range(self.args.num_train_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % self.args.logging_steps == 0:
                    print(f"Step {global_step}: Loss = {loss.item():.4f}")
                
                if global_step % self.args.save_steps == 0:
                    self._save_checkpoint(global_step)
                
                if self.eval_dataloader and global_step % self.args.eval_steps == 0:
                    self._evaluate()
            
            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.args.num_train_epochs}: Avg Loss = {avg_loss:.4f}")
    
    def _save_checkpoint(self, step):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)
    
    def _evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.eval_dataloader)
        print(f"Validation Loss: {avg_loss:.4f}")
        self.model.train()


def fine_tune_blip(train_dataset_path: str,
                   val_dataset_path: Optional[str] = None,
                   output_dir: str = "models/finetuned_blip",
                   **kwargs) -> str:
    """
    Convenience function to fine-tune BLIP.
    
    Args:
        train_dataset_path: Path to training dataset
        val_dataset_path: Optional path to validation dataset
        output_dir: Output directory for fine-tuned model
        **kwargs: Additional training arguments
        
    Returns:
        Path to fine-tuned model
    """
    tuner = BlipFineTuner()
    return tuner.fine_tune(
        train_dataset_path,
        val_dataset_path,
        output_dir,
        **kwargs
    )

