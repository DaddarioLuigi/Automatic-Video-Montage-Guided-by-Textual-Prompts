#!/usr/bin/env python3
"""
Script to fine-tune BLIP and CLIP models for video montage pipeline.

Usage:
    python scripts/finetune_models.py --model blip --dataset data/processed/dataset.json
    python scripts/finetune_models.py --model clip --dataset data/processed/dataset.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training import fine_tune_blip, fine_tune_clip


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BLIP or CLIP models")
    parser.add_argument(
        '--model',
        type=str,
        choices=['blip', 'clip'],
        required=True,
        help='Model to fine-tune (blip or clip)'
    )
    parser.add_argument(
        '--train_dataset',
        type=str,
        required=True,
        help='Path to training dataset JSON file'
    )
    parser.add_argument(
        '--val_dataset',
        type=str,
        default=None,
        help='Path to validation dataset JSON file (optional)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for fine-tuned model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = f"models/finetuned_{args.model}"
    
    print("=" * 60)
    print(f"Fine-Tuning {args.model.upper()}")
    print("=" * 60)
    print(f"Training dataset: {args.train_dataset}")
    print(f"Validation dataset: {args.val_dataset or 'None'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    try:
        if args.model == 'blip':
            output_path = fine_tune_blip(
                train_dataset_path=args.train_dataset,
                val_dataset_path=args.val_dataset,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size
            )
        elif args.model == 'clip':
            output_path = fine_tune_clip(
                train_dataset_path=args.train_dataset,
                val_dataset_path=args.val_dataset,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size
            )
        
        print(f"\nFine-tuning completed!")
        print(f"Model saved to: {output_path}")
        print(f"\nTo use the fine-tuned model, update your pipeline:")
        print(f"  - BLIP: Use model_path='{output_path}' in CaptionGenerator")
        print(f"  - CLIP: Load model from '{output_path}' in CLIPMatcher")
        
    except Exception as e:
        print(f"\nError during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

