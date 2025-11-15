#!/usr/bin/env python3
"""
Main script for running the video montage pipeline.

Example usage:
    python main.py --video_path video.mp4 --prompts "adding ingredients" "closing box" "plating dish"
"""

import argparse
import sys
from pathlib import Path

from src.pipeline import create_montage


def main():
    parser = argparse.ArgumentParser(
        description="Automatic Video Montage Guided by Textual Prompts"
    )
    parser.add_argument(
        '--video_path',
        type=str,
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--prompts',
        type=str,
        nargs='+',
        required=True,
        help='Textual prompts describing desired scenes'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='outputs/final_montage.mp4',
        help='Path for output montage video (default: outputs/final_montage.mp4)'
    )
    parser.add_argument(
        '--similarity_threshold',
        type=float,
        default=0.25,
        help='Minimum CLIP similarity score for segment selection (default: 0.25)'
    )
    parser.add_argument(
        '--pixel_change_threshold',
        type=int,
        default=25,
        help='Threshold for pixel change detection (default: 25)'
    )
    parser.add_argument(
        '--no_semantic_filtering',
        action='store_true',
        help='Disable semantic filtering of captions'
    )
    parser.add_argument(
        '--no_analysis',
        action='store_true',
        help='Skip analysis and experiments'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip generating visualization plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    try:
        output_path = create_montage(
            video_path=args.video_path,
            prompts=args.prompts,
            output_path=args.output_path,
            similarity_threshold=args.similarity_threshold,
            pixel_change_threshold=args.pixel_change_threshold,
            enable_semantic_filtering=not args.no_semantic_filtering,
            enable_analysis=not args.no_analysis,
            enable_plots=not args.no_plots
        )
        
        if output_path:
            print(f"\nSuccess! Montage saved to: {output_path}")
            sys.exit(0)
        else:
            print("\nError: No segments were selected. Try lowering the similarity threshold.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

