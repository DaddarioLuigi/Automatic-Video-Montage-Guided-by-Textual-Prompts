#!/usr/bin/env python3
"""
Script to run the complete pipeline on a video.
Usage: python run_complete_pipeline.py [video_path] [prompt1] [prompt2] ...
"""

import sys
from pathlib import Path

from src.pipeline import VideoMontagePipeline

def main():
    if len(sys.argv) < 2:
        video_path = "data/videos/1.mp4"
        prompts = [
            "adding the ingredients in the sandwich",
            "closing the box",
            "plating the dish"
        ]
        report_path = "results/reference_example_report.json"
    else:
        video_path = sys.argv[1]
        prompts = sys.argv[2:] if len(sys.argv) > 2 else [
            "adding the ingredients in the sandwich",
            "closing the box",
            "plating the dish"
        ]
        report_path = "results/reference_example_report.json"
    
    if not Path(video_path).exists():
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("RUNNING COMPLETE PIPELINE")
    print("=" * 60)
    print(f"\nVideo: {video_path}")
    print(f"Prompts: {prompts}\n")
    
    try:
        pipeline = VideoMontagePipeline(video_path)
        
        output_path = pipeline.run_complete_pipeline(
            prompts=prompts,
            similarity_threshold=0.25,
            enable_semantic_filtering=True,
            enable_analysis=True,
            enable_plots=True,
            output_path="outputs/test_montage.mp4",
            report_path=report_path
        )
        
        if output_path:
            print(f"\n{'='*60}")
            print(f"SUCCESS! Montage saved to: {output_path}")
            print(f"{'='*60}")
        else:
            print("\nWarning: No segments were selected.")
            print("Try lowering the similarity_threshold.")
            
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


