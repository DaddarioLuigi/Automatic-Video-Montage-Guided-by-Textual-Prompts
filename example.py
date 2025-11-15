#!/usr/bin/env python3
"""
Example script demonstrating how to use the video montage pipeline.

This script shows both simple and advanced usage patterns.
"""

from src.pipeline import VideoMontagePipeline, create_montage


def example_simple():
    """Simple example using the convenience function."""
    print("=" * 60)
    print("SIMPLE EXAMPLE")
    print("=" * 60)
    
    output_path = create_montage(
        video_path="path/to/your/video.mp4",
        prompts=[
            "adding the ingredients in the sandwich",
            "closing the box",
            "plating the dish"
        ],
        output_path="outputs/example_montage.mp4",
        similarity_threshold=0.25
    )
    
    print(f"Montage created at: {output_path}")


def example_advanced():
    """Advanced example with full pipeline control."""
    print("=" * 60)
    print("ADVANCED EXAMPLE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = VideoMontagePipeline(
        video_path="path/to/your/video.mp4",
        device="cuda"  # or "cpu"
    )
    
    # Run complete pipeline with custom parameters
    output_path = pipeline.run_complete_pipeline(
        prompts=[
            "cooking food",
            "serving dish",
            "preparing meal"
        ],
        pixel_change_threshold=25,
        motion_pixel_threshold=None,  # Auto-calculate
        similarity_threshold=0.30,     # More strict selection
        enable_semantic_filtering=True,
        output_path="outputs/advanced_montage.mp4",
        enable_analysis=True,
        enable_plots=True
    )
    
    print(f"Montage created at: {output_path}")
    
    # Access pipeline results
    print(f"\nDetected {len(pipeline.motion_segments)} motion segments")
    print(f"Generated {len(pipeline.captions)} captions")
    print(f"Selected {len(pipeline.selected_segments)} segments for montage")


def example_step_by_step():
    """Example showing step-by-step pipeline execution."""
    print("=" * 60)
    print("STEP-BY-STEP EXAMPLE")
    print("=" * 60)
    
    pipeline = VideoMontagePipeline("path/to/your/video.mp4")
    
    # Step 1: Motion Detection
    print("\n[1] Running motion detection...")
    cap = __import__('cv2').VideoCapture(pipeline.video_path)
    motions = pipeline.motion_detector.analyze(cap)
    cap = __import__('cv2').VideoCapture(pipeline.video_path)
    pipeline.motion_segments = pipeline.motion_detector.detect_segments(
        cap, pipeline.motion_detector.suggested_threshold
    )
    print(f"   Detected {len(pipeline.motion_segments)} segments")
    
    # Step 2: Frame Extraction
    print("\n[2] Extracting frames...")
    pipeline.frames = pipeline.frame_extractor.extract_center_frames(
        pipeline.motion_segments
    )
    print(f"   Extracted {len(pipeline.frames)} frames")
    
    # Step 3: Caption Generation
    print("\n[3] Generating captions...")
    pipeline.captions = pipeline.caption_generator.generate(pipeline.frames)
    print(f"   Generated {len(pipeline.captions)} captions")
    
    # Step 4: Prompt Processing
    print("\n[4] Processing prompts...")
    prompts = ["adding ingredients", "closing box", "plating dish"]
    parsed = pipeline.prompt_parser.parse(prompts)
    filtered = pipeline.prompt_parser.filter_captions(pipeline.captions)
    print(f"   Filtered to {len(filtered)} relevant captions")
    
    # Step 5: CLIP Matching
    print("\n[5] Computing similarities...")
    pipeline.clip_matcher.encode_prompts(prompts)
    pipeline.similarities = pipeline.clip_matcher.compute_similarities(
        filtered
    )
    pipeline.selected_segments = pipeline.clip_matcher.select_segments(
        pipeline.similarities,
        pipeline.motion_segments,
        threshold=0.25
    )
    print(f"   Selected {len(pipeline.selected_segments)} segments")
    
    # Step 6: Analysis
    print("\n[6] Running analysis...")
    analysis = pipeline.analyzer.analyze(
        pipeline.motion_segments,
        pipeline.frames,
        pipeline.captions,
        pipeline.similarities,
        pipeline.selected_segments
    )
    pipeline.analyzer.print_summary()
    
    # Step 7: Video Assembly
    print("\n[7] Assembling montage...")
    output_path = pipeline.video_assembler.assemble(
        pipeline.selected_segments,
        "outputs/step_by_step_montage.mp4"
    )
    print(f"   Montage created at: {output_path}")


if __name__ == "__main__":
    print("\nNOTE: Update the video paths in these examples before running!")
    print("\nAvailable examples:")
    print("  1. example_simple() - Simple usage")
    print("  2. example_advanced() - Advanced usage with custom parameters")
    print("  3. example_step_by_step() - Manual step-by-step execution")
    print("\nUncomment the example you want to run:\n")
    
    # Uncomment the example you want to run:
    # example_simple()
    # example_advanced()
    # example_step_by_step()

