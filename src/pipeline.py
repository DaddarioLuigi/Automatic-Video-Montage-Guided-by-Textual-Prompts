"""
Main Pipeline for Automatic Video Montage Generation

This module orchestrates the complete pipeline:
1. Motion Detection
2. Frame Extraction
3. Caption Generation
4. Prompt Parsing & Semantic Filtering
5. CLIP-based Semantic Matching
6. Analysis & Experiments
7. Video Assembly
"""

import cv2
import torch
from typing import List, Optional

from .motion_detection import MotionDetector
from .frame_extraction import FrameExtractor
from .caption_generation import CaptionGenerator
from .prompt_processing import PromptParser
from .clip_matching import CLIPMatcher
from .analysis import PipelineAnalyzer
from .video_assembly import VideoAssembler


class VideoMontagePipeline:
    """Main pipeline class for automatic video montage generation."""
    
    def __init__(self, video_path: str, device: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            video_path: Path to the input video file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.video_path = video_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.motion_detector = MotionDetector()
        self.frame_extractor = FrameExtractor(video_path)
        self.caption_generator = CaptionGenerator(device=device)
        # Use advanced parsing by default (with fallback to basic if not available)
        self.prompt_parser = PromptParser(
            use_advanced_parsing=True,
            use_semantic_filtering=True
        )
        self.clip_matcher = CLIPMatcher(device=device)
        self.analyzer = PipelineAnalyzer(self.fps)
        self.video_assembler = VideoAssembler(video_path, self.fps)
        
        self.motion_segments = None
        self.frames = None
        self.captions = None
        self.similarities = None
        self.selected_segments = None
    
    def run_complete_pipeline(self,
                             prompts: List[str],
                             pixel_change_threshold: int = 25,
                             motion_pixel_threshold: Optional[int] = None,
                             similarity_threshold: float = 0.25,
                             enable_semantic_filtering: bool = True,
                             output_path: str = "outputs/final_montage.mp4",
                             enable_analysis: bool = True,
                             enable_plots: bool = True) -> str:
        """
        Run the complete pipeline from motion detection to video assembly.
        
        Args:
            prompts: List of textual prompts describing desired scenes
            pixel_change_threshold: Threshold for pixel change detection
            motion_pixel_threshold: Threshold for motion detection (auto-calculated if None)
            similarity_threshold: Minimum CLIP similarity score for selection
            enable_semantic_filtering: Whether to apply semantic filtering
            output_path: Path for the output montage video
            enable_analysis: Whether to perform analysis
            enable_plots: Whether to generate visualization plots
            
        Returns:
            Path to the created montage video
        """
        print("=" * 60)
        print("AUTOMATIC VIDEO MONTAGE PIPELINE")
        print("=" * 60)
        
        print("\n[1/7] Motion Detection...")
        motions = self.motion_detector.analyze(self.video_path)
        
        if motion_pixel_threshold is None:
            motion_pixel_threshold = self.motion_detector.suggested_threshold
        
        self.motion_segments = self.motion_detector.detect_segments(
            self.video_path, motion_pixel_threshold
        )
        print(f"   Detected {len(self.motion_segments)} motion segments")
        print("\n[2/7] Frame Extraction...")
        self.frames = self.frame_extractor.extract_center_frames(self.motion_segments)
        print(f"   Extracted {len(self.frames)} representative frames")
        print("\n[3/7] Caption Generation...")
        self.captions = self.caption_generator.generate(self.frames)
        print("\n[4/7] Prompt Parsing & Semantic Filtering...")
        parsed_prompts = self.prompt_parser.parse(prompts)
        keywords = self.prompt_parser.get_keywords()
        verbs = self.prompt_parser.get_verbs()
        print(f"   Extracted {len(keywords)} keywords")
        print(f"   Found {len(verbs)} action verbs")
        
        # Show advanced parsing info if available
        entities = self.prompt_parser.get_entities()
        semantic_roles = self.prompt_parser.get_semantic_roles()
        if entities:
            print(f"   Found {len(entities)} named entities")
        if semantic_roles:
            print(f"   Extracted {len(semantic_roles)} semantic roles (subject-verb-object)")
        
        if enable_semantic_filtering:
            # Use semantic filtering with embeddings if available
            filtered_captions = self.prompt_parser.filter_captions(
                self.captions,
                semantic_threshold=0.3,
                use_semantic=True
            )
            print(f"   Filtered to {len(filtered_captions)} captions using semantic similarity")
            captions_for_matching = filtered_captions
        else:
            captions_for_matching = self.captions
        
        print("\n[5/7] CLIP-based Semantic Matching...")
        self.clip_matcher.encode_prompts(prompts)
        self.similarities = self.clip_matcher.compute_similarities(captions_for_matching)
        self.selected_segments = self.clip_matcher.select_segments(
            self.similarities, self.motion_segments, similarity_threshold
        )
        print(f"   Selected {len(self.selected_segments)} segments")
        
        if enable_analysis:
            print("\n[6/7] Analysis & Experiments...")
            analysis_results = self.analyzer.analyze(
                self.motion_segments, self.frames, self.captions,
                self.similarities, self.selected_segments
            )
            self.analyzer.print_summary()
            
            if enable_plots:
                from .analysis.analyzer import plot_analysis_results
                plot_analysis_results(
                    self.similarities,
                    self.selected_segments, self.fps
                )
        else:
            print("\n[6/7] Analysis & Experiments... (skipped)")
        
        print("\n[7/7] Video Assembly...")
        if not self.selected_segments:
            print("   WARNING: No segments selected. Montage cannot be created.")
            return None
        
        if output_path is None:
            print("   Skipping video assembly (output_path is None)")
            return None
        
        output_path = self.video_assembler.assemble(
            self.selected_segments, output_path
        )
        print(f"   Montage created: {output_path}")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return output_path
    
    def run_step_by_step(self):
        """Run pipeline step by step with interactive prompts."""
        pass


def create_montage(video_path: str,
                   prompts: List[str],
                   output_path: str = "outputs/final_montage.mp4",
                   **kwargs) -> str:
    """
    Convenience function to create a video montage.
    
    Args:
        video_path: Path to input video
        prompts: List of textual prompts
        output_path: Path for output montage
        **kwargs: Additional pipeline parameters
        
    Returns:
        Path to created montage
    """
    pipeline = VideoMontagePipeline(video_path)
    return pipeline.run_complete_pipeline(prompts, output_path=output_path, **kwargs)

