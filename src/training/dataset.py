"""
Dataset Creation for Fine-Tuning

This module helps create datasets for fine-tuning BLIP and CLIP
from video frames and annotations.
"""

import json
from pathlib import Path
from typing import List
from PIL import Image

from ..pipeline import VideoMontagePipeline
from ..frame_extraction import extract_center_frames


def create_finetuning_dataset(video_path: str,
                               annotations_path: str,
                               output_dir: str = "data/processed",
                               frames_per_video: int = 10) -> str:
    """
    Create a dataset for fine-tuning from annotated videos.
    
    Args:
        video_path: Path to video file
        annotations_path: Path to JSON file with annotations
                          Format: {"frames": [{"frame_idx": int, "caption": str, ...}]}
        output_dir: Directory to save processed dataset
        frames_per_video: Number of frames to extract per video
        
    Returns:
        Path to created dataset directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    pipeline = VideoMontagePipeline(video_path)

    pipeline.motion_detector.analyze(video_path, show_plots=False)
    motion_segments = pipeline.motion_detector.detect_segments(
        video_path,
        motion_pixel_threshold=pipeline.motion_detector.suggested_threshold,
        show_plots=False,
    )
    
    frames = extract_center_frames(video_path, motion_segments)
    
    dataset = []
    for idx, image in frames[:frames_per_video]:
        annotation = None
        for ann in annotations.get('frames', []):
            if abs(ann['frame_idx'] - idx) < 10:  # Within 10 frames
                annotation = ann
                break
        
        if annotation:
            dataset_entry = {
                'image_path': str(output_path / f"frame_{idx}.jpg"),
                'caption': annotation['caption'],
                'frame_idx': idx,
                'video_path': video_path
            }
            dataset.append(dataset_entry)
            image.save(dataset_entry['image_path'])
    
    dataset_path = output_path / "dataset.json"
    with open(dataset_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created dataset with {len(dataset)} samples")
    print(f"Dataset saved to: {dataset_path}")
    
    return str(dataset_path)


class VideoFrameDataset:
    """Dataset class for video frame-caption pairs."""
    
    def __init__(self, dataset_path: str, transform=None):
        """
        Initialize dataset.
        
        Args:
            dataset_path: Path to dataset JSON file
            transform: Optional image transformations
        """
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        image = Image.open(entry['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        caption = entry['caption']
        
        return {
            'image': image,
            'caption': caption,
            'frame_idx': entry.get('frame_idx', 0)
        }

