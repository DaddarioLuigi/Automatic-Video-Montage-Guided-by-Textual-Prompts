"""
Configuration module for the video montage pipeline.

Default parameters and configuration settings.
"""

# Motion Detection Parameters
MOTION_DETECTION = {
    'pixel_change_threshold': 25,
    'motion_pixel_threshold': None,  # Auto-calculated from 75th percentile
    'percentile_threshold': 75
}

# Caption Generation Parameters
CAPTION_GENERATION = {
    'model_name': 'Salesforce/blip-image-captioning-base',
    'context_prompt': None,
    'max_length': 50
}

# Prompt Processing Parameters
PROMPT_PROCESSING = {
    'min_keyword_matches': 1,
    'enable_semantic_filtering': True
}

# CLIP Matching Parameters
CLIP_MATCHING = {
    'model_name': 'ViT-B/32',
    'similarity_threshold': 0.25
}

# Analysis Parameters
ANALYSIS = {
    'enable_analysis': True,
    'enable_plots': True,
    'threshold_sensitivity_range': (0.15, 0.45, 0.05)
}

# Video Assembly Parameters
VIDEO_ASSEMBLY = {
    'output_format': 'mp4',
    'codec': 'libx264',
    'audio_codec': 'aac',
    'preserve_audio': True
}

# Default Pipeline Configuration
DEFAULT_CONFIG = {
    **MOTION_DETECTION,
    **CAPTION_GENERATION,
    **PROMPT_PROCESSING,
    **CLIP_MATCHING,
    **ANALYSIS,
    **VIDEO_ASSEMBLY
}

