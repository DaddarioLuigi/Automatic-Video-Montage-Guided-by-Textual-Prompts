# Project Structure

This document describes the organization of the video montage project after restructuring.

## Directory Structure

```
video_summarization/
│
├── src/                              # Main source code
│   ├── __init__.py                  # Package initialization
│   ├── pipeline.py                  # Main pipeline orchestrator
│   │
│   ├── motion_detection/            # Motion detection module
│   │   ├── __init__.py
│   │   └── detector.py              # Motion detection algorithms
│   │
│   ├── frame_extraction/            # Frame extraction module
│   │   ├── __init__.py
│   │   └── extractor.py             # Frame extraction from segments
│   │
│   ├── caption_generation/          # Caption generation module
│   │   ├── __init__.py
│   │   └── generator.py             # BLIP-based caption generation
│   │
│   ├── prompt_processing/           # Prompt parsing & filtering
│   │   ├── __init__.py
│   │   └── parser.py                # Prompt parsing and semantic filtering
│   │
│   ├── clip_matching/              # CLIP matching module
│   │   ├── __init__.py
│   │   └── matcher.py               # CLIP-based similarity matching
│   │
│   ├── analysis/                    # Analysis & experiments module
│   │   ├── __init__.py
│   │   └── analyzer.py              # Pipeline analysis and experiments
│   │
│   └── video_assembly/              # Video assembly module
│       ├── __init__.py
│       └── assembler.py             # Video montage assembly
│
├── notebooks/                       # Jupyter notebooks
│   └── Automatic_Video_Montage_Guided_by_Textual_Prompts.ipynb
│
├── config/                          # Configuration files
│   ├── __init__.py
│   └── config.py                    # Default parameters and settings
│
├── outputs/                          # Generated montage videos
│   └── (montage outputs go here)
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   └── video_utils.py                # Video utility functions
│
├── main.py                          # Command-line interface
├── example.py                       # Example usage scripts
├── requirements.txt                # Python dependencies
├── README.md                        # Project documentation
└── STRUCTURE.md                     # This file
```

## Pipeline Stages

The pipeline implements the following 7 stages:

### 1. Motion Detection (`src/motion_detection/`)
- **Purpose**: Identify dynamic segments in videos
- **Key Functions**:
  - `analyze_motion()`: Analyze motion and suggest thresholds
  - `detect_motion()`: Detect motion segments
- **Algorithm**: Frame difference analysis with statistical thresholding

### 2. Frame Extraction (`src/frame_extraction/`)
- **Purpose**: Extract representative frames from motion segments
- **Key Functions**:
  - `extract_center_frames()`: Extract center frame from each segment
- **Strategy**: Uses center frame of each segment as representative

### 3. Caption Generation (`src/caption_generation/`)
- **Purpose**: Generate textual descriptions of video frames
- **Key Functions**:
  - `generate_captions()`: Generate captions using BLIP
- **Model**: Salesforce/blip-image-captioning-base

### 4. Prompt Parsing & Semantic Filtering (`src/prompt_processing/`)
- **Purpose**: Parse user prompts and filter captions semantically
- **Key Functions**:
  - `parse_prompts()`: Extract keywords, verbs, nouns from prompts
  - `filter_captions()`: Filter captions based on keyword matches
- **Features**: Extracts semantic information for focused matching

### 5. Semantic Matching (`src/clip_matching/`)
- **Purpose**: Compute similarity between prompts and captions
- **Key Functions**:
  - `compute_similarities()`: Compute CLIP similarity scores
  - `select_segments()`: Select segments based on similarity threshold
- **Model**: CLIP ViT-B/32

### 6. Analysis & Experiments (`src/analysis/`)
- **Purpose**: Provide quantitative analysis and threshold experiments
- **Key Functions**:
  - `analyze_pipeline()`: Complete pipeline analysis
  - `threshold_sensitivity_analysis()`: Threshold sensitivity experiments
  - `plot_analysis_results()`: Visualization plots
- **Features**: Statistical analysis, visualizations, threshold experiments

### 7. Video Assembly (`src/video_assembly/`)
- **Purpose**: Assemble selected segments into final montage
- **Key Functions**:
  - `assemble_montage()`: Concatenate selected segments
- **Output**: Final montage video file

## Usage Patterns

### 1. Command-Line Interface
```bash
python main.py --video_path video.mp4 --prompts "prompt1" "prompt2"
```

### 2. Simple Python API
```python
from src.pipeline import create_montage
create_montage("video.mp4", ["prompt1", "prompt2"])
```

### 3. Advanced Python API
```python
from src.pipeline import VideoMontagePipeline
pipeline = VideoMontagePipeline("video.mp4")
pipeline.run_complete_pipeline(["prompt1", "prompt2"])
```

### 4. Step-by-Step Control
```python
# Use individual modules for fine-grained control
from src.motion_detection import MotionDetector
from src.frame_extraction import FrameExtractor
# ... etc
```

## Benefits of This Structure

1. **Modularity**: Each stage is in its own module, making it easy to modify or extend
2. **Reusability**: Modules can be used independently or together
3. **Testability**: Each module can be tested in isolation
4. **Maintainability**: Clear separation of concerns
5. **Extensibility**: Easy to add new features or replace components
6. **Documentation**: Each module is self-documented with docstrings

## Migration from Notebook

The original notebook code has been extracted into modules:
- All functionality is preserved
- Code is more organized and reusable
- Notebook is still available in `notebooks/` for interactive use
- Command-line and Python API provide easier integration


