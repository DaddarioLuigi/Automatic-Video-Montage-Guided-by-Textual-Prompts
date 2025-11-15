# Automatic Video Montage Guided by Textual Prompts

This project implements an automatic video montage system that selects and merges the most relevant scenes from a video based on user-provided natural language prompts. The pipeline combines motion detection, image captioning, semantic filtering, and similarity matching using CLIP to identify and assemble video segments that match textual descriptions.

## Features

- **Motion Detection**: Automatically identifies dynamic segments in videos using frame difference analysis
- **Frame Extraction**: Extracts representative frames from motion segments for analysis
- **Image Captioning**: Generates detailed captions for video frames using BLIP (Bootstrapping Language-Image Pre-training)
- **Prompt Parsing & Semantic Filtering**: Parses user prompts to extract keywords and filters captions based on semantic rules
- **Semantic Similarity Matching**: Uses CLIP (Contrastive Language-Image Pre-training) to match user prompts with video content
- **Analysis & Experiments**: Provides detailed quantitative analysis and threshold sensitivity experiments
- **Automatic Montage Generation**: Assembles selected video segments into a cohesive final montage

## Project Structure

```
video_summarization/
├── src/                        # Source code modules
│   ├── motion_detection/       # Motion detection algorithms
│   ├── frame_extraction/       # Frame extraction from segments
│   ├── caption_generation/     # BLIP-based caption generation
│   ├── prompt_processing/      # Prompt parsing and semantic filtering
│   ├── clip_matching/          # CLIP-based similarity matching
│   ├── analysis/               # Pipeline analysis and experiments
│   ├── video_assembly/         # Video montage assembly
│   └── pipeline.py             # Main pipeline orchestrator
├── notebooks/                  # Jupyter notebooks
│   └── Automatic_Video_Montage_Guided_by_Textual_Prompts.ipynb
├── config/                     # Configuration files
│   └── config.py               # Default parameters
├── outputs/                    # Generated montage videos
├── utils/                      # Utility functions
├── main.py                     # Command-line interface
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/video_summarization.git
   cd video_summarization
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   For memory-efficient inference with 8-bit quantization:

   ```bash
   pip install bitsandbytes
   ```

   Then uncomment the bitsandbytes line in `requirements.txt`.

## Usage

### Command-Line Interface

The easiest way to use the pipeline is through the command-line interface:

```bash
python main.py --video_path video.mp4 --prompts "adding ingredients" "closing box" "plating dish" --output_path outputs/montage.mp4
```

**Available options:**
- `--video_path`: Path to input video file (required)
- `--prompts`: One or more textual prompts describing desired scenes (required)
- `--output_path`: Path for output montage video (default: `outputs/final_montage.mp4`)
- `--similarity_threshold`: Minimum CLIP similarity score (default: 0.25)
- `--pixel_change_threshold`: Threshold for pixel change detection (default: 25)
- `--no_semantic_filtering`: Disable semantic filtering
- `--no_analysis`: Skip analysis and experiments
- `--no_plots`: Skip visualization plots

### Python API

You can also use the pipeline programmatically:

```python
from src.pipeline import create_montage

# Simple usage
output_path = create_montage(
    video_path="video.mp4",
    prompts=[
        "adding the ingredients in the sandwich",
        "closing the box",
        "plating the dish"
    ],
    output_path="outputs/montage.mp4"
)

# Advanced usage with custom parameters
from src.pipeline import VideoMontagePipeline

pipeline = VideoMontagePipeline("video.mp4", device="cuda")
output_path = pipeline.run_complete_pipeline(
    prompts=["your prompts here"],
    similarity_threshold=0.30,
    pixel_change_threshold=25,
    enable_semantic_filtering=True,
    enable_analysis=True,
    enable_plots=True
)
```

### Jupyter Notebooks

Two notebooks are available:

1. **Original Development Notebook** (`notebooks/Automatic_Video_Montage_Guided_by_Textual_Prompts.ipynb`): Original implementation and exploration

2. **Research Analysis Notebook** (`notebooks/Research_Analysis_and_Experiments.ipynb`): Complete research pipeline including:
   - Pipeline execution and analysis
   - Baseline comparisons
   - Ablation studies
   - Threshold sensitivity analysis
   - Publication-quality visualizations
   - LaTeX table generation for papers

## How It Works

The pipeline consists of seven main stages:

1. **Motion Detection**: Analyzes the video to detect segments with significant motion using pixel-level frame differences. A statistical threshold (75th percentile) is automatically computed to identify dynamic portions of the video.

2. **Frame Extraction**: Extracts center frames from each detected motion segment to serve as representative images for content analysis.

3. **Caption Generation**: Uses BLIP image captioning model to generate descriptive captions for each extracted frame, capturing the visual content and context.

4. **Prompt Parsing & Semantic Filtering**: Parses user-provided prompts to extract keywords, verbs, and nouns. Filters captions based on semantic rules to focus on relevant content.

5. **Semantic Matching**: Uses CLIP to encode both prompts and captions, then computes cosine similarity scores to identify the most relevant segments.

6. **Analysis & Experiments**: Provides detailed quantitative analysis including:
   - Similarity score distributions
   - Threshold sensitivity analysis
   - Temporal distribution of selected segments
   - Caption quality metrics
   - Pipeline performance summary

7. **Video Assembly**: Selects segments with similarity scores above the threshold and concatenates them into the final montage video.

## Configuration Parameters

### Motion Detection
- `pixel_change_threshold`: Threshold for detecting pixel changes between frames (default: 25)
- `motion_pixel_threshold`: Minimum number of changed pixels to consider motion (auto-calculated as 75th percentile)

### Caption Generation
- `model_name`: BLIP model to use (default: "Salesforce/blip-image-captioning-base")
- `context_prompt`: Optional context prompt for captioning

### Prompt Processing
- `min_keyword_matches`: Minimum keyword matches for semantic filtering (default: 1)
- `enable_semantic_filtering`: Enable/disable semantic filtering (default: True)

### CLIP Matching
- `model_name`: CLIP model to use (default: "ViT-B/32")
- `similarity_threshold`: Minimum similarity score for selection (default: 0.25)

## Technical Details

### Motion Detection Algorithm

The motion detection uses frame difference analysis:
- Computes absolute difference between consecutive grayscale frames
- Counts pixels where the difference exceeds a threshold
- Identifies segments where motion exceeds a statistical threshold (75th percentile)

### Caption Generation

Uses BLIP (Salesforce/blip-image-captioning-base) for generating frame descriptions. The model can be configured with:
- Float16 precision for memory efficiency
- 8-bit quantization using bitsandbytes for further memory reduction
- Custom context prompts for context-aware captioning

### Prompt Parsing & Semantic Filtering

The prompt parser:
- Extracts keywords, verbs, and nouns from user prompts
- Applies semantic filtering to captions based on keyword matches
- Helps focus the matching process on relevant content

### Semantic Matching

CLIP (ViT-B/32) is used to compute semantic similarity between:
- User-provided textual prompts
- Generated image captions

Segments with cosine similarity above the threshold are selected for the final montage.

## Requirements

- Python 3.8+
- GPU acceleration is highly recommended for faster processing
- Sufficient memory for loading CLIP and BLIP models (approximately 2-4GB VRAM)
- Video input should be in a standard format (MP4, AVI, etc.)

## Dependencies

- PyTorch and TorchVision
- CLIP (OpenAI's implementation)
- transformers (for BLIP models)
- moviepy (for video processing)
- opencv-python (for video I/O)
- numpy, matplotlib, PIL

See `requirements.txt` for complete list with versions.

## Example

```bash
# Basic usage
python main.py \
    --video_path input_video.mp4 \
    --prompts "cooking food" "serving dish" "plating meal" \
    --output_path outputs/cooking_montage.mp4

# With custom threshold
python main.py \
    --video_path input_video.mp4 \
    --prompts "adding ingredients" "closing container" \
    --similarity_threshold 0.30 \
    --output_path outputs/higher_quality_montage.mp4
```

## Research Features

This project includes comprehensive research tools:

### Evaluation Metrics
- **Classification Metrics**: Precision, Recall, F1 Score
- **Coverage Metrics**: Coverage ratio, temporal coverage
- **Diversity Metrics**: Vocabulary size, caption diversity
- **Coherence Metrics**: Temporal coherence score

### Baseline Methods
- Random selection
- Uniform temporal sampling
- First N segments
- Motion intensity-based selection

### Experiments
- **Baseline Comparison**: Compare proposed method with baseline approaches
- **Ablation Study**: Analyze contribution of each component
- **Threshold Sensitivity**: Explore optimal parameter values

### Visualization Tools
- Publication-quality figures (300 DPI)
- Heatmaps, bar charts, sensitivity plots
- LaTeX table generation for papers

See `notebooks/Research_Analysis_and_Experiments.ipynb` for complete research pipeline.

### Running Experiments

```bash
# Run all experiments
python run_experiments.py \
    --video_path video.mp4 \
    --prompts "prompt1" "prompt2" \
    --experiments all \
    --generate_plots \
    --generate_latex
```

## Fine-Tuning for Domain-Specific Improvement

The pipeline supports fine-tuning BLIP and CLIP models for domain-specific videos:

### Quick Fine-Tuning

```bash
# Fine-tune BLIP for better domain-specific captions
python scripts/finetune_models.py \
    --model blip \
    --train_dataset data/processed/dataset.json \
    --epochs 3

# Fine-tune CLIP for better semantic matching
python scripts/finetune_models.py \
    --model clip \
    --train_dataset data/processed/dataset.json \
    --epochs 5
```

**Expected Improvements:**
- BLIP: +15-25% improvement in caption quality
- CLIP: +20-30% improvement in matching accuracy
- Overall: Better montage quality for specific domains

See `README_FINETUNING.md` and `FINETUNING_PROPOSAL.md` for detailed guides.

## Future Improvements

- Support for temporal context in captioning (using previous frames)
- Enhanced prompt customization options
- Automatic threshold optimization
- Support for multiple video inputs
- Transition effects between segments
- Audio track preservation in final montage
- Support for video streaming
- Multi-language prompt support
- End-to-end training of entire pipeline

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
