# Automatic Video Montage Guided by Textual Prompts

This project implements an automatic video montage system that selects and merges the most relevant scenes from a video based on user-provided natural language prompts. The pipeline combines motion detection, image captioning, advanced NLP processing, semantic filtering, and similarity matching using CLIP to identify and assemble video segments that match textual descriptions.

## Quick Start (Short)

- **Install**:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

- **Run (CLI)**:

```bash
python main.py \
    --video_path video.mp4 \
    --prompts "adding ingredients" "closing box" "plating dish" \
    --output_path outputs/montage.mp4
```

- **Run (Python)**:

```python
from src.pipeline import create_montage

output_path = create_montage(
    video_path="video.mp4",
    prompts=[
        "adding the ingredients in the sandwich",
        "closing the box",
        "plating the dish"
    ],
    output_path="outputs/montage.mp4"
)
```

- **More docs**: open the section below.

<details>
<summary><strong>Full README (expanded)</strong></summary>

## Features

### Core Pipeline Features

- **Motion Detection**: Automatically identifies dynamic segments in videos using frame difference analysis with statistical thresholding
- **Frame Extraction**: Extracts representative frames from motion segments for analysis
- **Image Captioning**: Generates detailed captions for video frames using BLIP (Bootstrapping Language-Image Pre-training)
- **Advanced NLP Processing**: 
  - **Linguistic Parsing**: spaCy-based parsing with POS tagging, Named Entity Recognition (NER), and dependency parsing
  - **Semantic Role Extraction**: Identifies subject-verb-object relations
  - **Action Phrase Extraction**: Extracts verb-object pairs from prompts
- **Semantic Filtering**: 
  - **Keyword-based filtering**: Traditional keyword matching
  - **Embedding-based filtering**: Advanced semantic similarity using sentence transformers
  - **Synonym expansion**: Automatically finds semantically similar keywords
- **Semantic Similarity Matching**: Uses CLIP (Contrastive Language-Image Pre-training) to match user prompts with video content
- **Text Generation & Summarization**: 
  - Caption summarization (individual and collective)
  - Alternative description generation
  - Prompt expansion and variation
- **Analysis & Experiments**: Provides detailed quantitative analysis and threshold sensitivity experiments
- **Automatic Montage Generation**: Assembles selected video segments into a cohesive final montage

### Research & Development Features

- **Fine-tuning Support**: Fine-tune BLIP and CLIP models for domain-specific videos (see **[`README_FINETUNING.md`](README_FINETUNING.md)**)
- **Comprehensive Evaluation Metrics**: Precision, Recall, F1, Coverage, Diversity, Coherence
- **Baseline Comparisons**: Compare with random, uniform, and motion-based selection
- **Ablation Studies**: Analyze contribution of each pipeline component
- **Threshold Sensitivity Analysis**: Explore optimal parameter values
- **Publication-quality Visualizations**: Generate figures and LaTeX tables for papers

## Project Structure

```
video_summarization/
├── src/                              # Source code modules
│   ├── motion_detection/             # Motion detection algorithms
│   │   └── detector.py
│   ├── frame_extraction/             # Frame extraction from segments
│   │   └── extractor.py
│   ├── caption_generation/          # BLIP-based caption generation
│   │   └── generator.py
│   ├── prompt_processing/            # Advanced NLP processing
│   │   ├── parser.py                 # Main parser (with fallback support)
│   │   ├── advanced_parsing.py      # spaCy-based linguistic parsing
│   │   ├── semantic_filtering.py    # Embedding-based semantic filtering
│   │   └── text_generation.py       # Text summarization and generation
│   ├── clip_matching/                # CLIP-based similarity matching
│   │   └── matcher.py
│   ├── analysis/                     # Pipeline analysis and experiments
│   │   └── analyzer.py
│   ├── metrics/                      # Evaluation metrics
│   │   └── evaluator.py
│   ├── baselines/                     # Baseline methods
│   │   └── baselines.py
│   ├── experiments/                  # Experiment runner and visualization
│   │   ├── experiment_runner.py
│   │   └── visualization.py
│   ├── training/                     # Fine-tuning modules
│   │   ├── blip_finetuning.py
│   │   ├── clip_finetuning.py
│   │   └── dataset.py
│   ├── video_assembly/               # Video montage assembly
│   │   └── assembler.py
│   └── pipeline.py                   # Main pipeline orchestrator
├── notebooks/                        # Jupyter notebooks
│   └── Research_Analysis_and_Experiments.ipynb
├── scripts/                          # Utility scripts
│   └── finetune_models.py
├── config/                           # Configuration files
│   └── config.py
├── data/                             # Data directories
│   ├── videos/
│   ├── processed/
│   └── annotations/
├── outputs/                          # Generated montage videos
├── results/                          # Experiment results
│   ├── experiments/
│   └── figures/
├── utils/                            # Utility functions
│   └── video_utils.py
├── main.py                           # Command-line interface
├── example.py                        # Basic example
├── example_advanced_nlp.py          # Advanced NLP examples
├── run_experiments.py                # Experiment runner script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/video_summarization.git
cd video_summarization
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install spaCy Language Model

For advanced NLP features, download the English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

For better accuracy (optional, larger models):

```bash
python -m spacy download en_core_web_md   # Medium model (~40MB)
python -m spacy download en_core_web_lg   # Large model (~560MB)
```

### 4. Optional: Memory-Efficient Inference

For 8-bit quantization support (reduces memory usage):

```bash
pip install bitsandbytes
```

Then uncomment the `bitsandbytes` line in `requirements.txt`.

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: Highly recommended for faster processing (CUDA-compatible)
- **Memory**: 
  - Minimum: 4GB RAM (CPU mode)
  - Recommended: 8GB+ RAM, 2-4GB VRAM (GPU mode)
- **Video Formats**: MP4, AVI, MOV, and other formats supported by OpenCV

## Usage

### Command-Line Interface

The easiest way to use the pipeline:

```bash
python main.py \
    --video_path video.mp4 \
    --prompts "adding ingredients" "closing box" "plating dish" \
    --output_path outputs/montage.mp4
```

**Available Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--video_path` | Path to input video file | Required |
| `--prompts` | One or more textual prompts | Required |
| `--output_path` | Path for output montage | `outputs/final_montage.mp4` |
| `--similarity_threshold` | Minimum CLIP similarity score | `0.25` |
| `--pixel_change_threshold` | Threshold for pixel change detection | `25` |
| `--no_semantic_filtering` | Disable semantic filtering | `False` |
| `--no_analysis` | Skip analysis and experiments | `False` |
| `--no_plots` | Skip visualization plots | `False` |

### Python API

#### Simple Usage

```python
from src.pipeline import create_montage

output_path = create_montage(
    video_path="video.mp4",
    prompts=[
        "adding the ingredients in the sandwich",
        "closing the box",
        "plating the dish"
    ],
    output_path="outputs/montage.mp4"
)
```

#### Advanced Usage

```python
from src.pipeline import VideoMontagePipeline

pipeline = VideoMontagePipeline("video.mp4", device="cuda")
output_path = pipeline.run_complete_pipeline(
    prompts=["adding ingredients", "closing box"],
    similarity_threshold=0.30,
    pixel_change_threshold=25,
    enable_semantic_filtering=True,  # Uses embeddings if available
    enable_analysis=True,
    enable_plots=True
)
```

#### Using Advanced NLP Features Directly

```python
from src.prompt_processing import (
    AdvancedLinguisticParser,
    SemanticFilter,
    TextGenerator,
    PromptParser
)

# Advanced linguistic parsing
parser = AdvancedLinguisticParser(model_name="en_core_web_sm")
parsed = parser.parse_prompts(["adding ingredients to the sandwich"])
print(parsed['aggregated']['semantic_roles'])  # Subject-verb-object relations

# Semantic filtering with embeddings
semantic_filter = SemanticFilter(model_name="all-MiniLM-L6-v2")
semantic_filter.encode_prompts(["adding ingredients"])
filtered = semantic_filter.filter_captions_semantic(
    captions,
    similarity_threshold=0.3
)

# Text generation and summarization
text_generator = TextGenerator()
summarized = text_generator.summarize_captions(captions)
alternatives = text_generator.generate_alternative_descriptions("a person cooking")
```

See `example_advanced_nlp.py` for complete examples of all NLP features.

## How It Works

The pipeline consists of seven main stages:

### 1. Motion Detection
Analyzes the video to detect segments with significant motion using pixel-level frame differences. A statistical threshold (75th percentile) is automatically computed to identify dynamic portions of the video.

**Algorithm**: Frame difference analysis
- Computes absolute difference between consecutive grayscale frames
- Counts pixels where difference exceeds threshold
- Identifies segments where motion exceeds statistical threshold

### 2. Frame Extraction
Extracts center frames from each detected motion segment to serve as representative images for content analysis.

**Strategy**: Center frame extraction (optimal balance between representativeness and efficiency)

### 3. Caption Generation
Uses BLIP (Salesforce/blip-image-captioning-base) to generate descriptive captions for each extracted frame, capturing visual content and context.

**Model**: BLIP-base (990M parameters, fast inference, high quality)

### 4. Prompt Parsing & Semantic Filtering

#### Basic Parsing (Fallback)
- Extracts keywords, verbs, and nouns using regex
- Simple keyword matching for filtering

#### Advanced Parsing (spaCy) - **NEW**
- **POS Tagging**: Part-of-speech tags for all tokens
- **Named Entity Recognition**: Identifies persons, organizations, locations
- **Dependency Parsing**: Syntactic dependency relationships
- **Semantic Role Extraction**: Subject-verb-object relations
- **Action Phrase Extraction**: Verb-object pairs

#### Semantic Filtering

**Keyword-based** (Basic):
- Filters captions based on keyword matches
- Fast but limited to exact matches

**Embedding-based** (Advanced) - **NEW**:
- Uses sentence transformers for semantic similarity
- Understands synonyms and paraphrases
- More accurate matching beyond keywords
- Models: `all-MiniLM-L6-v2` (fast), `all-mpnet-base-v2` (better quality)

### 5. Semantic Matching
Uses CLIP (ViT-B/32) to encode both prompts and captions, then computes cosine similarity scores to identify the most relevant segments.

**Model**: CLIP ViT-B/32 (338MB, optimal speed/accuracy trade-off)

### 6. Analysis & Experiments
Provides detailed quantitative analysis including:
- Similarity score distributions
- Threshold sensitivity analysis
- Temporal distribution of selected segments
- Caption quality metrics
- Pipeline performance summary

### 7. Video Assembly
Selects segments with similarity scores above the threshold and concatenates them into the final montage video.

## Configuration Parameters

### Motion Detection
- `pixel_change_threshold`: Threshold for detecting pixel changes (default: 25)
- `motion_pixel_threshold`: Minimum changed pixels for motion (auto-calculated as 75th percentile)

### Caption Generation
- `model_name`: BLIP model (default: "Salesforce/blip-image-captioning-base")
- `context_prompt`: Optional context for captioning
- `finetuned_model_path`: Path to fine-tuned model (optional)

### Prompt Processing
- `use_advanced_parsing`: Enable spaCy parsing (default: True)
- `use_semantic_filtering`: Enable embedding-based filtering (default: True)
- `spacy_model`: spaCy model name (default: "en_core_web_sm")
- `embedding_model`: Sentence transformer model (default: "all-MiniLM-L6-v2")
- `min_keyword_matches`: Minimum keyword matches for basic filtering (default: 1)
- `semantic_threshold`: Similarity threshold for semantic filtering (default: 0.3)

### CLIP Matching
- `model_name`: CLIP model (default: "ViT-B/32")
- `similarity_threshold`: Minimum similarity score for selection (default: 0.25)

### Text Generation
- `summarization_model`: Model for summarization (default: "facebook/bart-large-cnn")
- Options: `"t5-small"` (fast), `"google/pegasus-xsum"` (abstractive)

## Research Features

### Evaluation Metrics

- **Classification Metrics**: Precision, Recall, F1 Score
- **Coverage Metrics**: Coverage ratio, temporal coverage, segment coverage
- **Diversity Metrics**: Vocabulary size, unique words ratio, caption diversity (entropy-based)
- **Coherence Metrics**: Temporal coherence score, gap duration analysis

### Baseline Methods

- **Random Selection**: Randomly sample N segments
- **Uniform Temporal Sampling**: Uniformly distribute segments across timeline
- **First N Segments**: Select first N segments temporally
- **Motion Intensity-based**: Select segments with highest motion

### Experiments

Run comprehensive experiments:

```bash
python run_experiments.py \
    --video_path video.mp4 \
    --prompts "prompt1" "prompt2" \
    --experiments all \
    --generate_plots \
    --generate_latex
```

**Available Experiments:**
- Baseline comparison
- Ablation study (with/without semantic filtering, different models)
- Threshold sensitivity analysis
- Component contribution analysis

### Visualization Tools

- Publication-quality figures (300 DPI)
- Heatmaps, bar charts, sensitivity plots
- LaTeX table generation for papers
- Temporal distribution visualizations

## Fine-Tuning for Domain-Specific Improvement

The pipeline supports fine-tuning BLIP and CLIP models for domain-specific videos:

**Fine-tuning guide**: follow **[`README_FINETUNING.md`](README_FINETUNING.md)** for dataset preparation, training commands, and how to use fine-tuned checkpoints.

### Quick Fine-Tuning

```bash
# Fine-tune BLIP for better domain-specific captions
python scripts/finetune_models.py \
    --model blip \
    --train_dataset data/processed/dataset.json \
    --epochs 3 \
    --learning_rate 5e-5

# Fine-tune CLIP for better semantic matching
python scripts/finetune_models.py \
    --model clip \
    --train_dataset data/processed/dataset.json \
    --epochs 5 \
    --learning_rate 1e-5
```

**Expected Improvements:**
- BLIP: +15-25% improvement in caption quality on domain-specific videos
- CLIP: +20-30% improvement in matching accuracy
- Overall: Better montage quality for specific domains (cooking, sports, etc.)

See **[`README_FINETUNING.md`](README_FINETUNING.md)** 

## Examples

### Basic Example

```bash
python main.py \
    --video_path input_video.mp4 \
    --prompts "cooking food" "serving dish" "plating meal" \
    --output_path outputs/cooking_montage.mp4
```

### Advanced Example with Custom Thresholds

```bash
python main.py \
    --video_path input_video.mp4 \
    --prompts "adding ingredients" "closing container" \
    --similarity_threshold 0.30 \
    --pixel_change_threshold 30 \
    --output_path outputs/higher_quality_montage.mp4
```

### Using Advanced NLP Features

```python
# See example_advanced_nlp.py for complete examples
python example_advanced_nlp.py
```

## Technical Details

### Models Used

| Component | Model | Size | Speed | Quality |
|-----------|-------|------|-------|----------|
| **Captioning** | BLIP-base | 990M | Fast | High |
| **Matching** | CLIP ViT-B/32 | 338MB | Fast | High |
| **NLP Parsing** | spaCy en_core_web_sm | 20MB | Very Fast | Good |
| **Semantic Filtering** | all-MiniLM-L6-v2 | 80MB | Fast | Good |
| **Summarization** | BART-large-CNN | 1.5GB | Medium | High |

### Performance Considerations

**Memory Usage:**
- Minimum (CPU): ~2GB RAM
- Recommended (GPU): 8GB+ RAM, 2-4GB VRAM
- With all features: ~4-6GB RAM, 3-5GB VRAM

**Speed (on GPU):**
- Motion detection: ~1-5 seconds per minute of video
- Caption generation: ~0.5-2 seconds per frame
- NLP parsing: ~10-50ms per prompt
- Semantic filtering: ~5-20ms per caption
- CLIP matching: ~10-50ms per caption
- Total: ~30-60 seconds per minute of video

**Speed (on CPU):**
- 3-5× slower than GPU
- Still usable for short videos (< 5 minutes)

## Dependencies

### Core Dependencies
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- CLIP (OpenAI's implementation)
- transformers >= 4.30.0
- opencv-python >= 4.8.0
- moviepy >= 1.0.3
- numpy >= 1.21.0
- Pillow >= 9.0.0
- matplotlib >= 3.5.0

### NLP Dependencies
- spacy >= 3.7.0
- sentence-transformers >= 2.2.0
- scikit-learn >= 1.3.0

### Utilities
- tqdm >= 4.65.0
- accelerate >= 0.20.0

See `requirements.txt` for complete list with versions.

## Troubleshooting

### spaCy Model Not Found

```bash
python -m spacy download en_core_web_sm
```

### Out of Memory

- Use smaller models: `en_core_web_sm` instead of `en_core_web_lg`
- Use `all-MiniLM-L6-v2` instead of `all-mpnet-base-v2`
- Use CPU instead of GPU for smaller models
- Reduce batch sizes in fine-tuning

### Slow Performance

- Use GPU for sentence transformers and summarization
- Use smaller models if speed is critical
- Disable advanced features if not needed:
  ```python
  parser = PromptParser(use_advanced_parsing=False, use_semantic_filtering=False)
  ```

### Import Errors

If advanced NLP features are not available, the pipeline automatically falls back to basic parsing. This ensures backward compatibility.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

</details>
