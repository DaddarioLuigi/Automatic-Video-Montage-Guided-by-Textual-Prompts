# Automatic Video Montage Guided by Textual Prompts

This project implements an automatic video montage system that selects and merges the most relevant scenes from a video based on user-provided natural language prompts. The pipeline combines motion detection, image captioning, and semantic similarity matching using CLIP to identify and assemble video segments that match textual descriptions.

## Features

- **Motion Detection**: Automatically identifies dynamic segments in videos using frame difference analysis
- **Image Captioning**: Generates detailed captions for video frames using BLIP (Bootstrapping Language-Image Pre-training)
- **Semantic Similarity Matching**: Uses CLIP (Contrastive Language-Image Pre-training) to match user prompts with video content
- **Automatic Montage Generation**: Assembles selected video segments into a cohesive final montage

## How It Works

The pipeline consists of four main stages:

1. **Motion Detection**: Analyzes the video to detect segments with significant motion using pixel-level frame differences. A statistical threshold (75th percentile) is automatically computed to identify dynamic portions of the video.

2. **Frame Extraction**: Extracts center frames from each detected motion segment to serve as representative images for content analysis.

3. **Caption Generation**: Uses BLIP image captioning model to generate descriptive captions for each extracted frame, capturing the visual content and context.

4. **Semantic Matching and Montage**: Encodes user-provided textual prompts using CLIP and computes cosine similarity with the generated captions. Segments with similarity scores above a threshold are selected and concatenated into the final montage video.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/video_summarization.git
   cd video_summarization
   ```

2. Install required dependencies:

   ```bash
   pip install git+https://github.com/openai/CLIP.git
   pip install transformers torchvision moviepy opencv-python
   ```

   For memory-efficient inference with 8-bit quantization:

   ```bash
   pip install bitsandbytes
   ```

## Dependencies

- Python 3.8+
- CLIP (OpenAI's implementation)
- transformers (for BLIP models)
- torch and torchvision
- moviepy
- opencv-python
- numpy
- matplotlib
- PIL (Pillow)

## Usage

The project is implemented as a Jupyter notebook (`Automatic_Video_Montage_Guided_by_Textual_Prompts.ipynb`). To use it:

1. Open the notebook in Jupyter or Google Colab
2. Mount your Google Drive (if using Colab) or update the video path
3. Configure the video input path
4. Run the motion detection cells to analyze your video
5. Set your textual prompts describing the scenes you want to include
6. Execute the pipeline to generate the final montage

### Example Prompts

```
prompts = [
    "adding the ingredients in the sandwich",
    "closing the box",
    "plating the dish"
]
```

### Configuration Parameters

- `pixel_change_threshold`: Threshold for detecting pixel changes between frames (default: 25)
- `motion_pixel_threshold`: Minimum number of changed pixels to consider a segment as containing motion (auto-calculated as 75th percentile)
- `similarity_threshold`: Minimum CLIP similarity score for segment selection (default: 0.25)

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
- Custom prompts for context-aware captioning

### Semantic Matching

CLIP (ViT-B/32) is used to compute semantic similarity between:
- User-provided textual prompts
- Generated image captions

Segments with cosine similarity above the threshold are selected for the final montage.

## Requirements

- GPU acceleration is highly recommended for faster processing
- Sufficient memory for loading CLIP and BLIP models (approximately 2-4GB VRAM)
- Video input should be in a standard format (MP4, AVI, etc.)

## Future Improvements

- Support for temporal context in captioning (using previous frames)
- Enhanced prompt customization options
- Automatic threshold optimization
- Support for multiple video inputs
- Transition effects between segments
- Audio track preservation in final montage

## License

This project is licensed under the MIT License.