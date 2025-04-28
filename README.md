# Automatic Video Montage Guided by Textual Prompts

This project implements an **automatic video montage tool** that selects and edits video clips based on **textual prompts**.  
It analyzes video content (optionally also audio), matches it with the given prompt, and intelligently constructs a final montage, enhancing creativity and saving editing time.

## Features

- **Text-to-video matching**: Selects clips based on how well they match a textual prompt.
- **Speech analysis** (optional): Considers spoken words in videos when matching.
- **Automatic montage creation**: Generates a final video sequence based on relevance scores.
- **Customizable parameters**: Adjust number of clips, clip length, and scoring thresholds.
- **Lightweight and easy to extend**.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/automatic-video-montage.git
   cd automatic-video-montage
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *(You'll need packages like `moviepy`, `transformers`, `torch`, and optionally `whisper` for speech analysis.)*

## Usage

```bash
python automatic_video_montage_guided_by_textual_prompts.py --input_dir <path_to_your_clips> --prompt "<your_text_prompt>" --output_path <output_video_path>
```

### Example

```bash
python automatic_video_montage_guided_by_textual_prompts.py --input_dir ./videos --prompt "sunset at the beach" --output_path ./final_montage.mp4
```

### Important arguments

- `--input_dir`: Folder containing the input video clips.
- `--prompt`: Text prompt guiding the montage (e.g., "people smiling", "city at night").
- `--output_path`: Where the final montage will be saved.
- `--use_audio`: (optional) If set, the model also uses the audio track to improve matching.
- `--clip_duration`: (optional) Duration of selected clips in seconds.
- `--number_of_clips`: (optional) How many clips to select for the final video.

## How It Works

1. **Clip Extraction**: Splits input videos into short clips.
2. **Content Encoding**: Encodes video frames (and optionally audio) using pretrained models.
3. **Prompt Encoding**: Encodes the input textual prompt.
4. **Similarity Scoring**: Calculates similarity between prompt and clips.
5. **Selection and Montage**: Selects top-matching clips and assembles them into a smooth video.

## Requirements

- Python 3.8+
- `moviepy`
- `transformers`
- `torch`
- `whisper` (if using speech-to-text)
- `scikit-learn`
- `numpy`
- `opencv-python`

You can install them via:

```bash
pip install moviepy transformers torch scikit-learn numpy opencv-python
```

*(Optional for audio analysis)*

```bash
pip install git+https://github.com/openai/whisper.git 
```

## Notes

- For better accuracy, you can fine-tune the model thresholds for your specific dataset.
- GPU acceleration is recommended for faster video processing and prompt matching.

## Future Improvements

- Add support for smoother transitions (crossfade).
- Implement dynamic clip selection based on prompt complexity.
- Allow background music addition.

## License

This project is licensed under the [MIT License](LICENSE).
