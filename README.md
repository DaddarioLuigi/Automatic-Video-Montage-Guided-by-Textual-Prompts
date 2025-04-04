# Automatic Video Summarization using CLIP and Motion Detection

## Overview

This repository provides a pipeline for automatic video summarization that extracts salient segments from videos—particularly cooking videos—by combining semantic filtering with OpenAI's CLIP and motion detection using frame differencing. The system is designed to work with videos stored on Google Drive and outputs a concise, dynamic final montage.

## Features

- **Motion Detection:** Filters out static scenes by analyzing frame-to-frame differences.
- **Semantic Filtering with CLIP:** Uses OpenAI's CLIP model to compute cosine similarity between video frames and a user-defined text prompt.
- **Segment Extraction and Merging:** Automatically extracts video segments around key frames and merges segments that are temporally close.
- **Duration Limiting:** Option to specify a maximum total duration for the final summary.
- **Flexible Output Formats:** Supports both horizontal and vertical (9:16) video formats.
- **Google Drive Integration:** Easily process videos stored in Google Drive.

## How It Works

1. **Frame Sampling and Preprocessing:**  
   The video is sampled at a fixed rate. Each frame is converted to grayscale for motion detection and preprocessed for CLIP if semantic analysis is enabled.

2. **Dual Filtering:**  
   For each frame, the system computes:
   - The **motion magnitude** \(M_t\) by comparing consecutive frames.
   - The **cosine similarity** between the frame and the provided text prompt using CLIP (if enabled).  
   A frame is selected as a candidate if it exceeds both the motion threshold and, when applicable, the CLIP similarity threshold.

3. **Segment Extraction and Merging:**  
   Selected frames are expanded into segments of a specified duration (\(\Delta t\)). If segments are close in time, they are merged to maintain temporal continuity.

4. **Final Assembly:**  
   The selected segments are concatenated using MoviePy to produce the final video summary, with an optional limit on total duration.

## Installation

### Prerequisites

- Python 3.7+
- Google Colab (recommended) for easy integration with Google Drive
- Required Libraries: OpenCV, MoviePy, OpenAI CLIP, and optionally OpenAI Whisper

### Steps

Clone the repository and install the dependencies:
```bash
git clone https://github.com/your_username/your_repo.git
cd your_repo
pip install -r requirements.txt
```

Alternatively, if you are using Google Colab, run the following in a cell:
```python
!pip install -q opencv-python moviepy openai-whisper
!pip install -q git+https://github.com/openai/CLIP.git
!apt-get -qq install -y ffmpeg
```

## Setup

1. **Mount Google Drive:**  
   If your videos are stored on Google Drive, mount your drive using:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Configure Folders:**  
   Set the input folder (where your videos are located) and the output folder (where the summarized video will be saved) in the configuration parameters of the code.

## Usage

1. **Configure Parameters:**  
   Edit the configuration parameters in the provided Python script or Colab notebook:
   - **Text Prompt:** Specify a prompt (e.g., "while cooking") or leave it blank to use only motion detection.
   - **Thresholds:** Adjust the motion threshold (\(\tau_{\text{motion}}\)) and CLIP similarity threshold (\(\tau_{\text{clip}}\)).
   - **Clip Duration and Merging Gap:** Set the duration for each extracted segment and the merging gap.
   - **Output Format:** Choose whether to force the final output into a vertical (9:16) format.
   
2. **Run the Pipeline:**  
   Execute the script or notebook. The system will process each video file, extract candidate segments, merge them as needed, and generate the final video summary.

3. **Output:**  
   The final video is saved in the specified output folder on Google Drive.

## Theoretical Background

### CLIP and Contrastive Learning

CLIP (Contrastive Language–Image Pretraining) maps images and text into a shared embedding space using a dual encoder architecture. For a given image \( x \) and text prompt \( y \), the cosine similarity is computed as:
\[
\text{sim}(x, y) = \frac{f(x) \cdot g(y)}{\|f(x)\| \, \|g(y)\|}
\]
This similarity measure is key for determining the semantic relevance of a frame to the provided text prompt.

### Motion Detection

Motion detection is performed by comparing consecutive frames. Let \( F_t \) and \( F_{t-1} \) be the grayscale frames at times \( t \) and \( t-1 \), respectively. The difference is computed as:
\[
D_t(i,j) = \left| F_t(i,j) - F_{t-1}(i,j) \right|
\]
The motion magnitude \( M_t \) is calculated by:
\[
M_t = \sum_{i,j} \mathbb{1}\{ D_t(i,j) > \delta \}
\]
A frame is considered dynamic if \( M_t > \tau_{\text{motion}} \).

## CLIP Architecture

This system leverages the CLIP model as described in [Understanding OpenAI's CLIP Model](https://medium.com/@paluchasz/understanding-openais-clip-model-6b52bade3fa3). The architecture consists of:

- **Image Encoder:** Processes the input image using a CNN or Vision Transformer (ViT) to generate a feature vector.
- **Text Encoder:** Processes tokenized text using a Transformer to generate a corresponding feature vector.

Both encoders output embeddings in a shared space, where cosine similarity is used to compare their semantic content.

### Referenced Figures

- **Figure 1: CLIP Architecture Diagram**  
  *Place an image named `clip_architecture.png` here. This image should illustrate the dual encoder structure (image encoder and text encoder) and their shared embedding space.*

- **Figure 2: CLIP Flow Diagram**  
  *Place an image named `clip_flow.png` here. This image should depict the contrastive training process, showing how image-text pairs are aligned.*

## References

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., et al. (2021). [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
- Paluchasz, P. (n.d.). [Understanding OpenAI's CLIP Model](https://medium.com/@paluchasz/understanding-openais-clip-model-6b52bade3fa3).
- Truong, B. T., & Venkatesh, S. (2007). [Video Summarization: A Survey](https://dl.acm.org/doi/10.1145/1214999.1215001). ACM Transactions on Multimedia Computing, Communications, and Applications, 3(1).
- Mahasseni, B., Lam, M., & Saquib, M. S. (2017). [Unsupervised Video Summarization with Adversarial LSTM Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Mahasseni_Unsupervised_Video_Summarization_CVPR_2017_paper.html).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with any enhancements or bug fixes.

## Contact

For any questions or feedback, please contact [luigi@luxdada.it](mailto:luigi@luxdada.it).
```

Feel free to adjust the file paths, contact details, and any parameters to fit your project specifics.
