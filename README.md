## 🎙️ Direct Speech-to-Image Synthesis via Cross-Modal Latent Transformers
<div align="center">

[Adhithyaa Natarajan](https://dl.acm.org/doi/10.1145/3767308.3835860#), [Balamurugan Palaniappan](https://dl.acm.org/doi/10.1145/3767308.3835860#)

**Accepted in ACM MM'26**

[![Paper](https://img.shields.io/badge/Paper-ACM_DL-red.svg)](https://dl.acm.org/doi/epdf/10.1145/3767308.3835860) [![Project Page](https://img.shields.io/badge/Project_Page-Visit-green.svg)](https://adhithyaa15.github.io/Speech2Image/)


</div>

## 📋 Table of Contents
- [🎯 Overview](#-overview)
- [📁 Project Structure](#-project-structure)
- [🔧 Environment Setup](#-environment-setup)
- [💾 Checkpoint](#-checkpoint)
- [⚡ Quick Start](#-quick-start)
- [📊 Data](#-data)
- [🎤 Generating Speech Data](#-generating-speech-data)
- [🚀 Training](#-training)
- [📝 Citation](#-citation)


## 🎯 Overview

This project aims to map speech audio directly to the embedding space of Stable Diffusion 3.5, enabling speech-driven image generation. The model learns to predict the CLIP and T5 conditioning embeddings used by SD3.5's MMDiT transformer purely from the acoustic signal.

> **At inference, only an audio `.wav` file is needed.**

Two model families are provided:

| Model | Input features |
|-------|----------------|
| **SPIMSyn** | Whisper encoder features |
| **SPIMSyn-D** | DMEL (discrete mel) embeddings |

Each family has three branches, one per SD3.5 conditioning encoder:

| Branch | Target encoder | Role |
|--------|----------------|------|
| **Semantic** | CLIP-L | Low-level visual semantics |
| **Compositional** | CLIP-G | Compositional structure |
| **Linguistic** | T5 | Linguistic / caption-level semantics |

<!--
TODO: add an architecture figure
<div align="center">
  <img src="/Figures/architecture.png" alt="Architecture" width="800"/>
  <p><em>Direct Speech-to-Image Framework Architecture</em></p>
</div>
-->

## 📁 Project Structure

```
DirectSpeechtoImage/
├── run_inference.py                  # Run inference from a .wav file
├── load_pipeline.py                  # Pipeline bootstrap loader
├── requirements.txt                  # Python dependencies
├── data_samples/                     # Sample audio + caption pairs
│   ├── speechs/                      # .wav files (named {id}_voice{n}_{voice}.wav)
├── SPIMSyn/                          # Whisper_encoder-input models
│   ├── Compositional_branch/         # Whisper_encoder → CLIP-G  (77 × 1280)
│   ├── Linguistic_branch/            # Whisper_encoder → T5      (256 × 4096)
│   └── Semantic_branch/              # Whisper_encoder → CLIP-L  (77 × 768)
└── SPIMSyn-D/                        # DMEL-input models
    ├── Compositional_branch/         # DMEL → CLIP-G     (77 × 1280)
    ├── Linguistic_branch/            # DMEL → T5         (256 × 4096)
    └── Semantic_branch/              # DMEL → CLIP-G     (77 × 1280)
```

## 🔧 Environment Setup

### Prerequisites

* **CUDA-compatible GPU**
* **CUDA 11.8** (for other CUDA versions, replace the `cu118` packages in `requirements.txt` accordingly)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Adhithyaa15/DirectSpeechtoImage.git
cd DirectSpeechtoImage

# 2. Install dependencies
pip install -r requirements.txt
```

## 💾 Checkpoint

Download the combined checkpoint `combined_speech2image_checkpoint.pt`:

| File | Download Link |
|------|---------------|
| **combined_speech2image_checkpoint.pt** | [🔗 Google Drive](https://drive.google.com/file/d/1Psid1b5y2eGd1uWcPPzPXhD0u5G89NjK/view?usp=sharing) |

It bundles everything needed for inference:

| Component | Description |
|-----------|-------------|
| **Model** | Our trained Speech2Image model weights |
| **Audio processor** | Whisper model encoder weights + tokenizer |
| **Transformer** | SD3.5 MMDiT weights |
| **VAE** | SD3.5 decoder weights |
| **Scheduler** | FlowMatchEulerDiscreteScheduler config |

## ⚡ Quick Start

### Python (run_inference.py)

```python
import torch
from load_pipeline import load

pipe = load("combined_speech2image_checkpoint.pt")
pipe = pipe.to("cuda")

image = pipe("audio.wav").images[0]
image.save("output.png")
```

### Command line

```bash
python run_inference.py
```

> Edit `run_inference.py` to change the audio path and output path.

## 📊 Data

### Dataset

Training data is sourced from LAION-5B. Each sample consists of an image caption (`.txt`) and a synthesised speech file (`.wav`) generated from that caption.

| Dataset | Description | Link |
|---------|-------------|------|
| **LAION-5B** | Large-scale image–caption pairs | [🔗 LAION](https://laion.ai/blog/laion-5b/) |

### Sample Data

`data_samples/` contains samples for quick testing:

```
data_samples/
├── speechs/
│   ├── 58230899_voice3_Madhur.wav    # "Crown tart with red fruits"
│   ├── 22913992_voice2_Mohan.wav
│   └── ...
```

File naming: `{sample_id}_voice{index}_{voice_name}.wav`

## 🎤 Generating Speech Data

Speech was generated from LAION captions using **Microsoft Edge TTS** via the `edge-tts` Python library.

### Voices

| Index | Voice | Gender |
|-------|-------|--------|
| 0 | `en-IN-NeerjaNeural` | Female |
| 1 | `en-IN-PrabhatNeural` | Male |
| 2 | `te-IN-MohanNeural` | Male |
| 3 | `hi-IN-MadhurNeural` | Male |
| 4 | `hi-IN-SwaraNeural` | Female |
| 5 | `en-US-AriaNeural` | Female |
| 6 | `en-US-DavisNeural` | Male |
| 7 | `en-GB-SoniaNeural` | Female |
| 8 | `en-AU-NatashaNeural` | Female |
| 9 | `en-US-JennyNeural` | Female |

##### ✅ Notes:

* Each caption is spoken at ~240 WPM to keep audio duration around 7 seconds.
* Output files are saved as `{sample_id}_voice{index}_{voice_name}.wav`.

## 🚀 Training

Training instructions is in the README inside each branch folder for both pipelines:

| Model | Branch | Mapping |
|-------|--------|---------|
| **SPIMSyn** | Semantic | Whisper encoder → CLIP-L |
| **SPIMSyn** | Compositional | Whisper encoder → CLIP-G |
| **SPIMSyn** | Linguistic | Whisper encoder → T5 |
| **SPIMSyn-D** | Semantic | DMEL → CLIP-G | 
| **SPIMSyn-D** | Compositional | DMEL → CLIP-G | 
| **SPIMSyn-D** | Linguistic | DMEL → T5 | 

## 📝 Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{10.1145/3767308.3835860,
author = {Natarajan, Adhithyaa and Palaniappan, Balamurugan},
title = {Direct Speech-to-Image Synthesis via Cross-Modal Latent Transformers},
year = {2026},
isbn = {9798400722134},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3767308.3835860},
doi = {10.1145/3767308.3835860},
booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
pages = {7826–7835},
numpages = {10},
keywords = {speech to image generation, text to image generation, stable diffusion, multimodal learning, latent transformer},
location = {Rio de Janeiro, Brazil},
series = {MM '26}
}
```


<div align="center">

**⭐ Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/Adhithyaa15/DirectSpeechtoImage.svg?style=social&label=Star)](https://github.com/Adhithyaa15/DirectSpeechtoImage)
[![GitHub forks](https://img.shields.io/github/forks/Adhithyaa15/DirectSpeechtoImage.svg?style=social&label=Fork)](https://github.com/Adhithyaa15/DirectSpeechtoImage/fork)

</div>
