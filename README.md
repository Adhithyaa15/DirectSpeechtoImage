# Direct Speech-to-Image Synthesis

Generate images directly from speech audio using a single unified checkpoint — no text input required at inference time.

---

## Overview

This project trains neural models that map speech audio directly to the embedding space of Stable Diffusion 3.5, enabling speech-driven image generation. The model learns to predict the CLIP and T5 conditioning embeddings used by SD3.5's MMDiT transformer purely from the acoustic signal.

**At inference, only an audio `.wav` file is needed.**

---

## Project Structure

```
Direct_Speech2Image/
├── run_inference.py                  # Run inference from a .wav file
├── load_pipeline.py                  # Pipeline bootstrap loader
├── requirements.txt                  # Python dependencies
├── data_samples/                     # Sample audio + caption pairs
│   ├── speechs/                      # .wav files (named {id}_voice{n}_{voice}.wav)
│   └── captions/                     # .txt files (named {id}.txt)
├── SPIMSyn/                          # Whisper_encoder-input models
│   ├── Compositional_branch/         # Whisper_encoder → CLIP-G  (77 × 1280)
│   ├── Linguistic_branch/            # Whisper_encoder → T5      (256 × 4096)
│   └── Semantic_branch/              # Whisper_encoder → CLIP-L  (77 × 768)
└── SPIMSyn-D/                        # DMEL-input models
    ├── Compositional_branch/         # DMEL → CLIP-G     (77 × 1280)
    ├── Linguistic_branch/            # DMEL → T5         (256 × 4096)
    └── Semantic_branch/              # DMEL → CLIP-G     (77 × 1280)
```

**SPIMSyn** uses Whisper encoder features as input.  
**SPIMSyn-D** uses DMEL (discrete mel) embeddings as input.

Each branch targets a different SD3.5 conditioning encoder:
- **Semantic** — CLIP-L (low-level visual semantics)
- **Compositional** — CLIP-G (compositional structure)
- **Linguistic** — T5 (linguistic / caption-level semantics)

---

## Quick Start — Inference

```python
import torch
from load_pipeline import load

pipe = load("combined_speech2image_checkpoint.pt")
pipe = pipe.to("cuda")

image = pipe("audio.wav").images[0]
image.save("output.png")
```

Or via command line:

```bash
python run_inference.py
```

> Edit `run_inference.py` to change the audio path and output path.

---

## Installation

```bash
pip install -r requirements.txt
```

> Requires CUDA 11.8. For other CUDA versions, replace the `cu118` packages accordingly.

---

## Data

### Dataset

Training data is sourced from [LAION-5B](https://laion.ai/blog/laion-5b/). Each sample consists of:
- An image caption (`.txt`)
- A synthesised speech file (`.wav`) generated from the caption

### Sample Data

`data_samples/` contains a small set of paired samples for quick testing:

```
data_samples/
├── speechs/
│   ├── 58230899_voice3_Madhur.wav    # "Crown tart with red fruits"
│   ├── 22913992_voice2_Mohan.wav
│   └── ...
└── captions/
    ├── 58230899.txt
    ├── 22913992.txt
    └── ...
```

File naming: `{sample_id}_voice{index}_{voice_name}.wav`

---

## Generating Speech Data

Speech was generated from LAION captions using **Microsoft Edge TTS** via the `edge-tts` Python library.

> **Important:** Speech generation requires internet access. Run on a **login node**, not a compute node.

**Voices used (Indian/diverse English accents):**

| Index | Voice | Accent |
|-------|-------|--------|
| 0 | `en-IN-NeerjaNeural` | Indian English, female |
| 1 | `en-IN-PrabhatNeural` | Indian English, male |
| 2 | `te-IN-MohanNeural` | Telugu, male |
| 3 | `hi-IN-MadhurNeural` | Hindi, male |
| 4 | `hi-IN-SwaraNeural` | Hindi, female |
| 5 | `en-US-AriaNeural` | US English, female |
| 6 | `en-US-DavisNeural` | US English, male |
| 7 | `en-GB-SoniaNeural` | British, female |
| 8 | `en-AU-NatashaNeural` | Australian, female |
| 9 | `en-US-JennyNeural` | US English, female |

Each caption is spoken at ~240 WPM to keep audio duration around 7 seconds. Output files are saved as `{sample_id}_voice{index}_{voice_name}.wav`.


## Training

See the README inside each branch folder for training instructions:

- [SPIMSyn/Compositional_branch](SPIMSyn/Compositional_branch/README.md)
- [SPIMSyn/Linguistic_branch](SPIMSyn/Linguistic_branch/README.md)
- [SPIMSyn/Semantic_branch](SPIMSyn/Semantic_branch/README.md)
- [SPIMSyn-D/Compositional_branch](SPIMSyn-D/Compositional_branch/README.md)
- [SPIMSyn-D/Linguistic_branch](SPIMSyn-D/Linguistic_branch/README.md)
- [SPIMSyn-D/Semantic_branch](SPIMSyn-D/Semantic_branch/README.md)

---

## Checkpoint

The combined checkpoint (`combined_speech2image_checkpoint.pt`) bundles:

| Component | Description |
|-----------|-------------|
| Model | Our Trained Speech2Image model weights |
| Audio processor | Whisper model encoder weights + tokenizer |
| Transformer | SD3.5 MMDiT weights |
| VAE | SD3.5 decoder weights |
| Scheduler | FlowMatchEulerDiscreteScheduler config |

please download the checkpoint at "https://drive.google.com/file/d/1Psid1b5y2eGd1uWcPPzPXhD0u5G89NjK/view?usp=sharing"

