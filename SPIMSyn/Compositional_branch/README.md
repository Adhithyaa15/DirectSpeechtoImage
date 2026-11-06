# SPIMSyn — Compositional Branch

**Input:** Speech audio (Whisper features)  
**Output:** CLIP-G like embeddings — `[77 × 1280]`  
**Model:** `Speech2ClipGModel` — frozen Whisper encoder + cross-attention decoder

---

## Setup

Update paths in `config.py`:

```python
SPEECH_DIR      = "/path/to/speech_files"       # .wav files
CAPTION_DIR     = "/path/to/caption_files"       # .txt files
CLIP_G_ENCODER_PATH = "/path/to/sd35"            # SD3.5 model dir
OUTPUT_DIR      = "/path/to/output"
```

---

## Training

**Single GPU:**
```bash
cd SPIMSyn/Compositional_branch
python train.py
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 train.py
```

Checkpoints are saved to `OUTPUT_DIR/checkpoints/` every `SAVE_INTERVAL` epochs. The best model is always saved as `best_model.pt`.

To resume from a checkpoint, add to `config.py`:
```python
resume_checkpoint = "{checkpoint path}"
```

---

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and paths |
| `model.py` | `Speech2ClipGModel` — Whisper encoder + cross-attention decoder |
| `dataset.py` | Loads `.wav` files and computes CLIP-G targets on-the-fly |
| `train.py` | Training loop with DDP support |
| `losses.py` | MSE + Cosine + CrossEntropy + Contrastive loss |
| `load_centroids.py` | K-means centroid loading utilities |
