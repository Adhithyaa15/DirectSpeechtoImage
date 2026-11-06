# SPIMSyn — Semantic Branch

**Input:** Speech audio (Whisper features)  
**Output:** CLIP-L like embeddings — `[77 × 768]`  
**Model:** `Speech2ImageModel` — frozen Whisper encoder + cross-attention decoder

This branch is the trained checkpoint used in the combined inference pipeline.

---

## Setup

Update paths in `config.py`:

```python
SPEECH_DIR   = "/path/to/speech_files"
OUTPUT_DIR   = "/path/to/output"
```

---

## Training

**Single GPU:**
```bash
cd SPIMSyn/Semantic_branch
python train.py
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 train.py
```

Checkpoints saved to `OUTPUT_DIR/checkpoints/`. Best model saved as `best_model.pt`.

To resume:
```python
# in config.py
resume_checkpoint = "{checkpoint path}"
```

---

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and paths |
| `model.py` | `Speech2ImageModel` — Whisper encoder + cross-attention decoder |
| `dataset.py` | Loads `.wav` + computes CLIP-L targets on-the-fly |
| `train.py` | Training loop with DDP support |
| `losses.py` | MSE + Cosine + CrossEntropy + Contrastive loss |
| `load_centroids.py` | K-means centroid loading utilities |
