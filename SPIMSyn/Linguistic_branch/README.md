# SPIMSyn — Linguistic Branch

**Input:** Speech audio (Whisper features)  
**Output:** T5 like embeddings — `[256 × 4096]`  
**Model:** `Speech2T5Model` — frozen Whisper encoder + cross-attention decoder

---

## Setup

Update paths in `config.py`:

```python
SPEECH_DIR       = "/path/to/speech_files"
OUTPUT_DIR       = "/path/to/output"
```

---

## Training

**Single GPU:**
```bash
cd SPIMSyn/Linguistic_branch
python train.py
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 train.py
```

Checkpoints are saved to `OUTPUT_DIR/checkpoints/` every `SAVE_INTERVAL` epochs.

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
| `model.py` | `Speech2T5Model` — Whisper encoder + MLP projection + decoder |
| `dataset.py` | Loads `.wav` files and computes T5 targets on-the-fly |
| `train.py` | Training loop with DDP support |
| `losses.py` | MSE + Cosine + CrossEntropy + Contrastive loss |
| `load_centroids.py` | K-means centroid loading utilities |
