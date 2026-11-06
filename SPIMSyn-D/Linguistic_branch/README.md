# SPIMSyn-D — Linguistic Branch

**Input:** DMEL embeddings `[T × 80]`  
**Output:** T5 like embeddings — `[256 × 4096]`  
**Model:** DmelInputProjection → cross-attention decoder

---

## Setup

Update paths in `config.py`:

```python
DMEL_DIR    = "/path/to/dmel_embeddings"
OUTPUT_DIR  = "/path/to/output"
```

T5 embeddings are computed on-the-fly from captions using the SD3.5 T5 encoder.

---

## Training

**Single GPU:**
```bash
cd SPIMSyn-D/Linguistic_branch
python train.py
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 train.py
```

**Epoch-based resampling:** Fixed 50k for epochs 1–50, fresh random 50k per epoch thereafter from the full ~265k pool.

---

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and paths |
| `model.py` | DMEL-input model targeting T5 dimension (4096) |
| `dataset.py` | Loads DMEL + computes T5 targets on-the-fly |
| `train.py` | Training loop with epoch-aware resampling |
| `losses.py` | MSE + Cosine + CrossEntropy + Contrastive loss |
| `load_centroids.py` | K-means centroid loading utilities |
