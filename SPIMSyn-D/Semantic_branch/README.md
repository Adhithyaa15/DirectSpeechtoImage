# SPIMSyn-D — Semantic Branch

**Input:** DMEL embeddings `[T × 80]`  
**Output:** CLIP-G like embeddings — `[77 × 1280]`  
**Model:** DmelInputProjection + cross-attention decoder (semantic alignment variant)

---

## Setup

Update paths in `config.py`:

```python
DMEL_DIR    = "/path/to/dmel_embeddings"
OUTPUT_DIR  = "/path/to/output"
```

---

## Training

**Single GPU:**
```bash
cd SPIMSyn-D/Semantic_branch
python train.py
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 train.py
```

---

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and paths |
| `model.py` | `Speech2ImageModel` with DMEL input projection |
| `dataset.py` | Loads DMEL inputs and pre-computed CLIP-G targets |
| `train.py` | Training loop with DDP and epoch-aware resampling |
| `losses.py` | MSE + Cosine + CrossEntropy + Contrastive loss |
| `load_centroids.py` | K-means centroid loading utilities |
