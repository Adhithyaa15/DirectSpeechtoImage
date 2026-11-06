# SPIMSyn-D — Compositional Branch

**Input:** DMEL embeddings `[T × 80]` (discrete mel)  
**Output:** CLIP-G like embeddings — `[77 × 1280]`  
**Model:** `Speech2ImageModel` — DmelInputProjection + cross-attention decoder

---

## Setup

Update paths in `config.py`:

```python
DMEL_DIR    = "/path/to/dmel_embeddings"
OUTPUT_DIR  = "/path/to/output"
```

DMEL embeddings are int8 arrays of shape `[T, 80]` stored as `.npy` files.  
CLIP-G embeddings are pre-computed as `.npy` files of shape `[77, 1280]`.

---

## Training

**Single GPU:**
```bash
cd SPIMSyn-D/Compositional_branch
python train.py
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=4 train.py
```

**Epoch-based resampling:** For the first `FIXED_EPOCHS` (default 50) epochs, a fixed random subset of 50k samples is used. From epoch 51 onwards, a fresh random 50k subset is drawn each epoch from the full ~265k pool.

---

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and paths |
| `model.py` | `Speech2ImageModel` with `DmelInputProjection` |
| `dataset.py` | Loads DMEL inputs and pre-computed CLIP-G targets |
| `train.py` | Training loop with epoch-aware resampling |
| `losses.py` | MSE + Cosine + CrossEntropy + Contrastive loss |
| `load_centroids.py` | K-means centroid loading utilities |
