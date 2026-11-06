import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from transformers import AutoTokenizer, T5EncoderModel
import warnings
import json
import random
warnings.filterwarnings('ignore')

# Set to use local files only (offline mode)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


class Speech2T5Dataset(Dataset):
    """
    Dataset that loads pre-computed DMEL embeddings [T, 80] and computes
    T5 text embeddings on-the-fly from captions.
    """

    # Shared T5 encoder instance (class variable — one copy per process)
    _shared_t5_encoder = None

    def __init__(
        self,
        dmel_dir,
        caption_dir,
        t5_encoder_path,
        t5_tokenizer_path,
        device='cuda',
        max_samples=None,
        index_file=None,
        split='train'
    ):
        """
        Args:
            dmel_dir:         Path to DMEL .npy files  (e.g. .../dmel_embeddings/dmel/)
            caption_dir:      Path to caption .txt files
            t5_encoder_path:  Path to SD3.5 T5 encoder
            t5_tokenizer_path:Path to SD3.5 T5 tokenizer
            device:           Device for T5 encoder ('cuda' or 'cpu')
            max_samples:      Max samples to use (ignored when index_file is provided)
            index_file:       Path to dataset index JSON (fixed reproducible split)
            split:            'train' or 'val'
        """
        self.dmel_dir = Path(dmel_dir)
        self.caption_dir = Path(caption_dir)
        self.split = split
        self.device_name = device

        # Paths stored for lazy loading in worker processes
        self.t5_encoder_path = t5_encoder_path
        self.t5_tokenizer_path = t5_tokenizer_path
        self._t5_tokenizer = None   # lazy-loaded per worker

        # Load dataset pairs
        if index_file and Path(index_file).exists():
            self.all_pairs = self._load_from_index(index_file, split)
            self.valid_pairs = self.all_pairs
            print(f"Dataset ({split}) loaded from index: {len(self.all_pairs)} samples")
        else:
            if index_file:
                print(f"⚠️  Index file not found: {index_file}")
                print(f"⚠️  Falling back to auto-discovery (non-reproducible)")
            self.all_pairs = self._find_valid_pairs(max_samples=None)
            self.valid_pairs = self.all_pairs
            print(f"Dataset ({split}) initialized with {len(self.all_pairs)} pairs")

    # ------------------------------------------------------------------
    # Pair discovery
    # ------------------------------------------------------------------

    def _load_from_index(self, index_file, split):
        with open(index_file, 'r') as f:
            index = json.load(f)

        if split not in ['train', 'val']:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")

        pairs = index[split]
        valid_pairs = []
        for pair in pairs:
            file_id = pair['id']
            dmel_path = self.dmel_dir / f"{file_id}.npy"
            caption_path = self.caption_dir / f"{file_id}.txt"
            valid_pairs.append({
                'id': file_id,
                'dmel_path': str(dmel_path),
                'caption_path': str(caption_path)
            })
        return valid_pairs

    def _find_valid_pairs(self, max_samples):
        valid_pairs = []
        caption_files = list(self.caption_dir.glob("*.txt"))
        total_files = len(caption_files)

        if max_samples and max_samples < total_files:
            random.seed(42)
            caption_files = random.sample(caption_files, max_samples)
            print(f"📊 Randomly sampled {max_samples} from {total_files} total (seed=42)")

        for caption_file in caption_files:
            file_id = caption_file.stem
            dmel_path = self.dmel_dir / f"{file_id}.npy"
            if dmel_path.exists():
                valid_pairs.append({
                    'id': file_id,
                    'dmel_path': str(dmel_path),
                    'caption_path': str(caption_file)
                })
        return valid_pairs

    def resample(self, max_samples=50000):
        """Randomly resample from all_pairs — call at the start of each epoch."""
        self.valid_pairs = random.sample(self.all_pairs, min(max_samples, len(self.all_pairs)))
        print(f"Resampled {len(self.valid_pairs)} samples from {len(self.all_pairs)}")

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def t5_encoder(self):
        """Load T5 encoder once per process and share across dataset instances."""
        if Speech2T5Dataset._shared_t5_encoder is None:
            device = torch.device(self.device_name if torch.cuda.is_available() else 'cpu')
            Speech2T5Dataset._shared_t5_encoder = T5EncoderModel.from_pretrained(
                self.t5_encoder_path,
                local_files_only=True,
                torch_dtype=torch.float32
            ).to(device)
            Speech2T5Dataset._shared_t5_encoder.eval()
            for param in Speech2T5Dataset._shared_t5_encoder.parameters():
                param.requires_grad = False
        return Speech2T5Dataset._shared_t5_encoder

    @property
    def t5_tokenizer(self):
        """Lazy-load tokenizer (one per worker process)."""
        if self._t5_tokenizer is None:
            self._t5_tokenizer = AutoTokenizer.from_pretrained(
                self.t5_tokenizer_path,
                local_files_only=True
            )
        return self._t5_tokenizer

    @property
    def device(self):
        return torch.device(self.device_name if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------------------
    # T5 embedding computation
    # ------------------------------------------------------------------

    def _compute_t5_embedding(self, caption):
        """
        Compute T5 embedding for a caption string on-the-fly.

        Returns:
            t5_embedding: [256, 4096] float32 tensor on CPU
        """
        with torch.no_grad():
            text_inputs = self.t5_tokenizer(
                caption,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            t5_outputs = self.t5_encoder(
                text_inputs.input_ids,
                output_hidden_states=False
            )
            # [1, 256, 4096] -> [256, 4096] on CPU in float32
            t5_embeds = t5_outputs.last_hidden_state.squeeze(0).cpu().float()

        return t5_embeds  # [256, 4096]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        """
        Returns:
            dmel_features:  [T, 80]     raw DMEL (variable T per sample)
            t5_embedding:   [256, 4096] T5 ground-truth embedding
            file_id:        str
        """
        pair = self.valid_pairs[idx]

        # ---- Load DMEL embedding ----
        try:
            dmel = np.load(pair['dmel_path'])          # [T, 80] or [1, T, 80]
            dmel = np.squeeze(dmel)                     # ensure [T, 80]
            if dmel.ndim == 1:
                dmel = dmel[np.newaxis, :]              # edge case: single frame
            dmel_features = torch.from_numpy(dmel).float()  # [T, 80]
        except Exception as e:
            print(f"Error loading DMEL {pair['dmel_path']}: {e}")
            dmel_features = torch.zeros(1, 80)          # fallback

        # ---- Load caption & compute T5 embedding ----
        try:
            with open(pair['caption_path'], 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            t5_embedding = self._compute_t5_embedding(caption)  # [256, 4096]
            if t5_embedding.shape != (256, 4096):
                print(f"Warning: unexpected T5 shape {t5_embedding.shape} for {pair['id']}")
                t5_embedding = torch.zeros(256, 4096)
        except Exception as e:
            print(f"Error computing T5 embedding for {pair['caption_path']}: {e}")
            t5_embedding = torch.zeros(256, 4096)

        return {
            'dmel_features': dmel_features,   # [T, 80]  — variable length
            't5_embedding': t5_embedding,      # [256, 4096]
            'file_id': pair['id']
        }


# ----------------------------------------------------------------------
# Collate: pad dmel along T dimension to the longest in the batch
# ----------------------------------------------------------------------

def collate_fn(batch):
    """
    Pad variable-length DMEL tensors to the longest T in the batch.
    Stack T5 embeddings (fixed shape).
    """
    # Pad dmel_features along time dimension
    max_t = max(item['dmel_features'].shape[0] for item in batch)
    padded_dmel = []
    for item in batch:
        t = item['dmel_features'].shape[0]
        pad_len = max_t - t
        if pad_len > 0:
            padding = torch.zeros(pad_len, 80)
            padded = torch.cat([item['dmel_features'], padding], dim=0)
        else:
            padded = item['dmel_features']
        padded_dmel.append(padded)

    dmel_features = torch.stack(padded_dmel)                         # [B, T_max, 80]
    t5_embeddings = torch.stack([item['t5_embedding'] for item in batch])  # [B, 256, 4096]
    file_ids = [item['file_id'] for item in batch]

    return {
        'dmel_features': dmel_features,
        't5_embeddings': t5_embeddings,
        'file_ids': file_ids
    }


# ----------------------------------------------------------------------
# Dataloader factory
# ----------------------------------------------------------------------

def create_dataloaders(
    dmel_dir,
    caption_dir,
    t5_encoder_path,
    t5_tokenizer_path,
    device='cuda',
    batch_size=8,
    num_workers=0,
    max_samples=50000,
    train_split=0.98,
    use_distributed=False,
    rank=0,
    world_size=1,
    index_file=None
):
    from torch.utils.data.distributed import DistributedSampler

    if index_file and Path(index_file).exists():
        if rank == 0:
            print(f"\n✓ Using fixed dataset index: {index_file}")

        train_dataset = Speech2T5Dataset(
            dmel_dir=dmel_dir,
            caption_dir=caption_dir,
            t5_encoder_path=t5_encoder_path,
            t5_tokenizer_path=t5_tokenizer_path,
            device=device,
            index_file=index_file,
            split='train'
        )
        val_dataset = Speech2T5Dataset(
            dmel_dir=dmel_dir,
            caption_dir=caption_dir,
            t5_encoder_path=t5_encoder_path,
            t5_tokenizer_path=t5_tokenizer_path,
            device=device,
            index_file=index_file,
            split='val'
        )

    else:
        # Auto-discovery + random split (non-reproducible)
        if index_file and rank == 0:
            print(f"\n⚠️  Index file not found: {index_file}")
            print(f"⚠️  Using auto-discovery + random split")

        caption_files = list(Path(caption_dir).glob("*.txt"))
        all_pairs = []
        for caption_file in caption_files:
            file_id = caption_file.stem
            dmel_path = Path(dmel_dir) / f"{file_id}.npy"
            if dmel_path.exists():
                all_pairs.append({
                    'id': file_id,
                    'dmel_path': str(dmel_path),
                    'caption_path': str(caption_file)
                })

        random.seed(42)
        random.shuffle(all_pairs)
        train_size = int(train_split * len(all_pairs))
        train_pairs = all_pairs[:train_size]
        val_pairs = all_pairs[train_size:]

        def _make_dataset(pairs, split_name):
            ds = Speech2T5Dataset.__new__(Speech2T5Dataset)
            ds.dmel_dir = Path(dmel_dir)
            ds.caption_dir = Path(caption_dir)
            ds.split = split_name
            ds.device_name = device
            ds.t5_encoder_path = t5_encoder_path
            ds.t5_tokenizer_path = t5_tokenizer_path
            ds._t5_tokenizer = None
            ds.all_pairs = pairs
            ds.valid_pairs = pairs
            return ds

        train_dataset = _make_dataset(train_pairs, 'train')
        val_dataset   = _make_dataset(val_pairs,   'val')

        if rank == 0:
            print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    # Distributed samplers
    if use_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                           rank=rank, shuffle=True,  seed=42)
        val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size,
                                           rank=rank, shuffle=False, seed=42)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


# ----------------------------------------------------------------------
# Quick smoke-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    dmel_dir      = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/dmel_embeddings/dmel"
    caption_dir   = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/captions"
    t5_encoder_path   = "/scratch/IITB/ai-at-ieor/20004596/sd35/text_encoder_3"
    t5_tokenizer_path = "/scratch/IITB/ai-at-ieor/20004596/sd35/tokenizer_3"

    dataset = Speech2T5Dataset(
        dmel_dir=dmel_dir,
        caption_dir=caption_dir,
        t5_encoder_path=t5_encoder_path,
        t5_tokenizer_path=t5_tokenizer_path,
        device='cuda',
        max_samples=10
    )

    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"DMEL features shape : {sample['dmel_features'].shape}")
        print(f"T5 embedding shape  : {sample['t5_embedding'].shape}")
        print(f"File ID             : {sample['file_id']}")