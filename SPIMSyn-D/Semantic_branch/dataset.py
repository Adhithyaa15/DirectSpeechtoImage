import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import warnings
import json
import random
warnings.filterwarnings('ignore')


class Speech2ImageDataset(Dataset):
    """Dataset for loading dmel embeddings and corresponding CLIP-L embeddings"""
    
    def __init__(
        self,
        dmel_dir,
        clip_l_dir,
        max_samples=None,
        sample_ids=None,       # Explicit list of IDs to use (overrides max_samples)
        index_file=None,
        split='train'
    ):
        """
        Args:
            dmel_dir: Path to dmel embedding files (.npy), shape (T, 80), dtype int8
            clip_l_dir: Path to CLIP-L embeddings directory
            max_samples: Maximum number of samples (used for initial discovery only)
            sample_ids: If provided, use exactly these IDs (for epoch-based resampling)
            index_file: Path to dataset index JSON file (if provided, uses fixed dataset)
            split: 'train' or 'val' (only used when index_file is provided)
        """
        self.dmel_dir = Path(dmel_dir)
        self.clip_l_dir = Path(clip_l_dir)
        self.split = split
        
        if index_file and Path(index_file).exists():
            self.valid_pairs = self._load_from_index(index_file, split)
            print(f"Dataset ({split}) loaded from index: {len(self.valid_pairs)} samples")
        elif sample_ids is not None:
            # Use explicitly provided IDs (epoch-based resampling)
            self.valid_pairs = self._pairs_from_ids(sample_ids)
            print(f"Dataset loaded from provided IDs: {len(self.valid_pairs)} samples")
        else:
            if index_file:
                print(f"⚠️  Index file not found: {index_file}, falling back to auto-discovery")
            self.valid_pairs = self._find_valid_pairs(max_samples)
            print(f"Dataset initialized with {len(self.valid_pairs)} valid pairs")
    
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
            clip_l_path = self.clip_l_dir / f"{file_id}_clip_l.npy"
            valid_pairs.append({
                'id': file_id,
                'dmel_path': str(dmel_path),
                'clip_l_path': str(clip_l_path)
            })
        return valid_pairs

    def _pairs_from_ids(self, sample_ids):
        valid_pairs = []
        for file_id in sample_ids:
            dmel_path = self.dmel_dir / f"{file_id}.npy"
            clip_l_path = self.clip_l_dir / f"{file_id}_clip_l.npy"
            valid_pairs.append({
                'id': file_id,
                'dmel_path': str(dmel_path),
                'clip_l_path': str(clip_l_path)
            })
        return valid_pairs

    def _find_valid_pairs(self, max_samples):
        valid_pairs = []
        clip_files = list(self.clip_l_dir.glob("*_clip_l.npy"))
        total_files = len(clip_files)
        
        if max_samples and max_samples < total_files:
            random.seed(42)
            clip_files = random.sample(clip_files, max_samples)
            print(f"📊 Randomly sampled {max_samples} from {total_files} total (seed=42)")
        
        for clip_file in clip_files:
            file_id = clip_file.stem.replace("_clip_l", "")
            dmel_path = self.dmel_dir / f"{file_id}.npy"
            if dmel_path.exists():
                valid_pairs.append({
                    'id': file_id,
                    'dmel_path': str(dmel_path),
                    'clip_l_path': str(clip_file)
                })
        return valid_pairs
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        
        # Load dmel embedding: (T, 80) int8 -> float32
        try:
            dmel = np.load(pair['dmel_path'])
            dmel_features = torch.from_numpy(dmel.astype(np.float32))  # [T, 80]
        except Exception as e:
            print(f"Error loading dmel {pair['dmel_path']}: {e}")
            dmel_features = torch.zeros(1, 80)
        
        # Load CLIP-L embedding: [77, 768]
        try:
            clip_l_embedding = np.load(pair['clip_l_path'])
            clip_l_embedding = torch.from_numpy(clip_l_embedding).float()
            if clip_l_embedding.shape != (77, 768):
                print(f"Warning: Unexpected CLIP-L shape {clip_l_embedding.shape}")
                clip_l_embedding = torch.zeros(77, 768)
        except Exception as e:
            print(f"Error loading CLIP-L embedding {pair['clip_l_path']}: {e}")
            clip_l_embedding = torch.zeros(77, 768)
        
        return {
            'dmel_features': dmel_features,        # [T, 80]
            'clip_l_embedding': clip_l_embedding,  # [77, 768]
            'file_id': pair['id']
        }


def collate_fn(batch):
    """Pad variable-length dmel sequences to max T in batch."""
    max_t = max(item['dmel_features'].shape[0] for item in batch)
    padded_dmel = []
    for item in batch:
        feat = item['dmel_features']  # [T, 80]
        pad_len = max_t - feat.shape[0]
        if pad_len > 0:
            feat = F.pad(feat, (0, 0, 0, pad_len))  # pad time dim
        padded_dmel.append(feat)
    
    return {
        'dmel_features': torch.stack(padded_dmel),                                       # [B, T_max, 80]
        'clip_l_embeddings': torch.stack([item['clip_l_embedding'] for item in batch]),  # [B, 77, 768]
        'file_ids': [item['file_id'] for item in batch]
    }


def discover_all_valid_ids(dmel_dir, clip_l_dir):
    """
    Discover all IDs that have both a dmel file and a CLIP-L file.
    Returns a sorted list of valid IDs from the full ~265k pool.
    """
    dmel_dir = Path(dmel_dir)
    clip_l_dir = Path(clip_l_dir)
    
    print("Discovering all valid dmel+CLIP-L pairs from full dataset (~265k)...")
    clip_files = list(clip_l_dir.glob("*_clip_l.npy"))
    valid_ids = []
    for clip_file in clip_files:
        file_id = clip_file.stem.replace("_clip_l", "")
        if (dmel_dir / f"{file_id}.npy").exists():
            valid_ids.append(file_id)
    valid_ids.sort()
    print(f"Found {len(valid_ids)} valid pairs in full dataset.")
    return valid_ids


def create_dataloaders(
    dmel_dir,
    clip_l_dir,
    batch_size=8,
    num_workers=4,
    train_split=0.98,
    use_distributed=False,
    rank=0,
    world_size=1,
    # Epoch-based sampling args
    epoch=1,
    fixed_epochs=50,
    max_samples=50000,
    all_valid_ids=None,      # Pass the full pool of IDs (pre-discovered)
    fixed_train_ids=None,    # Fixed train IDs for epochs 1-50 (pre-computed)
    fixed_val_ids=None,      # Fixed val IDs for epochs 1-50 (pre-computed)
):
    """
    Create train and validation dataloaders with epoch-aware sampling:
      - Epochs 1..fixed_epochs: fixed 50k sample (same every epoch)
      - Epochs fixed_epochs+1..200: fresh random 50k sample each epoch (different every epoch)

    Args:
        all_valid_ids: Full list of valid IDs (~265k). Must be provided.
        fixed_train_ids: Fixed train IDs for first 50 epochs. Pre-computed once in main().
        fixed_val_ids: Fixed val IDs for first 50 epochs. Pre-computed once in main().
        epoch: Current epoch number (1-indexed).
        fixed_epochs: Number of epochs to use fixed dataset (default 50).
        max_samples: Number of samples to draw (default 50000).
    """
    from torch.utils.data.distributed import DistributedSampler

    if epoch <= fixed_epochs:
        # Use the pre-fixed split for epochs 1-50
        train_ids = fixed_train_ids
        val_ids = fixed_val_ids
        if rank == 0:
            print(f"  [Dataset] Epoch {epoch}: Using fixed 50k sample "
                  f"({len(train_ids)} train / {len(val_ids)} val)")
    else:
        # Fresh random 50k sample — seed changes every epoch so samples differ
        epoch_seed = 1000 + epoch  # unique seed per epoch
        rng = random.Random(epoch_seed)
        sampled = rng.sample(all_valid_ids, min(max_samples, len(all_valid_ids)))
        train_size = int(train_split * len(sampled))
        train_ids = sampled[:train_size]
        val_ids   = sampled[train_size:]
        if rank == 0:
            print(f"  [Dataset] Epoch {epoch}: Fresh random 50k sample (seed={epoch_seed}), "
                  f"{len(train_ids)} train / {len(val_ids)} val")

    train_dataset = Speech2ImageDataset(
        dmel_dir=dmel_dir, clip_l_dir=clip_l_dir, sample_ids=train_ids
    )
    val_dataset = Speech2ImageDataset(
        dmel_dir=dmel_dir, clip_l_dir=clip_l_dir, sample_ids=val_ids
    )

    if use_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                           rank=rank, shuffle=True, seed=42)
        val_sampler   = DistributedSampler(val_dataset, num_replicas=world_size,
                                           rank=rank, shuffle=False, seed=42)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              sampler=val_sampler, num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader


if __name__ == "__main__":
    dmel_dir   = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/dmel_embeddings/dmel"
    clip_l_dir = "/scratch/IITB/ai-at-ieor/20004596/Speech2Image/embeddings/clip_l"

    dataset = Speech2ImageDataset(dmel_dir=dmel_dir, clip_l_dir=clip_l_dir, max_samples=10)
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Dmel features shape:    {sample['dmel_features'].shape}")
        print(f"CLIP-L embedding shape: {sample['clip_l_embedding'].shape}")
        print(f"File ID: {sample['file_id']}")