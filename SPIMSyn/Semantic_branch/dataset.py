import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import librosa
from transformers import WhisperFeatureExtractor
import warnings
import json
import random
warnings.filterwarnings('ignore')

# Set to use local files only (offline mode)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


class Speech2ImageDataset(Dataset):
    """Dataset for loading speech files and corresponding CLIP-L embeddings"""
    
    def __init__(
        self,
        speech_dir,
        clip_l_dir,
        feature_extractor=None,
        max_samples=None,
        sample_rate=16000,
        index_file=None,
        split='train'
    ):
        """
        Args:
            speech_dir: Path to speech files directory
            clip_l_dir: Path to CLIP-L embeddings directory
            feature_extractor: Whisper feature extractor
            max_samples: Maximum number of samples to load (None for all, ignored if index_file is provided)
            sample_rate: Audio sample rate
            index_file: Path to dataset index JSON file (if provided, uses fixed dataset)
            split: 'train' or 'val' (only used when index_file is provided)
        """
        self.speech_dir = Path(speech_dir)
        self.clip_l_dir = Path(clip_l_dir)
        self.sample_rate = sample_rate
        self.split = split
        
        # Initialize Whisper feature extractor
        if feature_extractor is None:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                "openai/whisper-medium",
                local_files_only=True
            )
        else:
            self.feature_extractor = feature_extractor
        
        # Load dataset based on index file or discovery
        if index_file and Path(index_file).exists():
            self.valid_pairs = self._load_from_index(index_file, split)
            print(f"Dataset ({split}) loaded from index: {len(self.valid_pairs)} samples")
        else:
            if index_file:
                print(f"⚠️  Index file not found: {index_file}")
                print(f"⚠️  Falling back to auto-discovery (non-reproducible)")
            self.valid_pairs = self._find_valid_pairs(max_samples)
            print(f"Dataset initialized with {len(self.valid_pairs)} valid pairs")
    
    def _load_from_index(self, index_file, split):
        """Load dataset from fixed index file"""
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        # Get pairs for the specified split
        if split not in ['train', 'val']:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
        
        pairs = index[split]
        
        # Convert relative paths to absolute using the actual base directories
        valid_pairs = []
        for pair in pairs:
            # Extract just the filename from the stored paths
            # JSON stores: 'Speechs/ID.wav' and 'clip_l/ID_clip_l.npy'
            speech_filename = Path(pair['speech_path']).name
            clip_l_filename = Path(pair['clip_l_path']).name
            
            # Build full paths using the actual speech_dir and clip_l_dir
            speech_path = self.speech_dir / speech_filename
            clip_l_path = self.clip_l_dir / clip_l_filename
            
            valid_pairs.append({
                'id': pair['id'],
                'speech_path': str(speech_path),
                'clip_l_path': str(clip_l_path)
            })
        
        return valid_pairs
        
    def _find_valid_pairs(self, max_samples):
        """Find all valid pairs of speech files and CLIP-L embeddings"""
        valid_pairs = []
        
        # Get all CLIP-L embedding files
        clip_files = list(self.clip_l_dir.glob("*_clip_l.npy"))
        total_files = len(clip_files)
        
        # Randomly sample if max_samples is specified
        if max_samples and max_samples < total_files:
            random.seed(42)  # For reproducibility
            clip_files = random.sample(clip_files, max_samples)
            print(f"📊 Randomly sampled {max_samples} files from {total_files} total files (seed=42)")
        
        for clip_file in clip_files:
            # Extract ID from filename (e.g., "38423_clip_l.npy" -> "38423")
            file_id = clip_file.stem.replace("_clip_l", "")
            
            # Look for corresponding speech file with various extensions
            speech_file = None
            for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                potential_speech = self.speech_dir / f"{file_id}{ext}"
                if potential_speech.exists():
                    speech_file = potential_speech
                    break
            
            if speech_file and speech_file.exists():
                valid_pairs.append({
                    'id': file_id,
                    'speech_path': str(speech_file),
                    'clip_l_path': str(clip_file)
                })
        
        return valid_pairs
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_features: Whisper input features [80, 3000]
            clip_l_embedding: CLIP-L embedding [77, 768]
            file_id: File identifier
        """
        pair = self.valid_pairs[idx]
        
        # Load and process audio
        try:
            audio, sr = librosa.load(pair['speech_path'], sr=self.sample_rate)
            
            # Handle very short audio
            if len(audio) < 400:  # Minimum for Whisper
                audio = np.pad(audio, (0, 400 - len(audio)))
            
            # Extract Whisper features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            input_features = inputs.input_features.squeeze(0)  # [80, 3000]
            
        except Exception as e:
            print(f"Error loading audio {pair['speech_path']}: {e}")
            # Return zeros as fallback
            input_features = torch.zeros(80, 3000)
        
        # Load CLIP-L embedding
        try:
            clip_l_embedding = np.load(pair['clip_l_path'])  # [77, 768]
            clip_l_embedding = torch.from_numpy(clip_l_embedding).float()
            
            # Ensure correct shape
            if clip_l_embedding.shape != (77, 768):
                print(f"Warning: Unexpected CLIP-L shape {clip_l_embedding.shape} for {pair['clip_l_path']}")
                clip_l_embedding = torch.zeros(77, 768)
        except Exception as e:
            print(f"Error loading CLIP-L embedding {pair['clip_l_path']}: {e}")
            clip_l_embedding = torch.zeros(77, 768)
        
        return {
            'input_features': input_features,
            'clip_l_embedding': clip_l_embedding,
            'file_id': pair['id']
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    input_features = torch.stack([item['input_features'] for item in batch])
    clip_l_embeddings = torch.stack([item['clip_l_embedding'] for item in batch])
    file_ids = [item['file_id'] for item in batch]
    
    return {
        'input_features': input_features,
        'clip_l_embeddings': clip_l_embeddings,
        'file_ids': file_ids
    }


def create_dataloaders(
    speech_dir,
    clip_l_dir,
    batch_size=8,
    num_workers=4,
    max_samples=50000,
    train_split=0.9,
    use_distributed=False,
    rank=0,
    world_size=1,
    index_file=None
):
    """
    Create train and validation dataloaders with optional DDP support
    
    Args:
        index_file: Path to dataset index JSON. If provided, uses fixed dataset split.
                   If None, falls back to auto-discovery with random_split.
    """
    from torch.utils.data.distributed import DistributedSampler
    
    # Check if index file exists
    if index_file and Path(index_file).exists():
        # Use fixed dataset from index file
        if rank == 0:
            print(f"\n✓ Using fixed dataset index: {index_file}")
        
        train_dataset = Speech2ImageDataset(
            speech_dir=speech_dir,
            clip_l_dir=clip_l_dir,
            index_file=index_file,
            split='train'
        )
        
        val_dataset = Speech2ImageDataset(
            speech_dir=speech_dir,
            clip_l_dir=clip_l_dir,
            index_file=index_file,
            split='val'
        )
    else:
        # Fall back to old behavior: auto-discovery + random_split
        if index_file and rank == 0:
            print(f"\n⚠️  Index file not found: {index_file}")
            print(f"⚠️  Using auto-discovery + random split (non-reproducible across runs)")
        
        # Create full dataset
        dataset = Speech2ImageDataset(
            speech_dir=speech_dir,
            clip_l_dir=clip_l_dir,
            max_samples=max_samples
        )
        
        # Split into train and validation
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.manual_seed(42)
        )
    
    # Create samplers for distributed training
    if use_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=42
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using distributed sampler
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


if __name__ == "__main__":
    # Test dataset loading
    speech_dir = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/Speechs"
    clip_l_dir = "/scratch/IITB/ai-at-ieor/20004596/Speech2Image/embeddings/clip_l"
    
    dataset = Speech2ImageDataset(
        speech_dir=speech_dir,
        clip_l_dir=clip_l_dir,
        max_samples=10
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Input features shape: {sample['input_features'].shape}")
        print(f"CLIP-L embedding shape: {sample['clip_l_embedding'].shape}")
        print(f"File ID: {sample['file_id']}")
