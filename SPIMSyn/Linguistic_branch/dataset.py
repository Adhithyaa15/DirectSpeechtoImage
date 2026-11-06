import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import librosa
from transformers import WhisperFeatureExtractor, AutoTokenizer, T5EncoderModel
import warnings
import json
import random
warnings.filterwarnings('ignore')

# Set to use local files only (offline mode)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


class Speech2T5Dataset(Dataset):
    """Dataset for loading speech files and computing T5 embeddings on-the-fly"""
    
    def __init__(
        self,
        speech_dir,
        caption_dir,
        t5_encoder_path,
        t5_tokenizer_path,
        device='cuda',
        feature_extractor=None,
        max_samples=None,
        sample_rate=16000,
        index_file=None,
        split='train'
    ):
        """
        Args:
            speech_dir: Path to speech files directory
            caption_dir: Path to caption text files directory
            t5_encoder_path: Path to SD3.5 T5 encoder
            t5_tokenizer_path: Path to SD3.5 T5 tokenizer
            device: Device for T5 encoder ('cuda' or 'cpu')
            feature_extractor: Whisper feature extractor
            max_samples: Maximum number of samples to load (None for all, ignored if index_file is provided)
            sample_rate: Audio sample rate
            index_file: Path to dataset index JSON file (if provided, uses fixed dataset)
            split: 'train' or 'val' (only used when index_file is provided)
        """
        self.speech_dir = Path(speech_dir)
        self.caption_dir = Path(caption_dir)
        self.sample_rate = sample_rate
        self.split = split
        self.device_name = device
        
        # Store paths for lazy loading (to avoid CUDA in forked workers)
        self.t5_encoder_path = t5_encoder_path
        self.t5_tokenizer_path = t5_tokenizer_path
        
        # These will be loaded lazily in each worker process
        self._t5_encoder = None
        self._t5_tokenizer = None
        
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
            # JSON stores: 'Speechs/ID.wav' and 'captions/ID.txt'
            speech_filename = Path(pair['speech_path']).name
            caption_filename = Path(pair['id'] + '.txt')  # Caption files are named as ID.txt
            
            # Build full paths using the actual speech_dir and caption_dir
            speech_path = self.speech_dir / speech_filename
            caption_path = self.caption_dir / caption_filename
            
            valid_pairs.append({
                'id': pair['id'],
                'speech_path': str(speech_path),
                'caption_path': str(caption_path)
            })
        
        return valid_pairs
        
    def _find_valid_pairs(self, max_samples):
        """Find all valid pairs of speech files and caption files"""
        valid_pairs = []
        
        # Get all caption files
        caption_files = list(self.caption_dir.glob("*.txt"))
        total_files = len(caption_files)
        
        # Randomly sample if max_samples is specified
        if max_samples and max_samples < total_files:
            random.seed(42)  # For reproducibility
            caption_files = random.sample(caption_files, max_samples)
            print(f"📊 Randomly sampled {max_samples} files from {total_files} total files (seed=42)")
        
        for caption_file in caption_files:
            # Extract ID from filename (e.g., "38423.txt" -> "38423")
            file_id = caption_file.stem
            
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
                    'caption_path': str(caption_file)
                })
        
        return valid_pairs
    
    @property
    def t5_encoder(self):
        """Lazy load T5 encoder (happens in main process with num_workers=0)"""
        if self._t5_encoder is None:
            # With num_workers=0, this runs in main process - safe to use GPU
            # T5 on GPU is 3-10x faster than CPU for inference
            device = torch.device(self.device_name if torch.cuda.is_available() else 'cpu')
            self._t5_encoder = T5EncoderModel.from_pretrained(
                self.t5_encoder_path,
                local_files_only=True,
                torch_dtype=torch.float32  # Force float32 (SD3.5 T5 is stored as float16)
            ).to(device)
            self._t5_encoder.eval()
            # Freeze T5 encoder
            for param in self._t5_encoder.parameters():
                param.requires_grad = False
        return self._t5_encoder
    
    @property
    def t5_tokenizer(self):
        """Lazy load T5 tokenizer (happens in each worker process separately)"""
        if self._t5_tokenizer is None:
            self._t5_tokenizer = AutoTokenizer.from_pretrained(
                self.t5_tokenizer_path,
                local_files_only=True
            )
        return self._t5_tokenizer
    
    @property
    def device(self):
        """Get device for T5 encoder (GPU with num_workers=0)"""
        return torch.device(self.device_name if torch.cuda.is_available() else 'cpu')
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def _compute_t5_embedding(self, caption):
        """
        Compute T5 embedding for a caption on-the-fly using SD3.5's T5 encoder.
        
        Args:
            caption: Text caption string
        
        Returns:
            t5_embedding: [256, 4096] T5 embedding (fixed size, padded/truncated)
        """
        with torch.no_grad():
            # Tokenize the caption with fixed max length 256
            text_inputs = self.t5_tokenizer(
                caption,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get T5 embeddings
            t5_outputs = self.t5_encoder(
                text_inputs.input_ids,
                output_hidden_states=False
            )
            t5_embeds = t5_outputs.last_hidden_state  # [1, 256, 4096]
            
            # Remove batch dimension, move to CPU, and ensure float32
            # (T5 encoder from SD3.5 is in float16, but training needs float32)
            t5_embeds = t5_embeds.squeeze(0).cpu().float()  # [256, 4096]
        
        return t5_embeds
    
    def __getitem__(self, idx):
        """
        Returns:
            input_features: Whisper input features [80, 3000]
            t5_embedding: T5 ground truth embedding [256, 4096]
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
        
        # Load caption and compute T5 embedding on-the-fly
        try:
            with open(pair['caption_path'], 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            # Compute T5 embedding [256, 4096]
            t5_embedding = self._compute_t5_embedding(caption)
            
            # Ensure correct shape
            if t5_embedding.shape != (256, 4096):
                print(f"Warning: Unexpected T5 shape {t5_embedding.shape} for {pair['caption_path']}")
                t5_embedding = torch.zeros(256, 4096)
        except Exception as e:
            print(f"Error computing T5 embedding for {pair['caption_path']}: {e}")
            t5_embedding = torch.zeros(256, 4096)
        
        return {
            'input_features': input_features,
            't5_embedding': t5_embedding,
            'file_id': pair['id']
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    input_features = torch.stack([item['input_features'] for item in batch])
    t5_embeddings = torch.stack([item['t5_embedding'] for item in batch])
    file_ids = [item['file_id'] for item in batch]
    
    return {
        'input_features': input_features,
        't5_embeddings': t5_embeddings,
        'file_ids': file_ids
    }


def create_dataloaders(
    speech_dir,
    caption_dir,
    t5_encoder_path,
    t5_tokenizer_path,
    device='cuda',
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
        speech_dir: Path to speech files directory
        caption_dir: Path to caption text files directory
        t5_encoder_path: Path to SD3.5 T5 encoder
        t5_tokenizer_path: Path to SD3.5 T5 tokenizer
        device: Device for T5 encoder
        index_file: Path to dataset index JSON. If provided, uses fixed dataset split.
                   If None, falls back to auto-discovery with random_split.
    """
    from torch.utils.data.distributed import DistributedSampler
    
    # Check if index file exists
    if index_file and Path(index_file).exists():
        # Use fixed dataset from index file
        if rank == 0:
            print(f"\n✓ Using fixed dataset index: {index_file}")
        
        train_dataset = Speech2T5Dataset(
            speech_dir=speech_dir,
            caption_dir=caption_dir,
            t5_encoder_path=t5_encoder_path,
            t5_tokenizer_path=t5_tokenizer_path,
            device=device,
            index_file=index_file,
            split='train'
        )
        
        val_dataset = Speech2T5Dataset(
            speech_dir=speech_dir,
            caption_dir=caption_dir,
            t5_encoder_path=t5_encoder_path,
            t5_tokenizer_path=t5_tokenizer_path,
            device=device,
            index_file=index_file,
            split='val'
        )
    else:
        # Fall back to old behavior: auto-discovery + random_split
        if index_file and rank == 0:
            print(f"\n⚠️  Index file not found: {index_file}")
            print(f"⚠️  Using auto-discovery + random split (non-reproducible across runs)")
        
        # Create full dataset
        dataset = Speech2T5Dataset(
            speech_dir=speech_dir,
            caption_dir=caption_dir,
            t5_encoder_path=t5_encoder_path,
            t5_tokenizer_path=t5_tokenizer_path,
            device=device,
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
    caption_dir = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/captions"
    t5_encoder_path = "/scratch/IITB/ai-at-ieor/20004596/sd35/text_encoder_3"
    t5_tokenizer_path = "/scratch/IITB/ai-at-ieor/20004596/sd35/tokenizer_3"
    
    dataset = Speech2T5Dataset(
        speech_dir=speech_dir,
        caption_dir=caption_dir,
        t5_encoder_path=t5_encoder_path,
        t5_tokenizer_path=t5_tokenizer_path,
        device='cuda',
        max_samples=10
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Input features shape: {sample['input_features'].shape}")
        print(f"T5 embedding shape: {sample['t5_embedding'].shape}")
        print(f"File ID: {sample['file_id']}")
