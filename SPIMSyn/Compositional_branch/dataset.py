import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
import librosa
from transformers import WhisperFeatureExtractor, CLIPTokenizer, CLIPTextModelWithProjection
import warnings
import random

warnings.filterwarnings('ignore')


class Speech2ClipGDataset(Dataset):
    def __init__(self, speech_dir, caption_dir, clip_g_encoder_path, device='cuda',
                 feature_extractor=None, max_samples=None, sample_rate=16000,
                 index_file=None, split='train'):
        self.speech_dir = Path(speech_dir)
        self.caption_dir = Path(caption_dir)
        self.sample_rate = sample_rate
        self.device_name = device
        self.clip_g_encoder_path = clip_g_encoder_path
        self._clip_g_encoder = None
        self._clip_g_tokenizer = None

        if feature_extractor is None:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                "openai/whisper-medium", local_files_only=True
            )
        else:
            self.feature_extractor = feature_extractor

        if index_file and Path(index_file).exists():
            import json
            with open(index_file) as f:
                idx = json.load(f)
            self.valid_pairs = [
                {
                    'id': p['id'],
                    'speech_path': str(self.speech_dir / Path(p['speech_path']).name),
                    'caption_path': str(self.caption_dir / (p['id'] + '.txt')),
                }
                for p in idx[split]
            ]
        else:
            self.valid_pairs = self._find_pairs(max_samples)

    def _find_pairs(self, max_samples):
        captions = list(self.caption_dir.glob("*.txt"))
        if max_samples and max_samples < len(captions):
            random.seed(42)
            captions = random.sample(captions, max_samples)
        pairs = []
        for cap in captions:
            fid = cap.stem
            for ext in ['.wav', '.mp3', '.flac']:
                sp = self.speech_dir / f"{fid}{ext}"
                if sp.exists():
                    pairs.append({'id': fid, 'speech_path': str(sp), 'caption_path': str(cap)})
                    break
        return pairs

    @property
    def clip_g_encoder(self):
        if self._clip_g_encoder is None:
            from diffusers import StableDiffusion3Pipeline
            pipe = StableDiffusion3Pipeline.from_pretrained(
                self.clip_g_encoder_path, local_files_only=True, torch_dtype=torch.float32
            )
            self._clip_g_encoder = pipe.text_encoder_2.eval()
            self._clip_g_tokenizer = pipe.tokenizer_2
            for p in self._clip_g_encoder.parameters():
                p.requires_grad = False
        return self._clip_g_encoder

    @property
    def clip_g_tokenizer(self):
        if self._clip_g_tokenizer is None:
            _ = self.clip_g_encoder
        return self._clip_g_tokenizer

    def _compute_clip_g(self, caption):
        dev = torch.device(self.device_name if torch.cuda.is_available() else 'cpu')
        enc = self.clip_g_encoder.to(dev)
        tok = self.clip_g_tokenizer(
            caption, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).to(dev)
        with torch.no_grad():
            out = enc(tok.input_ids, output_hidden_states=True)
        return out.hidden_states[-2].squeeze(0).cpu().float()

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        try:
            audio, _ = librosa.load(pair['speech_path'], sr=self.sample_rate)
            if len(audio) < 400:
                audio = np.pad(audio, (0, 400 - len(audio)))
            feats = self.feature_extractor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
            input_features = feats.input_features.squeeze(0)
        except Exception:
            input_features = torch.zeros(80, 3000)

        try:
            caption = Path(pair['caption_path']).read_text(encoding='utf-8').strip()
            clip_g = self._compute_clip_g(caption)
            if clip_g.shape != (77, 1280):
                clip_g = torch.zeros(77, 1280)
        except Exception:
            clip_g = torch.zeros(77, 1280)

        return {'input_features': input_features, 'clip_g_embedding': clip_g, 'file_id': pair['id']}


def collate_fn(batch):
    return {
        'input_features': torch.stack([b['input_features'] for b in batch]),
        'clip_g_embeddings': torch.stack([b['clip_g_embedding'] for b in batch]),
        'file_ids': [b['file_id'] for b in batch],
    }


def create_dataloaders(speech_dir, caption_dir, clip_g_encoder_path, device='cuda',
                       batch_size=8, num_workers=4, max_samples=50000, train_split=0.98,
                       use_distributed=False, rank=0, world_size=1, index_file=None):
    kwargs = dict(speech_dir=speech_dir, caption_dir=caption_dir,
                  clip_g_encoder_path=clip_g_encoder_path, device=device)

    if index_file and Path(index_file).exists():
        train_ds = Speech2ClipGDataset(**kwargs, index_file=index_file, split='train')
        val_ds   = Speech2ClipGDataset(**kwargs, index_file=index_file, split='val')
    else:
        full = Speech2ClipGDataset(**kwargs, max_samples=max_samples)
        n_train = int(train_split * len(full))
        train_ds, val_ds = torch.utils.data.random_split(
            full, [n_train, len(full) - n_train], generator=torch.manual_seed(42)
        )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_distributed else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False) if use_distributed else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              sampler=val_sampler,   num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader
