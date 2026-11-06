import os
import sys
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from datetime import datetime

from config import get_config
from model import Speech2ClipGModel
from dataset import create_dataloaders
from losses import CombinedCentroidLoss


def setup_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_distributed():
    dist.destroy_process_group()


class Trainer:
    def __init__(self, config, rank=0, world_size=1, use_distributed=False):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.use_distributed = use_distributed
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        if rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model = Speech2ClipGModel(
            whisper_model_name=config['whisper_model_name'],
            n_decoder_layers=config['n_decoder_layers'],
            n_state=config['n_state'],
            n_head=config['n_head'],
            mlp_ratio=config['mlp_ratio'],
            n_clusters=config['n_clusters'],
            centroids_path=config['centroids_path'],
        ).to(self.device)

        if use_distributed:
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=config['scheduler_t0'], T_mult=config['scheduler_t_mult'])
        self.criterion = CombinedCentroidLoss(
            mse_w=config['mse_weight'], cosine_w=config['cosine_weight'],
            ce_w=config['ce_weight'], contrastive_w=config['contrastive_weight'],
        )

        self.train_loader, self.val_loader = create_dataloaders(
            speech_dir=config['speech_dir'],
            caption_dir=config['caption_dir'],
            clip_g_encoder_path=config['clip_g_encoder_path'],
            device=config['device'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            max_samples=config['max_samples'],
            train_split=config['train_split'],
            use_distributed=use_distributed,
            rank=rank,
            world_size=world_size,
            index_file=config.get('dataset_index_file'),
        )

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        self.log_file = None
        if rank == 0:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = open(self.log_dir / f'training_log_{ts}.jsonl', 'w')

    def _get_centroids(self):
        m = self.model.module if self.use_distributed else self.model
        return m.centroids

    def _log(self, data):
        if self.rank == 0 and self.log_file:
            self.log_file.write(json.dumps(data) + '\n')
            self.log_file.flush()

    def train_epoch(self, epoch):
        self.model.train()
        if self.use_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        total_loss = 0
        for step, batch in enumerate(self.train_loader):
            feats = batch['input_features'].to(self.device)
            targets = batch['clip_g_embeddings'].to(self.device)
            pred, logits = self.model(feats, targets, use_teacher_forcing=True)
            loss = self.criterion(pred, targets, logits, self._get_centroids())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            total_loss += loss.item()
            if self.rank == 0 and step % self.config['log_interval'] == 0:
                self._log({'epoch': epoch, 'step': step, 'train_loss': loss.item()})
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                feats = batch['input_features'].to(self.device)
                targets = batch['clip_g_embeddings'].to(self.device)
                pred, logits = self.model(feats, targets, use_teacher_forcing=True)
                loss = self.criterion(pred, targets, logits, self._get_centroids())
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        if self.rank != 0:
            return
        m = self.model.module if self.use_distributed else self.model
        ckpt = {
            'epoch': epoch,
            'model_state_dict': m.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(ckpt, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / 'best_model.pt')

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        m = self.model.module if self.use_distributed else self.model
        m.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt.get('val_loss', float('inf'))

    def train(self):
        for epoch in range(self.start_epoch, self.config['num_epochs'] + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            if self.rank == 0:
                self._log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'best': is_best})
                print(f"Epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}{' *' if is_best else ''}")
                if epoch % self.config['save_interval'] == 0:
                    self.save_checkpoint(epoch, val_loss, is_best)
        if self.log_file:
            self.log_file.close()


def main():
    config = get_config()
    use_dist = 'LOCAL_RANK' in os.environ
    if use_dist:
        rank, world_size = setup_distributed()
    else:
        rank, world_size = 0, 1

    trainer = Trainer(config, rank=rank, world_size=world_size, use_distributed=use_dist)

    resume = config.get('resume_checkpoint')
    if resume and Path(resume).exists():
        trainer.load_checkpoint(resume)

    trainer.train()

    if use_dist:
        cleanup_distributed()


if __name__ == '__main__':
    main()
