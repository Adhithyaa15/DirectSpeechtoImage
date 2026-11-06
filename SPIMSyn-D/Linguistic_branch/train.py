import os

# Set cache directories to scratch to avoid home directory quota issues
os.environ['TORCH_HOME'] = '/scratch/IITB/ai-at-ieor/20004596/.torch_cache'
os.environ['TMPDIR']     = '/scratch/IITB/ai-at-ieor/20004596/tmp'

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

from model import Speech2T5Model
from dataset import create_dataloaders, collate_fn
from losses import CombinedCentroidLoss
import config as cfg
import csv


def visualize_teacher_forcing_matrix(target_embeddings, model, batch_idx=0, save_dir=None):
    """Visualize / log teacher-forcing structure (first batch of first epoch only)."""
    if batch_idx > 0:
        return

    actual_model = model.module if hasattr(model, 'module') else model
    B = target_embeddings.size(0)

    with torch.no_grad():
        shifted = actual_model.prepare_shifted_targets(target_embeddings)

    print("\n" + "="*80)
    print("TEACHER FORCING MATRIX STRUCTURE (First sample in batch)")
    print("="*80)
    print(f"Target shape : {target_embeddings.shape}  [B, 256, 4096]")
    print(f"Shifted shape: {shifted.shape}  [B, 256, 4096]")
    print("\nActual tensor values (first sample, first 5 dimensions):")
    print(f"  BOS token     : {shifted[0, 0, :5].cpu().numpy()}")
    print(f"  Target v1     : {target_embeddings[0, 0, :5].cpu().numpy()}")
    print(f"  Shifted pos 1 : {shifted[0, 1, :5].cpu().numpy()}")

    if save_dir:
        csv_dir = Path(save_dir) / 'autoregressive_matrices'
        csv_dir.mkdir(parents=True, exist_ok=True)

        for name, tensor in [('shifted_input', shifted), ('target', target_embeddings)]:
            csv_file = csv_dir / f'preparation_{name}.csv'
            arr = tensor[0].cpu().numpy()  # [256, 4096]
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Position'] + [f'Dim_{i}' for i in range(10)] + ['...'])
                for pos in range(256):
                    writer.writerow([pos] + arr[pos, :10].tolist() + ['...'])
        print(f"Saved teacher-forcing matrices to {csv_dir}")

    print("="*80 + "\n")


class Trainer:
    def __init__(self, config, rank=0, world_size=1):
        self.config       = config
        self.rank         = rank
        self.world_size   = world_size
        self.is_distributed = world_size > 1
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        # Output directories
        self.output_dir     = Path(config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.attention_dir  = self.output_dir / 'attention_maps'
        self.log_dir        = self.output_dir / 'logs'

        if self.rank == 0:
            for d in [self.checkpoint_dir, self.attention_dir, self.log_dir]:
                d.mkdir(parents=True, exist_ok=True)

        # ---- Model ----
        if self.rank == 0:
            print("Initializing Speech2T5Model (DMEL input)...")

        self.model = Speech2T5Model(
            n_decoder_layers=config['n_decoder_layers'],
            n_state=config['n_state'],
            n_head=config['n_head'],
            mlp_ratio=config['mlp_ratio'],
            n_clusters=config.get('n_clusters', 2950),
            centroids_path=config.get('centroids_path')
        ).to(self.device)

        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank,
                             find_unused_parameters=False)

        if self.rank == 0:
            m = self.model.module if self.is_distributed else self.model
            total, trainable = m.count_parameters()
            print(f"Total parameters     : {total:,}")
            print(f"Trainable parameters : {trainable:,}")

        # ---- Loss ----
        self.criterion = CombinedCentroidLoss(
            mse_weight=config.get('mse_weight', 1.0),
            cosine_weight=config.get('cosine_weight', 1.0),
            ce_weight=config.get('ce_weight', 0.5),
            contrastive_weight=config.get('contrastive_weight', 0.5)
        )
        if self.rank == 0:
            print(f"\nLoss weights — MSE:{config.get('mse_weight',1.0)}  "
                  f"Cos:{config.get('cosine_weight',1.0)}  "
                  f"CE:{config.get('ce_weight',0.5)}  "
                  f"Cont:{config.get('contrastive_weight',0.5)}")

        # ---- Optimizer & Scheduler ----
        m = self.model.module if self.is_distributed else self.model
        self.optimizer = AdamW(
            m.get_trainable_parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['scheduler_t0'],
            T_mult=config['scheduler_t_mult']
        )

        # Training state
        self.current_epoch = 0
        self.global_step   = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history   = []

    # ------------------------------------------------------------------

    def _build_loader(self, dataset, batch_size, shuffle, epoch_seed=None):
        """Helper to (re)build a DataLoader from an existing dataset."""
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank,
                shuffle=shuffle, seed=epoch_seed if epoch_seed is not None else 42
            )
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=self.config['num_workers'],
                              collate_fn=collate_fn, pin_memory=True)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=self.config['num_workers'],
                              collate_fn=collate_fn, pin_memory=True)

    # ------------------------------------------------------------------

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with per-epoch resampling."""
        self.model.train()

        # Resample 50 k from the full pool every epoch
        dataset = train_loader.dataset
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        dataset.resample(max_samples=self.config.get('max_samples', 50000))

        # Rebuild loader so the new length is reflected
        train_loader = self._build_loader(
            train_loader.dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            epoch_seed=epoch
        )

        if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        totals = dict(loss=0, mse=0, cos=0, ce=0, cont=0, acc=0)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}") \
               if self.rank == 0 else train_loader

        for batch_idx, batch in enumerate(pbar):
            dmel_features  = batch['dmel_features'].to(self.device)   # [B, T, 80]
            t5_embeddings  = batch['t5_embeddings'].to(self.device)   # [B, 256, 4096]

            # Visualise teacher-forcing once
            if epoch == 1 and batch_idx == 0 and self.rank == 0:
                visualize_teacher_forcing_matrix(
                    t5_embeddings, self.model, batch_idx, save_dir=self.output_dir
                )

            save_intermediate = (epoch == 1 and batch_idx == 0 and self.rank == 0)

            # Forward
            predictions, cluster_logits = self.model(
                dmel_features, t5_embeddings,
                return_attention=False,
                use_teacher_forcing=True,
                save_intermediate=save_intermediate,
                return_cluster_logits=True
            )

            m = self.model.module if self.is_distributed else self.model
            loss, loss_dict = self.criterion(
                predictions, cluster_logits, t5_embeddings, m.centroids
            )

            self.optimizer.zero_grad()
            loss.backward()

            if self.config['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    (self.model.module if self.is_distributed else self.model)
                    .get_trainable_parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()
            self.scheduler.step(epoch + batch_idx / len(train_loader))

            totals['loss'] += loss_dict['total_loss']
            totals['mse']  += loss_dict['mse_loss']
            totals['cos']  += loss_dict['cosine_loss']
            totals['ce']   += loss_dict['ce_loss']
            totals['cont'] += loss_dict['contrastive_loss']
            totals['acc']  += loss_dict['cluster_accuracy']
            self.global_step += 1

            if self.rank == 0:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'mse':  f"{loss_dict['mse_loss']:.4f}",
                    'cos':  f"{loss_dict['cosine_loss']:.4f}",
                    'ce':   f"{loss_dict['ce_loss']:.4f}",
                    'cont': f"{loss_dict['contrastive_loss']:.4f}",
                    'acc':  f"{loss_dict['cluster_accuracy']:.3f}",
                    'lr':   f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                self.log_metrics({
                    'epoch': epoch, 'step': self.global_step, 'batch_idx': batch_idx,
                    'train_loss':              loss_dict['total_loss'],
                    'train_mse_loss':          loss_dict['mse_loss'],
                    'train_cosine_loss':       loss_dict['cosine_loss'],
                    'train_cosine_sim':        loss_dict['avg_cosine_sim'],
                    'train_ce_loss':           loss_dict['ce_loss'],
                    'train_contrastive_loss':  loss_dict['contrastive_loss'],
                    'train_cluster_accuracy':  loss_dict['cluster_accuracy'],
                    'train_avg_positive_sim':  loss_dict['avg_positive_sim'],
                    'train_avg_negative_sim':  loss_dict['avg_negative_sim'],
                    'train_sim_margin':        loss_dict['sim_margin'],
                    'lr': self.optimizer.param_groups[0]['lr']
                }, log_type='train')

        n = len(train_loader)
        return {
            'loss':             totals['loss'] / n,
            'mse_loss':         totals['mse']  / n,
            'cosine_loss':      totals['cos']  / n,
            'ce_loss':          totals['ce']   / n,
            'contrastive_loss': totals['cont'] / n,
            'cluster_accuracy': totals['acc']  / n,
        }

    # ------------------------------------------------------------------

    def validate(self, val_loader, epoch, save_attention=False):
        """Validation loop."""
        self.model.eval()
        totals = dict(loss=0, mse=0, cos=0, ce=0, cont=0, acc=0)
        attention_maps_to_save = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation") if self.rank == 0 else val_loader

            for batch_idx, batch in enumerate(pbar):
                dmel_features = batch['dmel_features'].to(self.device)
                t5_embeddings = batch['t5_embeddings'].to(self.device)

                if save_attention:
                    predictions, attention_weights, cluster_logits = self.model(
                        dmel_features, t5_embeddings,
                        return_attention=True, use_teacher_forcing=True,
                        return_cluster_logits=True
                    )
                    if batch_idx == 0:
                        attention_maps_to_save.append({
                            'attention_weights': attention_weights.cpu().numpy(),
                            'file_ids': batch['file_ids']
                        })
                else:
                    predictions, cluster_logits = self.model(
                        dmel_features, t5_embeddings,
                        return_attention=False, use_teacher_forcing=True,
                        return_cluster_logits=True
                    )

                m = self.model.module if self.is_distributed else self.model
                loss, loss_dict = self.criterion(
                    predictions, cluster_logits, t5_embeddings, m.centroids
                )

                totals['loss'] += loss_dict['total_loss']
                totals['mse']  += loss_dict['mse_loss']
                totals['cos']  += loss_dict['cosine_loss']
                totals['ce']   += loss_dict['ce_loss']
                totals['cont'] += loss_dict['contrastive_loss']
                totals['acc']  += loss_dict['cluster_accuracy']

                if self.rank == 0:
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total_loss']:.4f}",
                        'mse':  f"{loss_dict['mse_loss']:.4f}",
                        'cos':  f"{loss_dict['cosine_loss']:.4f}",
                        'ce':   f"{loss_dict['ce_loss']:.4f}",
                        'cont': f"{loss_dict['contrastive_loss']:.4f}",
                        'acc':  f"{loss_dict['cluster_accuracy']:.3f}"
                    })
                    self.log_metrics({
                        'epoch': epoch, 'batch_idx': batch_idx,
                        'val_loss':             loss_dict['total_loss'],
                        'val_mse_loss':         loss_dict['mse_loss'],
                        'val_cosine_loss':      loss_dict['cosine_loss'],
                        'val_cosine_sim':       loss_dict['avg_cosine_sim'],
                        'val_ce_loss':          loss_dict['ce_loss'],
                        'val_contrastive_loss': loss_dict['contrastive_loss'],
                        'val_cluster_accuracy': loss_dict['cluster_accuracy'],
                        'val_avg_positive_sim': loss_dict['avg_positive_sim'],
                        'val_avg_negative_sim': loss_dict['avg_negative_sim'],
                        'val_sim_margin':       loss_dict['sim_margin']
                    }, log_type='val')

        if self.rank == 0 and save_attention and attention_maps_to_save:
            self.save_attention_maps(attention_maps_to_save, epoch)

        n = len(val_loader)
        return {
            'loss':             totals['loss'] / n,
            'mse_loss':         totals['mse']  / n,
            'cosine_loss':      totals['cos']  / n,
            'ce_loss':          totals['ce']   / n,
            'contrastive_loss': totals['cont'] / n,
            'cluster_accuracy': totals['acc']  / n,
        }

    # ------------------------------------------------------------------

    def save_attention_maps(self, attention_data, epoch):
        save_path = self.attention_dir / f'attention_epoch_{epoch}.npz'
        np.savez_compressed(
            save_path,
            attention_weights=attention_data[0]['attention_weights'],
            file_ids=attention_data[0]['file_ids'],
            epoch=epoch
        )
        print(f"Saved attention maps → {save_path}")

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        if self.rank != 0:
            return
        m = self.model.module if self.is_distributed else self.model
        checkpoint = {
            'epoch':                epoch,
            'global_step':          self.global_step,
            'model_state_dict':     m.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss':             val_loss,
            'best_val_loss':        self.best_val_loss,
            'config':               self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        m = self.model.module if self.is_distributed else self.model
        m.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step   = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {self.current_epoch}")

    def log_metrics(self, metrics, log_type='train'):
        log_file = self.log_dir / ('training_log.jsonl' if log_type == 'train'
                                   else 'validation_log.jsonl')
        with open(log_file, 'a') as f:
            json.dump(metrics, f)
            f.write('\n')

    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader):
        """Main training loop."""
        if self.rank == 0:
            print("\nStarting training...")
            print(f"Batch size per GPU : {self.config['batch_size']}")
            print(f"GPUs               : {self.world_size}")
            print(f"Effective batch    : {self.config['batch_size'] * self.world_size}")
            print(f"Epochs             : {self.config['num_epochs']}")

        current_train_loader = train_loader
        current_val_loader   = val_loader

        for epoch in range(self.current_epoch + 1, self.config['num_epochs'] + 1):

            # Build / rebuild dataloaders on first epoch (or after resume)
            if epoch == self.current_epoch + 1:
                current_train_loader, current_val_loader = create_dataloaders(
                    dmel_dir=self.config['dmel_dir'],
                    caption_dir=self.config['caption_dir'],
                    t5_encoder_path=self.config['t5_encoder_path'],
                    t5_tokenizer_path=self.config['t5_tokenizer_path'],
                    device='cuda',
                    batch_size=self.config['batch_size'],
                    num_workers=self.config['num_workers'],
                    max_samples=self.config['max_samples'],
                    train_split=self.config['train_split'],
                    use_distributed=self.is_distributed,
                    rank=self.rank,
                    world_size=self.world_size,
                    index_file=self.config.get('dataset_index_file')
                )
                if self.rank == 0:
                    n = len(current_train_loader.dataset.all_pairs)
                    print(f"✓ Dataloaders ready — {n} total train pairs")

            if self.rank == 0:
                print(f"\n{'='*60}\nEpoch {epoch}/{self.config['num_epochs']}\n{'='*60}")

            # Train
            train_metrics = self.train_epoch(current_train_loader, epoch)
            self.train_history.append(train_metrics)

            if self.rank == 0:
                print(f"Train — loss:{train_metrics['loss']:.4f}  "
                      f"mse:{train_metrics['mse_loss']:.4f}  "
                      f"cos:{train_metrics['cosine_loss']:.4f}  "
                      f"ce:{train_metrics['ce_loss']:.4f}  "
                      f"cont:{train_metrics['contrastive_loss']:.4f}  "
                      f"acc:{train_metrics['cluster_accuracy']:.3f}")

            # Validate
            val_metrics = self.validate(current_val_loader, epoch, save_attention=False)
            self.val_history.append(val_metrics)

            if self.rank == 0:
                print(f"Val   — loss:{val_metrics['loss']:.4f}  "
                      f"mse:{val_metrics['mse_loss']:.4f}  "
                      f"cos:{val_metrics['cosine_loss']:.4f}  "
                      f"ce:{val_metrics['ce_loss']:.4f}  "
                      f"cont:{val_metrics['contrastive_loss']:.4f}  "
                      f"acc:{val_metrics['cluster_accuracy']:.3f}")

            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            self.save_checkpoint(epoch, val_metrics['loss'], is_best)

            if self.rank == 0:
                self.log_metrics({
                    'epoch': epoch,
                    'train_loss':             train_metrics['loss'],
                    'train_mse_loss':         train_metrics['mse_loss'],
                    'train_cosine_loss':      train_metrics['cosine_loss'],
                    'train_ce_loss':          train_metrics['ce_loss'],
                    'train_contrastive_loss': train_metrics['contrastive_loss'],
                    'train_cluster_accuracy': train_metrics['cluster_accuracy'],
                }, log_type='train')
                self.log_metrics({
                    'epoch': epoch,
                    'val_loss':             val_metrics['loss'],
                    'val_mse_loss':         val_metrics['mse_loss'],
                    'val_cosine_loss':      val_metrics['cosine_loss'],
                    'val_ce_loss':          val_metrics['ce_loss'],
                    'val_contrastive_loss': val_metrics['contrastive_loss'],
                    'val_cluster_accuracy': val_metrics['cluster_accuracy'],
                    'best_val_loss':        self.best_val_loss,
                }, log_type='val')

            self.current_epoch = epoch

        if self.rank == 0:
            print(f"\n{'='*60}\nTraining complete! Best val loss: {self.best_val_loss:.4f}\n{'='*60}")


# ----------------------------------------------------------------------
# Distributed helpers
# ----------------------------------------------------------------------

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank       = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    rank, world_size, local_rank = setup_distributed()

    config = cfg.get_config()
    config.update({
        # Override / add runtime settings
        'resume_from': '/scratch/IITB/ai-at-ieor/20004596/T5_with_DMEL/outputs/checkpoints/latest.pt',
        'num_workers': 0,
        'batch_size':  4,
        'num_epochs':  200,
        'max_samples': 50000,
    })

    if rank == 0:
        print("\n" + "="*80)
        print("TRAINING CONFIGURATION  —  Speech → T5  (DMEL input)")
        print("="*80)
        print(f"  DMEL dir       : {config['dmel_dir']}")
        print(f"  Caption dir    : {config['caption_dir']}")
        print(f"  T5 encoder     : {config['t5_encoder_path']}")
        print(f"  Output dir     : {config['output_dir']}")
        print(f"  Decoder layers : {config['n_decoder_layers']}")
        print(f"  Hidden dim     : {config['n_state']}")
        print(f"  Attn heads     : {config['n_head']}")
        print(f"  Clusters       : {config['n_clusters']}")
        print(f"  Epochs         : {config['num_epochs']}")
        print(f"  LR             : {config['learning_rate']}")
        print(f"  Batch/GPU      : {config['batch_size']}  ×  {world_size} GPUs")
        print("="*80 + "\n")

    trainer = Trainer(config, rank=rank, world_size=world_size)

    resume_path = config.get('resume_from')
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            print(f"Resuming from: {resume_path}")
        trainer.load_checkpoint(resume_path)

    try:
        trainer.train(None, None)
    except Exception as e:
        print(f"Training crashed: {e}")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()