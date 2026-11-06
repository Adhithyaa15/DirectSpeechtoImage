import os

os.environ['TORCH_HOME'] = '/scratch/IITB/ai-at-ieor/20004596/.torch_cache'
os.environ['TMPDIR'] = '/scratch/IITB/ai-at-ieor/20004596/tmp'

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
import random

from model import Speech2ImageModel
from dataset import create_dataloaders, discover_all_valid_ids
from losses import CombinedCentroidLoss
import config as cfg
import csv


def visualize_teacher_forcing_matrix(target_embeddings, model, batch_idx=0, save_dir=None):
    if batch_idx > 0:
        return
    actual_model = model.module if hasattr(model, 'module') else model
    B = target_embeddings.size(0)
    with torch.no_grad():
        shifted = actual_model.prepare_shifted_targets(target_embeddings)
    
    print("\n" + "="*80)
    print("TEACHER FORCING MATRIX STRUCTURE (First sample in batch)")
    print("="*80)
    print("Target shape: ", target_embeddings.shape, " [B, 77, 1280]")
    print("Shifted input shape:", shifted.shape, " [B, 77, 1280]")
    print("\nFor autoregressive training at each position i:")
    print("  Decoder Input (shifted):  [<BOS>, v1, v2, ..., v(i-1)]")
    print("  Predict:                  v(i)")
    print("  Ground Truth Target:      v(i)")
    print("\nExample positions (showing first 5 tokens):")
    print("  Position | Decoder Input           | Predict  | Target")
    print("  ---------|-------------------------|----------|--------")
    print("     0     | <BOS>                   |   v1     |  v1")
    print("     1     | <BOS>, v1               |   v2     |  v2")
    print("     2     | <BOS>, v1, v2           |   v3     |  v3")
    print("     3     | <BOS>, v1, v2, v3       |   v4     |  v4")
    print("     4     | <BOS>, v1, v2, v3, v4   |   v5     |  v5")
    print("    ...    | ...                     |  ...     | ...")
    print("    76     | <BOS>, v1, ..., v76     |  v77     | v77")
    print("\nActual tensor values (first sample, first 5 dimensions):")
    print(f"  BOS token:     {shifted[0, 0, :5].cpu().numpy()}")
    print(f"  Target v1:     {target_embeddings[0, 0, :5].cpu().numpy()}")
    print(f"  Shifted pos 1: {shifted[0, 1, :5].cpu().numpy()}")
    print(f"  Target v2:     {target_embeddings[0, 1, :5].cpu().numpy()}")
    
    if save_dir:
        csv_dir = Path(save_dir) / 'autoregressive_matrices'
        csv_dir.mkdir(parents=True, exist_ok=True)
        shifted_csv = csv_dir / 'preparation_shifted_input.csv'
        with open(shifted_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Position'] + [f'Dim_{i}' for i in range(min(1280, 10))] + ['...'])
            shifted_np = shifted[0].cpu().numpy()
            for pos in range(77):
                writer.writerow([pos] + shifted_np[pos, :10].tolist() + ['...'])
        target_csv = csv_dir / 'preparation_target.csv'
        with open(target_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Position'] + [f'Dim_{i}' for i in range(min(1280, 10))] + ['...'])
            target_np = target_embeddings[0].cpu().numpy()
            for pos in range(77):
                writer.writerow([pos] + target_np[pos, :10].tolist() + ['...'])
    print("="*80 + "\n")


class Trainer:
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.attention_dir = self.output_dir / 'attention_maps'
        self.log_dir = self.output_dir / 'logs'
        
        if self.rank == 0:
            for dir_path in [self.checkpoint_dir, self.attention_dir, self.log_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        if self.rank == 0:
            print("Initializing model...")
        self.model = Speech2ImageModel(
            n_decoder_layers=config['n_decoder_layers'],
            n_state=config['n_state'],
            n_head=config['n_head'],
            mlp_ratio=config['mlp_ratio'],
            n_clusters=config.get('n_clusters', 590),
            centroids_path=config.get('centroids_path', 'auto')
        )
        self.model.to(self.device)
        
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank,
                             find_unused_parameters=False)
        
        if self.rank == 0:
            model_to_count = self.model.module if self.is_distributed else self.model
            total_params, trainable_params = model_to_count.count_parameters()
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        self.criterion = CombinedCentroidLoss(
            mse_weight=config.get('mse_weight', 1.0),
            cosine_weight=config.get('cosine_weight', 1.0),
            ce_weight=config.get('ce_weight', 0.5),
            contrastive_weight=config.get('contrastive_weight', 0.5)
        )
        if self.rank == 0:
            print(f"\nLoss Configuration:")
            print(f"  MSE weight: {config.get('mse_weight', 1.0)}")
            print(f"  Cosine weight: {config.get('cosine_weight', 1.0)}")
            print(f"  CE weight: {config.get('ce_weight', 0.5)}")
            print(f"  Contrastive weight: {config.get('contrastive_weight', 0.5)}")
        
        model_for_params = self.model.module if self.is_distributed else self.model
        self.optimizer = AdamW(
            model_for_params.get_trainable_parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['scheduler_t0'],
            T_mult=config['scheduler_t_mult']
        )
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0
        total_mse_loss = 0
        total_cosine_loss = 0
        total_ce_loss = 0
        total_contrastive_loss = 0
        total_cluster_accuracy = 0
        
        if self.rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            dmel_features    = batch['dmel_features'].to(self.device)
            clip_g_embeddings = batch['clip_g_embeddings'].to(self.device)
            
            if epoch == 1 and batch_idx == 0 and self.rank == 0:
                visualize_teacher_forcing_matrix(clip_g_embeddings, self.model,
                                                 batch_idx, save_dir=self.output_dir)
            
            save_intermediate = (epoch == 1 and batch_idx == 0 and self.rank == 0)
            
            predictions, cluster_logits = self.model(
                dmel_features, clip_g_embeddings,
                return_attention=False,
                use_teacher_forcing=True,
                save_intermediate=save_intermediate,
                return_cluster_logits=True
            )
            
            model_for_centroids = self.model.module if self.is_distributed else self.model
            centroids = model_for_centroids.centroids
            
            loss, loss_dict = self.criterion(predictions, cluster_logits, clip_g_embeddings, centroids)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.config['grad_clip'] > 0:
                model_for_params = self.model.module if self.is_distributed else self.model
                torch.nn.utils.clip_grad_norm_(
                    model_for_params.get_trainable_parameters(), self.config['grad_clip']
                )
            
            self.optimizer.step()
            self.scheduler.step(epoch + batch_idx / len(train_loader))
            
            total_loss += loss_dict['total_loss']
            total_mse_loss += loss_dict['mse_loss']
            total_cosine_loss += loss_dict['cosine_loss']
            total_ce_loss += loss_dict['ce_loss']
            total_contrastive_loss += loss_dict['contrastive_loss']
            total_cluster_accuracy += loss_dict['cluster_accuracy']
            self.global_step += 1
            
            if self.rank == 0:
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'mse': f"{loss_dict['mse_loss']:.4f}",
                    'cos': f"{loss_dict['cosine_loss']:.4f}",
                    'ce': f"{loss_dict['ce_loss']:.4f}",
                    'cont': f"{loss_dict['contrastive_loss']:.4f}",
                    'acc': f"{loss_dict['cluster_accuracy']:.3f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                self.log_metrics({
                    'epoch': epoch, 'step': self.global_step, 'batch_idx': batch_idx,
                    'train_loss': loss_dict['total_loss'],
                    'train_mse_loss': loss_dict['mse_loss'],
                    'train_cosine_loss': loss_dict['cosine_loss'],
                    'train_cosine_sim': loss_dict['avg_cosine_sim'],
                    'train_ce_loss': loss_dict['ce_loss'],
                    'train_contrastive_loss': loss_dict['contrastive_loss'],
                    'train_cluster_accuracy': loss_dict['cluster_accuracy'],
                    'train_avg_positive_sim': loss_dict['avg_positive_sim'],
                    'train_avg_negative_sim': loss_dict['avg_negative_sim'],
                    'train_sim_margin': loss_dict['sim_margin'],
                    'lr': self.optimizer.param_groups[0]['lr']
                })
        
        n = len(train_loader)
        return {
            'loss': total_loss / n,
            'mse_loss': total_mse_loss / n,
            'cosine_loss': total_cosine_loss / n,
            'ce_loss': total_ce_loss / n,
            'contrastive_loss': total_contrastive_loss / n,
            'cluster_accuracy': total_cluster_accuracy / n
        }
    
    def validate(self, val_loader, epoch, save_attention=False):
        self.model.eval()
        total_loss = 0
        total_mse_loss = 0
        total_cosine_loss = 0
        total_ce_loss = 0
        total_contrastive_loss = 0
        total_cluster_accuracy = 0
        attention_maps_to_save = []
        
        with torch.no_grad():
            if self.rank == 0:
                pbar = tqdm(val_loader, desc="Validation")
            else:
                pbar = val_loader
            
            for batch_idx, batch in enumerate(pbar):
                dmel_features    = batch['dmel_features'].to(self.device)
                clip_g_embeddings = batch['clip_g_embeddings'].to(self.device)
                
                if save_attention:
                    predictions, attention_weights, cluster_logits = self.model(
                        dmel_features, clip_g_embeddings,
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
                        dmel_features, clip_g_embeddings,
                        return_attention=False, use_teacher_forcing=True,
                        return_cluster_logits=True
                    )
                
                model_for_centroids = self.model.module if self.is_distributed else self.model
                centroids = model_for_centroids.centroids
                loss, loss_dict = self.criterion(predictions, cluster_logits, clip_g_embeddings, centroids)
                
                total_loss += loss_dict['total_loss']
                total_mse_loss += loss_dict['mse_loss']
                total_cosine_loss += loss_dict['cosine_loss']
                total_ce_loss += loss_dict['ce_loss']
                total_contrastive_loss += loss_dict['contrastive_loss']
                total_cluster_accuracy += loss_dict['cluster_accuracy']
                
                if self.rank == 0:
                    pbar.set_postfix({
                        'loss': f"{loss_dict['total_loss']:.4f}",
                        'mse': f"{loss_dict['mse_loss']:.4f}",
                        'cos': f"{loss_dict['cosine_loss']:.4f}",
                        'ce': f"{loss_dict['ce_loss']:.4f}",
                        'cont': f"{loss_dict['contrastive_loss']:.4f}",
                        'acc': f"{loss_dict['cluster_accuracy']:.3f}"
                    })
                    self.log_metrics({
                        'epoch': epoch, 'batch_idx': batch_idx,
                        'val_loss': loss_dict['total_loss'],
                        'val_mse_loss': loss_dict['mse_loss'],
                        'val_cosine_loss': loss_dict['cosine_loss'],
                        'val_cosine_sim': loss_dict['avg_cosine_sim'],
                        'val_ce_loss': loss_dict['ce_loss'],
                        'val_contrastive_loss': loss_dict['contrastive_loss'],
                        'val_cluster_accuracy': loss_dict['cluster_accuracy'],
                        'val_avg_positive_sim': loss_dict['avg_positive_sim'],
                        'val_avg_negative_sim': loss_dict['avg_negative_sim'],
                        'val_sim_margin': loss_dict['sim_margin']
                    }, log_type='val')
        
        if self.rank == 0 and save_attention and attention_maps_to_save:
            self.save_attention_maps(attention_maps_to_save, epoch)
        
        n = len(val_loader)
        return {
            'loss': total_loss / n,
            'mse_loss': total_mse_loss / n,
            'cosine_loss': total_cosine_loss / n,
            'ce_loss': total_ce_loss / n,
            'contrastive_loss': total_contrastive_loss / n,
            'cluster_accuracy': total_cluster_accuracy / n
        }
    
    def save_attention_maps(self, attention_data, epoch):
        save_path = self.attention_dir / f'attention_epoch_{epoch}.npz'
        all_attention = attention_data[0]['attention_weights']
        all_file_ids  = attention_data[0]['file_ids']
        np.savez_compressed(save_path, attention_weights=all_attention,
                            file_ids=all_file_ids, epoch=epoch)
        print(f"Saved attention maps to {save_path}, shape: {all_attention.shape}")
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        if self.rank != 0:
            return
        model_to_save = self.model.module if self.is_distributed else self.model
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        #if epoch % self.config['save_interval'] == 0:
        #    torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
        #if is_best:
        #    torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
        #    print(f"Saved best model with val loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_to_load = self.model.module if self.is_distributed else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def log_metrics(self, metrics, log_type='train'):
        log_file = self.log_dir / ('training_log.jsonl' if log_type == 'train' else 'validation_log.jsonl')
        with open(log_file, 'a') as f:
            json.dump(metrics, f)
            f.write('\n')
    
    def train(self, all_valid_ids, fixed_train_ids, fixed_val_ids):
        """Main training loop. Dataloaders are rebuilt each epoch for epochs > fixed_epochs."""
        config = self.config
        fixed_epochs = config.get('fixed_epochs', 50)
        
        if self.rank == 0:
            print("\nStarting training...")
            print(f"Full pool size: {len(all_valid_ids)}")
            print(f"Fixed sample epochs: 1-{fixed_epochs} ({len(fixed_train_ids)} train / {len(fixed_val_ids)} val)")
            print(f"Random sample epochs: {fixed_epochs+1}-{config['num_epochs']} (fresh 50k each epoch)")
            print(f"Batch size: {config['batch_size']} per GPU  |  GPUs: {self.world_size}")
            print(f"Epochs: {config['num_epochs']}")
        
        for epoch in range(self.current_epoch + 1, config['num_epochs'] + 1):
            if self.rank == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{config['num_epochs']}")
                print(f"{'='*60}")
            
            # Rebuild dataloaders each epoch (cheap for fixed; necessary for random)
            train_loader, val_loader = create_dataloaders(
                dmel_dir=config['dmel_dir'],
                clip_g_dir=config['clip_g_dir'],
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                train_split=config['train_split'],
                use_distributed=self.is_distributed,
                rank=self.rank,
                world_size=self.world_size,
                epoch=epoch,
                fixed_epochs=fixed_epochs,
                max_samples=config['max_samples'],
                all_valid_ids=all_valid_ids,
                fixed_train_ids=fixed_train_ids,
                fixed_val_ids=fixed_val_ids,
            )
            
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history.append(train_metrics)
            
            if self.rank == 0:
                print(f"\nTraining - Loss: {train_metrics['loss']:.4f}, "
                      f"MSE: {train_metrics['mse_loss']:.4f}, "
                      f"Cos: {train_metrics.get('cosine_loss', 0):.4f}, "
                      f"CE: {train_metrics.get('ce_loss', 0):.4f}, "
                      f"Cont: {train_metrics.get('contrastive_loss', 0):.4f}, "
                      f"Acc: {train_metrics.get('cluster_accuracy', 0):.3f}")
            
            val_metrics = self.validate(val_loader, epoch, save_attention=False)
            self.val_history.append(val_metrics)
            
            if self.rank == 0:
                print(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                      f"MSE: {val_metrics['mse_loss']:.4f}, "
                      f"Cos: {val_metrics.get('cosine_loss', 0):.4f}, "
                      f"CE: {val_metrics.get('ce_loss', 0):.4f}, "
                      f"Cont: {val_metrics.get('contrastive_loss', 0):.4f}, "
                      f"Acc: {val_metrics.get('cluster_accuracy', 0):.3f}")
            
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            self.save_checkpoint(epoch, val_metrics['loss'], is_best)
            
            if self.rank == 0:
                self.log_metrics({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_mse_loss': train_metrics['mse_loss'],
                    'train_cosine_loss': train_metrics.get('cosine_loss', 0),
                    'train_ce_loss': train_metrics.get('ce_loss', 0),
                    'train_contrastive_loss': train_metrics.get('contrastive_loss', 0),
                    'train_cluster_accuracy': train_metrics.get('cluster_accuracy', 0)
                }, log_type='train')
                self.log_metrics({
                    'epoch': epoch,
                    'val_loss': val_metrics['loss'],
                    'val_mse_loss': val_metrics['mse_loss'],
                    'val_cosine_loss': val_metrics.get('cosine_loss', 0),
                    'val_ce_loss': val_metrics.get('ce_loss', 0),
                    'val_contrastive_loss': val_metrics.get('contrastive_loss', 0),
                    'val_cluster_accuracy': val_metrics.get('cluster_accuracy', 0),
                    'best_val_loss': self.best_val_loss
                }, log_type='val')
            
            self.current_epoch = epoch
        
        if self.rank == 0:
            print("\n" + "="*60)
            print("Training completed!")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print("="*60)


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    rank, world_size, local_rank = setup_distributed()
    config = cfg.get_config()
    config.update({
        'dmel_dir':    '/scratch/IITB/ai-at-ieor/20004596/laion5b_data/dmel_embeddings/dmel',
        'clip_g_dir':  '/scratch/IITB/ai-at-ieor/20004596/Speech2Image/embeddings/clip_g',
        'output_dir':  '/scratch/IITB/ai-at-ieor/20004596/CLIP_G_with_DMEL/outputs',
        'resume_from': '/scratch/IITB/ai-at-ieor/20004596/CLIP_G_with_DMEL/outputs/checkpoints/latest.pt',
        'max_samples':  50000,
        'fixed_epochs': 50,
        'train_split':  0.98,
        'batch_size':   8,
        'num_workers':  4,
        'num_epochs':   1000,
        'learning_rate': 1e-4,
        'weight_decay':  0.01,
        'grad_clip':     1.0,
        'scheduler_t0':  10,
        'scheduler_t_mult': 2,
        'log_interval':  100,
        'save_interval': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    
    if rank == 0:
        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        print(f"📊 Dataset:")
        print(f"   Dmel dir:    {config['dmel_dir']}")
        print(f"   CLIP-G dir:  {config['clip_g_dir']}")
        print(f"   Max samples: {config['max_samples']} per epoch")
        print(f"   Fixed epochs: 1-{config['fixed_epochs']} (same 50k), "
              f"{config['fixed_epochs']+1}-{config['num_epochs']} (fresh 50k each)")
        print(f"   Train split: {config['train_split']}")
        print(f"   Batch size:  {config['batch_size']}")
        print(f"\n🧠 Model:")
        print(f"   Input: DmelInputProjection (80->380->680->980->1280)")
        print(f"   Decoder layers: {config['n_decoder_layers']}")
        print(f"   Hidden dim:     {config['n_state']}")
        print(f"   Attention heads:{config['n_head']}")
        print(f"   Clusters:       {config['n_clusters']}")
        print(f"\n🎯 Training:")
        print(f"   Epochs: {config['num_epochs']}")
        print(f"   LR: {config['learning_rate']}  WD: {config['weight_decay']}  Grad clip: {config['grad_clip']}")
        print(f"\n💾 Loss Weights:")
        print(f"   MSE: {config['mse_weight']}  Cosine: {config['cosine_weight']}  "
              f"CE: {config['ce_weight']}  Contrastive: {config['contrastive_weight']}")
        print(f"\n🔧 Distributed: world_size={world_size}, rank={rank}")
        print("="*80 + "\n")
    
    # ----------------------------------------------------------------
    # Pre-discover all valid IDs once (expensive scan done only once)
    # ----------------------------------------------------------------
    all_valid_ids = discover_all_valid_ids(config['dmel_dir'], config['clip_g_dir'])
    
    # Pre-compute the fixed 50k split for epochs 1-50 (seed=42)
    rng = random.Random(42)
    fixed_sample = rng.sample(all_valid_ids, min(config['max_samples'], len(all_valid_ids)))
    train_size   = int(config['train_split'] * len(fixed_sample))
    fixed_train_ids = fixed_sample[:train_size]
    fixed_val_ids   = fixed_sample[train_size:]
    if rank == 0:
        print(f"Fixed sample (epochs 1-{config['fixed_epochs']}): "
              f"{len(fixed_train_ids)} train / {len(fixed_val_ids)} val")
    
    # ----------------------------------------------------------------
    # Initialize trainer
    # ----------------------------------------------------------------
    trainer = Trainer(config, rank=rank, world_size=world_size)
    
    resume_path = config.get('resume_from', None)
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            print(f"\nResuming from checkpoint: {resume_path}")
        try:
            trainer.load_checkpoint(resume_path)
        except RuntimeError as e:
            if rank == 0:
                print(f"\n⚠️  Checkpoint incompatible (old Whisper model?), starting fresh.")
                print(f"   Reason: {str(e)[:200]}")
    
    try:
        trainer.train(all_valid_ids, fixed_train_ids, fixed_val_ids)
    except Exception as e:
        print("Training crashed:", e)
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()