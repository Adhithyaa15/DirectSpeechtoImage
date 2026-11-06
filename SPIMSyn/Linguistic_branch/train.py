import os

# Set cache directories to scratch to avoid home directory quota issues
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

from model import Speech2T5Model
from dataset import create_dataloaders
from losses import CombinedCentroidLoss
import config as cfg
import csv


def visualize_teacher_forcing_matrix(target_embeddings, model, batch_idx=0, save_dir=None):
    """
    Visualize the teacher forcing structure for debugging.
    Shows how targets are shifted to create the autoregressive training matrix.
    Also saves to CSV for detailed inspection.
    """
    if batch_idx > 0:  # Only print once per epoch
        return
    
    # Handle DDP-wrapped model
    actual_model = model.module if hasattr(model, 'module') else model
    
    B = target_embeddings.size(0)
    with torch.no_grad():
        shifted = actual_model.prepare_shifted_targets(target_embeddings)
    
    print("\n" + "="*80)
    print("TEACHER FORCING MATRIX STRUCTURE (First sample in batch)")
    print("="*80)
    print("Target shape: ", target_embeddings.shape, " [B, 256, 4096]")
    print("Shifted input shape:", shifted.shape, " [B, 256, 4096]")
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
    print("    255    | <BOS>, v1, ..., v255    |  v256    | v256")
    
    # Show actual tensor values for verification (first token's first 5 dims)
    print("\nActual tensor values (first sample, first 5 dimensions):")
    print(f"  BOS token:     {shifted[0, 0, :5].cpu().numpy()}")
    print(f"  Target v1:     {target_embeddings[0, 0, :5].cpu().numpy()}")
    print(f"  Shifted pos 1: {shifted[0, 1, :5].cpu().numpy()}")
    print(f"  Target v2:     {target_embeddings[0, 1, :5].cpu().numpy()}")
    
    # Save to CSV
    if save_dir:
        csv_dir = Path(save_dir) / 'autoregressive_matrices'
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Save shifted input matrix (first sample, all 256 positions, first 10 dims for readability)
        shifted_csv = csv_dir / 'preparation_shifted_input.csv'
        with open(shifted_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['Position'] + [f'Dim_{i}' for i in range(min(4096, 10))] + ['...'])
            # Data
            shifted_np = shifted[0].cpu().numpy()  # [256, 4096]
            for pos in range(256):
                row = [pos] + shifted_np[pos, :10].tolist() + ['...']
                writer.writerow(row)
        print(f"\nSaved shifted input matrix to: {shifted_csv}")
        
        # Save target matrix
        target_csv = csv_dir / 'preparation_target.csv'
        with open(target_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Position'] + [f'Dim_{i}' for i in range(min(4096, 10))] + ['...'])
            target_np = target_embeddings[0].cpu().numpy()  # [256, 4096]
            for pos in range(256):
                row = [pos] + target_np[pos, :10].tolist() + ['...']
                writer.writerow(row)
        print(f"Saved target matrix to: {target_csv}")
    
    print("="*80 + "\n")


class Trainer:
    def __init__(self, config, rank=0, world_size=1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories (only on rank 0)
        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.attention_dir = self.output_dir / 'attention_maps'
        self.log_dir = self.output_dir / 'logs'
        
        if self.rank == 0:
            for dir_path in [self.checkpoint_dir, self.attention_dir, self.log_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        if self.rank == 0:
            print("Initializing model...")
        self.model = Speech2T5Model(
            whisper_model_name=config['whisper_model_name'],
            n_decoder_layers=config['n_decoder_layers'],
            n_state=config['n_state'],
            n_head=config['n_head'],
            mlp_ratio=config['mlp_ratio'],
            n_clusters=config.get('n_clusters', 2950),
            centroids_path=config.get('centroids_path', 'auto')
        )
        self.model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        
        # Print model info (only on rank 0)
        if self.rank == 0:
            model_to_count = self.model.module if self.is_distributed else self.model
            total_params, trainable_params = model_to_count.count_parameters()
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize loss (Combined MSE + Cosine + Cluster CE + Contrastive)
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
        
        # Initialize optimizer
        model_for_params = self.model.module if self.is_distributed else self.model
        self.optimizer = AdamW(
            model_for_params.get_trainable_parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config['scheduler_t0'],
            T_mult=config['scheduler_t_mult']
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # History
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Set epoch for DistributedSampler
        if self.is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0
        total_mse_loss = 0
        total_cosine_loss = 0
        total_ce_loss = 0
        total_contrastive_loss = 0
        total_cluster_accuracy = 0
        
        # Only show progress bar on rank 0
        if self.rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_features = batch['input_features'].to(self.device)
            t5_embeddings = batch['t5_embeddings'].to(self.device)
            
            # Visualize teacher forcing structure (only first batch of first epoch)
            if epoch == 1 and batch_idx == 0 and self.rank == 0:
                visualize_teacher_forcing_matrix(t5_embeddings, self.model, batch_idx, save_dir=self.output_dir)
            
            # Forward pass with teacher forcing (autoregressive training)
            # Save intermediate outputs for first batch only
            save_intermediate = (epoch == 1 and batch_idx == 0 and self.rank == 0)
            
            # Forward pass
            predictions, cluster_logits = self.model(
                input_features, t5_embeddings, 
                return_attention=False, 
                use_teacher_forcing=True,
                save_intermediate=save_intermediate,
                return_cluster_logits=True
            )
            
            # Get centroids from model
            model_for_centroids = self.model.module if self.is_distributed else self.model
            centroids = model_for_centroids.centroids
            
            # Compute loss
            loss, loss_dict = self.criterion(predictions, cluster_logits, t5_embeddings, centroids)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['grad_clip'] > 0:
                model_for_params = self.model.module if self.is_distributed else self.model
                torch.nn.utils.clip_grad_norm_(
                    model_for_params.get_trainable_parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            self.scheduler.step(epoch + batch_idx / len(train_loader))
            
            # Update metrics
            total_loss += loss_dict['total_loss']
            total_mse_loss += loss_dict['mse_loss']
            total_cosine_loss += loss_dict['cosine_loss']
            total_ce_loss += loss_dict['ce_loss']
            total_contrastive_loss += loss_dict['contrastive_loss']
            total_cluster_accuracy += loss_dict['cluster_accuracy']
            self.global_step += 1
            
            # Update progress bar (only rank 0)
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
            
            # Log every batch (only rank 0)
            if self.rank == 0:
                self.log_metrics({
                    'epoch': epoch,
                    'step': self.global_step,
                    'batch_idx': batch_idx,
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
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_mse_loss = total_mse_loss / len(train_loader)
        avg_cosine_loss = total_cosine_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_contrastive_loss = total_contrastive_loss / len(train_loader)
        avg_cluster_accuracy = total_cluster_accuracy / len(train_loader)
        
        return {
            'loss': avg_loss,
            'mse_loss': avg_mse_loss,
            'cosine_loss': avg_cosine_loss,
            'ce_loss': avg_ce_loss,
            'contrastive_loss': avg_contrastive_loss,
            'cluster_accuracy': avg_cluster_accuracy
        }
    
    def validate(self, val_loader, epoch, save_attention=False):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mse_loss = 0
        total_cosine_loss = 0
        total_ce_loss = 0
        total_contrastive_loss = 0
        total_cluster_accuracy = 0
        
        attention_maps_to_save = []
        
        with torch.no_grad():
            # Only show progress bar on rank 0
            if self.rank == 0:
                pbar = tqdm(val_loader, desc="Validation")
            else:
                pbar = val_loader
            
            for batch_idx, batch in enumerate(pbar):
                input_features = batch['input_features'].to(self.device)
                t5_embeddings = batch['t5_embeddings'].to(self.device)
                
                # Forward pass with teacher forcing and optional attention
                if save_attention:
                    predictions, attention_weights, cluster_logits = self.model(
                        input_features,
                        t5_embeddings,
                        return_attention=True,
                        use_teacher_forcing=True,
                        return_cluster_logits=True
                    )
                    
                    # Store attention maps for first batch
                    if batch_idx == 0:
                        attention_maps_to_save.append({
                            'attention_weights': attention_weights.cpu().numpy(),
                            'file_ids': batch['file_ids']
                        })
                else:
                    predictions, cluster_logits = self.model(
                        input_features, t5_embeddings, 
                        return_attention=False, 
                        use_teacher_forcing=True,
                        return_cluster_logits=True
                    )
                
                # Get centroids from model
                model_for_centroids = self.model.module if self.is_distributed else self.model
                centroids = model_for_centroids.centroids
                
                # Compute loss
                loss, loss_dict = self.criterion(predictions, cluster_logits, t5_embeddings, centroids)
                
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
                    
                    # Log every validation batch
                    self.log_metrics({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
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
        
        # Save attention maps (only rank 0)
        if self.rank == 0 and save_attention and attention_maps_to_save:
            self.save_attention_maps(attention_maps_to_save, epoch)
        
        # Calculate average losses
        avg_loss = total_loss / len(val_loader)
        avg_mse_loss = total_mse_loss / len(val_loader)
        avg_cosine_loss = total_cosine_loss / len(val_loader)
        avg_ce_loss = total_ce_loss / len(val_loader)
        avg_contrastive_loss = total_contrastive_loss / len(val_loader)
        avg_cluster_accuracy = total_cluster_accuracy / len(val_loader)
        
        return {
            'loss': avg_loss,
            'mse_loss': avg_mse_loss,
            'cosine_loss': avg_cosine_loss,
            'ce_loss': avg_ce_loss,
            'contrastive_loss': avg_contrastive_loss,
            'cluster_accuracy': avg_cluster_accuracy
        }
    
    def save_attention_maps(self, attention_data, epoch):
        """Save cross attention maps from the final decoder block"""
        save_path = self.attention_dir / f'attention_epoch_{epoch}.npz'
        
        # Concatenate all attention weights
        all_attention = attention_data[0]['attention_weights']  # [B, n_heads, 256, 1500]
        all_file_ids = attention_data[0]['file_ids']
        
        # Save as compressed numpy file
        np.savez_compressed(
            save_path,
            attention_weights=all_attention,
            file_ids=all_file_ids,
            epoch=epoch
        )
        
        print(f"Saved attention maps to {save_path}")
        print(f"Attention shape: {all_attention.shape}")
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint (only on rank 0)"""
        if self.rank != 0:
            return
        
        # Get state dict from unwrapped model
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
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        if epoch % self.config['save_interval'] == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load into unwrapped model if DDP
        model_to_load = self.model.module if self.is_distributed else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def log_metrics(self, metrics, log_type='train'):
        """
        Log metrics to JSONL file (one JSON object per line for each batch)
        
        Files created:
        - outputs/logs/training_log.jsonl: All training batch metrics
        - outputs/logs/validation_log.jsonl: All validation batch metrics
        
        Each line contains complete loss breakdown for that batch including:
        total_loss, mse_loss, cosine_loss, ce_loss, contrastive_loss, 
        cluster_accuracy, similarities, etc.
        """
        if log_type == 'train':
            log_file = self.log_dir / 'training_log.jsonl'
        else:
            log_file = self.log_dir / 'validation_log.jsonl'
        
        with open(log_file, 'a') as f:
            json.dump(metrics, f)
            f.write('\n')
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        if self.rank == 0:
            print("\nStarting training...")
            print(f"Training samples: {len(train_loader.dataset)}")
            print(f"Validation samples: {len(val_loader.dataset)}")
            print(f"Batch size: {self.config['batch_size']} per GPU")
            print(f"Number of GPUs: {self.world_size}")
            print(f"Effective batch size: {self.config['batch_size'] * self.world_size}")
            print(f"Number of epochs: {self.config['num_epochs']}")
            print(f"\nBatch-wise logs will be saved to:")
            print(f"  Training: {self.log_dir / 'training_log.jsonl'}")
            print(f"  Validation: {self.log_dir / 'validation_log.jsonl'}")
        
        for epoch in range(self.current_epoch + 1, self.config['num_epochs'] + 1):
            if self.rank == 0:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch}/{self.config['num_epochs']}")
                print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history.append(train_metrics)
            
            if self.rank == 0:
                print(f"\nTraining - Loss: {train_metrics['loss']:.4f}, "
                      f"MSE: {train_metrics['mse_loss']:.4f}, "
                      f"Cos: {train_metrics.get('cosine_loss', 0):.4f}, "
                      f"CE: {train_metrics.get('ce_loss', 0):.4f}, "
                      f"Cont: {train_metrics.get('contrastive_loss', 0):.4f}, "
                      f"Acc: {train_metrics.get('cluster_accuracy', 0):.3f}")
            
            # Validate (save attention maps at the end of each epoch)
            val_metrics = self.validate(val_loader, epoch, save_attention=True)
            self.val_history.append(val_metrics)
            
            if self.rank == 0:
                print(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                      f"MSE: {val_metrics['mse_loss']:.4f}, "
                      f"Cos: {val_metrics.get('cosine_loss', 0):.4f}, "
                      f"CE: {val_metrics.get('ce_loss', 0):.4f}, "
                      f"Cont: {val_metrics.get('contrastive_loss', 0):.4f}, "
                      f"Acc: {val_metrics.get('cluster_accuracy', 0):.3f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(epoch, val_metrics['loss'], is_best)
            
            # Log epoch metrics (only rank 0)
            if self.rank == 0:
                # Log training epoch summary
                self.log_metrics({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_mse_loss': train_metrics['mse_loss'],
                    'train_cosine_loss': train_metrics.get('cosine_loss', 0),
                    'train_ce_loss': train_metrics.get('ce_loss', 0),
                    'train_contrastive_loss': train_metrics.get('contrastive_loss', 0),
                    'train_cluster_accuracy': train_metrics.get('cluster_accuracy', 0)
                }, log_type='train')
                
                # Log validation epoch summary
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
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Load base configuration from config.py
    config = cfg.get_config()
    
    # Override/add training-specific settings
    config.update({
        # Paths
        'speech_dir': '/scratch/IITB/ai-at-ieor/20004596/laion5b_data/Speechs',
        'caption_dir': '/scratch/IITB/ai-at-ieor/20004596/laion5b_data/captions',
        't5_encoder_path': '/scratch/IITB/ai-at-ieor/20004596/sd35/text_encoder_3',
        't5_tokenizer_path': '/scratch/IITB/ai-at-ieor/20004596/sd35/tokenizer_3',
        'output_dir': '/scratch/IITB/ai-at-ieor/20004596/T5_Alignment_V0/outputs',
        'resume_from': '/scratch/IITB/ai-at-ieor/20004596/T5_Alignment_V0/outputs/checkpoints/latest.pt',

        
        # Dataset
        # Use the full dataset (no max_samples limit)
        'max_samples': 50000,
        'train_split': 0.98,
        'batch_size': 4,  # Reduced from 8 to avoid OOM (T5 encoder + training model)
        'num_workers': 0,  # Single process - no multiprocessing overhead for testing
        
        # Training
        # Run for 200 full epochs over the dataset
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        
        # Scheduler
        'scheduler_t0': 10,
        'scheduler_t_mult': 2,
        
        # Logging
        'log_interval': 100,
        'save_interval': 15,
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    })
    
    if rank == 0:
        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        print(f"📊 Dataset:")
        print(f"   Speech dir: {config['speech_dir']}")
        print(f"   Caption dir: {config['caption_dir']}")
        print(f"   T5 encoder: {config['t5_encoder_path']}")
        print(f"   T5 tokenizer: {config['t5_tokenizer_path']}")
        print(f"   Max samples: {config['max_samples']}")
        print(f"   Train split: {config['train_split']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"\n🧠 Model:")
        print(f"   Whisper: {config['whisper_model_name']}")
        print(f"   Decoder layers: {config['n_decoder_layers']}")
        print(f"   Hidden dim: {config['n_state']}")
        print(f"   Attention heads: {config['n_head']}")
        print(f"   Clusters: {config['n_clusters']}")
        print(f"\n🎯 Training:")
        print(f"   Epochs: {config['num_epochs']}")
        print(f"   Learning rate: {config['learning_rate']}")
        print(f"   Weight decay: {config['weight_decay']}")
        print(f"   Grad clip: {config['grad_clip']}")
        print(f"\n💾 Loss Weights:")
        print(f"   MSE: {config['mse_weight']}")
        print(f"   Cosine: {config['cosine_weight']}")
        print(f"   CE (cluster): {config['ce_weight']}")
        print(f"   Contrastive: {config['contrastive_weight']}")
        print(f"\n🔧 Distributed:")
        print(f"   World size: {world_size}")
        print(f"   Rank: {rank}")
        print("="*80 + "\n")
    
    # Create dataloaders
    if rank == 0:
        print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        speech_dir=config['speech_dir'],
        caption_dir=config['caption_dir'],
        t5_encoder_path=config['t5_encoder_path'],
        t5_tokenizer_path=config['t5_tokenizer_path'],
        device='cuda',
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        max_samples=config['max_samples'],
        train_split=config['train_split'],
        use_distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
        index_file=config.get('dataset_index_file', None)
    )
    
    # Initialize trainer
    trainer = Trainer(config, rank=rank, world_size=world_size)

    # Resume if checkpoint exists
    resume_path = config.get('resume_from', None)
    if resume_path and os.path.exists(resume_path):
        if rank == 0:
            print(f"\nResuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(resume_path)

    # Train
    try:
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print("Training crashed:", e)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
