import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import csv
from pathlib import Path
import numpy as np

# Import centroid loading utilities
try:
    from load_centroids import load_centroids_from_csv, find_latest_centroid_file
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from load_centroids import load_centroids_from_csv, find_latest_centroid_file


class LearnedPositionalEmbedding(nn.Module):
    """Learned (trainable) positional embeddings [max_positions, embedding_dim]."""
    def __init__(self, max_positions=77, embedding_dim=1280):
        super().__init__()
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_positions, embedding_dim)
        nn.init.normal_(pe, mean=0.0, std=0.02)
        self.pe = nn.Parameter(pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class DmelInputProjection(nn.Module):
    """Project dmel embeddings from [B, T, 80] to [B, T, 768] for cross-attention."""
    def __init__(self, input_dim=80, output_dim=1280):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 380)
        self.fc2 = nn.Linear(380, 680)
        self.fc3 = nn.Linear(680, 980)
        self.fc4 = nn.Linear(980, output_dim)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x: [B, T, 80]
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        x = self.gelu(self.fc3(x))
        x = self.fc4(x)
        x = self.layer_norm(x)
        return x  # [B, T, 1280]


class MultiHeadAttention(nn.Module):
    """Multi-head attention module similar to Whisper implementation"""
    def __init__(self, n_state=1280, n_head=12):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head
        
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
    
    def create_causal_mask(self, seq_len, device):
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask
        
    def forward(self, x, xa=None, mask=None, return_attention=False, use_causal_mask=False,
                save_self_attn=False, save_dir=None, block_id=None):
        q = self.query(x)
        
        if xa is None:
            k = self.key(x)
            v = self.value(x)
            if use_causal_mask and mask is None:
                mask = self.create_causal_mask(x.size(1), x.device)
        else:
            k = self.key(xa)
            v = self.value(xa)
        
        B, seq_len, _ = q.shape
        _, seq_len_kv, _ = k.shape
        
        q = q.view(B, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len_kv, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len_kv, self.n_head, self.head_dim).transpose(1, 2)
        
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if mask is not None:
            attn_weights = attn_weights + mask
        
        if save_self_attn and xa is None and save_dir and block_id is not None:
            self._save_self_attention_to_csv(attn_weights, save_dir, block_id)
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.n_state)
        output = self.out(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output
    
    def _save_self_attention_to_csv(self, attn_weights, save_dir, block_id):
        csv_dir = Path(save_dir) / 'self_attention_weights'
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_file = csv_dir / f'self_attn_block_{block_id}_after_mask.csv'
        attn_matrix = attn_weights[0, 0].detach().cpu().numpy()
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Query/Key'] + [f'K_{i}' for i in range(attn_matrix.shape[1])])
            for i, row in enumerate(attn_matrix):
                writer.writerow([f'Q_{i}'] + row.tolist())


class ResidualAttentionBlock(nn.Module):
    """Transformer decoder block with self-attention and cross-attention"""
    def __init__(self, n_state=1280, n_head=12, mlp_ratio=4.0):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(n_state, n_head)
        self.cross_attn_ln = nn.LayerNorm(n_state)
        
        mlp_dim = int(n_state * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(n_state, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)
        
    def forward(self, x, xa, return_attention=False, use_causal_mask=True,
                save_self_attn=False, save_dir=None, block_id=None):
        # Self-attention with causal masking
        x = x + self.attn(self.attn_ln(x), use_causal_mask=use_causal_mask,
                          save_self_attn=save_self_attn, save_dir=save_dir, block_id=block_id)
        # Cross-attention
        if return_attention:
            cross_attn_out, attn_weights = self.cross_attn(
                self.cross_attn_ln(x), xa, return_attention=True
            )
            x = x + cross_attn_out
        else:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
            attn_weights = None
        # MLP
        x = x + self.mlp(self.mlp_ln(x))
        
        if return_attention:
            return x, attn_weights
        return x


class CustomDecoder(nn.Module):
    """Custom decoder with 8 blocks"""
    def __init__(self, n_layers=8, n_state=1280, n_head=12, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, mlp_ratio)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(n_state)
        self.n_layers = n_layers
        
    def forward(self, x, xa, return_attention=False, use_causal_mask=True,
                save_intermediate=False, save_dir=None, save_self_attn=False):
        attention_weights = None
        for i, block in enumerate(self.blocks):
            if return_attention and i == self.n_layers - 1:
                x, attention_weights = block(x, xa, return_attention=True,
                                             use_causal_mask=use_causal_mask,
                                             save_self_attn=save_self_attn,
                                             save_dir=save_dir, block_id=i)
            else:
                x = block(x, xa, return_attention=False, use_causal_mask=use_causal_mask,
                          save_self_attn=save_self_attn, save_dir=save_dir, block_id=i)
            if save_intermediate and save_dir:
                self._save_block_output_to_csv(x, i, save_dir)
        x = self.ln(x)
        if return_attention:
            return x, attention_weights
        return x
    
    def _save_block_output_to_csv(self, x, block_idx, save_dir):
        csv_dir = Path(save_dir) / 'autoregressive_matrices'
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_file = csv_dir / f'decoder_block_{block_idx}_output.csv'
        x_np = x[0].detach().cpu().numpy()
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Position'] + [f'Dim_{i}' for i in range(min(1280, 10))] + ['...'])
            for pos in range(77):
                writer.writerow([pos] + x_np[pos, :10].tolist() + ['...'])


class Speech2ImageModel(nn.Module):
    """Complete Speech to Image model (dmel → CLIP-G embeddings)"""
    def __init__(
        self,
        n_decoder_layers=8,
        n_state=1280,
        n_head=12,
        mlp_ratio=4.0,
        n_clusters=590,
        centroids_path=None
    ):
        super().__init__()
        
        # Dmel input projection: [B, T, 80] -> [B, T, 1280]
        print("Initializing DmelInputProjection (80 -> 380 -> 680 -> 980 -> 1280)...")
        self.dmel_projection = DmelInputProjection(input_dim=80, output_dim=1280)
        
        # Custom decoder (unchanged)
        self.decoder = CustomDecoder(
            n_layers=n_decoder_layers,
            n_state=n_state,
            n_head=n_head,
            mlp_ratio=mlp_ratio
        )
        
        # Learned positional embeddings for decoder input (CLIP tokens)
        self.positional_embedding = LearnedPositionalEmbedding(
            max_positions=77,
            embedding_dim=n_state
        )
        
        # Learnable BOS token for autoregressive generation
        self.bos_token = nn.Parameter(torch.randn(1, 1, n_state))
        
        # Cluster Classification Head
        self.n_clusters = n_clusters
        self.cluster_head = nn.Sequential(
            nn.Linear(n_state, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, n_clusters)
        )
        
        # Load and register centroids as frozen buffer
        if centroids_path is not None:
            if centroids_path == 'auto':
                base_dir = '/scratch/IITB/ai-at-ieor/20004596/CLIP_G_with_DMEL/outputs/kmeans'
                centroid_file = find_latest_centroid_file(base_dir, n_clusters=n_clusters, format='csv')
                if centroid_file is None:
                    print(f"WARNING: No centroid file found in {base_dir}. Cluster losses disabled.")
                    self.register_buffer('centroids', None)
                else:
                    centroids = load_centroids_from_csv(centroid_file)
                    self.register_buffer('centroids', torch.from_numpy(centroids).float())
                    print(f"✓ Loaded {self.n_clusters} centroids as frozen buffer")
            else:
                centroids = load_centroids_from_csv(centroids_path)
                self.register_buffer('centroids', torch.from_numpy(centroids).float())
                print(f"✓ Loaded {self.n_clusters} centroids from {centroids_path}")
        else:
            self.register_buffer('centroids', None)
            print("WARNING: No centroids loaded. Cluster losses disabled.")
    
    def prepare_shifted_targets(self, target_embeddings):
        """
        Prepare shifted targets for teacher forcing.
        Input:  [v1, v2, ..., v77]
        Output: [<BOS>, v1, v2, ..., v76]
        """
        B = target_embeddings.size(0)
        bos_tokens = self.bos_token.expand(B, 1, -1)  # [B, 1, 1280]
        shifted = torch.cat([bos_tokens, target_embeddings[:, :-1, :]], dim=1)  # [B, 77, 1280]
        return shifted
        
    def forward(self, dmel_features, target_embeddings, return_attention=False,
                use_teacher_forcing=True, save_intermediate=False, return_cluster_logits=False):
        """
        dmel_features: [B, T, 80]  (variable T, padded in batch)
        target_embeddings: [B, 77, 1280]
        """
        # Project dmel to decoder dimension
        projected_features = self.dmel_projection(dmel_features)  # [B, T, 1280]
        
        # Prepare decoder input with teacher forcing
        if use_teacher_forcing:
            decoder_input = self.prepare_shifted_targets(target_embeddings)  # [B, 77, 1280]
        else:
            decoder_input = target_embeddings
        
        # Add positional embeddings
        decoder_input = self.positional_embedding(decoder_input)  # [B, 77, 1280]
        
        # Save dir for intermediate outputs
        save_dir = None
        save_self_attn = False
        if save_intermediate:
            save_dir = '/scratch/IITB/ai-at-ieor/20004596/CLIP_L_with_DMEL/outputs'
            save_self_attn = True
        
        # Decoder
        if return_attention:
            output, attention_weights = self.decoder(
                decoder_input, projected_features,
                return_attention=True, use_causal_mask=use_teacher_forcing,
                save_intermediate=save_intermediate, save_dir=save_dir, save_self_attn=save_self_attn
            )
        else:
            output = self.decoder(
                decoder_input, projected_features,
                use_causal_mask=use_teacher_forcing,
                save_intermediate=save_intermediate, save_dir=save_dir, save_self_attn=save_self_attn
            )
        
        # Cluster logits
        if return_cluster_logits and self.centroids is not None:
            cluster_logits = self.cluster_head(output)  # [B, 77, n_clusters]
            if return_attention:
                return output, attention_weights, cluster_logits
            else:
                return output, cluster_logits
        
        if return_attention:
            return output, attention_weights
        return output
    
    @torch.no_grad()
    def generate(self, dmel_features, max_length=77, temperature=1.0):
        """
        Autoregressive generation for inference.
        dmel_features: [B, T, 80]
        Returns: generated_embeddings [B, max_length, 1280]
        """
        self.eval()
        B = dmel_features.size(0)
        
        # Project dmel
        projected_features = self.dmel_projection(dmel_features)  # [B, T, 1280]
        
        # Start with BOS token
        current_sequence = self.bos_token.expand(B, -1, -1)  # [B, 1, 1280]
        current_sequence = self.positional_embedding(current_sequence)
        
        for i in range(max_length):
            decoder_output = self.decoder(
                current_sequence, projected_features,
                return_attention=False, use_causal_mask=True,
                save_intermediate=False, save_dir=None, save_self_attn=False
            )
            next_token = decoder_output[:, -1:, :]  # [B, 1, 1280]
            current_sequence = torch.cat([current_sequence, next_token], dim=1)
        
        return current_sequence[:, 1:, :]  # [B, 77, 1280]
    
    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable