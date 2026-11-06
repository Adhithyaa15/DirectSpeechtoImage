import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel
import math
import os
import csv
from pathlib import Path
import numpy as np

# Set to use local files only (offline mode)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Import centroid loading utilities
try:
    from load_centroids import load_centroids_from_csv, find_latest_centroid_file
except ImportError:
    # Fallback if imported from different location
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from load_centroids import load_centroids_from_csv, find_latest_centroid_file


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned (trainable) positional embeddings.
    Stores a parameter table of shape [max_positions, embedding_dim].
    """
    def __init__(self, max_positions=256, embedding_dim=4096):
        super().__init__()
        self.max_positions = max_positions
        self.embedding_dim = embedding_dim

        # Learnable positional embeddings: register as Parameter so they're trainable
        pe = torch.zeros(max_positions, embedding_dim)
        # Small random init like many token embeddings
        nn.init.normal_(pe, mean=0.0, std=0.02)
        self.pe = nn.Parameter(pe)

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, embedding_dim]
        Returns:
            x + positional embeddings: [B, seq_len, embedding_dim]
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class MLPProjection(nn.Module):
    """MLP to project Whisper encoder output from [B, 1500, 1024] to [B, 1500, 4096]"""
    def __init__(self, input_dim=1024, hidden_dim=2560, output_dim=4096):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # x: [B, 1500, 1024]
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x  # [B, 1500, 4096]


class MultiHeadAttention(nn.Module):
    """Multi-head attention module similar to Whisper implementation"""
    def __init__(self, n_state=4096, n_head=16):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head
        
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
    
    def create_causal_mask(self, seq_len, device):
        """Create causal mask for autoregressive attention"""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)  # Upper triangular (excluding diagonal)
        return mask
        
    def forward(self, x, xa=None, mask=None, return_attention=False, use_causal_mask=False, 
                save_self_attn=False, save_dir=None, block_id=None):
        """
        x: query input [B, seq_len, n_state]
        xa: key/value input for cross attention [B, seq_len_kv, n_state]. If None, self-attention
        mask: attention mask
        return_attention: whether to return attention weights
        use_causal_mask: whether to apply causal masking (for autoregressive self-attention)
        save_self_attn: whether to save self-attention weights to CSV
        save_dir: directory to save CSV files
        block_id: decoder block ID for naming the CSV file
        """
        q = self.query(x)
        
        if xa is None:
            # Self-attention
            k = self.key(x)
            v = self.value(x)
            # Apply causal mask for autoregressive generation
            if use_causal_mask and mask is None:
                mask = self.create_causal_mask(x.size(1), x.device)
        else:
            # Cross-attention
            k = self.key(xa)
            v = self.value(xa)
        
        # Reshape for multi-head attention
        B, seq_len, _ = q.shape
        _, seq_len_kv, _ = k.shape
        
        q = q.view(B, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, seq_len, head_dim]
        k = k.view(B, seq_len_kv, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, seq_len_kv, head_dim]
        v = v.view(B, seq_len_kv, self.n_head, self.head_dim).transpose(1, 2)  # [B, n_head, seq_len_kv, head_dim]
        
        # Attention scores
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, n_head, seq_len, seq_len_kv]
        
        if mask is not None:
            attn_weights = attn_weights + mask
        
        # Save self-attention weights after masking (before softmax)
        if save_self_attn and xa is None and save_dir and block_id is not None:
            self._save_self_attention_to_csv(attn_weights, save_dir, block_id)
        
        attn_weights = F.softmax(attn_weights, dim=-1)  # [B, n_head, seq_len, seq_len_kv]
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, n_head, seq_len, head_dim]
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.n_state)
        
        # Output projection
        output = self.out(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output
    
    def _save_self_attention_to_csv(self, attn_weights, save_dir, block_id):
        """
        Save self-attention weights to CSV after masking (before softmax).
        Saves first sample, first head for readability.
        
        Args:
            attn_weights: [B, n_head, seq_len, seq_len] attention scores after masking
            save_dir: directory to save CSV
            block_id: decoder block identifier
        """
        csv_dir = Path(save_dir) / 'self_attention_weights'
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = csv_dir / f'self_attn_block_{block_id}_after_mask.csv'
        
        # Extract first sample, first head: [seq_len, seq_len]
        attn_matrix = attn_weights[0, 0].detach().cpu().numpy()
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header row
            header = ['Query/Key'] + [f'K_{i}' for i in range(attn_matrix.shape[1])]
            writer.writerow(header)
            # Data rows
            for i, row in enumerate(attn_matrix):
                writer.writerow([f'Q_{i}'] + row.tolist())


class ResidualAttentionBlock(nn.Module):
    """Transformer decoder block with self-attention and cross-attention"""
    def __init__(self, n_state=4096, n_head=16, mlp_ratio=4.0):
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
        """
        x: target embeddings [B, 256, 4096]
        xa: source embeddings from MLP projection [B, 1500, 4096]
        return_attention: whether to return cross attention weights
        use_causal_mask: whether to use causal masking in self-attention
        save_self_attn: whether to save self-attention weights
        save_dir: directory to save attention weights
        block_id: block identifier for file naming
        """
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
    """Custom decoder with 6 blocks"""
    def __init__(self, n_layers=6, n_state=4096, n_head=16, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualAttentionBlock(n_state, n_head, mlp_ratio)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(n_state)
        self.n_layers = n_layers
        
    def forward(self, x, xa, return_attention=False, use_causal_mask=True, save_intermediate=False, 
                save_dir=None, save_self_attn=False):
        """
        x: target embeddings [B, 256, 4096]
        xa: source embeddings from MLP projection [B, 1500, 4096]
        return_attention: if True, return attention from last block
        use_causal_mask: whether to use causal masking in self-attention
        save_intermediate: if True, save outputs after each block to CSV
        save_dir: directory to save intermediate outputs
        save_self_attn: if True, save self-attention weights from all blocks
        """
        attention_weights = None
        
        for i, block in enumerate(self.blocks):
            # Only return attention from the last block
            if return_attention and i == self.n_layers - 1:
                x, attention_weights = block(x, xa, return_attention=True, use_causal_mask=use_causal_mask,
                                            save_self_attn=save_self_attn, save_dir=save_dir, block_id=i)
            else:
                x = block(x, xa, return_attention=False, use_causal_mask=use_causal_mask,
                         save_self_attn=save_self_attn, save_dir=save_dir, block_id=i)
            
            # Save intermediate output after cross-attention
            if save_intermediate and save_dir:
                self._save_block_output_to_csv(x, i, save_dir)
        
        x = self.ln(x)
        
        if return_attention:
            return x, attention_weights
        return x
    
    def _save_block_output_to_csv(self, x, block_idx, save_dir):
        """Save decoder block output to CSV (first sample only)"""
        import csv
        from pathlib import Path
        
        csv_dir = Path(save_dir) / 'autoregressive_matrices'
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = csv_dir / f'decoder_block_{block_idx}_output.csv'
        
        # Save first sample, all 256 positions, first 10 dimensions
        x_np = x[0].detach().cpu().numpy()  # [256, 4096]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['Position'] + [f'Dim_{i}' for i in range(min(4096, 10))] + ['...'])
            # Data
            for pos in range(256):
                row = [pos] + x_np[pos, :10].tolist() + ['...']
                writer.writerow(row)


class Speech2T5Model(nn.Module):
    """Complete Speech to T5 model"""
    def __init__(
        self,
        whisper_model_name="openai/whisper-medium",
        n_decoder_layers=6,
        n_state=4096,
        n_head=16,
        mlp_ratio=4.0,
        n_clusters=2950,
        centroids_path=None
    ):
        super().__init__()
        
        # Load Whisper encoder (frozen)
        print(f"Loading Whisper model: {whisper_model_name}")
        self.whisper = WhisperModel.from_pretrained(
            whisper_model_name,
            local_files_only=True
        )
        
        # Freeze Whisper encoder
        for param in self.whisper.parameters():
            param.requires_grad = False
        
        # MLP projection layer
        self.mlp_projection = MLPProjection(input_dim=1024, output_dim=4096)
        
        # Custom decoder
        self.decoder = CustomDecoder(
            n_layers=n_decoder_layers,
            n_state=n_state,
            n_head=n_head,
            mlp_ratio=mlp_ratio
        )
        
        # Learned positional embeddings for decoder input (T5 tokens)
        self.positional_embedding = LearnedPositionalEmbedding(
            max_positions=256,  # 256 T5 tokens
            embedding_dim=n_state
        )
        
        # Learnable BOS token for autoregressive generation
        self.bos_token = nn.Parameter(torch.randn(1, 1, n_state))
        
        # ============================================================
        # Load and register centroids as frozen buffer
        # This must be done before cluster head to get correct n_clusters
        # ============================================================
        self.n_clusters = n_clusters  # Default from config
        if centroids_path is not None:
            if centroids_path == 'auto':
                # Auto-detect latest centroid file (ignore n_clusters parameter)
                base_dir = '/scratch/IITB/ai-at-ieor/20004596/T5_Alignment_V0/outputs/kmeans'
                centroid_file = find_latest_centroid_file(base_dir, n_clusters=None, format='csv')
                if centroid_file is None:
                    print(f"WARNING: No centroid file found in {base_dir}")
                    print(f"Cluster losses will not be available.")
                    self.register_buffer('centroids', None)
                else:
                    centroids = load_centroids_from_csv(centroid_file)
                    # Update n_clusters to match loaded centroids
                    self.n_clusters = centroids.shape[0]
                    centroids_tensor = torch.from_numpy(centroids).float()
                    self.register_buffer('centroids', centroids_tensor)
                    print(f"✓ Loaded {self.n_clusters} centroids from {centroid_file.name}")
            else:
                # Load from specified path
                centroids = load_centroids_from_csv(centroids_path)
                self.n_clusters = centroids.shape[0]
                centroids_tensor = torch.from_numpy(centroids).float()
                self.register_buffer('centroids', centroids_tensor)
                print(f"✓ Loaded {self.n_clusters} centroids from {centroids_path}")
        else:
            # No centroids - cluster losses won't work
            self.register_buffer('centroids', None)
            print("WARNING: No centroids loaded. Cluster losses disabled.")
        
        # ============================================================
        # Cluster Classification Head (created after loading centroids)
        # ============================================================
        self.cluster_head = nn.Sequential(
            nn.Linear(n_state, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, self.n_clusters)
        )
    
    def prepare_shifted_targets(self, target_embeddings):
        """
        Prepare shifted targets for teacher forcing.
        Input:  [v1, v2, v3, v4, ..., v256]
        Output: [<BOS>, v1, v2, v3, ..., v255]  (prepend BOS, remove last)
        
        Args:
            target_embeddings: [B, 256, 4096]
        Returns:
            shifted_targets: [B, 256, 4096] with BOS prepended
        """
        B = target_embeddings.size(0)
        # Expand BOS token for batch
        bos_tokens = self.bos_token.expand(B, 1, -1)  # [B, 1, 4096]
        # Prepend BOS, remove last token
        shifted = torch.cat([bos_tokens, target_embeddings[:, :-1, :]], dim=1)  # [B, 256, 4096]
        return shifted
        
    def forward(self, input_features, target_embeddings, return_attention=False, use_teacher_forcing=True, save_intermediate=False, return_cluster_logits=False):
        """
        input_features: Whisper input features [B, 80, 3000]
        target_embeddings: Ground truth T5 embeddings [B, 256, 4096]
        return_attention: whether to return cross attention from last decoder block
        use_teacher_forcing: if True, use shifted targets for autoregressive training
        save_intermediate: if True, save intermediate decoder outputs to CSV
        return_cluster_logits: if True, return cluster prediction logits [B, 256, n_clusters]
        """
        # Whisper encoder (frozen)
        with torch.no_grad():
            encoder_outputs = self.whisper.encoder(input_features)
            encoder_hidden_states = encoder_outputs.last_hidden_state  # [B, 1500, 1024]
        
        # MLP projection
        projected_features = self.mlp_projection(encoder_hidden_states)  # [B, 1500, 4096]
        
        # Prepare decoder input with teacher forcing
        if use_teacher_forcing:
            decoder_input = self.prepare_shifted_targets(target_embeddings)  # [B, 256, 4096]
        else:
            decoder_input = target_embeddings
        
        # Add positional embeddings to decoder input
        decoder_input = self.positional_embedding(decoder_input)  # [B, 256, 4096]
        
        # Determine save directory if saving intermediates
        save_dir = None
        save_self_attn = False
        if save_intermediate:
            # Use a default path relative to the project
            save_dir = '/scratch/IITB/ai-at-ieor/20004596/T5_Alignment_V0/outputs'
            save_self_attn = True  # Also save self-attention weights when saving intermediates
        
        # Decoder with causal masking
        if return_attention:
            output, attention_weights = self.decoder(
                decoder_input,
                projected_features,
                return_attention=True,
                use_causal_mask=use_teacher_forcing,
                save_intermediate=save_intermediate,
                save_dir=save_dir,
                save_self_attn=save_self_attn
            )
        else:
            output = self.decoder(
                decoder_input, 
                projected_features, 
                use_causal_mask=use_teacher_forcing,
                save_intermediate=save_intermediate,
                save_dir=save_dir,
                save_self_attn=save_self_attn
            )
        
        # Compute cluster logits if requested
        if return_cluster_logits and self.centroids is not None:
            cluster_logits = self.cluster_head(output)  # [B, 256, n_clusters]
            
            if return_attention:
                return output, attention_weights, cluster_logits
            else:
                return output, cluster_logits
        
        # Original return behavior
        if return_attention:
            return output, attention_weights
        return output
    
    @torch.no_grad()
    def generate(self, input_features, max_length=256, temperature=1.0):
        """
        Autoregressive generation for inference (no teacher forcing).
        
        Args:
            input_features: Whisper input features [B, 80, 3000]
            max_length: Number of T5 tokens to generate (default 256)
            temperature: Sampling temperature (not used for embeddings, kept for API consistency)
        
        Returns:
            generated_embeddings: [B, max_length, 4096] - Generated T5 embeddings
        """
        self.eval()
        B = input_features.size(0)
        device = input_features.device
        
        # Encode audio with frozen Whisper
        encoder_outputs = self.whisper.encoder(input_features)
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [B, 1500, 1024]
        
        # Project to decoder dimension
        projected_features = self.mlp_projection(encoder_hidden_states)  # [B, 1500, 4096]
        
        # Initialize with BOS token and apply positional embedding
        current_sequence = self.bos_token.expand(B, -1, -1)  # [B, 1, 4096]
        current_sequence = self.positional_embedding(current_sequence)  # Apply PE to BOS
        
        # Autoregressive generation loop
        for i in range(max_length):
            # Predict next token given current sequence
            decoder_output = self.decoder(
                current_sequence,
                projected_features,
                return_attention=False,
                use_causal_mask=True,
                save_intermediate=False,
                save_dir=None,
                save_self_attn=False
            )  # [B, seq_len, 4096]
            
            # Get the prediction for the next token (from the last position's output)
            next_token = decoder_output[:, -1:, :]  # [B, 1, 4096]
            
            # Append predicted token to sequence
            current_sequence = torch.cat([current_sequence, next_token], dim=1)  # [B, seq_len+1, 4096]
        
        # Remove BOS token and return generated embeddings
        generated_embeddings = current_sequence[:, 1:, :]  # [B, 256, 4096]
        
        return generated_embeddings
    
    def get_trainable_parameters(self):
        """Return only trainable parameters"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
