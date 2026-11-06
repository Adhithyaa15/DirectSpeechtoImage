import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel
import math
import os

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

try:
    from load_centroids import load_centroids_from_csv, find_latest_centroid_file
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from load_centroids import load_centroids_from_csv, find_latest_centroid_file


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_positions=77, embedding_dim=1280):
        super().__init__()
        pe = torch.zeros(max_positions, embedding_dim)
        nn.init.normal_(pe, mean=0.0, std=0.02)
        self.pe = nn.Parameter(pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class MLPProjection(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=1280):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return self.layer_norm(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state=1280, n_head=10):
        super().__init__()
        self.n_head = n_head
        self.n_state = n_state
        self.head_dim = n_state // n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x, xa=None, mask=None, use_causal_mask=False):
        q = self.query(x)
        if xa is None:
            k, v = self.key(x), self.value(x)
            if use_causal_mask and mask is None:
                mask = torch.full((x.size(1), x.size(1)), float('-inf'), device=x.device)
                mask = torch.triu(mask, diagonal=1)
        else:
            k, v = self.key(xa), self.value(xa)

        B, T, _ = q.shape
        Tkv = k.size(1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, Tkv, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, Tkv, self.n_head, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.n_state)
        return self.out(out)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state=1280, n_head=10, mlp_ratio=4.0):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(n_state, n_head)
        self.cross_attn_ln = nn.LayerNorm(n_state)
        mlp_dim = int(n_state * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(n_state, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, n_state))
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x, xa, use_causal_mask=True):
        x = x + self.attn(self.attn_ln(x), use_causal_mask=use_causal_mask)
        x = x + self.cross_attn(self.cross_attn_ln(x), xa)
        return x + self.mlp(self.mlp_ln(x))


class CustomDecoder(nn.Module):
    def __init__(self, n_layers=8, n_state=1280, n_head=10, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head, mlp_ratio) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_state)

    def forward(self, x, xa, use_causal_mask=True):
        for block in self.blocks:
            x = block(x, xa, use_causal_mask=use_causal_mask)
        return self.ln(x)


class Speech2ClipGModel(nn.Module):
    def __init__(
        self,
        whisper_model_name="openai/whisper-medium",
        n_decoder_layers=8,
        n_state=1280,
        n_head=10,
        mlp_ratio=4.0,
        n_clusters=590,
        centroids_path=None,
    ):
        super().__init__()

        self.whisper = WhisperModel.from_pretrained(whisper_model_name, local_files_only=True)
        for p in self.whisper.parameters():
            p.requires_grad = False

        self.mlp_projection = MLPProjection(input_dim=1024, hidden_dim=2048, output_dim=n_state)
        self.decoder = CustomDecoder(n_layers=n_decoder_layers, n_state=n_state, n_head=n_head, mlp_ratio=mlp_ratio)
        self.positional_embedding = LearnedPositionalEmbedding(max_positions=77, embedding_dim=n_state)
        self.bos_token = nn.Parameter(torch.randn(1, 1, n_state))

        self.n_clusters = n_clusters
        if centroids_path is not None:
            if centroids_path == 'auto':
                base_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'kmeans')
                centroid_file = find_latest_centroid_file(base_dir, n_clusters=None, format='csv')
                if centroid_file is None:
                    self.register_buffer('centroids', None)
                else:
                    c = torch.from_numpy(load_centroids_from_csv(centroid_file)).float()
                    self.n_clusters = c.shape[0]
                    self.register_buffer('centroids', c)
            else:
                c = torch.from_numpy(load_centroids_from_csv(centroids_path)).float()
                self.register_buffer('centroids', c)
        else:
            self.register_buffer('centroids', None)

        self.cluster_head = nn.Sequential(
            nn.Linear(n_state, n_state // 2),
            nn.GELU(),
            nn.Linear(n_state // 2, self.n_clusters),
        )

    def forward(self, input_features, target_embeddings=None, use_teacher_forcing=True):
        enc = self.whisper.encoder(input_features).last_hidden_state
        xa = self.mlp_projection(enc)

        if use_teacher_forcing and target_embeddings is not None:
            B = input_features.size(0)
            bos = self.bos_token.expand(B, -1, -1)
            shifted = torch.cat([bos, target_embeddings[:, :-1, :]], dim=1)
            shifted = self.positional_embedding(shifted)
            out = self.decoder(shifted, xa, use_causal_mask=True)
        else:
            out = self._autoregressive(xa, input_features.size(0), input_features.device)

        cluster_logits = self.cluster_head(out) if self.centroids is not None else None
        return out, cluster_logits

    def _autoregressive(self, xa, B, device):
        seq = self.bos_token.expand(B, -1, -1)
        seq = self.positional_embedding(seq)
        for _ in range(77):
            dec = self.decoder(seq, xa, use_causal_mask=True)
            seq = torch.cat([seq, dec[:, -1:, :]], dim=1)
        return seq[:, 1:, :]

    def generate(self, input_features, max_length=77):
        self.eval()
        B = input_features.size(0)
        enc = self.whisper.encoder(input_features).last_hidden_state
        xa = self.mlp_projection(enc)
        seq = self.bos_token.expand(B, -1, -1)
        seq = self.positional_embedding(seq)
        with torch.no_grad():
            for _ in range(max_length):
                dec = self.decoder(seq, xa, use_causal_mask=True)
                seq = torch.cat([seq, dec[:, -1:, :]], dim=1)
        return seq[:, 1:, :]
