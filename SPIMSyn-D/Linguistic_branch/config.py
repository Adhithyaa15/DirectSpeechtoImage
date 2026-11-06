# Configuration file for Speech2Image T5 Alignment training (DMEL input)

# Paths
DMEL_DIR          = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/dmel_embeddings/dmel"
CAPTION_DIR       = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/captions"
T5_ENCODER_PATH   = "/scratch/IITB/ai-at-ieor/20004596/sd35/text_encoder_3"
T5_TOKENIZER_PATH = "/scratch/IITB/ai-at-ieor/20004596/sd35/tokenizer_3"
OUTPUT_DIR        = "/scratch/IITB/ai-at-ieor/20004596/T5_with_DMEL/outputs"

# Model Configuration  (no Whisper — DMEL input replaces it)
N_DECODER_LAYERS = 8
N_STATE          = 4096   # T5 hidden dimension
N_HEAD           = 16     # 4096 / 16 = 256 per head
MLP_RATIO        = 4.0
MAX_SEQ_LENGTH   = 256    # T5 sequence length

# T5 Configuration
T5_HIDDEN_DIM = 4096
T5_MAX_LENGTH  = 256

# DMEL Configuration
DMEL_INPUT_DIM  = 80      # Raw DMEL feature dimension
DMEL_HIDDEN_DIM = 2048    # DmelInputProjection hidden layer

# Cluster Configuration
N_CLUSTERS     = 2950
CENTROIDS_PATH = "/scratch/IITB/ai-at-ieor/20004596/T5_with_DMEL/outputs/kmeans/t5_kmeans_centroids_2950clusters_20260220_103908.csv"

# Dataset Configuration
DATASET_INDEX_FILE = None   # Set to a JSON path to use a fixed reproducible split
MAX_SAMPLES  = 50000
TRAIN_SPLIT  = 0.98
BATCH_SIZE   = 4
NUM_WORKERS  = 0            # Keep 0 — T5 encoder runs in main process

# Training Configuration
NUM_EPOCHS    = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 0.01
GRAD_CLIP     = 1.0

# Loss Configuration
MSE_WEIGHT         = 1.0
COSINE_WEIGHT      = 1.0
CE_WEIGHT          = 0.5
CONTRASTIVE_WEIGHT = 0.5

# Scheduler Configuration
SCHEDULER_T0    = 10
SCHEDULER_T_MULT = 2

# Logging Configuration
LOG_INTERVAL  = 100
SAVE_INTERVAL = 15


def get_config():
    """Return configuration as dictionary."""
    return {
        # Paths
        'dmel_dir':           DMEL_DIR,
        'caption_dir':        CAPTION_DIR,
        't5_encoder_path':    T5_ENCODER_PATH,
        't5_tokenizer_path':  T5_TOKENIZER_PATH,
        'output_dir':         OUTPUT_DIR,

        # Model
        'n_decoder_layers': N_DECODER_LAYERS,
        'n_state':          N_STATE,
        'n_head':           N_HEAD,
        'mlp_ratio':        MLP_RATIO,
        'max_seq_length':   MAX_SEQ_LENGTH,

        # T5
        't5_hidden_dim': T5_HIDDEN_DIM,
        't5_max_length': T5_MAX_LENGTH,

        # DMEL
        'dmel_input_dim':  DMEL_INPUT_DIM,
        'dmel_hidden_dim': DMEL_HIDDEN_DIM,

        # Cluster
        'n_clusters':     N_CLUSTERS,
        'centroids_path': CENTROIDS_PATH,

        # Dataset
        'dataset_index_file': DATASET_INDEX_FILE,
        'max_samples':  MAX_SAMPLES,
        'train_split':  TRAIN_SPLIT,
        'batch_size':   BATCH_SIZE,
        'num_workers':  NUM_WORKERS,

        # Training
        'num_epochs':    NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay':  WEIGHT_DECAY,
        'grad_clip':     GRAD_CLIP,

        # Loss
        'mse_weight':         MSE_WEIGHT,
        'cosine_weight':      COSINE_WEIGHT,
        'ce_weight':          CE_WEIGHT,
        'contrastive_weight': CONTRASTIVE_WEIGHT,

        # Scheduler
        'scheduler_t0':    SCHEDULER_T0,
        'scheduler_t_mult': SCHEDULER_T_MULT,

        # Logging
        'log_interval':  LOG_INTERVAL,
        'save_interval': SAVE_INTERVAL,

        # Device
        'device': 'cuda',
    }