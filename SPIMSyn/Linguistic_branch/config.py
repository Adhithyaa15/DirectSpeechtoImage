# Configuration file for Speech2Image T5 Alignment training

# Paths
SPEECH_DIR = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/Speechs"
CAPTION_DIR = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/captions"
T5_ENCODER_PATH = "/scratch/IITB/ai-at-ieor/20004596/sd35/text_encoder_3"
T5_TOKENIZER_PATH = "/scratch/IITB/ai-at-ieor/20004596/sd35/tokenizer_3"
OUTPUT_DIR = "/scratch/IITB/ai-at-ieor/20004596/T5_Alignment_V0/outputs"

# Model Configuration
WHISPER_MODEL_NAME = "openai/whisper-medium"
N_DECODER_LAYERS = 8
N_STATE = 4096  # Hidden dimension for T5 (4096)
N_HEAD = 16    # Number of attention heads (4096 / 256 = 16)
MLP_RATIO = 4.0
MAX_SEQ_LENGTH = 256  # T5 sequence length

# T5 Configuration
T5_HIDDEN_DIM = 4096  # T5 embedding dimension
T5_MAX_LENGTH = 256   # Fixed T5 sequence length

# Cluster Configuration
N_CLUSTERS = 2950  # Number of K-means clusters (will be auto-determined by kmeans_clustering.py, update after running)
CENTROIDS_PATH = 'auto'  # 'auto' = use latest centroid file from outputs/kmeans/

# Dataset Configuration
DATASET_INDEX_FILE = "/scratch/IITB/ai-at-ieor/20004596/T5_Alignment_V0/dataset_index_50k.json"  # Fixed dataset index
MAX_SAMPLES = 50000  # Only used if index file doesn't exist
TRAIN_SPLIT = 0.98
BATCH_SIZE = 4  # Reduced batch size due to larger embeddings
NUM_WORKERS = 4

# Training Configuration
NUM_EPOCHS = 20
LEARNING_RATE = 5e-5  # Lower learning rate for larger model
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Loss Configuration
MSE_WEIGHT = 1.0
COSINE_WEIGHT = 1.0  # Cosine similarity loss (directional alignment)
CE_WEIGHT = 0.5  # Cluster classification cross-entropy loss
CONTRASTIVE_WEIGHT = 0.5  # Contrastive loss (pull toward correct centroid)

# Scheduler Configuration
SCHEDULER_T0 = 10
SCHEDULER_T_MULT = 2

# Logging Configuration
LOG_INTERVAL = 100
SAVE_INTERVAL = 15


def get_config():
    """Return configuration as dictionary"""
    config = {
        # Paths
        'speech_dir': SPEECH_DIR,
        'caption_dir': CAPTION_DIR,
        't5_encoder_path': T5_ENCODER_PATH,
        't5_tokenizer_path': T5_TOKENIZER_PATH,
        'output_dir': OUTPUT_DIR,
        
        # Model
        'whisper_model_name': WHISPER_MODEL_NAME,
        'n_decoder_layers': N_DECODER_LAYERS,
        'n_state': N_STATE,
        'n_head': N_HEAD,
        'mlp_ratio': MLP_RATIO,
        'max_seq_length': MAX_SEQ_LENGTH,
        
        # T5
        't5_hidden_dim': T5_HIDDEN_DIM,
        't5_max_length': T5_MAX_LENGTH,
        
        # Cluster (disabled)
        'n_clusters': N_CLUSTERS,
        'centroids_path': CENTROIDS_PATH,
        
        # Dataset
        'dataset_index_file': DATASET_INDEX_FILE,
        'max_samples': MAX_SAMPLES,
        'train_split': TRAIN_SPLIT,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        
        # Training
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'grad_clip': GRAD_CLIP,
        
        # Loss
        'mse_weight': MSE_WEIGHT,
        'cosine_weight': COSINE_WEIGHT,
        'ce_weight': CE_WEIGHT,
        'contrastive_weight': CONTRASTIVE_WEIGHT,
        
        # Scheduler
        'scheduler_t0': SCHEDULER_T0,
        'scheduler_t_mult': SCHEDULER_T_MULT,
        
        # Logging
        'log_interval': LOG_INTERVAL,
        'save_interval': SAVE_INTERVAL,
        
        # Device
        'device': 'cuda'
    }
    
    return config
