# Configuration file for Speech2Image Alignment training

# Paths
DMEL_DIR = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/dmel_embeddings/dmel"
CLIP_G_DIR = "/scratch/IITB/ai-at-ieor/20004596/Speech2Image/embeddings/clip_g"
OUTPUT_DIR = "/scratch/IITB/ai-at-ieor/20004596/CLIP_G_with_DMEL/outputs"

# Model Configuration
N_DECODER_LAYERS = 8
N_STATE = 1280  # Hidden dimension
N_HEAD = 10    # Number of attention heads
MLP_RATIO = 4.0

# Cluster Configuration
N_CLUSTERS = 590
CENTROIDS_PATH = 'auto'

# Dataset Configuration
TOTAL_SAMPLES = 265000   # Approximate full pool size
MAX_SAMPLES = 50000      # Samples drawn per epoch
FIXED_EPOCHS = 50        # Epochs 1-50: fixed sample; 51-200: fresh random each epoch
TRAIN_SPLIT = 0.98
BATCH_SIZE = 8
NUM_WORKERS = 4

# Training Configuration
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Loss Configuration
MSE_WEIGHT = 1.0
COSINE_WEIGHT = 1.0
CE_WEIGHT = 0.5
CONTRASTIVE_WEIGHT = 0.5

# Scheduler Configuration
SCHEDULER_T0 = 10
SCHEDULER_T_MULT = 2

# Logging Configuration
LOG_INTERVAL = 100
SAVE_INTERVAL = 5


def get_config():
    config = {
        # Paths
        'dmel_dir': DMEL_DIR,
        'clip_g_dir': CLIP_G_DIR,
        'output_dir': OUTPUT_DIR,
        
        # Model
        'n_decoder_layers': N_DECODER_LAYERS,
        'n_state': N_STATE,
        'n_head': N_HEAD,
        'mlp_ratio': MLP_RATIO,
        
        # Cluster
        'n_clusters': N_CLUSTERS,
        'centroids_path': CENTROIDS_PATH,
        
        # Dataset
        'total_samples': TOTAL_SAMPLES,
        'max_samples': MAX_SAMPLES,
        'fixed_epochs': FIXED_EPOCHS,
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