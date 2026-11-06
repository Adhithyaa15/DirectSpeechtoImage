# Configuration file for Speech2Image Alignment training

# Paths
SPEECH_DIR = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/Speechs"
CLIP_L_DIR = "/scratch/IITB/ai-at-ieor/20004596/Speech2Image/embeddings/clip_l"
OUTPUT_DIR = "/scratch/IITB/ai-at-ieor/20004596/CLIP_L_Alignment_V1/outputs"

# Model Configuration
WHISPER_MODEL_NAME = "openai/whisper-medium"
N_DECODER_LAYERS = 6
N_STATE = 768  # Hidden dimension
N_HEAD = 12    # Number of attention heads
MLP_RATIO = 4.0

# Cluster Configuration
N_CLUSTERS = 980  # Number of K-means clusters
CENTROIDS_PATH = 'auto'  # 'auto' = use latest centroid file from outputs/kmeans/

# Dataset Configuration
DATASET_INDEX_FILE = "/scratch/IITB/ai-at-ieor/20004596/CLIP_L_Alignment_V1/dataset_index_50k.json"  # Fixed dataset index
MAX_SAMPLES = 50000  # Only used if index file doesn't exist
TRAIN_SPLIT = 0.98
BATCH_SIZE = 8
NUM_WORKERS = 4

# Training Configuration
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
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
SAVE_INTERVAL = 5


def get_config():
    """Return configuration as dictionary"""
    config = {
        # Paths
        'speech_dir': SPEECH_DIR,
        'clip_l_dir': CLIP_L_DIR,
        'output_dir': OUTPUT_DIR,
        
        # Model
        'whisper_model_name': WHISPER_MODEL_NAME,
        'n_decoder_layers': N_DECODER_LAYERS,
        'n_state': N_STATE,
        'n_head': N_HEAD,
        'mlp_ratio': MLP_RATIO,
        
        # Cluster
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
