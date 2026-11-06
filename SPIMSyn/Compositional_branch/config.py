SPEECH_DIR = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/train/speech"
CAPTION_DIR = "/scratch/IITB/ai-at-ieor/20004596/laion5b_data/train/captions"
CLIP_G_ENCODER_PATH = "/scratch/IITB/ai-at-ieor/20004596/sd35"
OUTPUT_DIR = "/scratch/IITB/ai-at-ieor/20004596/CLIP_G_Alignment_V1/outputs"

WHISPER_MODEL_NAME = "openai/whisper-medium"

N_DECODER_LAYERS = 8
N_STATE = 1280
N_HEAD = 10
MLP_RATIO = 4.0

N_CLUSTERS = 590
CENTROIDS_PATH = "auto"

DATASET_INDEX_FILE = None
MAX_SAMPLES = 50000
TRAIN_SPLIT = 0.98
BATCH_SIZE = 8
NUM_WORKERS = 4

NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

MSE_WEIGHT = 1.0
COSINE_WEIGHT = 1.0
CE_WEIGHT = 0.5
CONTRASTIVE_WEIGHT = 0.5

SCHEDULER_T0 = 10
SCHEDULER_T_MULT = 2

LOG_INTERVAL = 100
SAVE_INTERVAL = 15


def get_config():
    return {
        'speech_dir': SPEECH_DIR,
        'caption_dir': CAPTION_DIR,
        'clip_g_encoder_path': CLIP_G_ENCODER_PATH,
        'output_dir': OUTPUT_DIR,
        'whisper_model_name': WHISPER_MODEL_NAME,
        'n_decoder_layers': N_DECODER_LAYERS,
        'n_state': N_STATE,
        'n_head': N_HEAD,
        'mlp_ratio': MLP_RATIO,
        'n_clusters': N_CLUSTERS,
        'centroids_path': CENTROIDS_PATH,
        'dataset_index_file': DATASET_INDEX_FILE,
        'max_samples': MAX_SAMPLES,
        'train_split': TRAIN_SPLIT,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'grad_clip': GRAD_CLIP,
        'mse_weight': MSE_WEIGHT,
        'cosine_weight': COSINE_WEIGHT,
        'ce_weight': CE_WEIGHT,
        'contrastive_weight': CONTRASTIVE_WEIGHT,
        'scheduler_t0': SCHEDULER_T0,
        'scheduler_t_mult': SCHEDULER_T_MULT,
        'log_interval': LOG_INTERVAL,
        'save_interval': SAVE_INTERVAL,
        'device': 'cuda',
    }
