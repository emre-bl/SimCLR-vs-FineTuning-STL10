import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
DATA_DIR = "./data"
RESULTS_DIR = "./results" 

os.makedirs(RESULTS_DIR, exist_ok=True)

# Baseline
EPOCHS_BASELINE = 100 
BATCH_SIZE_BASELINE = 64
LR_BASELINE = 0.001

# Fine-Tuning 
EPOCHS_FINETUNE = 25
BATCH_SIZE_FINETUNE = 32
LR_FINETUNE_BACKBONE = 1e-5
LR_FINETUNE_HEAD = 1e-3

# Self-Supervised (SimCLR)
# Stage 1: Pre-training
EPOCHS_SIMCLR = 100
BATCH_SIZE_SIMCLR = 128
LR_SIMCLR = 3e-4
PROJECTION_DIM = 128
TEMPERATURE = 0.5

# Stage 2: Linear Evaluation
EPOCHS_LINEAR = 50
BATCH_SIZE_LINEAR = 128
LR_LINEAR = 1e-3