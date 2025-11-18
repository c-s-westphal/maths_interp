"""
Configuration file for arithmetic reasoning interpretability experiment.
"""
import os
import torch
import random
import numpy as np

# Disable HF transfer to avoid errors
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

# Random seeds for reproducibility
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Model configuration
MODEL_NAME = "EleutherAI/pythia-160m"

# Dataset configuration
OPERATIONS = ["add", "sub", "mul", "max", "min"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# Number of examples per (operation, difficulty) pair
# For smaller subset: ~200-300 examples per pair → ~10k total
EXAMPLES_PER_OP_DIFFICULTY = 200  # 5 ops × 3 difficulties × 200 = 3000 examples

# Difficulty ranges (inclusive)
DIFFICULTY_RANGES = {
    "easy": (1, 99),       # 1-2 digits
    "medium": (100, 999),  # 3 digits
    "hard": (1000, 9999)   # 4 digits
}

# Probe configuration
PROBE_HIDDEN_DIM = 64
PROBE_LEARNING_RATE = 0.001
PROBE_EPOCHS = 50
PROBE_BATCH_SIZE = 128
PROBE_VALIDATION_SPLIT = 0.2

# Interaction predictor configuration
INTERACTION_PREDICTOR_HIDDEN_DIM = 128
INTERACTION_LEARNING_RATE = 0.001
INTERACTION_EPOCHS = 30
INTERACTION_BATCH_SIZE = 128

# Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# File paths
DATASET_PATH = os.path.join(DATA_DIR, "arithmetic_dataset.pkl")
HIDDEN_STATES_PATH = os.path.join(RESULTS_DIR, "hidden_states.npz")
FEATURES_PATH = os.path.join(RESULTS_DIR, "features.npz")
PROBES_DIR = os.path.join(RESULTS_DIR, "probes")
INTERACTION_SCORES_PATH = os.path.join(RESULTS_DIR, "interaction_scores.csv")

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, PROBES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
