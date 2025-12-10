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
# Options (in order of arithmetic capability):
#
# SMALL MODELS (fast but limited math ability):
# - "EleutherAI/pythia-160m" - Too small, can't do arithmetic (~26% on easy)
# - "EleutherAI/pythia-410m" - Small but may work for easy examples
# - "EleutherAI/pythia-1b" - Better, ~26% accuracy on easy arithmetic
# - "gpt2" (124M) - Too small
# - "gpt2-medium" (355M) - Decent arithmetic ability
# - "gpt2-large" (774M) - Good arithmetic
#
# MEDIUM MODELS (good math, needs ~8-12GB GPU):
# - "EleutherAI/pythia-2.8b" - Larger Pythia, better at arithmetic
# - "microsoft/phi-2" (2.7B) - Excellent reasoning/math for size, HIGHLY RECOMMENDED
# - "Qwen/Qwen2-1.5B" - Good at math, efficient
#
# LARGE MODELS (excellent math, needs ~16-32GB GPU):
# - "EleutherAI/pythia-6.9b" - Largest Pythia
# - "Qwen/Qwen2-7B" - Very good at math and reasoning
# - "deepseek-ai/deepseek-math-7b-base" - Specialized for math (best accuracy)
#
# Current selection:
MODEL_NAME = "microsoft/phi-2"  # Excellent math performance, 2.7B params

# Dataset configuration
OPERATIONS = ["add", "sub", "mul", "div", "max", "min"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard", "very_hard"]  # Added very_hard

# Number of examples per (operation, difficulty) pair
# For larger dataset: 6 ops × 4 difficulties × 1500 = 36,000 examples
EXAMPLES_PER_OP_DIFFICULTY = 1500

# Difficulty ranges (inclusive) - expanded with harder examples
DIFFICULTY_RANGES = {
    "easy": (10, 99),              # 2 digits
    "medium": (100, 9999),         # 3-4 digits
    "hard": (10000, 99999),        # 5 digits
    "very_hard": (100000, 999999)  # 6 digits
}

# Prompt format options
USE_FEW_SHOT = True  # Add few-shot examples to help the model
# Updated few-shot examples with diverse difficulty levels (including harder examples)
FEW_SHOT_EXAMPLES = {
    "add": "12 + 34 = 46\n456 + 789 = 1245\n12345 + 67890 = 80235\n",
    "sub": "98 - 45 = 53\n8765 - 1234 = 7531\n67890 - 12345 = 55545\n",
    "mul": "12 * 8 = 96\n45 * 23 = 1035\n123 * 456 = 56088\n",
    "div": "96 / 8 = 12\n1035 / 23 = 45\n56088 / 456 = 123\n",
    "max": "max(73, 28) = 73\nmax(4567, 7890) = 7890\nmax(123456, 567890) = 567890\n",
    "min": "min(73, 28) = 28\nmin(4567, 7890) = 4567\nmin(123456, 567890) = 123456\n"
}

# Probe configuration
PROBE_HIDDEN_DIM = 64
PROBE_LEARNING_RATE = 0.001
PROBE_EPOCHS = 50
PROBE_BATCH_SIZE = 128
PROBE_VALIDATION_SPLIT = 0.2

# MI Synergy configuration
MI_KSG_NEIGHBORS = 3  # k for KSG estimator
MI_PCA_COMPONENTS = [3, 5]  # PCA dimensions to try

# Paths - Use absolute paths relative to project root
# Get the project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

# File paths
DATASET_PATH = os.path.join(DATA_DIR, "arithmetic_dataset.pkl")
HIDDEN_STATES_PATH = os.path.join(RESULTS_DIR, "hidden_states.npz")
FEATURES_PATH = os.path.join(RESULTS_DIR, "features.npz")
PROBES_DIR = os.path.join(RESULTS_DIR, "probes")
MI_SYNERGY_PATH = os.path.join(RESULTS_DIR, "mi_synergy_scalar_probe.csv")

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, PROBES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
