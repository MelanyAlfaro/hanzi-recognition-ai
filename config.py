"""Central place for paths and hyperparameters.

Every value that might need tuning lives here, so no "magic numbers"
are scattered across the code. 
"""

from pathlib import Path

import torch

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

# __file__ is the path of this file (config.py).
# .resolve() turns it into an absolute path.
# .parent is the folder that contains it -> the project root.
ROOT = Path(__file__).resolve().parent


FONTS_DIR = ROOT / "data" / "fonts"            # .ttf/.ttc files used to draw characters
PROCESSED_DIR = ROOT / "data" / "processed"    # generated dataset (images, labels)
RESULTS_DIR = ROOT / "results"                 # saved metrics, figures

# ----------------------------------------------------------------------
# Dataset Output file names
# ----------------------------------------------------------------------

IMG_FILE = "images.npy"
LABELS_FILE = "labels.npy"
META_FILE = "metadata.csv"
GENERATION_FILE = "generation.json"

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

# Side length, in pixels, of each square image (64 x 64).
# Must be divisible by 8: the CNN halves the image 3 times (64 -> 32 -> 16 -> 8).
IMG_SIZE = 64

# Seed for random number generators. Fixing it makes experiments reproducible:
# same seed -> same data split, same initial weights, same results.
SEED = 42

# Characters assigned to each split (chosen by hand; see the paper's methodology)
TRAIN_CHARS = ["林", "森", "昌", "晶", "朋", "杳", "棚", "品", "呆", "唱"]
VAL_CHARS   = ["杲", "呂"]
TEST_CHARS  = ["明", "杏"]


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

# Number of images processed together before updating the weights once.
BATCH_SIZE = 64

# Number of full passes over the training set.
EPOCHS = 50

# How big each weight adjustment is. Too high: unstable. Too low: very slow.
LEARNING_RATE = 1e-3   # = 0.001

# The network outputs a probability per radical (0 to 1).
# If probability >= THRESHOLD, we predict "radical present". (tau in the paper)
THRESHOLD = 0.5

# Use the GPU if there is one (Colab), otherwise the CPU .
# Models and data must be moved to this device with .to(config.DEVICE).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# When True, the trainer prints tensor shapes and loss for every batch of the first epoch.
DEBUG = True

# Learning-rate warmup: the first epochs use a much smaller learning rate
WARMUP_EPOCHS = 2            # 0 disables warmup
WARMUP_START_FACTOR = 1e-3   # first epoch uses LEARNING_RATE * 1e-3

# Set the seed for everything so it is all the same
import random
import numpy as np


def apply_seed(seed: int = SEED) -> None:
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


apply_seed()
