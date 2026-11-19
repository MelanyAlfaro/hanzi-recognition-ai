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
# Dataset
# ----------------------------------------------------------------------

# Side length, in pixels, of each square image (64 x 64).
# Must be divisible by 8: the CNN halves the image 3 times (64 -> 32 -> 16 -> 8).
IMG_SIZE = 64

# Seed for random number generators. Fixing it makes experiments reproducible:
# same seed -> same data split, same initial weights, same results.
SEED = 42

# Fraction of the data used for each purpose. Must sum to 1.0.
TRAIN_FRAC = 0.7   # learn from these
VAL_FRAC = 0.15    # monitor overfitting during training
TEST_FRAC = 0.15   # final evaluation, only used once at the end


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