"""Throwaway smoke test: tiny in-memory dataset, trains both approaches."""

import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import TensorDataset

import config
from src.tools.metrics import evaluate_predictions
from src.training.binary_relevance import BinaryRelevanceClassifier
from src.training.unified import UnifiedClassifier

# ---- Toy data: your 14 characters, label = indices into RADICALS ----
RADICALS = ["木", "日", "月", "口"]
CHAR_TO_RADICALS = {
    "林": [0], "森": [0], "昌": [1], "晶": [1], "朋": [2], "明": [1, 2],
    "杳": [0, 1], "杲": [0, 1], "棚": [0, 2], "品": [3], "呂": [3],
    "杏": [0, 3], "呆": [0, 3], "唱": [1, 3],
}

# Uses the first font file found in data/fonts
FONT_PATH = next(p for p in config.FONTS_DIR.iterdir()
                 if p.suffix.lower() in (".ttf", ".otf", ".ttc"))


def render(char):
    """Draws one character as a (IMG_SIZE, IMG_SIZE) grayscale numpy array."""
    img = Image.new("L", (config.IMG_SIZE, config.IMG_SIZE), color=255)  # white canvas
    font = ImageFont.truetype(str(FONT_PATH), int(config.IMG_SIZE * 0.8))
    ImageDraw.Draw(img).text(
        (config.IMG_SIZE // 2, config.IMG_SIZE // 2), char,
        font=font, fill=0, anchor="mm")                                  # black, centered
    return np.array(img)


def make_dataset():
    X, Y = [], []
    for char, idxs in CHAR_TO_RADICALS.items():
        label = np.zeros(len(RADICALS), dtype=np.float32)
        label[idxs] = 1.0                       # e.g. 明 -> [0, 1, 1, 0]
        X.append(render(char))
        Y.append(label)
    # (N, H, W) uint8 -> (N, 1, H, W) float in [0, 1]. The 1 is the channel.
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1) / 255.0
    Y = torch.tensor(np.array(Y))
    return TensorDataset(X, Y)


def run(name, classifier, ds):
    start = time.time()
    classifier.fit(ds)
    elapsed = time.time() - start

    x, y = ds[:]                                # all images and labels at once
    y_pred = classifier.predict(x)
    results = evaluate_predictions(y.numpy(), y_pred.numpy())

    print(f"\n=== {name} ===")
    print(f"parameters: {classifier.count_parameters():,}   time: {elapsed:.1f}s")
    print(f"F1 macro: {results['f1_macro']:.3f}   Hamming: {results['hamming_loss']:.3f}   "
          f"exact match: {results['exact_match']:.3f}")
    print("F1 per radical:", dict(zip(RADICALS, np.round(results["f1_per_radical"], 3))))


if __name__ == "__main__":
    config.EPOCHS = 150          # 14 images = 1 step per epoch, so we need many epochs
    # config.DEBUG = True        # uncomment to see shapes per batch

    torch.manual_seed(config.SEED)
    ds = make_dataset()
    print("Dataset:", ds.tensors[0].shape, ds.tensors[1].shape)

    q = len(RADICALS)
    run("Unified", UnifiedClassifier(q), ds)
    run("Binary Relevance", BinaryRelevanceClassifier(q), ds)