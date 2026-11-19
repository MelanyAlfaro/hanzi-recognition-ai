"""CNN shared by both approaches (unified: n_outputs=q, Binary Relevance: n_outputs=1)."""

import torch.nn as nn
import config


class CNN(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        # Feature extractor: learns shapes, strokes and patterns
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        # After 3 poolings the image side is IMG_SIZE / 8 (64 -> 8)
        side = config.IMG_SIZE // 8

        # Classifier head: turns the features into n_outputs scores
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * side * side, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x):
        # Returns raw scores (logits), NOT probabilities.
        # BCEWithLogitsLoss applies the sigmoid internally.
        return self.head(self.features(x))