
"""Shared Convolutional Neural Network (CNN) architecture.

This CNN is used by both classification approaches:

- Unified: one network with q outputs, one per radical.
- Binary Relevance: one network with a single output per radical.

The architecture consists of two main components:

1. Feature extractor: learns visual patterns from input images.
2. Classifier head: transforms the extracted features into raw scores
   (logits), one for each output label.
"""

import torch
from torch import nn

import config


class CNN(nn.Module):
    """Convolutional Neural Network for radical detection.

    The network receives grayscale images and extracts visual features
    using convolutional and pooling layers. These features are then
    passed to a classifier head that produces one logit per output.

    Attributes:
        n_outputs (int): Number of output neurons.
        features (nn.Sequential): Convolutional feature extractor.
        head (nn.Sequential): Classifier that produces the output logits.
    """

    def __init__(self, n_outputs: int) -> None:
        """Initialize the CNN architecture.

        Args:
            n_outputs (int): Number of output neurons.
                - Unified: equal to the total number of radicals (q).
                - Binary Relevance: 1, since each network detects one radical.
        """
        super().__init__()

        self.n_outputs = n_outputs

        # Feature extractor: learns shapes, strokes and patterns.
        # Input shape: (B, 1, IMG_SIZE, IMG_SIZE)
        #
        # Each convolution extracts visual features.
        # ReLU introduces non-linearity.
        # MaxPool2d reduces the spatial dimensions by half.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        # After 3 pooling operations, each spatial dimension is divided by 8.
        # For IMG_SIZE = 64: 64 -> 32 -> 16 -> 8.
        feature_map_size = config.IMG_SIZE // 8

        # Classifier head: transforms extracted features into n_outputs scores.
        # Flatten converts the feature maps into a one-dimensional vector
        # per image before passing them to the fully connected layers.
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feature_map_size * feature_map_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_outputs),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network.

        Args:
            images (torch.Tensor): Batch of grayscale images with shape
                (B, 1, H, W), where:
                    B = batch size
                    1 = number of channels
                    H = image height
                    W = image width

        Returns:
            torch.Tensor: Raw output scores (logits) with shape
                (B, n_outputs).

                These values are not probabilities. When training, 
                the sigmoid operation is applied
                internally by the loss function.
        """
        extracted_features = self.features(images)
        output_logits = self.head(extracted_features)

        return output_logits
