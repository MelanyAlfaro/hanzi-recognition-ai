
"""Unified multi-label classification approach using a single CNN.

A single neural network predicts the presence or absence of all q radicals
for each input image. The network has q output neurons, one per radical.

All outputs share the same feature extractor. This allows the model to
learn visual features (such as strokes, corners, and shapes) that may be
useful for detecting multiple radicals.

Unlike a multi-class classifier, this approach allows multiple radicals
to be present in the same image.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset

import config
from src.training.cnn import CNN
from src.training.trainer import Trainer
from src.training.classifier import Classifier


class UnifiedClassifier(Classifier):
    """Unified multi-label classifier based on a shared CNN.

    The classifier uses one CNN with q output neurons, where q is the
    number of radicals to detect.

    Attributes:
        n_labels (int): Number of radicals the model can detect.
        model (CNN): Convolutional neural network used for classification.
    """

    def __init__(self, n_labels: int) -> None:
        """Initialize the unified classifier.

        Args:
            n_labels (int): Number of radicals (output labels) to predict.
                Must be a positive integer.
        """
        # n_labels = q = number of radicals (4 in our case).
        self.n_labels = n_labels

        # We create ONE network whose last layer has q outputs.
        # At this point its weights are random, so it predicts garbage.
        self.model = CNN(n_outputs=n_labels)

    def fit(
        self,
        train_ds: Dataset,
        val_ds: Dataset | None = None
    ) -> "UnifiedClassifier":
        """Train the CNN using all radical labels simultaneously.

        The trainer receives label_idx=None (the default), which means
        "use the full label vector". So the single network is trained
        on all q radicals at once, and the loss is averaged over all of them.

        Args:
            train_ds (Dataset): Training dataset containing images and
                their complete multi-label vectors.
            val_ds (Dataset | None ): Optional validation dataset used
                to monitor model performance during training.

        Returns:
            UnifiedClassifier: The trained classifier instance.
                Returning self allows method chaining.
        """
        Trainer().train(self.model, train_ds, val_ds)

        # Returning self lets us chain calls: UnifiedClassifier(4).fit(ds)
        return self

    def predict_proba(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the probability of each radical for every image.

        Args:
            images (torch.Tensor): Batch of input images with shape
                (N, 1, H, W), where:
                    N = number of images
                    1 = number of image channels (grayscale)
                    H = image height
                    W = image width

        Returns:
            torch.Tensor: Probability tensor with shape (N, q).
                Each value is between 0 and 1 and represents the
                predicted probability of one radical being present.
                The returned tensor is located on the CPU.
        """
        self.model.to(config.DEVICE)

        # eval() switches the network to prediction mode.
        # Why: Dropout randomly turns neurons off while training; when
        # predicting we want stable, repeatable answers, so it must be off.
        self.model.eval()

        # no_grad() tells PyTorch "we are not learning now". It saves memory
        # and time because it skips the bookkeeping needed for backward().
        with torch.no_grad():
            # The network must be on the same device as the data (CPU/GPU).
            logits = self.model(images.to(config.DEVICE))
            # logits shape: (N, q), containing raw scores.

            # The CNN outputs logits (any real number). The sigmoid squeezes
            # each one into the range 0-1 so it can be read as a probability.
            probabilities = torch.sigmoid(logits)

        # .cpu() brings the result back to the CPU, which is what
        # numpy and scikit-learn (metrics) expect.
        return probabilities.cpu()

    def predict(
        self,
        images: torch.Tensor,
        threshold: float | None = None
    ) -> torch.Tensor:
        """Predict whether each radical is present or absent.

        A radical is predicted as present (1) when its probability is
        greater than or equal to the specified threshold.

        Args:
            images (torch.Tensor): Batch of input images with shape
                (N, 1, H, W).
            threshold (float | None): Decision threshold between 0 and 1.
                If None, the value configured in config.THRESHOLD is used.
                A lower threshold makes the classifier more likely to
                predict radicals as present.

        Returns:
            torch.Tensor: Binary prediction tensor with shape (N, q).
                Each element is either:
                    1 = radical predicted as present
                    0 = radical predicted as absent.
                The returned tensor is located on the CPU.
        """
        # If no threshold is given, use the one from config.py.
        # (Read at call time, so changing config.THRESHOLD takes effect.)
        if threshold is None:
            threshold = config.THRESHOLD

        # (probabilities >= threshold) is a True/False tensor;
        # .int() makes it 1/0.
        return (self.predict_proba(images) >= threshold).int()

    def count_parameters(self) -> int:
        """Count the total number of parameters in the CNN.

        This includes both weights and biases from all network layers.

        Returns:
            int: Total number of model parameters.

        Note:
            This method counts all parameters, including any that might
            have requires_grad=False. In the current architecture, all
            parameters are expected to be trainable.

        Useful for the paper: it lets us compare the size of both approaches.
        p.numel() = number of elements in one parameter tensor.
        """
        return sum(parameter.numel() for parameter in self.model.parameters())

    
    def save(self, weights_path: str | Path) -> None:
        """Save the learned weights of the network to a file.

        Args:
            weights_path (str | Path): Destination file, usually with the
                .pt extension (e.g. results/unified.pt). Missing parent
                folders are created automatically.
        """
        weights_path = Path(weights_path)

        # Create the destination folder (e.g. results/) if it does not exist.
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        # state_dict() is a dictionary {layer name: tensor of weights}.
        torch.save(self.model.state_dict(), weights_path)

    def load(self, weights_path: str | Path) -> "UnifiedClassifier":
        """Load previously saved weights into this classifier's network.

        The classifier must have been created with the same number of
        labels (q) that was used when the weights were saved. Otherwise the
        layer sizes do not match and PyTorch raises an error.

        Args:
            weights_path (str | Path): File created by save().

        Returns:
            UnifiedClassifier: This same instance, so calls can be chained:
                UnifiedClassifier(q).load(path).predict(images)

        Raises:
            FileNotFoundError: If weights_path does not exist.
            RuntimeError: If the saved weights do not match the architecture
                (for example, a different number of labels).
        """
        # map_location lets a file saved on a GPU (Colab) be loaded on a
        # machine without one, and vice versa.
        saved_weights = torch.load(weights_path, map_location=config.DEVICE)

        # Copy the saved values into the layers of the existing network.
        self.model.load_state_dict(saved_weights)
        self.model.to(config.DEVICE)

        return self
