
"""Binary Relevance multi-label classification approach using independent CNNs.

Instead of using one network to detect all q radicals, this approach
trains q independent CNNs, each specialized in detecting one radical.

Each network has a single output neuron and learns to answer one binary
classification question: "Is radical j present in the image?".

The networks do not share weights or learned features, and they do not
use each other's predictions.

During prediction, the probabilities produced by all q networks are
combined to reconstruct the complete multi-label prediction vector.
"""
from pathlib import Path

import torch
from torch.utils.data import Dataset

import config
from src.training.cnn import CNN
from src.training.trainer import Trainer


class BinaryRelevanceClassifier:
    """Binary Relevance classifier composed of independent CNNs.

    The classifier creates one CNN per radical. Each network has a single
    output and is trained independently using only the corresponding
    column of the multi-label target vector.

    Attributes:
        n_labels (int): Number of radicals the model can detect.
        models (list[CNN]): List of independent CNNs, one per radical.
    """

    def __init__(self, n_labels: int) -> None:
        """Initialize the Binary Relevance classifier.

        Args:
            n_labels (int): Number of radicals (output labels) to predict.
                Must be a positive integer.
        """
        # n_labels = q = number of radicals (4 in our case).
        self.n_labels = n_labels

        # Create q independent networks, each with a single output neuron.
        # Each call to CNN(1) creates a new network with its own random weights.
        # All networks share the same architecture, but not their parameters.
        #
        # Using the same architecture as Unified keeps the comparison fair.
        self.models = [
            CNN(n_outputs=1)
            for _ in range(n_labels)
        ]

    def fit(
        self,
        train_ds: Dataset,
        val_ds: Dataset | None = None
    ) -> "BinaryRelevanceClassifier":
        """Train each CNN independently using its corresponding radical label.

        All networks receive the same training images. However, each network
        is trained using only one column of the multi-label target vector.

        The label_idx argument tells the trainer which label column to use.
        For example, label_idx=0 selects the first radical's labels.

        Args:
            train_ds (Dataset): Training dataset containing images and
                their complete multi-label vectors.
            val_ds (Dataset | None): Optional validation dataset used
                to monitor model performance during training.

        Returns:
            BinaryRelevanceClassifier: The trained classifier instance.
                Returning self allows method chaining.
        """

        trainer = Trainer()    # one instance, reused for the q networks

        for label_idx, model in enumerate(self.models):
            print(
                f"\n--- Classifier {label_idx + 1}/{self.n_labels} "
                f"(radical index {label_idx}) ---"
            )

            # label_idx selects the corresponding column of the target vector.
            # This turns the multi-label problem into an independent
            # binary classification task for the current network.
            #
            # The model is modified in place, so the trained weights remain
            # stored in self.models[label_idx].
            trainer.train(
                model,
                train_ds,
                val_ds,
                label_idx=label_idx
            )

        # Returning self lets us chain calls:
        # BinaryRelevanceClassifier(4).fit(train_ds)
        return self

    def predict_proba(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the probability of each radical for every image.

        Each independent CNN processes the same batch of images and
        produces one probability per image. The resulting columns are
        concatenated to reconstruct the complete probability matrix.

        Args:
            images (torch.Tensor): Batch of input images with shape
                (N, 1, H, W), where:
                    N = number of images
                    1 = number of image channels (grayscale)
                    H = image height
                    W = image width

        Returns:
            torch.Tensor: Probability tensor with shape (N, q).
                Each column contains the probabilities predicted by
                one specialist network.
                Each value is between 0 and 1.
                The returned tensor is located on the CPU.
        """
        # The same batch of images goes to every specialist,
        # so move it to the configured device only once.
        images = images.to(config.DEVICE)

        # Each element will contain one specialist's probability column.
        # Every column has shape (N, 1).
        probability_columns = []

        # no_grad() tells PyTorch "we are not learning now". It saves memory
        # and time because it skips the bookkeeping needed for backward().
        with torch.no_grad():

            for model in self.models:
                model.to(config.DEVICE)

                # eval() switches the network to prediction mode.
                # Dropout is disabled to obtain stable predictions.
                model.eval()

                # The specialist produces one logit per image.
                logits = model(images)  # Shape: (N, 1)

                # Sigmoid transforms the raw logits into probabilities.
                probabilities = torch.sigmoid(logits)

                # Store this specialist's probability column.
                probability_columns.append(probabilities)

        # Concatenate the q columns side by side along dimension 1.
        # q tensors of shape (N, 1) become one tensor of shape (N, q).
        #
        # .cpu() brings the result back to the CPU, which is what
        # NumPy and scikit-learn (metrics) expect.
        return torch.cat(probability_columns, dim=1).cpu()

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
                A lower threshold makes the classifiers more likely to
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

        # Obtain the probabilities from all specialist networks.
        probabilities = self.predict_proba(images)

        # Compare each probability against the threshold.
        # The boolean results are converted into integer predictions (1/0).
        return (probabilities >= threshold).int()

    def count_parameters(self) -> int:
        """Count the total number of parameters across all CNNs.

        This includes the weights and biases of every specialist network.

        Returns:
            int: Total number of parameters across all q networks.

        Note:
            This method counts all parameters, including any that might
            have requires_grad=False. In the current architecture, all
            parameters are expected to be trainable.

            Since each network has its own parameters, the total is
            approximately q times the parameter count of one CNN.
            It may differ slightly from q times the Unified model's
            parameter count because their output layers have different sizes.
        """
        return sum(
            parameter.numel()
            for model in self.models
            for parameter in model.parameters()
        )
    

    def save(self, weights_path: str | Path) -> None:
        """Save the weights of all q networks in a single file.

        The file contains a list with q state dictionaries. The position in
        the list matches the radical index (position 0 is the first radical,
        and so on), so the order must never change.

        Args:
            weights_path (str | Path): Destination file, usually with the
                .pt extension (e.g. results/binary_relevance.pt). Missing
                parent folders are created automatically.

        Returns:
            None
        """
        weights_path = Path(weights_path)
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        # One state dictionary per network, in radical order.
        network_weights = [model.state_dict() for model in self.models]

        torch.save(network_weights, weights_path)
        

    def load(self, weights_path: str | Path) -> "BinaryRelevanceClassifier":
        """Load previously saved weights into the q networks.

        The classifier must have been created with the same number of
        labels (q) that was used when the weights were saved.

        Args:
            weights_path (str | Path): File created by save().

        Returns:
            BinaryRelevanceClassifier: This same instance, so calls can be
                chained: BinaryRelevanceClassifier(q).load(path).predict(images)

        Raises:
            FileNotFoundError: If weights_path does not exist.
            ValueError: If the file holds a different number of networks
                than this classifier has.
            RuntimeError: If the saved weights do not match the architecture.
        """
        network_weights = torch.load(weights_path, map_location=config.DEVICE)

        # Fail early with a clear message instead of a confusing mismatch later.
        if len(network_weights) != len(self.models):
            raise ValueError(
                f"The file contains {len(network_weights)} networks, "
                f"but this classifier has {len(self.models)}."
            )

        # Network j receives the weights that were saved at position j.
        for model, saved_weights in zip(self.models, network_weights):
            model.load_state_dict(saved_weights)
            model.to(config.DEVICE)

        return self