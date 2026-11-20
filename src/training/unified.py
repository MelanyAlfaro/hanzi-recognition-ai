"""Unified multi-label approach: ONE CNN with q outputs, one per radical.

Idea: a single network looks at the image and answers all q questions
("is radical 1 present? is radical 2 present? ...") at the same time.
All the weights are shared, so the features learned for one radical
(strokes, corners, shapes) can also help to detect the others.
"""

import torch

import config
from src.training.cnn import CNN
from src.training.trainer import train_model


class UnifiedClassifier:
    def __init__(self, n_labels):
        # n_labels = q = number of radicals (4 in our case).
        self.n_labels = n_labels

        # We create ONE network whose last layer has q outputs.
        # At this point its weights are random, so it predicts garbage.
        self.model = CNN(n_outputs=n_labels)

    def fit(self, train_ds, val_ds=None):
        """Trains the network on the training set.

        The trainer receives label_idx=None (the default), which means
        "use the full label vector". So the single network is trained
        on all q radicals at once, and the loss is averaged over all of them.
        """
        train_model(self.model, train_ds, val_ds)

        # Returning self lets us chain calls: UnifiedClassifier(4).fit(ds)
        return self

    def predict_proba(self, X):
        """Returns the probability of each radical for each image.

        X:      tensor of shape (N, 1, H, W)  ->  N images
        return: tensor of shape (N, q)        ->  one probability per radical
        """
        # eval() switches the network to prediction mode.
        # Why: Dropout randomly turns neurons off while training; when
        # predicting we want stable, repeatable answers, so it must be off.
        self.model.eval()

        # no_grad() tells PyTorch "we are not learning now". It saves memory
        # and time because it skips the bookkeeping needed for backward().
        with torch.no_grad():
            # The network must be on the same device as the data (CPU/GPU).
            logits = self.model(X.to(config.DEVICE))   # raw scores, shape (N, q)

            # The CNN outputs logits (any real number). The sigmoid squeezes
            # each one into the range 0-1 so it can be read as a probability.
            probs = torch.sigmoid(logits)

        # .cpu() brings the result back to the CPU, which is what
        # numpy and scikit-learn (metrics) expect.
        return probs.cpu()

    def predict(self, X, threshold=None):
        """Returns hard decisions: 1 = radical present, 0 = absent.

        A radical is predicted present when its probability >= threshold
        (the tau of the theoretical framework).
        """
        # If no threshold is given, use the one from config.py.
        # (Read at call time, so changing config.THRESHOLD takes effect.)
        if threshold is None:
            threshold = config.THRESHOLD

        # (probs >= threshold) is a True/False tensor; .int() makes it 1/0.
        return (self.predict_proba(X) >= threshold).int()

    def count_parameters(self):
        """Total number of trainable weights.

        Useful for the paper: it lets us compare the size of both approaches.
        p.numel() = number of elements in one weight tensor.
        """
        return sum(p.numel() for p in self.model.parameters())