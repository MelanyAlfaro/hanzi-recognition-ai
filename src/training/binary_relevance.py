"""Binary Relevance approach: q independent CNNs, each with ONE output.

Idea: instead of one network answering everything, we train one separate
specialist per radical. Specialist j only answers "is radical j present?".
Specialists share nothing (no weights, no features) and never see each
other's answers. At prediction time we ask all of them and glue the q
answers together to rebuild the multi-label vector.
"""

import torch

import config
from src.training.cnn import CNN
from src.training.trainer import train_model


class BinaryRelevanceClassifier:
    def __init__(self, n_labels):
        # n_labels = q = number of radicals (4 in our case).
        self.n_labels = n_labels

        # A LIST of q networks. Each one is a brand-new CNN(1) with its own
        # random weights (the list comprehension calls CNN(1) q times, so
        # they are q different objects, not the same network repeated).
        # Same architecture as the unified model, except for the single output;
        # that keeps the comparison fair.
        self.models = [CNN(n_outputs=1) for _ in range(n_labels)]

    def fit(self, train_ds, val_ds=None):
        """Trains the q networks one after another.

        Network j learns ONLY radical j. The images are the same for all of
        them; what changes is which column of the label vector it receives.
        """
        for j, model in enumerate(self.models):
            print(f"\n--- Classifier {j + 1}/{self.n_labels} (radical index {j}) ---")

            # label_idx=j makes the trainer keep only column j of the labels,
            # turning the multi-label problem into a plain yes/no problem.
            # The model is modified in place, so self.models[j] ends up trained.
            train_model(model, train_ds, val_ds, label_idx=j)

        return self

    def predict_proba(self, X):
        """Returns the probability of each radical for each image.

        X:      tensor of shape (N, 1, H, W)  ->  N images
        return: tensor of shape (N, q)        ->  one probability per radical
        """
        # The same batch of images goes to every specialist, so move it once.
        X = X.to(config.DEVICE)

        columns = []   # will hold q tensors of shape (N, 1)

        with torch.no_grad():                    # prediction only, no learning
            for model in self.models:
                model.eval()                     # Dropout off for stable answers

                # Specialist j answers "is radical j present?" for all N images.
                # model(X) has shape (N, 1); the sigmoid makes it a probability.
                columns.append(torch.sigmoid(model(X)))

        # Gluing q columns of shape (N, 1) side by side (dim=1) gives (N, q).
        # This is exactly the "combine the q outputs into one vector" step
        # described in the theoretical framework for Binary Relevance.
        return torch.cat(columns, dim=1).cpu()

    def predict(self, X, threshold=None):
        """Returns hard decisions: 1 = radical present, 0 = absent."""
        if threshold is None:
            threshold = config.THRESHOLD
        return (self.predict_proba(X) >= threshold).int()

    def count_parameters(self):
        """Total weights summed over ALL q networks.

        Expect roughly q times the unified model: that is the cost of
        not sharing parameters.
        """
        return sum(p.numel() for m in self.models for p in m.parameters())