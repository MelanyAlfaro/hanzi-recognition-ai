"""Multi-label evaluation metrics."""

import numpy as np
from sklearn.metrics import f1_score, hamming_loss


def evaluate_predictions(y_true, y_pred):
    """y_true, y_pred: arrays of shape (N, q) with 0/1 values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        # One F1 per radical (answers "which radical is hardest?")
        "f1_per_radical": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        # Fraction of individual radical decisions that are wrong
        "hamming_loss": hamming_loss(y_true, y_pred),
        # Fraction of characters where ALL radicals are correct
        "exact_match": float((y_true == y_pred).all(axis=1).mean()),
    }