from abc import ABC, abstractmethod

from torch.utils.data import Dataset
import torch
from pathlib import Path

class Classifier(ABC):
    @abstractmethod
    def __init__(self, n_labels: int) -> None:
        # n_labels = q = number of radicals (4 in our case).
        self.n_labels = n_labels


    @abstractmethod
    def fit(self,
            train_ds: Dataset,
            val_ds: Dataset | None = None
    ) -> "Classifier":
        pass


    @abstractmethod
    def predict_proba(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        pass


    @abstractmethod
    def predict(
        self,
        images: torch.Tensor,
        threshold: float | None = None
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def count_parameters(self) -> int:
        pass


    @abstractmethod
    def save(self, weights_path: str | Path) -> None:
        pass


    @abstractmethod
    def load(self, weights_path: str | Path) -> "Classifier":
        pass
