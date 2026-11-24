"""Reading the generated dataset: ``RadicalDataset(split)``.

Contract with the Trainer and the classifiers:
    image  : float32 ``[1, IMG_SIZE, IMG_SIZE]``, values in [0, 1], ink = 1
    labels : float32 ``[q]``, 0/1, columns in the order of ``RADICALS``

Files read from ``config.PROCESSED_DIR``:
    images.npy       uint8 ``[N, H, W]``, ink = 255 (white), background = 0 (black)
    labels.npy       uint8 ``[N, q]``
    metadata.csv     row i describes image i: character, codepoint, font,
                     variant, augmented, split
    generation.json  generation parameters (includes the order of RADICALS)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config
from src.tools.radicals import RADICALS

SPLITS: tuple[str, ...] = ("train", "val", "test")



class RadicalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """In-memory dataset of character images and their radical labels.

    Everything is loaded into RAM when the object is created.

    Attributes:
        root: Folder the data was read from.
        radicals: Label column names, in the order of ``RADICALS``.
        split: The requested split, or None.
        meta: Metadata rows of the selected images.
        x: Images, uint8 tensor ``[N, 1, H, W]``, ink = 255 (white).
        y: Labels of radical presence, float32 tensor ``[N, q]``.
    """

    def __init__(
        self,
        split: str | None = None,
        characters: Sequence[str] | None = None,
        root: str | Path | None = None,
        clean_only: bool = False,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Load the selected images and labels.

        Select the data with ``split`` OR with ``characters`` (useful for
        k-fold). With neither, everything is loaded.

        Args:
            split: One of ``"train"``, ``"val"`` or ``"test"``.
            characters: Characters to keep. Cannot be combined with ``split``.
            root: Dataset folder. Defaults to ``config.PROCESSED_DIR``.
            clean_only: If True, keep only glyphs without augmentation.
            transform: Optional function applied to each image tensor.

        Raises:
            ValueError: If both ``split`` and ``characters`` are given, if
                ``split`` is not valid or has not been assigned yet, if a
                character is not in the dataset, if the selection is empty, or
                if the stored arrays have different lengths.
            FileNotFoundError: If a generated file is missing.
        """
        self.root = Path(root) if root else default_root()
        self.radicals = list(read_info(self.root)["radicals"])
        self.transform = transform

        meta, images, labels = self._load_data()
        self._validate_data(meta, images, labels)
        mask = self._build_selection_mask(
            meta, split, characters, clean_only
        )

        self.split = split
        self.meta = meta[mask].reset_index(drop=True)
        self.x = torch.from_numpy(images[mask]).unsqueeze(1)
        self.y = torch.from_numpy(labels[mask]).to(torch.float32)

    def _load_data(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load metadata, images, and labels from disk.
        
        Returns:
            A tuple with the read metadata, images array, and labels array
        """
        meta = read_metadata(self.root)
        images = np.load(self.root / config.IMG_FILE)
        labels = np.load(self.root / config.LABELS_FILE)
        return meta, images, labels

    @staticmethod
    def _validate_data(
        meta: pd.DataFrame,
        images: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Validate that all stored arrays have matching lengths.
        
        Args:
            meta: Metadata of generated images.
            images: Numpy array of images with shape [N, H, W].
            labels: Numpy array of the labels with shape [N, q].

        Raises:
            ValueError: If the three arguments don't have the same length.
                As this would mean that there is no correspondence between the registers.
        """
        if not (len(meta) == len(images) == len(labels)):
            raise ValueError(
                f"metadata ({len(meta)}), images ({len(images)}) and "
                f"labels ({len(labels)}) do not have the same length"
            )

    @staticmethod
    def _validate_selection(
        split: str | None,
        characters: Sequence[str] | None,
    ) -> None:
        """Validate mutually exclusive selection arguments.
        
        Args:
            split: One of ``"train"``, ``"val"`` or ``"test"``.
            
        Raises:
            ValueError: If split and characters are both passed (Split is exclusive)
        """
        if split is not None and characters is not None:
            raise ValueError("Pass split or characters, not both")

    def _build_selection_mask(
        self,
        meta: pd.DataFrame,
        split: str | None,
        characters: Sequence[str] | None,
        clean_only: bool,
    ) -> np.ndarray:
        """Build a boolean mask for the requested dataset selection.
        
        Args:
            meta: Metadata of generated images
            split: One of ``"train"``, ``"val"`` or ``"test"``.
            characters: Characters to keep. Cannot be combined with ``split``.
            clean_only: If True, keep only glyphs without augmentation.

        Returns:
            The created selection mask based on the arguments
            
        Raises:
            ValueError: If the passed split is not one of the three valid ones, or
                if the split column is empty in the metadata file due to lack of assignments, or
                if the selected characters are not in the dataset, or
                if the final selection has no images
        """
        self._validate_selection(split, characters)

        mask = np.ones(len(meta), dtype=bool)

        # Select by split
        if split is not None:
            if split not in SPLITS:
                raise ValueError(
                    f"split must be one of {SPLITS}, not {split!r}"
                )
            if (meta["split"] == "").all():
                raise ValueError(
                    "The split column is empty; run assign_splits() first"
                )
            mask &= (meta["split"] == split).to_numpy()

        # Select by characters
        elif characters is not None:
            missing = sorted(set(characters) - set(meta["character"]))
            if missing:
                raise ValueError(
                    f"Characters not in the dataset: {''.join(missing)}"
                )
            mask &= meta["character"].isin(characters).to_numpy()

        # Keep glyphs without augmentation
        if clean_only:
            mask &= (meta["augmented"] == 0).to_numpy()

        if not mask.any():
            raise ValueError("The selection contains no images")

        return mask

    @property
    def n_radicals(self) -> int:
        """Number of radicals ``q``, i.e. the length of each label vector.

        Returns:
            ``len(self.radicals)``.
        """
        return len(self.radicals)

    def __len__(self) -> int:
        """Return the number of images in the selection.

        Returns:
            The number of images.
        """
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one image and its labels.

        Args:
            idx: Index of the image inside the selection.

        Returns:
            A tuple ``(image, labels)``: ``image`` is float32 ``[1, H, W]`` in
            [0, 1] with ink = 1, and ``labels`` is float32 ``[q]`` with 0/1 values.
        """
        image = self.x[idx].float().div_(255.0)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.y[idx]
    
# [HELPERS]
def default_root() -> Path:
    """Return the default dataset folder, ``config.PROCESSED_DIR``.

    Returns:
        Path to the folder that holds the generated files.

    Raises:
        ValueError: If ``config`` cannot be imported or has no ``PROCESSED_DIR``.
    """
    try:
        import config
        return Path(config.PROCESSED_DIR)
    except (ImportError, AttributeError):
        raise ValueError("Pass root=... or define config.PROCESSED_DIR") from None


def read_info(root: str | Path) -> dict[str, Any]:
    """Read ``generation.json`` and check that ``RADICALS`` did not change.

    Args:
        root: Folder that holds the generated files.

    Returns:
        The parsed contents of ``generation.json``.

    Raises:
        FileNotFoundError: If ``generation.json`` does not exist.
        ValueError: If the radical order stored on disk differs from ``RADICALS``.
    """
    path = Path(root) / config.GENERATION_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run src/scripts/build_dataset.py")
    info: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if info["radicals"] != RADICALS:
        raise ValueError(
            f"The order of RADICALS changed since the dataset was generated: "
            f"{info['radicals']} (disk) vs {RADICALS} (code). Regenerate the dataset."
        )
    return info


def read_metadata(root: str | Path) -> pd.DataFrame:
    """Read ``metadata.csv``.

    Args:
        root: Folder that holds the generated files.

    Returns:
        One row per image. An unassigned ``split`` is the empty string, not NaN.

    Raises:
        FileNotFoundError: If ``metadata.csv`` does not exist.
    """
    path = Path(root) / config.META_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run src/scripts/build_dataset.py")
    
    # keep_default_na=False keeps an empty `split` as "" instead of NaN.
    return pd.read_csv(
        path,
        dtype={"character": str, "codepoint": str, "font": str, "split": str},
        keep_default_na=False,
    )
