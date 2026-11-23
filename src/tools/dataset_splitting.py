"""Dataset splitting: assign characters to training, validation, or test.

The split is assigned PER CHARACTER with ``assign_splits``, to the metadata file.

Since it only rewrites the ``split`` column, 
changing the fold does not require regenerating the images.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.tools.radical_dataset import read_info, read_metadata


def _build_character_mapping(
    train_chars: Sequence[str],
    val_chars: Sequence[str],
    test_chars: Sequence[str],
) -> dict[str, str]:
    """Build a mapping between characters and their assigned splits.

    Args:
        train_chars: Characters assigned to training.
        val_chars: Characters assigned to validation.
        test_chars: Characters assigned to testing.

    Returns:
        A dictionary mapping each character to its split name.

    Raises:
        ValueError: If a character appears in more than one split.
    """
    mapping: dict[str, str] = {}

    lists = {
        "train": train_chars,
        "val": val_chars,
        "test": test_chars,
    }

    for split, chars in lists.items():
        for char in chars:
            if char in mapping:
                raise ValueError(
                    f"{char!r} appears in '{mapping[char]}' "
                    f"and in '{split}'"
                )

            mapping[char] = split

    return mapping


def _validate_character_coverage(
    mapping: dict[str, str],
    existing: set[str],
) -> None:
    """Validate that every dataset character is assigned exactly once.

    Args:
        mapping: Character-to-split mapping.
        existing: Set of characters present in the dataset.

    Raises:
        ValueError: If a mapped character does not exist in the dataset.
        ValueError: If a dataset character has not been assigned.
    """
    unknown = set(mapping) - existing

    if unknown:
        raise ValueError(
            f"Characters not in the dataset: {''.join(sorted(unknown))}"
        )

    unassigned = existing - set(mapping)

    if unassigned:
        raise ValueError(
            f"{''.join(sorted(unassigned))} is not assigned to any split"
        )


def _count_radicals(
    labels: np.ndarray,
    mask: np.ndarray,
    radicals: Sequence[str],
) -> dict[str, int]:
    """Count the number of images containing each radical in a split.

    Args:
        labels: Binary label matrix where each row represents an image
            and each column represents a radical.
        mask: Boolean array indicating which images belong to the split.
        radicals: Ordered sequence of radical names corresponding to
            the columns in the label matrix.

    Returns:
        A dictionary mapping each radical to its number of occurrences
        in the selected images.
    """
    counts = labels[mask].sum(axis=0).astype(int)

    return dict(zip(radicals, counts.tolist()))


def _validate_radical_coverage(
    counts: dict[str, int],
    split: str,
) -> None:
    """Validate that a split contains examples of every radical.

    Args:
        counts: Dictionary mapping radicals to their occurrence counts.
        split: Name of the split being validated.

    Raises:
        ValueError: If a radical has no examples in the training split.
    """
    # Get radicals that don't appear in split
    missing = [radical for radical, count in counts.items() if count == 0]

    # If no radical is missing in the split, it's all good
    if not missing:
        return

    # Train must use all of the radicals
    if split == "train":
        raise ValueError(
            f"No examples in train of: {''.join(missing)}"
        )


    print(
        f"[warning] no examples in {split} of: {''.join(missing)}"
    )


def _print_split_summary(
    split: str,
    characters: Sequence[str],
    image_count: int,
    radical_counts: dict[str, int],
) -> None:
    """Print a summary of the characters, images, and radicals in a split.

    Args:
        split: Name of the split.
        characters: Characters assigned to the split.
        image_count: Total number of images assigned to the split.
        radical_counts: Dictionary mapping radicals to their occurrence
            counts in the split.
    """
    print(
        f"{split:5s} "
        f"characters={''.join(characters) or '-':<12} "
        f"images={image_count:<5} "
        f"radicals={radical_counts}"
    )


def _save_metadata(
    metadata: pd.DataFrame,
    root: Path,
) -> None:
    """Save updated metadata to disk using an atomic file replacement.

    Args:
        metadata: Updated metadata DataFrame.
        root: Directory where metadata.csv will be saved.

    Raises:
        OSError: If writing the temporary file or replacing the original
            metadata file fails.
    """
    temporary_file = root / "metadata.csv.tmp"

    metadata.to_csv(temporary_file, index=False)
    temporary_file.replace(root / "metadata.csv")


def assign_splits(
    root: str | Path,
    train_chars: Sequence[str],
    val_chars: Sequence[str],
    test_chars: Sequence[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """Assign dataset characters to training, validation, and testing.

    All images belonging to the same character are assigned to the same
    split. Images are not modified, allowing splits to be reassigned
    without regenerating the dataset.

    Args:
        root: Directory containing the generated dataset files.
        train_chars: Characters assigned to training.
        val_chars: Characters assigned to validation.
        test_chars: Characters assigned to testing.
        verbose: Whether to print the number of images and radical
            occurrences in each split.

    Returns:
        The updated metadata DataFrame, with its split column assigned.
        The updated metadata is also saved to metadata.csv.

    Raises:
        ValueError: If a character appears in multiple splits.
        ValueError: If a listed character does not exist in the dataset.
        ValueError: If a dataset character is not assigned to a split.
        ValueError: If a radical has no examples in the training split.
        FileNotFoundError: If a required dataset file is missing.
        OSError: If the updated metadata cannot be saved.
    """
    root = Path(root)

    radicals = read_info(root)["radicals"]
    metadata = read_metadata(root)  # Data for every image
    labels = np.load(root / "labels.npy")  # 0/1 label vectors

    character_lists = {
        "train": train_chars,
        "val": val_chars,
        "test": test_chars,
    }

    mapping = _build_character_mapping(
        train_chars,
        val_chars,
        test_chars,
    )

    _validate_character_coverage(
        mapping,
        set(metadata["character"]),
    )

    if len(labels) != len(metadata):
        raise ValueError(
            "labels.npy and metadata.csv have different numbers of rows"
        )

    # Assign a split to every image based on its character (overwrites).
    metadata["split"] = metadata["character"].map(mapping)

    for split, characters in character_lists.items():
        # Mask is a true or false array, 
        # with True in the positions assigned to the current split
        # and False in the positions not assigned to the split
        mask = (metadata["split"] == split).to_numpy()

        # Skip empty split (if there are no Trues in the mask)
        if not mask.any():
            continue

        # Use mask to count the number of images containing each radical in a split
        counts = _count_radicals(labels, mask, radicals)

        if verbose:
            _print_split_summary(
                split,
                characters,
                int(mask.sum()),
                counts,
            )

        # Verify that split has each radical
        _validate_radical_coverage(counts, split)

    _save_metadata(metadata, root)

    return metadata
