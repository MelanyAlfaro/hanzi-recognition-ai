"""Renders generated character images in the images.npy file
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import config


class DataVisualizer:
    """Loads and visualizes generated character images."""

    def __init__(
        self,
        dataset_dir: str | Path,
        save_dir: str | Path | None = None,
    ) -> None:
        """Initialize the visualizer and load the dataset.

        Args:
            dataset_dir: Directory containing images.npy and metadata.csv.
            save_dir: Optional directory for saving rendered images.
            
        Raises:
            ValueError: If images and metadata don't have the same number of entries
            FileNotFoundError: If dataset dir is not present
        """
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"{self.dataset_dir} is missing; run src/scripts/build_dataset.py")
                                    
        self.save_dir = Path(save_dir) if save_dir else None

        self.images = np.load(self.dataset_dir / config.IMG_FILE)
        self.metadata = pd.read_csv(self.dataset_dir / config.META_FILE)

        if len(self.images) != len(self.metadata):
            raise ValueError(
                "Images and metadata must have the same number of entries"
            )

    def visualize(
        self,
        character: str | None = None,
        n: int = 25,
        cols: int = 5,
        save: bool = False,
    ) -> None:
        """Render the requested character images.

        Args:
            character: Character to visualize. If None, visualize the
                first n images in the dataset.
            n: Maximum number of images to render.
            cols: Number of columns in the visualization grid.
            save: Whether to save the selected images individually.

        Raises:
            ValueError: If no images match the requested selection.
        """
        indices = self._select_indices(character, n)

        rows = int(np.ceil(len(indices) / cols))

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 2.5, rows * 2.5),
            squeeze=False,
        )

        for ax in axes.flat:
            ax.axis("off")

        for ax, idx in zip(axes.flat, indices):
            row = self.metadata.iloc[idx]

            ax.imshow(
                self.images[idx],
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            ax.set_title(
                f"{row['character']} | {row['font']}\n"
                f"Variant: {row['variant']} | "
                f"Aug: {row['augmented']}"
            )

            if save:
                self._save_image(idx)

        fig.tight_layout()
        plt.show()

    def _select_indices(
        self,
        character: str | None,
        n: int,
    ) -> np.ndarray:
        """Select image indices matching the requested character.
        
            Args:
                character: character to visualize, or None if selection is by indices
                n: number of visualizations requested
                
            Returns:
                The indices with the image 
        """
        if n <= 0:
            raise ValueError("n must be greater than zero")

        indices = np.arange(len(self.images))

        # Filter indices by character if requested
        if character is not None:
            indices = self.metadata.index[
                self.metadata["character"] == character
            ].to_numpy()

        # Grab only n indices
        indices = indices[:n]

        if len(indices) == 0:
            raise ValueError(
                f"No images found for character {character!r}"
            )

        return indices

    def _save_image(self, idx: int) -> None:
        """Save an individual image using its metadata as the filename.
        
            Args:
                idx: Index of image to save.
        """
        if self.save_dir is None:
            raise ValueError("No save directory was configured")

        # Make save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Obtain metadata of image at the index
        row = self.metadata.iloc[idx]

        # Build file name for image with metadata information
        filename = (
            f"{row['character']}_"
            f"{row['font']}_"
            f"variant-{row['variant']}_"
            f"{idx}.png"
        )

        Image.fromarray(self.images[idx]).save(
            self.save_dir / filename
        )

    def get_characters(self) -> list[str]:
        """Return all unique characters available in the dataset."""
        return sorted(self.metadata["character"].unique())
