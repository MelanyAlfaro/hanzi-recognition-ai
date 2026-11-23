"""Dataset generation: draw each character with several fonts plus mild augmentation.

Typical use (``src/scripts/build_dataset.py`` does exactly this):

    builder = DatasetBuilder.from_font_dir(FONTS_DIR, image_size=config.IMG_SIZE, seed=config.SEED)
    builder.generate(config.PROCESSED_DIR, images_per_char=350)

Output files in ``out_dir``:
    images.npy       uint8 ``[N, H, W]``, ink = 255 (white), background = 0 (black)
    labels.npy       uint8 ``[N, q]``, columns in the order of ``RADICALS``
    metadata.csv     row i describes image i: character, codepoint, font,
                     variant, augmented, split
    generation.json  seed, fonts, parameters, counts and library versions

Labels come from ``CHAR_TO_RADICALS`` (``src/tools/radicals.py``). 
"""
from __future__ import annotations

import datetime
import json
import platform
import random
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import albumentations as A
import numpy as np
import numpy.typing as npt
import pandas as pd
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

import config
from src.tools.radicals import CHAR_TO_RADICALS, RADICALS, label_vector

FONT_EXTS: set[str] = {".ttf", ".otf", ".ttc"}
MAX_ATTEMPTS: int = 20  # augmentation retries per image before falling back to the clean glyph


class DatasetBuilder:
    """Render characters with several fonts and write them to disk.

    Attributes:
        chars: Characters to generate.
        font_paths: Font files to draw with.
        font_names: Mapping from font path to its unique short name (file stem).
        image_size: Side of the square images, in pixels.
        glyph_ratio: Glyph size relative to the image side.
        seed: Base seed for the augmentation.
    """

    def __init__(
        self,
        font_paths: Sequence[str | Path],
        chars: Sequence[str] | None = None,
        image_size: int = config.IMG_SIZE,
        glyph_ratio: float = 0.75,
        seed: int = config.SEED,
    ) -> None:
        """Create a builder and check which fonts cover which characters.

        Args:
            font_paths: Font files to draw with.
            chars: Characters to generate. Defaults to all of ``CHAR_TO_RADICALS``.
            image_size: Side of the square images (use ``config.IMG_SIZE``).
            glyph_ratio: Glyph size relative to the image side.
            seed: Base seed for the augmentation (use ``config.SEED``).

        Raises:
            KeyError: If a character has no entry in ``CHAR_TO_RADICALS``.
            ValueError: If two fonts share the same file name (stem).
        """
        # Chars to generate
        self.chars: list[str] = list(chars) if chars is not None else list(CHAR_TO_RADICALS)
        
        # Check for unregistered chars
        unknown = [c for c in self.chars if c not in CHAR_TO_RADICALS]
        if unknown:
            raise KeyError(f"No radicals defined in CHAR_TO_RADICALS for: {''.join(unknown)}")
        
        # Font paths to draw with
        self.font_paths: list[str] = [str(p) for p in font_paths]
        
        # Get font names and check for duplicate fonts
        names = {fp: Path(fp).stem for fp in self.font_paths}
        if len(set(names.values())) != len(names):
            raise ValueError("Two fonts share the same file name (stem)")
        
        # Set attributes
        self.font_names: dict[str, str] = names
        self.image_size: int = image_size
        self.glyph_ratio: float = glyph_ratio
        self.seed: int = seed
        
        # Map characters to fonts that cover it
        self._coverage: dict[str, list[str]] = self._build_coverage()

    @classmethod
    def from_font_dir(cls, font_dir: str | Path, **kwargs: Any) -> DatasetBuilder:
        """Alternative constructor that creates a builder from every font file found under a folder.

        Instead of passing a list of font directories, only a folder directory is needed.
        
        Args:
            font_dir: Folder searched recursively for ``.ttf``, ``.otf`` and ``.ttc`` files.
            **kwargs: Passed to the constructor (``chars``, ``image_size``, ...).

        Returns:
            A new ``DatasetBuilder``.

        Raises:
            FileNotFoundError: If the folder contains no font files.
        """
        # Recursively search for font files with extensions
        fonts = sorted(p for p in Path(font_dir).rglob("*") if p.suffix.lower() in FONT_EXTS)
        
        # Raise error if no fonts are found at all
        if not fonts:
            raise FileNotFoundError(f"No fonts found in {font_dir}")
        
        # Call constructor and return created DatasetBuilder
        return cls(fonts, **kwargs) # cls: reference to DatasetBuilder class itself

    def _build_coverage(self) -> dict[str, list[str]]:
        """Find which fonts have a glyph for each character (avoids blank 'tofu').

        Returns:
            Mapping from character to the paths of the fonts that cover it.
        """
        coverage: dict[str, list[str]] = {c: [] for c in self.chars}
        for fp in self.font_paths:
            try:
                codepoints = _cmap(fp)
            except Exception as e:             # corrupt or unreadable font
                print(f"[warning] skipping {fp}: {e}")
                continue
            for c in self.chars:
                if ord(c) in codepoints:
                    coverage[c].append(fp)
        return coverage

    def coverage_report(self) -> dict[str, int]:
        """Count how many fonts cover each character.

        Returns:
            Mapping from character to the number of fonts that have its glyph.
        """
        return {c: len(fonts) for c, fonts in self._coverage.items()}

    def render(self, ch: str, font_path: str) -> npt.NDArray[np.uint8]:
        """Draw one centered glyph.

        Args:
            ch: The character to draw.
            font_path: Path to the font to draw it with.

        Returns:
            A uint8 array ``[H, W]`` with ink = 255 on a background of 0.
        """
        n = self.image_size
        font = _load_font(font_path, int(n * self.glyph_ratio))
        
        # Create image with Pillow
        # mode: "L" mode in Pillow means 8-bit grayscale (luminance).
        # size: n x n, default config.IMG_SIZE
        # color: 0 for black background, consistent with computer vision conventions.
        img = Image.new("L", (n, n), 0)
        
        # Add character with applied font, fill=255 for white strokes 
        ImageDraw.Draw(img).text((n / 2, n / 2), ch, font=font, fill=255, anchor="mm")
        
        # Return as numpy uint8 array
        return np.array(img, dtype=np.uint8)
    
    
    def generate(
        self,
        out_dir: str | Path,
        images_per_char: int = 350,
        augment: A.Compose | bool | None = None,
        seed: int | None = None,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Generate the dataset and write it to out_dir.

        Every character gets exactly images_per_char images, distributed as
        evenly as possible over the fonts that cover it.

        Variant 0 of each character-font combination is the clean glyph;
        the others are augmented using deterministic seeds derived from
        (seed, character, font, variant, attempt).

        Augmented images that fail the plausibility filter are retried.
        The split column stays empty until assign_splits is called.

        Args:
            out_dir: Directory to write images.npy, labels.npy, metadata.csv,
                and generation.json into.
            images_per_char: Number of images per character.
            augment: Albumentations pipeline. None or True uses default_augment().
            seed: Base seed. Defaults to the builder's seed.
            overwrite: If True, replace an existing dataset in out_dir.

        Returns:
            Metadata DataFrame containing one row per generated image.

        Raises:
            ValueError: If augmentation is disabled, a character has no font
                coverage, or images_per_char is smaller than the number of
                fonts covering a character.
            FileExistsError: If out_dir already holds a dataset and overwrite
                is False.
        """
        aug = self._validate_generation(images_per_char, augment)
        seed = self.seed if seed is None else seed

        files = self._prepare_output_dir(out_dir, overwrite)

        imgs, labels, rows, rejected, fallbacks = self._generate_images(
            images_per_char,
            aug,
            seed,
        )

        # Save metadata
        meta = pd.DataFrame(rows)

        np.save(files["images"], np.stack(imgs).astype(np.uint8))
        np.save(files["labels"], np.array(labels, dtype=np.uint8))
        meta.to_csv(files["meta"], index=False)
        
        # Build and save generation information
        info = self._build_generation_info(
            meta,
            aug,
            seed,
            images_per_char,
            rejected,
            fallbacks,
        )

        self._save_generation_info(files["info"], info)

        print(
            f"[ok] {len(meta)} images in {Path(out_dir)} "
            f"(rejected attempts: {rejected}, "
            f"fallbacks to clean glyph: {fallbacks})"
        )

        return meta

    # AUXILIARY METHODS FOR GENERATE
    def _validate_generation(
        self,
        images_per_char: int,
        augment: A.Compose | bool | None,
    ) -> A.Compose:
        """Validate the generation configuration and resolve augmentation.

        Args:
            images_per_char: Number of images to generate per character.
            augment: Albumentations pipeline. None or True uses default_augment().

        Returns:
            The resolved Albumentations augmentation pipeline.

        Raises:
            ValueError: If augmentation is disabled, a character has no font
                coverage, or images_per_char is smaller than the number of
                fonts covering a character.
        """
        if augment is False:
            raise ValueError("generate needs augmentation to produce variants")

        no_coverage = [c for c in self.chars if not self._coverage[c]]
        if no_coverage:
            raise ValueError(f"No font covers: {''.join(no_coverage)}")

        if len({len(fonts) for fonts in self._coverage.values()}) > 1:
            print(f"[warning] uneven font coverage: {self.coverage_report()}")

        for c in self.chars:
            if images_per_char < len(self._coverage[c]):
                raise ValueError(
                    f"images_per_char={images_per_char} "
                    f"< {len(self._coverage[c])} fonts for {c}"
                )

        return default_augment() if augment is None or augment is True else augment


    def _prepare_output_dir(
        self,
        out_dir: str | Path,
        overwrite: bool,
    ) -> dict[str, Path]:
        """Prepare the output directory and resolve dataset file paths.

        If the directory already contains dataset files, they are preserved
        unless overwrite is True.

        Args:
            out_dir: Directory where the dataset will be stored.
            overwrite: Whether to replace existing dataset files.

        Returns:
            Dictionary mapping dataset file identifiers to their paths.

        Raises:
            FileExistsError: If dataset files already exist and overwrite is False.
        """
        out_dir = Path(out_dir)

        files = {
            "images": out_dir / config.IMG_FILE,
            "labels": out_dir / config.LABELS_FILE,
            "meta": out_dir / config.META_FILE,
            "info": out_dir / config.GENERATION_FILE,
        }

        if any(p.exists() for p in files.values()):
            if not overwrite:
                raise FileExistsError(
                    f"{out_dir} already holds a dataset; use overwrite=True"
                )

            for p in files.values():
                p.unlink(missing_ok=True)

        out_dir.mkdir(parents=True, exist_ok=True)

        return files


    def _generate_images(
        self,
        images_per_char: int,
        aug: A.Compose,
        seed: int,
    ) -> tuple[
        list[npt.NDArray[np.uint8]],
        list[list[int]],
        list[dict[str, Any]],
        int,
        int,
    ]:
        """Generate character images, labels, and associated metadata.
        
        See documentation for generation for details about generation.

        Args:
            images_per_char: Number of images to generate per character.
            aug: Albumentations augmentation pipeline.
            seed: Base random seed used for deterministic augmentation.

        Returns:
            Tuple containing:
                - Generated images as uint8 arrays.
                - Radical label vectors for each image.
                - Metadata dictionaries describing each image.
                - Number of rejected augmentation attempts.
                - Number of fallbacks to clean glyphs.
        """
        imgs: list[npt.NDArray[np.uint8]] = []
        labels: list[list[int]] = []
        rows: list[dict[str, Any]] = []

        rejected = fallbacks = 0

        # Generate images for each char
        for c in self.chars:
            fonts = self._coverage[c]

            # Determine fonts to use, distributed as evenly as possible
            base_n, extra = divmod(images_per_char, len(fonts))
            code = f"{ord(c):04X}"

            # fi = font index, fp = font path
            for fi, fp in enumerate(fonts):
                name = self.font_names[fp]
                base = self.render(c, fp) # Create base image for character

                # Determine amount of images to create with current font
                # Adds 1 if current font index is less than the extra
                # Otherwise, all the extra images have been generated.
                font_images = base_n + (1 if fi < extra else 0)

                # Generate each image
                for v in range(font_images):
                    img = base

                    if v > 0:
                        for attempt in range(MAX_ATTEMPTS):
                            # Create unique key to convert into deterministic seed
                            key = f"{seed}|{code}|{name}|{v}|{attempt}"
                            _seed_augment(aug, zlib.crc32(key.encode()))

                            # Generate augmented image from base
                            candidate = aug(image=base)["image"]

                            # If augmentation did not change the image, retry
                            if np.array_equal(candidate, base):
                                continue

                            # If augmentation passes plausability test, select the candidate and break 
                            if _plausible(candidate, base):
                                img = candidate
                                break

                            rejected += 1
                        else:
                            fallbacks += 1

                    # Add image array
                    imgs.append(img)
                    # Add the label vector (0/1 with radicals) associated to this image
                    labels.append(label_vector(c))

                    # Add metadata
                    rows.append({
                        "character": c,
                        "codepoint": f"U+{code}",
                        "font": name,
                        "variant": v,
                        "augmented": int(not np.array_equal(img, base)), # 1 if augmented
                        "split": "", # No split assigned for now
                    })

        return imgs, labels, rows, rejected, fallbacks


    def _build_generation_info(
        self,
        meta: pd.DataFrame,
        aug: A.Compose,
        seed: int,
        images_per_char: int,
        rejected: int,
        fallbacks: int,
    ) -> dict[str, Any]:
        """Build the generation configuration and statistics dictionary.

        The resulting dictionary describes the dataset generation process,
        including its configuration, fonts, augmentation pipeline, plausibility
        filter, generation statistics, and software versions.

        Args:
            meta: Metadata table describing the generated images.
            aug: Albumentations augmentation pipeline.
            seed: Base random seed used during generation.
            images_per_char: Number of images generated per character.
            rejected: Number of rejected augmentation attempts.
            fallbacks: Number of times the clean glyph was used as a fallback.

        Returns:
            Dictionary containing the dataset generation information.
        """
        try:
            aug_description = aug.to_dict()
        except Exception:
            aug_description = repr(aug)

        return {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "image_size": self.image_size,
            "glyph_ratio": self.glyph_ratio,
            "images_per_char": images_per_char,
            "storage": (
                f"{config.IMG_FILE} uint8 [N,H,W] ink=255 background=0; "
                f"{config.LABELS_FILE} uint8 [N,q]; "
                f"row i of {config.META_FILE} describes image i"
            ),
            "radicals": RADICALS,
            "char_to_radicals": {
                c: CHAR_TO_RADICALS[c] for c in self.chars
            },
            "fonts": [
                {"name": self.font_names[fp], "path": fp}
                for fp in self.font_paths
                if any(fp in self._coverage[c] for c in self.chars)
            ],
            "augmentation": aug_description,
            "plausibility_filter": {
                "ink_ratio": [0.7, 1.5],
                "border": "no ink",
                "max_attempts": MAX_ATTEMPTS,
            },
            "stats": {
                "rejected_attempts": rejected,
                "fallbacks_to_clean": fallbacks,
            },
            "counts": {
                "total": len(meta),
                "per_char": meta["character"].value_counts().to_dict(),
            },
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pillow": Image.__version__,
                "albumentations": A.__version__,
            },
        }


    def _save_generation_info(
        self,
        path: Path,
        info: dict[str, Any],
    ) -> None:
        """Serialize and save generation information as a JSON file.

        Args:
            path: Destination path for generation.json.
            info: Dictionary containing generation configuration and statistics.
        """
        path.write_text(
            json.dumps(info, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    
# ---------------------------------------------------------------------------
# Fonts (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _cmap(font_path: str) -> frozenset[int]:
    """Return the Unicode codepoints that a font actually covers.

    Args:
        font_path: Path to a ``.ttf``, ``.otf`` or ``.ttc`` file.

    Returns:
        The set of codepoints with a glyph in the font.
    """
    tt = TTFont(font_path, fontNumber=0, lazy=True)
    return frozenset((tt.getBestCmap() or {}).keys())


@lru_cache(maxsize=256)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font at a given pixel size (first face of a collection).

    Args:
        font_path: Path to the font file.
        size: Font size in pixels.

    Returns:
        The loaded FreeType font.
    """
    return ImageFont.truetype(font_path, size, index=0)


# ---------------------------------------------------------------------------
# Mild augmentation. No flips or strong warps: they can erase a stroke and
# make the label false.
# ---------------------------------------------------------------------------
def _gauss_noise(p: float) -> A.BasicTransform:
    """Build a mild Gaussian noise transform for Albumentations 1.x and 2.x.

    Handles compatibility between Albumentations versions.
    
    Args:
        p: Probability of applying the transform.

    Returns:
        The noise transform.
    """
    try:
        return A.GaussNoise(std_range=(0.01, 0.05), p=p)   # 2.x
    except TypeError:
        return A.GaussNoise(var_limit=(5.0, 25.0), p=p)    # 1.x


def default_augment() -> A.Compose:
    """Build the default mild augmentation pipeline.

    Returns:
        A pipeline with a small rotation (+-5 degrees), shifts (+-5 %), size
        changes (+-8 %), slightly thicker or thinner strokes, blur, noise and a
        brightness/contrast change.
    """
    return A.Compose( # Compose chains transformations and applies them sequentially
        [
            # Rotate, scale, and shift
            A.Affine(rotate=(-5, 5), scale=(0.92, 1.08),
                     translate_percent=(-0.05, 0.05), p=0.8),
            # Slightly thicker or thinner strokes.
            A.OneOf([A.Morphological(scale=(2, 2), operation="dilation"),
                     A.Morphological(scale=(2, 2), operation="erosion")], p=0.15),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            _gauss_noise(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ]
    )


def _seed_augment(aug: A.Compose, seed: int) -> None:
    """Seed an augmentation pipeline so that its next output is reproducible.

    Args:
        aug: The pipeline to seed.
        seed: Seed value (non-negative, below 2**32).
    """
    random.seed(seed)
    np.random.seed(seed)                       # Albumentations 1.x uses global state
    if hasattr(aug, "set_random_seed"):        # Albumentations 2.x has its own generator
        aug.set_random_seed(seed)


def _plausible(img: npt.NDArray[np.uint8], base: npt.NDArray[np.uint8]) -> bool:
    """Check that an augmented glyph probably still shows all its strokes.

    The amount of ink must stay between 70 % and 150 % of the clean glyph and
    no ink may touch the image border (which would mean the glyph was cut).

    Args:
        img: The augmented image, ink = 255.
        base: The clean image of the same glyph.

    Returns:
        True if the augmented image is acceptable, False otherwise.
    """
    # Check that amount of ink stays between 70 % and 150 % of the clean glyph
    ink, ref = int((img > 127).sum()), int((base > 127).sum())
    if ref == 0 or not (0.7 * ref <= ink <= 1.5 * ref):
        return False
    
    # Check that glyph does not touch border (is not cut)
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    
    return int(border.max()) < 128
