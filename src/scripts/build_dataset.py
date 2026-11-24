"""Generate the dataset once and assign the split by character.

From the project root:

    python -m src.scripts.build_dataset
    python -m src.scripts.build_dataset --fonts fonts --per-char 350 --overwrite

To change fold without regenerating images, call ``assign_splits()`` again
with other character lists (this script is not needed for that).
"""
import argparse
from pathlib import Path

import config
from src.tools.dataset_builder import DatasetBuilder
from src.tools.dataset_splitting import assign_splits


def main() -> None:
    """Parse the command line, generate the dataset and assign the splits.

    Raises:
        FileNotFoundError: If the fonts folder contains no font files.
        FileExistsError: If the output folder already holds a dataset and
            ``--overwrite`` was not given.
        ValueError: If the character lists in ``config`` are inconsistent
            (see ``assign_splits``).
    """
    # Parse Args
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--fonts", type=Path, default=getattr(config, "FONTS_DIR", config.FONTS_DIR),
                        help="folder with .ttf/.otf/.ttc fonts (default: <root>/fonts)")
    parser.add_argument("--out", type=Path, default=Path(config.PROCESSED_DIR),
                        help="output folder (default: config.PROCESSED_DIR)")
    parser.add_argument("--per-char", type=int, default=350, help="images per character")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing dataset in the output folder")
    args = parser.parse_args()

    # Create builder
    builder = DatasetBuilder.from_font_dir(args.fonts, image_size=config.IMG_SIZE,
                                           seed=getattr(config, "SEED", 0))
    
    # Report amount of fonts used
    n_fonts = len(builder.font_paths)
    print(f"{n_fonts} fonts in {args.fonts}")
    
    if n_fonts < 5:
        print("[warning] 5 or more fonts are recommended (serif and sans)")

    try:
        # Generate dataset
        builder.generate(args.out, images_per_char=args.per_char, overwrite=args.overwrite)
        # Assign splits according to the config arrays of characters
        assign_splits(args.out, config.TRAIN_CHARS, config.VAL_CHARS, config.TEST_CHARS)
    except Exception as e:
        print(f"ERROR: {e}")
        


if __name__ == "__main__":
    main()