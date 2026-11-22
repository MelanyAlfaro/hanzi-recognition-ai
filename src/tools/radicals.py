"""Project radicals and the radicals contained in each character.

IMPORTANT: the order of ``RADICALS`` defines the columns of the label vector
``[木, 日, 月, 口]`` and must NEVER change after the dataset is generated
(``generation.json`` stores this order and ``RadicalDataset`` checks it).

The task is presence/absence, not counting: 林 and 木 have the same label.
"""
from __future__ import annotations

RADICALS: list[str] = ["木", "日", "月", "口"]

CHAR_TO_RADICALS: dict[str, list[str]] = {
    "林": ["木"],
    "森": ["木"],
    "昌": ["日"],
    "晶": ["日"],
    "朋": ["月"],
    "杳": ["木", "日"],
    "棚": ["木", "月"],
    "品": ["口"],
    "呆": ["木", "口"],
    "唱": ["日", "口"],
    "杲": ["木", "日"],
    "呂": ["口"],
    "明": ["日", "月"],
    "杏": ["木", "口"],
}


def label_vector(char: str) -> list[int]:
    """Build the 0/1 label vector of a character.

    Args:
        char: A character listed in ``CHAR_TO_RADICALS``.

    Returns:
        A list of length ``len(RADICALS)`` with 1 where the character contains
        the radical and 0 elsewhere. For example, 明 gives ``[0, 1, 1, 0]``.

    Raises:
        KeyError: If ``char`` is not listed in ``CHAR_TO_RADICALS``.
    """
    present = CHAR_TO_RADICALS[char]
    return [1 if radical in present else 0 for radical in RADICALS]


def _validate_mapping() -> None:
    """Check that ``CHAR_TO_RADICALS`` only uses radicals from ``RADICALS``.

    Raises:
        ValueError: If a character lists a radical that is not in ``RADICALS``.
    """
    for char, radicals in CHAR_TO_RADICALS.items():
        unknown = [r for r in radicals if r not in RADICALS]
        if unknown:
            raise ValueError(f"{char}: unknown radicals {unknown}")


# Fail at import time on a typo instead of producing wrong labels later.
_validate_mapping()