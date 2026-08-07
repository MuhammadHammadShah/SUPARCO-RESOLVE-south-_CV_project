from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def validate_dataset(
    image_dir: str,
    csv_file: str,
    sample_column: str,
    targets: list[str],
) -> None:

    image_dir = Path(image_dir)
    csv_file = Path(csv_file)

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    if not csv_file.exists():
        raise FileNotFoundError(csv_file)

    df = pd.read_csv(csv_file)

    if sample_column not in df.columns:
        raise ValueError(f"{sample_column} missing from csv.")

    for target in targets:

        if target not in df.columns:
            raise ValueError(f"{target} missing from csv.")

    print("✔ Dataset validation successful.")