from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


import re


def get_group_id(sample_id: str) -> str:
    """
    Convert an augmented image Sample ID into its original
    soil-sample group.

    Examples:
        1Ah_m_x_a   -> 1Ah
        1Ah_m_x1_a  -> 1Ah
        1Ah_m_x2_c  -> 1Ah
        10Bv_x2_c   -> 10Bv
        13v_x1_b    -> 13v
    """

    sample_id = str(sample_id).strip()

    match = re.match(r"^(\d+[A-Za-z]+)", sample_id)

    if not match:
        raise ValueError(
            f"Could not determine group ID from Sample ID: {sample_id}"
        )

    return match.group(1)


class SoilRegressionDataset(Dataset):
    """
    Dataset for multi-output soil mineral regression.

    Each image is paired with six target values:
        Cd, Cu, Ni, Mn, Fe, Zn

    Target normalization should be performed BEFORE creating this dataset.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        targets: list[str],
        sample_column: str,
        image_extension: str = ".jpg",
        transforms=None,
    ) -> None:

        self.df = dataframe.reset_index(drop=True)

        self.image_dir = Path(image_dir)

        self.targets = targets

        self.sample_column = sample_column

        self.image_extension = image_extension

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):

        row = self.df.iloc[index]

        # --------------------------------------------------
        # Image path
        # --------------------------------------------------

        sample_id = str(row[self.sample_column])

        image_path = (
            self.image_dir
            / f"{sample_id}{self.image_extension}"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        image = np.asarray(image)

        # --------------------------------------------------
        # Image transforms
        # --------------------------------------------------

        if self.transforms is not None:
            image = self.transforms(
                image=image
            )["image"]

        # --------------------------------------------------
        # Targets
        # --------------------------------------------------

        target_values = row[self.targets].to_numpy(
            dtype=np.float32
        )

        target = torch.tensor(
            target_values,
            dtype=torch.float32,
        )

        return image, target