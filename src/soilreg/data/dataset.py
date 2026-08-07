from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class SoilRegressionDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        targets: list[str],
        sample_column: str,
        image_extension: str = ".jpg",
        transforms=None,
    ):

        self.df = dataframe.reset_index(drop=True)

        self.image_dir = Path(image_dir)

        self.targets = targets

        self.sample_column = sample_column

        self.image_extension = image_extension

        self.transforms = transforms

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        sample_id = str(row[self.sample_column])

        image_path = self.image_dir / f"{sample_id}{self.image_extension}"

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image = Image.open(image_path).convert("RGB")

        image = np.array(image)

        if self.transforms is not None:

            image = self.transforms(image=image)["image"]

        target = torch.tensor(
            row[self.targets].values.astype(np.float32),
            dtype=torch.float32,
        )

        return image, target