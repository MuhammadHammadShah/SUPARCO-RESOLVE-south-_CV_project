from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from soilreg.preprocessing import TargetScaler

from .dataset import SoilRegressionDataset
from .transforms import (
    get_train_transforms,
    get_valid_transforms,
)


class SoilDataModule:

    def __init__(
        self,
        image_dir: str,
        csv_file: str,
        targets: list[str],
        sample_column: str,
        image_extension: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ):

        self.image_dir = image_dir
        self.csv_file = csv_file
        self.targets = targets
        self.sample_column = sample_column
        self.image_extension = image_extension
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.seed = seed

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.target_scaler = TargetScaler()

    def setup(self):

        # ---------------------------------------------------------
        # Read CSV
        # ---------------------------------------------------------

        df = pd.read_csv(
            self.csv_file
        )

        # ---------------------------------------------------------
        # Split
        # ---------------------------------------------------------

        train_df, temp_df = train_test_split(
            df,
            test_size=1 - self.train_ratio,
            random_state=self.seed,
            shuffle=True,
        )

        remaining = (
            self.val_ratio
            + self.test_ratio
        )

        test_fraction = (
            self.test_ratio
            / remaining
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_fraction,
            random_state=self.seed,
            shuffle=True,
        )

        # ---------------------------------------------------------
        # Normalize targets
        # ---------------------------------------------------------

        self.target_scaler.fit(
            train_df,
            self.targets,
        )

        train_df = self.target_scaler.transform(
            train_df,
            self.targets,
        )

        val_df = self.target_scaler.transform(
            val_df,
            self.targets,
        )

        test_df = self.target_scaler.transform(
            test_df,
            self.targets,
        )

        # ---------------------------------------------------------
        # Save scaler
        # ---------------------------------------------------------

        self.target_scaler.save(
            "artifacts/scaler.pkl"
        )

        # ---------------------------------------------------------
        # Datasets
        # ---------------------------------------------------------

        self.train_dataset = SoilRegressionDataset(
            dataframe=train_df,
            image_dir=self.image_dir,
            targets=self.targets,
            sample_column=self.sample_column,
            image_extension=self.image_extension,
            transforms=get_train_transforms(
                self.image_size
            ),
        )

        self.val_dataset = SoilRegressionDataset(
            dataframe=val_df,
            image_dir=self.image_dir,
            targets=self.targets,
            sample_column=self.sample_column,
            image_extension=self.image_extension,
            transforms=get_valid_transforms(
                self.image_size
            ),
        )

        self.test_dataset = SoilRegressionDataset(
            dataframe=test_df,
            image_dir=self.image_dir,
            targets=self.targets,
            sample_column=self.sample_column,
            image_extension=self.image_extension,
            transforms=get_valid_transforms(
                self.image_size
            ),
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )