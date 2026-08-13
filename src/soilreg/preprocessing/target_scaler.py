from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TargetScaler:

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(
        self,
        dataframe: pd.DataFrame,
        targets: list[str],
    ):
        self.scaler.fit(
            dataframe[targets]
        )

    def transform(
        self,
        dataframe: pd.DataFrame,
        targets: list[str],
    ) -> pd.DataFrame:

        dataframe = dataframe.copy()

        dataframe[targets] = self.scaler.transform(
            dataframe[targets]
        )

        return dataframe

    def inverse_transform(
        self,
        values,
    ):
        return self.scaler.inverse_transform(values)

    def save(
        self,
        path: str | Path,
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self.scaler,
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ):

        scaler = cls()

        scaler.scaler = joblib.load(path)

        return scaler