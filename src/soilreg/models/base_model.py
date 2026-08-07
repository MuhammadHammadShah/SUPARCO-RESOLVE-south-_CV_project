from __future__ import annotations

import torch.nn as nn

from .heads.regression_head import RegressionHead


class BaseRegressionModel(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_outputs: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = backbone

        self.regression_head = RegressionHead(
            in_features=feature_dim,
            out_features=num_outputs,
            dropout=dropout,
        )

    def forward(self, x):

        features = self.backbone(x)

        predictions = self.regression_head(features)

        return predictions