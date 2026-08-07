from __future__ import annotations

import torch.nn as nn


class RegressionHead(nn.Module):
    """
    Generic regression head.

    Input:
        Feature vector from backbone

    Output:
        Mineral composition predictions
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),

            nn.BatchNorm1d(hidden_dim),

            nn.ReLU(inplace=True),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):

        return self.head(x)