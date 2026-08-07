from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)

from ..base_model import BaseRegressionModel


def build_resnet50(
    num_outputs: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> BaseRegressionModel:
    """
    Build a ResNet50 regression model.

    Args:
        num_outputs: Number of regression targets.
        pretrained: Load ImageNet pretrained weights.
        dropout: Dropout used in regression head.

    Returns:
        BaseRegressionModel
    """

    weights = ResNet50_Weights.DEFAULT if pretrained else None

    model = resnet50(weights=weights)

    feature_dim = model.fc.in_features

    # Remove classification layer
    model.fc = nn.Identity()

    return BaseRegressionModel(
        backbone=model,
        feature_dim=feature_dim,
        num_outputs=num_outputs,
        dropout=dropout,
    )