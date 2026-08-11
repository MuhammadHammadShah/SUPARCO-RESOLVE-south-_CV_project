from __future__ import annotations

import torch.nn as nn

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)

from ..base_model import BaseRegressionModel


def build_convnext_tiny(
    num_outputs: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> BaseRegressionModel:

    weights = (
        ConvNeXt_Tiny_Weights.DEFAULT
        if pretrained
        else None
    )

    model = convnext_tiny(weights=weights)

    feature_dim = model.classifier[2].in_features

    # Remove original classification head
    model.classifier = nn.Identity()

    return BaseRegressionModel(
        backbone=model,
        feature_dim=feature_dim,
        num_outputs=num_outputs,
        dropout=dropout,
    )