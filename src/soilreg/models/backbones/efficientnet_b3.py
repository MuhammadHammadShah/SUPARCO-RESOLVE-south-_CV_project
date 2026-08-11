from __future__ import annotations

import timm

from ..base_model import BaseRegressionModel


def build_efficientnet_b3(
    num_outputs: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> BaseRegressionModel:
    """
    Build an EfficientNet-B3 regression model.

    Args:
        num_outputs: Number of regression targets.
        pretrained: Load ImageNet pretrained weights.
        dropout: Dropout used in regression head.

    Returns:
        BaseRegressionModel
    """

    model = timm.create_model(
        "efficientnet_b3",
        pretrained=pretrained,
        num_classes=0,
    )

    feature_dim = model.num_features

    return BaseRegressionModel(
        backbone=model,
        feature_dim=feature_dim,
        num_outputs=num_outputs,
        dropout=dropout,
    )