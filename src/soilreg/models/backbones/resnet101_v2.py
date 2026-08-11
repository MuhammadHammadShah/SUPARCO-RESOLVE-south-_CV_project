from __future__ import annotations

import timm

from ..base_model import BaseRegressionModel


def build_resnet101_v2(
    num_outputs: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> BaseRegressionModel:
    """
    Build a ResNet-101-V2 regression model.

    ResNet-V2 uses the pre-activation residual design.

    Args:
        num_outputs: Number of regression targets.
        pretrained: Load ImageNet pretrained weights.
        dropout: Dropout used in regression head.

    Returns:
        BaseRegressionModel
    """

    model_name = "resnetv2_101"

    model = timm.create_model(
        model_name,
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