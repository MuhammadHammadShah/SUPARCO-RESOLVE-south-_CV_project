from __future__ import annotations

import timm

from ..base_model import BaseRegressionModel


def build_vit_base(
    num_outputs: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> BaseRegressionModel:
    """
    Build a Vision Transformer (ViT-Base) regression model.

    Args:
        num_outputs: Number of regression targets.
        pretrained: Load ImageNet pretrained weights.
        dropout: Dropout used in regression head.

    Returns:
        BaseRegressionModel
    """

    model = timm.create_model(
        "vit_base_patch16_224",
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