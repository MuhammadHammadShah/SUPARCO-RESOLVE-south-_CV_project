from __future__ import annotations

import torch.nn as nn

from torchvision.models import (
    densenet121,
    DenseNet121_Weights,
)


def build_densenet121(
    num_outputs: int,
    pretrained: bool = True,
    dropout: float = 0.2,
):

    weights = (
        DenseNet121_Weights.DEFAULT
        if pretrained
        else None
    )

    model = densenet121(
        weights=weights
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(
            in_features,
            num_outputs,
        ),
    )

    return model