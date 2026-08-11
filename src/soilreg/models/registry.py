from __future__ import annotations

from .backbones.resnet import build_resnet50
from .backbones.resnet50_d import build_resnet50_d
from .backbones.resnet101_v2 import build_resnet101_v2
from .backbones.densenet121 import build_densenet121

from .backbones.efficientnet_b3 import build_efficientnet_b3
from .backbones.convnext_tiny import build_convnext_tiny
from .backbones.vit_base import build_vit_base


MODEL_REGISTRY = {
    "resnet50": build_resnet50,
    "resnet50_d": build_resnet50_d,
    "resnet101_v2": build_resnet101_v2,
    "densenet121": build_densenet121,
    "efficientnet_b3": build_efficientnet_b3,
    "convnext_tiny": build_convnext_tiny,
    "vit_base": build_vit_base,
}


def build_model(
    model_name,
    num_outputs,
    pretrained,
    dropout,
):

    print(f"Model: {model_name}")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_name](
        num_outputs=num_outputs,
        pretrained=pretrained,
        dropout=dropout,
    )