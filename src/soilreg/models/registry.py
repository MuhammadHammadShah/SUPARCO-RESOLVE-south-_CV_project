from __future__ import annotations

from .backbones.resnet import build_resnet50
from .backbones.densenet121 import build_densenet121
# from .backbones.efficientnet import build_efficientnet_b3
# from .backbones.convnext import build_convnext_tiny
# from .backbones.vit import build_vit_base

MODEL_REGISTRY = {
    "resnet50": build_resnet50,
    "densenet121": build_densenet121,
    # "efficientnet_b3": build_efficientnet_b3,
    # "convnext_tiny": build_convnext_tiny,
    # "vit_base": build_vit_base,
}


def build_model(
   model_name,
    num_outputs,
    pretrained,
    dropout,
):

    print(f"model_name = '{model_name}'")
    print(MODEL_REGISTRY.keys())

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    return MODEL_REGISTRY[model_name](  
    num_outputs=num_outputs,
    pretrained=pretrained,
    dropout=dropout,)