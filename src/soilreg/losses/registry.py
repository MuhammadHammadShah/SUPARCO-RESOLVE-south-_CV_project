from __future__ import annotations

from .losses import (
    huber_loss,
    mae_loss,
    mse_loss,
    smooth_l1_loss,
)

LOSS_REGISTRY = {
    "mse": mse_loss,
    "mae": mae_loss,
    "huber": huber_loss,
    "smooth_l1": smooth_l1_loss,
}


def build_loss(name: str):

    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}")

    return LOSS_REGISTRY[name]()