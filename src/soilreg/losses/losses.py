from __future__ import annotations

import torch.nn as nn


def mse_loss():
    return nn.MSELoss()


def mae_loss():
    return nn.L1Loss()


def huber_loss(delta: float = 1.0):
    return nn.HuberLoss(delta=delta)


def smooth_l1_loss(beta: float = 1.0):
    return nn.SmoothL1Loss(beta=beta)