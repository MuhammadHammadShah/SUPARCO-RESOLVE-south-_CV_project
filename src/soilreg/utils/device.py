from __future__ import annotations

import torch


def get_device(device: str = "auto") -> torch.device:

    if device == "cpu":
        return torch.device("cpu")

    if device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")

        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def gpu_name() -> str:

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    return "CPU"