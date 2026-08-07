from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import torch

from soilreg.models.registry import build_model
from soilreg.data.transforms import get_valid_transforms


class SoilMineralPredictor:

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str,
        num_outputs: int,
        targets: list[str],
        image_size: int = 224,
        pretrained: bool = False,
        dropout: float = 0.0,
        device: str = "auto",
    ):

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.targets = targets

        self.transforms = get_valid_transforms(
            image_size
        )

        self.model = build_model(
            model_name=model_name,
            num_outputs=num_outputs,
            pretrained=pretrained,
            dropout=dropout,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.model.to(self.device)

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
    ) -> dict[str, float]:

        image = Image.open(
            image_path
        ).convert("RGB")

        image = np.array(image)

        image = self.transforms(
            image=image
        )["image"]

        image = image.unsqueeze(0)

        image = image.to(self.device)

        outputs = self.model(image)

        outputs = outputs.squeeze(0)

        outputs = outputs.cpu().numpy()

        return {

            target: float(value)

            for target, value in zip(
                self.targets,
                outputs,
            )

        }