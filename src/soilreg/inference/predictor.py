from __future__ import annotations

import numpy as np
from PIL import Image

import torch

from soilreg.models.registry import build_model
from soilreg.data.transforms import get_valid_transforms
from soilreg.preprocessing import TargetScaler


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
        scaler_path: str = "artifacts/scaler.pkl",
    ):

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------

        if device == "auto":

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.targets = targets

        # ---------------------------------------------------------
        # Image transforms
        # ---------------------------------------------------------

        self.transforms = get_valid_transforms(
            image_size
        )

        # ---------------------------------------------------------
        # Build model
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Load target scaler
        # ---------------------------------------------------------

        self.target_scaler = TargetScaler()

        self.target_scaler.load(
            scaler_path
        )

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
    ) -> dict[str, float]:

        # ---------------------------------------------------------
        # Read image
        # ---------------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = np.array(image)

        image = self.transforms(
            image=image
        )["image"]

        image = image.unsqueeze(0)

        image = image.to(self.device)

        # ---------------------------------------------------------
        # Forward pass
        # ---------------------------------------------------------

        outputs = self.model(image)

        outputs = outputs.squeeze(0)

        outputs = outputs.cpu().numpy()

        # ---------------------------------------------------------
        # Convert back to original mineral values
        # ---------------------------------------------------------

        outputs = self.target_scaler.inverse_transform(
            outputs.reshape(1, -1)
        )[0]

        # ---------------------------------------------------------
        # Return dictionary
        # ---------------------------------------------------------

        return {

            target: float(value)

            for target, value in zip(
                self.targets,
                outputs,
            )

        }