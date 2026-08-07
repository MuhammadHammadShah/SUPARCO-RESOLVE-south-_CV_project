from __future__ import annotations

from soilreg.config.loader import (
    load_config,
    load_data_config,
)

from soilreg.inference import SoilMineralPredictor


def main():

    # ---------------------------------------------------------
    # Load configs
    # ---------------------------------------------------------

    cfg = load_config(
        "configs/config.yaml"
    )

    data_cfg = load_data_config(
        "configs/data/soil_images.yaml"
    )

    # ---------------------------------------------------------
    # Create predictor
    # ---------------------------------------------------------

    predictor = SoilMineralPredictor(

        checkpoint_path="artifacts/checkpoints/best.pt",

        model_name=cfg.model.name,

        num_outputs=len(data_cfg.targets),

        targets=data_cfg.targets,

        image_size=data_cfg.dataset.image_size,

        pretrained=False,

        dropout=cfg.model.dropout,

        device=cfg.device,

    )

    # ---------------------------------------------------------
    # Image to predict
    # ---------------------------------------------------------

    image_path = input(
        "Enter image path: "
    ).strip()

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    predictions = predictor.predict(
        image_path
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\nPredicted Mineral Composition")
    print("-" * 40)

    for mineral, value in predictions.items():

        print(
            f"{mineral:<5}: {value:.4f}"
        )


if __name__ == "__main__":
    main()