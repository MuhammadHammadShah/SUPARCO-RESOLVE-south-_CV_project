import os

# Suppress Albumentations update check network timeout warning
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from soilreg.config.loader import load_config, load_data_config
from soilreg.data import SoilDataModule


def main():
    cfg = load_config("configs/config.yaml")
    data_cfg = load_data_config("configs/data/soil_images.yaml")

    dm = SoilDataModule(
        image_dir=data_cfg.dataset.image_dir,
        csv_file=data_cfg.dataset.csv_file,
        targets=data_cfg.targets,
        sample_column=data_cfg.dataset.sample_id_column,
        image_extension=data_cfg.dataset.image_extension,
        image_size=data_cfg.dataset.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    dm.setup()

    loader = dm.train_dataloader()

    images, targets = next(iter(loader))

    print(f"Images batch shape:  {images.shape}")
    print(f"Targets batch shape: {targets.shape}")
    print(f"First sample targets: {targets[0]}")


if __name__ == "__main__":
    main()