from __future__ import annotations

from pathlib import Path

from soilreg.config.loader import (
    load_config,
    load_data_config,
)

from soilreg.data.datamodule import SoilDataModule
from soilreg.engine.trainer import Trainer
from soilreg.models.registry import build_model

from soilreg.utils.device import get_device
from soilreg.utils.logger import get_logger
from soilreg.utils.seed import seed_everything



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


    logger = get_logger()



    # ---------------------------------------------------------
    # Seed
    # ---------------------------------------------------------

    seed_everything(
        cfg.seed
    )



    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = get_device(
        cfg.device
    )


    logger.info(
        f"Using device: {device}"
    )



    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    datamodule = SoilDataModule(

        image_dir=data_cfg.dataset.image_dir,

        csv_file=data_cfg.dataset.csv_file,

        targets=data_cfg.targets,

        sample_column=data_cfg.dataset.sample_id_column,

        image_extension=data_cfg.dataset.image_extension,

        image_size=data_cfg.dataset.image_size,

        batch_size=cfg.training.batch_size,

        num_workers=cfg.num_workers,

        pin_memory=cfg.pin_memory,

        seed=cfg.seed,

        train_ratio=cfg.training.train_ratio,

        val_ratio=cfg.training.val_ratio,

        test_ratio=cfg.training.test_ratio,

    )



    datamodule.setup()



    train_loader = (
        datamodule.train_dataloader()
    )

    val_loader = (
        datamodule.val_dataloader()
    )



    logger.info(
        "Data loaded successfully."
    )



    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = build_model(

        model_name=cfg.model.name,

        num_outputs=len(data_cfg.targets),

        pretrained=cfg.model.pretrained,

        dropout=cfg.model.dropout,

    )



    logger.info(
        f"Model: {cfg.model.name}"
    )



    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    trainer = Trainer(

        model=model,

        config=cfg,

        device=device,

    )



    # ---------------------------------------------------------
    # Resume checkpoint
    # ---------------------------------------------------------

    checkpoint_path = (
        "artifacts/checkpoints/last.pt"
    )



    if Path(checkpoint_path).exists():


        logger.info(
            "Checkpoint found. Resuming training..."
        )


        trainer.load_checkpoint(
            checkpoint_path
        )


    else:


        logger.info(
            "No checkpoint found. Starting fresh training..."
        )



    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    trainer.fit(

        train_loader=train_loader,

        val_loader=val_loader,

    )



    logger.info(
        "Training completed successfully."
    )




if __name__ == "__main__":

    main()