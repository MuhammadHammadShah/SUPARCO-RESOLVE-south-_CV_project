from pathlib import Path

import yaml

from .schema import (
    Config,
    DataConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
)


def load_yaml(path: str | Path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str | Path) -> Config:

    cfg = load_yaml(config_path)

    return Config(
        seed=cfg["seed"],
        device=cfg["device"],
        experiment_name=cfg["experiment_name"],
        output_dir=cfg["output_dir"],
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        mixed_precision=cfg["mixed_precision"],
        model=ModelConfig(**cfg["model"]),
        training=TrainingConfig(**cfg["training"]),
        loss=LossConfig(**cfg["loss"]),
        optimizer=OptimizerConfig(**cfg["optimizer"]),
        scheduler=SchedulerConfig(**cfg["scheduler"]),
    )


def load_data_config(data_path: str | Path) -> DataConfig:

    cfg = load_yaml(data_path)

    return DataConfig(
        dataset=DatasetConfig(**cfg["dataset"]),
        targets=cfg["targets"],
    )