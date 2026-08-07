from dataclasses import dataclass
from typing import List


# ==========================================================
# Model
# ==========================================================

@dataclass
class ModelConfig:
    name: str
    pretrained: bool
    dropout: float


# ==========================================================
# Training
# ==========================================================

@dataclass
class TrainingConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float


# ==========================================================
# Loss
# ==========================================================

@dataclass
class LossConfig:
    name: str


# ==========================================================
# Optimizer
# ==========================================================

@dataclass
class OptimizerConfig:
    name: str


# ==========================================================
# Scheduler
# ==========================================================

@dataclass
class SchedulerConfig:
    name: str


# ==========================================================
# Main Config
# ==========================================================

@dataclass
class Config:
    seed: int
    device: str
    experiment_name: str
    output_dir: str

    num_workers: int
    pin_memory: bool
    mixed_precision: bool

    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig


# ==========================================================
# Dataset
# ==========================================================

@dataclass
class DatasetConfig:
    image_dir: str
    csv_file: str
    image_size: int
    image_extension: str
    sample_id_column: str


# ==========================================================
# Data Config
# ==========================================================

@dataclass
class DataConfig:
    dataset: DatasetConfig
    targets: List[str]