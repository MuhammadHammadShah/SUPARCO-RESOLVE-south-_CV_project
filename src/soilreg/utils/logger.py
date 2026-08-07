from __future__ import annotations

import logging
from pathlib import Path


def get_logger(
    name: str = "soilreg",
    log_dir: str = "artifacts/logs",
    filename: str = "train.log",
) -> logging.Logger:

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        Path(log_dir) / filename,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger