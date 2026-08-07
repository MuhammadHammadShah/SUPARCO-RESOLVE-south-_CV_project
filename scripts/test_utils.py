from soilreg.config.loader import load_config, load_data_config
from soilreg.utils import (
    get_logger,
    get_device,
    gpu_name,
    seed_everything,
)
from soilreg.data.validators import validate_dataset

cfg = load_config("configs/config.yaml")
data_cfg = load_data_config("configs/data/soil_images.yaml")

logger = get_logger()

seed_everything(cfg.seed)

device = get_device(cfg.device)

logger.info(f"Device : {device}")
logger.info(f"GPU    : {gpu_name()}")

validate_dataset(
    image_dir=data_cfg.dataset.image_dir,
    csv_file=data_cfg.dataset.csv_file,
    sample_column=data_cfg.dataset.sample_id_column,
    targets=data_cfg.targets,
)

logger.info("Everything works.")