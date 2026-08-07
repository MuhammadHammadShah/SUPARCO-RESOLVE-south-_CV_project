from soilreg.config.loader import load_config, load_data_config

cfg = load_config("configs/config.yaml")
data_cfg = load_data_config("configs/data/soil_images.yaml")

print(cfg)
print()
print(data_cfg)