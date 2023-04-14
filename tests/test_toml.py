import os

from settings import DATA_DIR
from src.get_toml import get_toml_data


config = get_toml_data(os.path.join(DATA_DIR, 'config.toml'))

print(config)