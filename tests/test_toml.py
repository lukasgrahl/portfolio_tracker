import os

from settings import PROJECT_ROOT
from src.get_toml import get_toml_data

from src.get_logger import get_logger
get_logger(log_level="DEBUG")

config = get_toml_data(os.path.join(PROJECT_ROOT, 'config.toml'))

print(config)