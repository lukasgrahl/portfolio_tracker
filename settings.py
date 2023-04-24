import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logger')

today_str = str(datetime.today().date())


LOGGING_DICT_CONFIG = {
    "version": 1,
    "formatters": {
        "simple": {"format": '%(asctime)s - %(name)s - %(levelname)s - %(message)s'},
        "detailed": {"format": '%(asctime)s %(module)-17s line:%(lineno)-4d ' \
                               '%(levelname)-8s %(message)s'}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": os.path.join(LOGS_DIR, "logs.log"),
            "mode": 'a'
        },
        "daily_file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": os.path.join(LOGS_DIR, f"logs_{today_str}.log"),
            "mode": 'a'
        },
        "error": {
            "class": "logging.FileHandler",
            "level": "WARN",
            "formatter": "detailed",
            "filename": os.path.join(LOGS_DIR, "errors.log"),
            "mode": 'a'
        },
        "daily_error": {
            "class": "logging.FileHandler",
            "level": "WARN",
            "formatter": "detailed",
            "filename": os.path.join(LOGS_DIR, f"errors_{today_str}.log"),
            "mode": 'a'
        }
    },
    "loggers": {
        "simple": {
            "level": "DEBUG",
            "handlers": ["console", ],
            "propagate": False
        },
        "file": {
            "level": "INFO",
            "handlers": ["file", ]
        },
        "error": {
            "level": "INFO",
            "handlers": ["error", ],
            "propagate": False
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["daily_file", "file", "error", "daily_error", "console", ]
    }
}