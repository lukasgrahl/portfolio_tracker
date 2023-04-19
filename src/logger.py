import logging
import logging.config
import os
import settings
from functools import wraps


def get_logger(name, config_dict=settings.LOGGING_DICT_CONFIG):
    logging.config.dictConfig(config_dict)
    return logging.getLogger(name=name)
    pass


def catch_and_log_errors(orig_func, config_dict=settings.LOGGING_DICT_CONFIG):
    logger = get_logger(orig_func.__name__, config_dict=config_dict)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        try:
            results = orig_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Got Exception {e}")
            logger.exception(e)
            results = None
        return results

    return wrapper
    pass
