import logging
import logging.config
import os
import settings
from functools import wraps

import logging
import os
logger = logging.getLogger()


def init_logging(file='run.log', console_level ='INFO', file_level='INFO', discard_old_info=True):
    path = os.path.join(os.path.dirname(__file__), file)

    if discard_old_info:
        with open(path, "w") as f:
            f.write('')

    console_format = '%(name)s | %(funcname)s | %(levelname)s | %(message)s'
    logging.basicConfig(level=console_level, format=console_format)

    file_handler = logging.FileHandler(path)
    file_handler.setLevel((file_level))
    file_format = '%(asctime) | %(name)s | %(funcname)s | %(levelname)s | %(message)s'
    file_handler.setFormatter(file_format)

    logger.addHandler(file_handler)

    logger.info(f'Begin Program | user {os.getlogin()}')


# def get_logger(name, config_dict=settings.LOGGING_DICT_CONFIG):
#     logging.config.dictConfig(config_dict)
#     return logging.getLogger(name=name)
#     pass
#
#
# def catch_and_log_errors(orig_func, config_dict=settings.LOGGING_DICT_CONFIG):
#     logger = get_logger(orig_func.__name__, config_dict=config_dict)
#
#     @wraps(orig_func)
#     def wrapper(*args, **kwargs):
#         try:
#             results = orig_func(*args, **kwargs)
#         except Exception as e:
#             logger.error(f"Got Exception {e}")
#             logger.exception(e)
#             results = None
#         return results
#
#     return wrapper
#     pass
