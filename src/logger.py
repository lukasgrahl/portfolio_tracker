import logging
import logging.config
import os
import settings
from functools import wraps

from settings import LOGS_DIR

import logging
import os
logger = logging.getLogger()


def init_logging(file, console_level='INFO', discard_old_info=True):
    """
    Function initialises and saves log file
    :param file: path\filename
    :param console_level: logging level for console print
    :param discard_old_info: if True old log file will be overwritten, otherwise appended
    """
    path = os.path.join(LOGS_DIR, file)

    if discard_old_info:
        with open(path, "w") as f:
            f.write('')

    logging.basicConfig(level=console_level)
    file_handler = logging.FileHandler(path)
    logger.addHandler(file_handler)

    logger.info(f'Begin Program | user {os.getlogin()}')
    pass