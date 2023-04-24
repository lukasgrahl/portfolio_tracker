import logging
import logging.config
import os
import settings
from functools import wraps

from settings import LOGS_DIR

import logging
import os
logger = logging.getLogger()


def init_logging(file, console_level='INFO', file_level='INFO', discard_old_info=True):
    path = os.path.join(LOGS_DIR, file)

    if discard_old_info:
        with open(path, "w") as f:
            f.write('')

    console_format = '%(funcname)s - %(message)s'
    logging.basicConfig(level=console_level) # format=console_format)

    file_handler = logging.FileHandler(path)
    file_format = logging.Formatter('%(asctime) %(name)s %(funcname)s %(levelname)s %(message)s')
    # file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    logger.info(f'Begin Program | user {os.getlogin()}')

    pass