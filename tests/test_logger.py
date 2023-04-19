import logging
import os
from settings import LOG_DIR

logger = logging.getLogger()

file = f'{__file__}.log'
level = 'WARNING'

# path = os.path.join(LOG_DIR, file)
# print(path)
# logging.info(path)

console_format = '%(name)s - %(levelname)s - %(message)s'

logging.basicConfig(filename='app.log', filemode='w', format=console_format, level=level)


logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')