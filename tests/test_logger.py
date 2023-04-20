import logging
import os

logger = logging.getLogger()
console_format = '%(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename='app.log', filemode='w', format=console_format, level='DEBUG')


logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')