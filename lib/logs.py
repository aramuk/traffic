import logging

logging.basicConfig(level=logging.INFO)

LOGGER_NAME = 'traffic'

get_logger = lambda: logging.getLogger(LOGGER_NAME)
