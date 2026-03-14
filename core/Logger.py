import logging
import sys

LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

logger = logging.getLogger("myapp")

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(LOGGING_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

logger.info("Logger is initialized")