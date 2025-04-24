import os
import sys
from datetime import datetime

from loguru import logger as _logger

from app.config import PROJECT_ROOT

_print_level = "INFO"


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """Adjust the log level to above level"""
    global _print_level
    _print_level = print_level

    current = datetime.now()
    formatted_time = current.strftime("%Y%m%d%H%M%S")
    formatted_date = current.strftime("%Y%m%d")
    log_name = (
        f"{name}_{formatted_time}" if name else formatted_time
    )  # name a log with prefix name

    log_dir = PROJECT_ROOT / f"logs/{formatted_date}"
    os.makedirs(log_dir, exist_ok=True)

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(PROJECT_ROOT / f"logs/{formatted_date}/{log_name}.log", level=logfile_level)
    return _logger


logger = define_log_level(print_level="INFO", logfile_level="DEBUG")


if __name__ == "__main__":
    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
