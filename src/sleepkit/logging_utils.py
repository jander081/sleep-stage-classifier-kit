# src/sleepkit/logging_utils.py

import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(
    log_name="sleepkit",
    log_dir="logs",
    log_file="sleepkit.log",
    max_bytes=5_000_000,
    backup_count=5,
    level=logging.INFO,
):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger