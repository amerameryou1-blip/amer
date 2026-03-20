"""
Logging setup helpers for Territorial.io bot.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from utils.file_utils import ensure_dir


def setup_logger(name: str, log_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to both console and a timestamped log file.

    Args:
        name (str): Logical logger name.
        log_dir (str | Path): Directory for output log files.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """

    resolved_log_dir = ensure_dir(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = resolved_log_dir / f"bot_{name}_{timestamp}.log"
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
