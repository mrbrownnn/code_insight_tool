"""
Structured logging utility for Code Insight Tool.
"""

import logging
import sys

from config import settings


def get_logger(name: str) -> logging.Logger:
    """Create a structured logger with consistent formatting.

    Args:
        name: Module name for the logger (typically __name__).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    return logger
