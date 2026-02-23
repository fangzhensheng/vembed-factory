"""
vembed.utils — Shared utility functions.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger for vembed-factory.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=getattr(logging, level.upper(), logging.INFO),
    )
