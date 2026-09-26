"""Logging setup shared by the command line entry points.

Ref: Sec. 4.2 (agent calls are logged with the ledger timestamp).
"""

from __future__ import annotations

import logging
import sys

ROOT_LOGGER = "crb_concordance"
DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: str | int = "INFO", *, stream: object | None = None) -> logging.Logger:
    logger = logging.getLogger(ROOT_LOGGER)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
    logger.handlers = [handler]
    logger.setLevel(level if isinstance(level, int) else logging.getLevelName(level.upper()))
    logger.propagate = False
    return logger


def get_logger(suffix: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_LOGGER}.{suffix}")
