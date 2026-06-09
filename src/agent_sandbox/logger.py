"""Minimal standalone structured logger backed by the stdlib ``logging`` module."""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured :class:`logging.Logger` for ``name``.

    The logger gets a single stream handler with a sensible default format the
    first time it is requested. The log level is taken from the ``ASB_LOG_LEVEL``
    environment variable, defaulting to ``INFO``.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(os.environ.get("ASB_LOG_LEVEL", "INFO").upper())
    return logger
