"""Minimal standalone structured logger backed by the stdlib ``logging`` module.

``agent_sandbox`` deliberately depends on nothing but the standard library, so
this is a thin wrapper over :mod:`logging` — no structlog, no Datadog, no boto3.
:func:`get_logger` returns an ordinary :class:`logging.Logger` whose level is
taken from the ``ASB_LOG_LEVEL`` environment variable (default ``INFO``).
"""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured :class:`logging.Logger` for *name*.

    The logger gets a single stream handler with a sensible default format the
    first time it is requested; subsequent calls reuse the same handler so no
    duplicates accumulate. The log level honours the ``ASB_LOG_LEVEL``
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
