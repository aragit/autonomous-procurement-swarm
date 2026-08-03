"""Structured logging for the swarm runtime.

Log volume is controlled by the ``SWARM_LOG_LEVEL`` environment variable:

- ``DEBUG`` — full event-level detail (every published and routed event).
- ``INFO``  — lifecycle and high-level routing only (default).

Modules log with structlog via :func:`get_logger`. Inside a host application
(e.g. the FastAPI service) the host's structlog configuration governs the
level; standalone tools such as the demo call :func:`configure_logging` so the
environment variable takes effect.
"""

import logging
import os
import sys
from typing import Any

import structlog

_LEVEL_ENV = "SWARM_LOG_LEVEL"
_DEFAULT_LEVEL = "INFO"
_SUPPORTED_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def resolve_log_level(value: str | None = None) -> str:
    """Resolve the effective log level.

    Reads ``SWARM_LOG_LEVEL`` from the environment (default ``INFO``) and
    normalizes it to an upper-case supported level, raising :class:`ValueError`
    for anything else.
    """
    raw = (value or os.environ.get(_LEVEL_ENV, _DEFAULT_LEVEL)).strip().upper()
    if raw not in _SUPPORTED_LEVELS:
        raise ValueError(
            f"Unsupported SWARM_LOG_LEVEL {raw!r}; expected one of {_SUPPORTED_LEVELS}"
        )
    return raw


def configure_logging(level: str | None = None) -> None:
    """Configure structlog for JSON output at the resolved ``level``.

    ``DEBUG`` enables the full event logs; ``INFO`` keeps only lifecycle and
    high-level entries. Safe to call repeatedly.
    """
    effective = resolve_log_level(level)
    numeric_level = getattr(logging, effective)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(stream=sys.stdout, level=numeric_level, format="%(message)s")


def get_logger(name: str) -> Any:
    """Return a structlog logger bound to ``name``."""
    return structlog.get_logger(name)
