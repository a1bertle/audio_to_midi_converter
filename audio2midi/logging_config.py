"""Shared structured error logging for application and evaluation entry points."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _IsoTimezoneFormatter(logging.Formatter):
    """Render logging timestamps as timezone-aware ISO 8601 values."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        del datefmt
        return datetime.fromtimestamp(record.created).astimezone().isoformat()


class _ContextFilter(logging.Filter):
    def __init__(self, context: str) -> None:
        super().__init__()
        self._context = context

    def filter(self, record: logging.LogRecord) -> bool:
        record.context = self._context
        return True


def configure_logging(level: str | int = "INFO", context: str = "") -> None:
    """Configure stderr plus rotating ``.logs/error.log`` handlers once."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    context_filter = _ContextFilter(context or "none")
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.addFilter(context_filter)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s | %(message)s"))

    logs_dir = Path(__file__).resolve().parents[1] / ".logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        logs_dir / "error.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
    )
    file_handler.setLevel(logging.ERROR)
    file_handler.addFilter(context_filter)
    file_handler.setFormatter(_IsoTimezoneFormatter(
        "%(asctime)s %(levelname)s %(name)s PID=%(process)d "
        "THREAD=%(threadName)s CONTEXT=%(context)s | %(message)s"
    ))

    root.addHandler(stderr_handler)
    root.addHandler(file_handler)

