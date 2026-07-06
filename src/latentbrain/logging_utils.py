from __future__ import annotations

import json as json_module
import logging
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json_module.dumps(payload, sort_keys=True)


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure standard library logging for LatentBrain entrypoints."""
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        msg = f"invalid logging level: {level}"
        raise ValueError(msg)

    formatter: logging.Formatter
    if json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if getattr(handler, "_latentbrain_managed", False):
            root_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.__dict__["_latentbrain_managed"] = True

    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)
