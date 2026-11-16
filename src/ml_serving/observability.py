"""Small JSON logging setup for correlating requests across services."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """Render each log event as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "service": record.name,
            "event": record.getMessage(),
        }
        event.update(getattr(record, "event_fields", {}))
        if record.exc_info:
            event["exception"] = self.formatException(record.exc_info)
        return json.dumps(event, separators=(",", ":"), sort_keys=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure JSON logs on stderr for the current service process."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
