"""
api/logging_config.py

Structured logging setup. Replaces ad-hoc print() statements with
proper leveled, timestamped, contextual logs.

In development: human-readable colored console output.
In production:  single-line JSON per log entry, ready to ship to
                 any log aggregator (CloudWatch, Datadog, Loki, etc.)

Usage:
    from api.logging_config import get_logger
    logger = get_logger(__name__)

    logger.info("User signed up", extra={"email": email})
    logger.warning("Rate limit hit", extra={"user": email, "endpoint": "/multi-upload"})
    logger.error("CSV ingestion failed", exc_info=True, extra={"user": email})
"""

import logging
import sys
import json
from datetime import datetime, timezone

from api.config import settings


class JSONFormatter(logging.Formatter):
    """Renders each log record as a single line of JSON."""

    RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Include any extra= fields passed by the caller
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)  # ensure it's serializable
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        return json.dumps(payload, default=str)


class PrettyFormatter(logging.Formatter):
    """Human-readable colored console output for local development."""

    COLORS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[41m",   # red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts    = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        base  = f"{color}[{ts}] {record.levelname:<8}{self.RESET} {record.name}: {record.getMessage()}"

        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in JSONFormatter.RESERVED and not k.startswith("_")
        }
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())

        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)

        return base


_configured = False


def configure_logging():
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)

    # Remove default handlers to avoid duplicate log lines
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if settings.LOG_JSON else PrettyFormatter())
    root.addHandler(handler)

    # Quiet down noisy third-party loggers in production
    if settings.is_production:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    root.info(
        "Logging configured",
        extra={"app_env": settings.APP_ENV, "log_level": settings.LOG_LEVEL, "json": settings.LOG_JSON},
    )


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)