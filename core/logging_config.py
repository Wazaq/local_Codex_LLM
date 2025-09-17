"""Structured logging helpers for the Codex bridge."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Render log records as JSON with consistent fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for field in ("session_id", "ai_id", "endpoint"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(app) -> logging.Logger:
    """Configure Flask app logging to emit structured JSON to file and concise console logs."""

    logger = app.logger
    logger.setLevel(logging.DEBUG)

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    logs_dir = os.path.abspath(logs_dir)
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        # If directory creation fails fall back to current directory
        logs_dir = os.getcwd()

    structured_path = os.path.join(logs_dir, "structured.log")

    structured_handler = logging.FileHandler(structured_path, encoding="utf-8")
    structured_handler.setFormatter(StructuredFormatter())
    structured_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    console_handler.setLevel(logging.INFO)

    existing_handlers = {type(h) for h in logger.handlers}
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == structured_handler.baseFilename for h in logger.handlers):
        logger.addHandler(structured_handler)
    if logging.StreamHandler not in existing_handlers:
        logger.addHandler(console_handler)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    return logger


def log_error(logger: logging.Logger, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log an error with context using structured extra fields."""

    logger.error("Error: %s", error, extra=context or {}, exc_info=True)


def log_request(
    logger: logging.Logger,
    endpoint: str,
    *,
    session_id: Optional[str] = None,
    ai_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log request metadata for tracing."""

    extra: Dict[str, Any] = {"endpoint": endpoint}
    if session_id:
        extra["session_id"] = session_id
    if ai_id:
        extra["ai_id"] = ai_id
    if details:
        extra.update(details)
    logger.info("Request to %s", endpoint, extra=extra)
