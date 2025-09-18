import time
from functools import wraps
from typing import Any, Optional

from flask import current_app, g, request

from core.errors import ErrorCode


def _get_dashboard_metrics():
    return current_app.config.get('dashboard_metrics')


def _extract_status_code(response: Any) -> int:
    if hasattr(response, 'status_code'):
        try:
            return int(response.status_code)
        except (TypeError, ValueError):
            return 200
    if isinstance(response, tuple) and len(response) >= 2:
        try:
            return int(response[1])
        except (TypeError, ValueError):
            return 200
    return 200


def track_request_metrics(func):
    """Decorator that records request metrics for admin dashboard visibility."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        metrics = _get_dashboard_metrics()
        start_time = time.time()
        try:
            response = func(*args, **kwargs)
            status_code = _extract_status_code(response)
            if metrics:
                metrics.record_request(
                    endpoint=request.endpoint or request.path or 'unknown',
                    method=request.method,
                    status_code=status_code,
                    duration=time.time() - start_time,
                )
            return response
        except Exception as exc:  # pragma: no cover - re-raised after recording
            duration = time.time() - start_time
            if metrics:
                endpoint = request.endpoint or request.path or 'unknown'
                metrics.record_request(
                    endpoint=endpoint,
                    method=request.method,
                    status_code=500,
                    duration=duration,
                )
                error_code = _extract_error_code(exc)
                metrics.record_error(
                    error_code=str(error_code),
                    endpoint=endpoint,
                    message=str(exc),
                )
            raise

    return wrapper


def _extract_error_code(exc: Exception) -> str:
    code: Optional[Any] = getattr(exc, 'code', None)
    if isinstance(code, ErrorCode):
        return code.value
    if isinstance(code, str):
        return code
    if isinstance(code, int):
        return str(code)
    if isinstance(exc, ErrorCode):
        return exc.value
    if hasattr(exc, 'status_code') and isinstance(exc.status_code, int):
        return str(exc.status_code)
    return ErrorCode.SYSTEM_ERROR.value


def setup_metrics_middleware(app):
    """Attach hooks that prepare per-request state for the dashboard metrics."""

    @app.before_request
    def _before_request():
        g.metrics_request_started = time.time()

    @app.after_request
    def _after_request(response):
        return response

    return app
