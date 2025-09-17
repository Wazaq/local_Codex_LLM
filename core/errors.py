"""Codex Bridge error definitions and helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .user_messages import get_user_friendly_error


class ErrorCode(Enum):
    """Enumerates standardized error codes returned by the bridge."""

    # Session errors
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_FULL = "SESSION_FULL"

    # AI communication errors
    AI_UNREACHABLE = "AI_UNREACHABLE"
    AI_TIMEOUT = "AI_TIMEOUT"
    AI_RATE_LIMITED = "AI_RATE_LIMITED"

    # Request errors
    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_PARAMETERS = "MISSING_PARAMETERS"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"

    # System errors
    SYSTEM_ERROR = "SYSTEM_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    STORAGE_ERROR = "STORAGE_ERROR"

    # MCP errors
    MCP_ERROR = "MCP_ERROR"
    MCP_TOOL_NOT_FOUND = "MCP_TOOL_NOT_FOUND"
    MCP_INVALID_PARAMS = "MCP_INVALID_PARAMS"


class CodexError(Exception):
    """Base exception that carries structured error information."""

    def __init__(
        self,
        code: ErrorCode,
        message: Optional[str],
        details: Optional[Dict[str, Any]] = None,
        recovery_suggestion: Optional[str] = None,
        *,
        status_code: int = 400,
        title: Optional[str] = None,
    ) -> None:
        self.code = code
        friendly = get_user_friendly_error(code.value)
        self.title = title or friendly.get("title")
        self.message = message or friendly.get("message")
        self.recovery_suggestion = recovery_suggestion or friendly.get("recovery")
        self.details = details or {}
        self.status_code = status_code
        self.timestamp = datetime.now(timezone.utc).isoformat()
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "error": True,
            "code": self.code.value,
            "message": self.message,
            "timestamp": self.timestamp,
        }
        if self.title:
            payload["title"] = self.title
        if self.details:
            payload["details"] = self.details
        if self.recovery_suggestion:
            payload["recovery"] = self.recovery_suggestion
        return payload


def create_error_response(
    code: ErrorCode,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    recovery_suggestion: Optional[str] = None,
    status_code: int = 400,
    *,
    title: Optional[str] = None,
) -> Tuple[Dict[str, Any], int]:
    """Return a standardized Flask-style error response tuple."""

    error = CodexError(
        code,
        message,
        details,
        recovery_suggestion,
        status_code=status_code,
        title=title,
    )
    return error.to_dict(), status_code


def session_not_found_error(session_id: str) -> Tuple[Dict[str, Any], int]:
    return create_error_response(
        ErrorCode.SESSION_NOT_FOUND,
        f"Session '{session_id}' not found",
        {"session_id": session_id},
        "Create a new session or verify the session ID.",
        status_code=404,
    )


def ai_unreachable_error(ai_name: str, details: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], int]:
    extra = {"ai_name": ai_name}
    if details:
        extra.update(details)
    return create_error_response(
        ErrorCode.AI_UNREACHABLE,
        f"Unable to reach AI '{ai_name}'",
        extra,
        "Check AI service status and try again.",
        status_code=503,
        title="Codex is unreachable",
    )


def invalid_request_error(message: str, details: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], int]:
    return create_error_response(
        ErrorCode.INVALID_REQUEST,
        message,
        details,
        "Check request format and required parameters.",
        status_code=400,
    )
