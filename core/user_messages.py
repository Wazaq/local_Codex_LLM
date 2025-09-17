"""User-friendly error message catalog for surface-level messaging."""

from __future__ import annotations

from typing import Dict


ERROR_MESSAGES: Dict[str, Dict[str, str]] = {
    "SESSION_NOT_FOUND": {
        "title": "Session Not Found",
        "message": "We couldn't find the selected session.",
        "recovery": "Refresh the session list or create a new session."
    },
    "SESSION_EXPIRED": {
        "title": "Session Expired",
        "message": "This conversation session has expired.",
        "recovery": "Create a fresh session to continue."
    },
    "AI_UNREACHABLE": {
        "title": "Codex is Unreachable",
        "message": "Codex didn't respond in time.",
        "recovery": "Please wait a moment and try sending again."
    },
    "AI_TIMEOUT": {
        "title": "Request Timed Out",
        "message": "Codex took too long to respond.",
        "recovery": "Try resending your request."
    },
    "AI_RATE_LIMITED": {
        "title": "Rate Limit Reached",
        "message": "Too many requests were sent in a short period.",
        "recovery": "Pause briefly before trying again."
    },
    "INVALID_REQUEST": {
        "title": "Invalid Request",
        "message": "Something in the request was unexpected.",
        "recovery": "Double-check the information and try again."
    },
    "MISSING_PARAMETERS": {
        "title": "Missing Information",
        "message": "Required information was missing from the request.",
        "recovery": "Fill in the missing details and try once more."
    },
    "MCP_ERROR": {
        "title": "Tool Bridge Error",
        "message": "The MCP bridge encountered an issue.",
        "recovery": "Retry shortly. If this keeps happening, contact support."
    },
    "SYSTEM_ERROR": {
        "title": "System Error",
        "message": "Something went wrong on our side.",
        "recovery": "Please try again. If the issue persists, reach out to support."
    }
}


def get_user_friendly_error(code: str) -> Dict[str, str]:
    """Return a user-friendly message bundle for an error code."""

    return ERROR_MESSAGES.get(code, {
        "title": "Unexpected Error",
        "message": "An unexpected error occurred.",
        "recovery": "Try again in a moment or contact support if it continues."
    })
