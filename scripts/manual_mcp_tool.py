#!/usr/bin/env python3
"""Manual helper for exercising MCP endpoints on a running bridge."""

import json
from typing import Any, Dict

import requests

BASE_URL = "http://localhost:8080"


def _post(path: str, payload: Dict[str, Any], *, timeout: float = 10.0) -> requests.Response:
    return requests.post(f"{BASE_URL}{path}", json=payload, timeout=timeout)


def exercise_mcp_tool(tool_name: str, params: Dict[str, Any], *, timeout: float = 10.0) -> None:
    request_payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": params},
        "id": tool_name,
    }
    try:
        response = _post("/mcp", request_payload, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        print(f"Testing {tool_name}: request failed -> {exc}")
        print("-" * 60)
        return
    print(f"Testing {tool_name} (status {response.status_code}):")
    try:
        data = response.json()
    except Exception:
        data = {"error": "invalid json", "body": response.text}
    print(json.dumps(data, indent=2))
    print("-" * 60)


def main() -> None:
    exercise_mcp_tool("get_sessions", {})

    create_resp = _post("/sessions", {"ai_ids": ["claude", "codex"]})
    if not create_resp.ok:
        print("Failed to create test session:", create_resp.text)
        return
    session_payload = create_resp.json().get("session") or {}
    session_id = session_payload.get("session_id")
    if not session_id:
        print("Session creation did not return an id")
        return
    print(f"Created session {session_id}")

    exercise_mcp_tool(
        "send_message_to_ai",
        {
            "speaker_name": "codex",
            "message": "Hello from MCP integration test!",
            "session_id": session_id,
        },
        timeout=65.0,
    )

    exercise_mcp_tool(
        "read_session_context",
        {
            "session_id": session_id,
            "limit": 10,
        },
    )


if __name__ == "__main__":
    main()
