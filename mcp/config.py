import os

MCP_CONFIG = {
    "server_name": "codex-bridge",
    "version": "1.0.0",
    "tools_enabled": ["send_message_to_ai", "get_sessions", "read_session_context"],
    "auth_required": os.environ.get("MCP_AUTH_REQUIRED", "false").lower() == "true",
    "rate_limit": {
        "requests_per_minute": 60,
        "burst_limit": 10,
    },
}
