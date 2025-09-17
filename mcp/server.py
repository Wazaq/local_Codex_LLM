from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

from flask import Flask

from .config import MCP_CONFIG

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """Structured error for MCP tool execution."""

    def __init__(self, message: str, *, code: int = -32000, data: Optional[Any] = None):
        super().__init__(message)
        self.code = code
        self.data = data


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]


class MCPServer:
    """Very small JSON-RPC dispatcher for MCP tool calls."""

    def __init__(self, server_name: str):
        self.server_name = server_name
        self._tools: Dict[str, Tool] = {}

    def add_tool(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> Dict[str, Any]:
        tools = []
        for tool in self._tools.values():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
                "inputSchema": tool.parameters,
            })
        return {"tools": tools, "server": self.server_name}

    def handle_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise MCPError("Invalid request payload", code=-32600)

        method = payload.get("method")
        request_id = payload.get("id")

        if method == "initialize":
            server_info = {
                "name": self.server_name,
                "version": MCP_CONFIG.get("version", "1.0.0"),
            }
            capabilities = {
                "tools": {
                    "refresh": False,
                }
            }
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "server": server_info,
                    "capabilities": capabilities,
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"ok": True}}

        if method == "tools/list":
            result = self.list_tools()
            return {"jsonrpc": "2.0", "id": request_id, "result": result}

        if method == "tools/call":
            params = payload.get("params") or {}
            name = params.get("name")
            if not name or name not in self._tools:
                raise MCPError("Tool not found", code=-32601)
            arguments = params.get("arguments") or {}
            tool = self._tools[name]
            result = self._invoke_tool(tool, arguments)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}

        raise MCPError(f"Unknown method: {method}", code=-32601)

    def _invoke_tool(self, tool: Tool, arguments: Dict[str, Any]) -> Any:
        try:
            result = tool.handler(**arguments)
            if inspect.isawaitable(result):
                return asyncio.run(_await_result(result))
            return result
        except MCPError:
            raise
        except TypeError as exc:
            raise MCPError(f"Invalid parameters for {tool.name}: {exc}", code=-32602) from exc
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Tool %s failed", tool.name)
            raise MCPError(f"Tool {tool.name} failed: {exc}") from exc


async def _await_result(result: Awaitable[Any]) -> Any:
    return await result


class CodexMCPServer:
    """Registers Codex-specific MCP tools against the Flask application."""

    def __init__(self, flask_app: Flask):
        self.app = flask_app
        self.server = MCPServer("codex-bridge")
        self.register_tools()

    def register_tools(self) -> None:
        """Register the core tool set exposed over MCP."""

        self.server.add_tool(
            Tool(
                name="send_message_to_ai",
                description="Send a message to any AI connected to the bridge",
                parameters={
                    "type": "object",
                    "properties": {
                        "speaker_name": {"type": "string", "description": "The Speakers Name (codex, claude, brent)"},
                        "message": {"type": "string", "description": "Message content"},
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "stream": {"type": "boolean", "description": "Use streaming endpoint", "default": False},
                    },
                    "required": ["speaker_name", "message", "session_id"],
                },
                handler=self.send_message_to_ai,
            )
        )

        self.server.add_tool(
            Tool(
                name="get_sessions",
                description="List all sessions for discovery and management",
                parameters={
                    "type": "object",
                    "properties": {
                        "active_only": {"type": "boolean", "description": "Only include active sessions", "default": True}
                    },
                },
                handler=self.get_sessions,
            )
        )

        self.server.add_tool(
            Tool(
                name="read_session_context",
                description="Return the conversation history for a session",
                parameters={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session identifier"},
                        "limit": {"type": "integer", "description": "Maximum messages to include", "default": 50},
                    },
                    "required": ["session_id"],
                },
                handler=self.read_session_context,
            )
        )

    def handle_mcp_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.server.handle_request(request_data)
        except MCPError as exc:
            logger.warning("MCP request error: %s", exc)
            return {
                "jsonrpc": "2.0",
                "error": {"code": exc.code, "message": str(exc), "data": exc.data},
                "id": request_data.get("id"),
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("Unhandled MCP error")
            return {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": "Internal error", "data": str(exc)},
                "id": request_data.get("id"),
            }

    async def send_message_to_ai(self, speaker_name: str, message: str, session_id: str, stream: bool = False) -> Dict[str, Any]:
        try:
            with self.app.test_client() as client:
                endpoint = "/chat-with-codex/stream" if stream else "/chat-with-codex"
                response = client.post(
                    endpoint,
                    json={"message": message, "session_id": session_id, "speaker_name": speaker_name},
                    headers={"X-AI-Id": speaker_name},
                )
                if stream:
                    payload: Any = response.get_data(as_text=True)
                else:
                    payload = response.get_json(silent=True)
                if response.status_code != 200:
                    raise MCPError(
                        f"AI communication failed: {response.status_code}",
                        code=-32010,
                        data={"status_code": response.status_code, "body": payload},
                    )
                return {
                    "success": True,
                    "speaker_name": speaker_name,
                    "status_code": response.status_code,
                    "response": payload,
                }
        except MCPError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("send_message_to_ai failure")
            raise MCPError(f"Failed to send message: {exc}") from exc

    async def get_sessions(self, active_only: bool = True) -> Dict[str, Any]:
        try:
            sessions_store = self.app.config.get("sessions")
            if sessions_store is None:
                raise MCPError("Session store unavailable", code=-32011)
            session_ids = sessions_store.list_ids()
            sessions: list[Dict[str, Any]] = []
            for sid in session_ids:
                session = sessions_store.get(sid)
                if not session:
                    continue
                if active_only and session.ttl_seconds > 0:
                    last_activity = session.last_updated or 0
                    if last_activity and (last_activity + session.ttl_seconds) < time.time():
                        continue
                context = {
                    "session_id": session.id,
                    "ai_ids": list(session.ai_ids or []),
                    "created_at": session.created_at,
                    "token_usage": session.token_usage,
                    "messages": len(session.messages or []),
                    "message_count": len(session.messages or []),
                    "last_activity": _ts_to_iso(session.last_updated),
                }
                sessions.append(context)
            return {"success": True, "sessions": sessions, "count": len(sessions)}
        except MCPError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("get_sessions failure")
            raise MCPError(f"Failed to get sessions: {exc}") from exc

    async def read_session_context(self, session_id: str, limit: int = 50) -> Dict[str, Any]:
        try:
            sessions_store = self.app.config.get("sessions")
            if sessions_store is None:
                raise MCPError("Session store unavailable", code=-32011)
            session = sessions_store.get(session_id)
            if not session:
                raise MCPError(f"Session {session_id} not found", code=-32004)
            messages_raw = list(session.messages or [])
            if limit and limit > 0:
                messages_raw = messages_raw[-limit:]
            messages = []
            for msg in messages_raw:
                role = "assistant" if (msg.get("from") == "ai") else "user"
                messages.append(
                    {
                        "role": role,
                        "content": msg.get("content"),
                        "timestamp": msg.get("ts"),
                        "ai_id": msg.get("ai_id"),
                    }
                )
            return {
                "success": True,
                "session_id": session.id,
                "summary": session.summary,
                "token_usage": session.token_usage,
                "messages": messages,
            }
        except MCPError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.exception("read_session_context failure")
            raise MCPError(f"Failed to read session context: {exc}") from exc


def _ts_to_iso(ts: Optional[float]) -> Optional[str]:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None
