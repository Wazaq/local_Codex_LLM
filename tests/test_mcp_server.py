import asyncio

import pytest
import requests

import routes.chat as chat_module
from mcp.server import MCPError


def test_handle_request_lists_tools(app):
    server = app.config["mcp_server"]
    result = server.handle_mcp_request({"jsonrpc": "2.0", "method": "tools/list", "id": "1"})
    assert "result" in result
    assert any(tool["name"] == "send_message_to_ai" for tool in result["result"]["tools"])


def test_handle_request_unknown_method(app):
    server = app.config["mcp_server"]
    response = server.handle_mcp_request({"jsonrpc": "2.0", "method": "does/not/exists", "id": "1"})
    assert response["error"]["code"] == -32601


def test_send_message_to_ai_success(monkeypatch, app):
    sessions = app.config["sessions"]
    session = sessions.create(ai_ids=["codex"])

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "Hello from MCP"}}

    monkeypatch.setattr(chat_module.requests, "post", lambda *args, **kwargs: FakeResponse())

    server = app.config["mcp_server"]
    result = asyncio.run(server.send_message_to_ai("codex", "hi", session.id))

    assert result["success"] is True
    assert result["response"]["response"] == "Hello from MCP"


def test_send_message_to_ai_session_missing(app):
    server = app.config["mcp_server"]
    with pytest.raises(MCPError) as exc:
        asyncio.run(server.send_message_to_ai("codex", "hi", "missing"))
    assert exc.value.code == -32004


def test_send_message_to_ai_handles_downstream_failure(monkeypatch, app):
    sessions = app.config["sessions"]
    session = sessions.create(ai_ids=["codex"])

    class FailureResponse:
        status_code = 503

        def raise_for_status(self):
            raise requests.exceptions.HTTPError("service unavailable")

        def json(self):
            return {"error": "downstream"}

    monkeypatch.setattr(chat_module.requests, "post", lambda *args, **kwargs: FailureResponse())

    server = app.config["mcp_server"]
    with pytest.raises(MCPError) as exc:
        asyncio.run(server.send_message_to_ai("codex", "hi", session.id))
    assert exc.value.code == -32010
