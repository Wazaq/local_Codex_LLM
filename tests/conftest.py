import pytest

import config
from main import create_app


class StubMCPClient:
    def __init__(self):
        self.calls = []
        self.mcp_url = "http://stubbed-mcp.local"

    def call_mcp_function(self, tool_name, arguments, timeout=None):
        self.calls.append((tool_name, arguments, timeout))
        return {"result": {"results": []}}

    def list_tools(self):
        return {"result": {"tools": []}}


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SESSION_BACKEND", "memory", raising=False)
    monkeypatch.setattr(config, "SESSION_STORAGE_PATH", str(tmp_path / "sessions.json"), raising=False)
    app = create_app()
    app.config.update(TESTING=True)
    stub_mcp = StubMCPClient()
    app.config["ail_client"] = stub_mcp
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def stub_mcp(app):
    return app.config["ail_client"]
