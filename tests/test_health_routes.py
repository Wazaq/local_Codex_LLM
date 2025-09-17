import requests

import routes.health as health_module


class DummyResponse:
    ok = True
    status_code = 200

    def json(self):
        return {"models": []}


def test_health_endpoint_basics(monkeypatch, client):
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: DummyResponse())
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "MCP Bridge running"
    summary = data.get("status_summary", {})
    assert summary.get("level") in {"healthy", "degraded", "unavailable"}
    assert "components" in summary


def test_health_handles_internal_errors(monkeypatch, client, app):
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: DummyResponse())

    def boom():  # pragma: no cover - helper to trigger error path
        raise RuntimeError("boom")

    monkeypatch.setattr(app.config["sessions"], "snapshot_stats", boom, raising=False)

    response = client.get("/health")
    assert response.status_code == 500
    data = response.get_json()
    assert data["code"] == "SYSTEM_ERROR"


def test_health_mcp_status(monkeypatch, client):
    monkeypatch.setattr(
        health_module,
        "_check_dependencies",
        lambda app: (True, 3, 42, False, None),
    )
    response = client.get("/health/mcp")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "degraded"
    assert data["mcp"]["tools_available"] == 3


def test_readyz_degraded(monkeypatch, client):
    monkeypatch.setattr(
        health_module,
        "_check_dependencies",
        lambda app: (False, 0, None, False, None),
    )
    response = client.get("/readyz")
    assert response.status_code == 503
    data = response.get_json()
    assert data["ready"] is False


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "codex_streaming_active" in response.get_data(as_text=True)


def test_config_endpoint(client):
    response = client.get("/config")
    assert response.status_code == 200
    data = response.get_json()
    assert "model" in data["config"]
