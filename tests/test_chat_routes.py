import json

import routes.chat as chat_module


def test_chat_missing_message_returns_structured_error(client):
    response = client.post("/chat-with-codex", json={"session_id": "abc"})

    assert response.status_code == 400
    data = response.get_json()
    assert data["code"] == "MISSING_PARAMETERS"
    assert data["error"] is True


def test_chat_invalid_session_returns_404(client):
    response = client.post(
        "/chat-with-codex",
        json={"session_id": "missing", "message": "hello"},
        headers={"X-AI-Id": "codex"},
    )

    assert response.status_code == 404
    data = response.get_json()
    assert data["code"] == "SESSION_NOT_FOUND"
    assert data["details"]["session_id"] == "missing"


def test_chat_success_returns_response(monkeypatch, app, client):
    sessions = app.config["sessions"]
    session = sessions.create(ai_ids=["codex"])  # ensure valid session exists

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": "Hello Test"}}

    monkeypatch.setattr(chat_module.requests, "post", lambda *args, **kwargs: FakeResponse())

    response = client.post(
        "/chat-with-codex",
        json={"session_id": session.id, "message": "ping", "ai_id": "codex"},
        headers={"X-AI-Id": "codex"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["response"] == "Hello Test"
    assert data["session_id"] == session.id


def test_chat_streaming_emits_tokens(monkeypatch, app, client):
    sessions = app.config["sessions"]
    session = sessions.create(ai_ids=["codex"])

    class FakeStreamResponse:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield json.dumps({"message": {"content": "Hello"}})
            yield json.dumps({"done": True})

    monkeypatch.setattr(chat_module.requests, "post", lambda *args, **kwargs: FakeStreamResponse())

    with app.app_context():
        response = client.post(
            "/chat-with-codex/stream",
            json={"session_id": session.id, "message": "ping", "ai_id": "codex"},
            headers={"X-AI-Id": "codex"},
            buffered=True,
        )
        body = response.get_data(as_text=True)
    assert "token" in body
    assert "response_end" in body
