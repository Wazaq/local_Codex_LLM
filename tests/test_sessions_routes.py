from core.session_store import Session


def test_create_session_success(client):
    response = client.post("/sessions", json={"ai_ids": ["codex"], "display_name": "Daily Standup"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["ok"] is True
    assert data["session"]["session_id"]
    assert data["session"]["display_name"] == "Daily Standup"


def test_create_session_invalid_payload(client):
    response = client.post("/sessions", json={"ai_ids": "codex"})
    assert response.status_code == 400
    data = response.get_json()
    assert data["code"] == "INVALID_REQUEST"


def test_get_session_not_found(client):
    response = client.get("/sessions/does-not-exist")
    assert response.status_code == 404
    data = response.get_json()
    assert data["code"] == "SESSION_NOT_FOUND"


def test_session_lifecycle(client, app):
    create = client.post("/sessions", json={"ai_ids": ["codex"], "display_name": "Lifecycle Test"})
    session_payload = create.get_json()["session"]
    session_id = session_payload["session_id"]
    assert session_payload["display_name"] == "Lifecycle Test"

    fetched = client.get(f"/sessions/{session_id}")
    assert fetched.status_code == 200
    assert fetched.get_json()["session"]["display_name"] == "Lifecycle Test"

    listed = client.get("/sessions")
    assert listed.status_code == 200
    payload = listed.get_json()
    found = next((s for s in payload["sessions"] if s["session_id"] == session_id), None)
    assert found is not None
    assert found["display_name"] == "Lifecycle Test"

    deleted = client.delete(f"/sessions/{session_id}")
    assert deleted.status_code == 200

    missing = client.get(f"/sessions/{session_id}")
    assert missing.status_code == 404


def test_add_session_message_validates_ai_id(client, app):
    create = client.post("/sessions", json={"ai_ids": ["codex"]})
    session_id = create.get_json()["session"]["session_id"]

    response = client.post(
        f"/sessions/{session_id}/messages",
        json={"from": "user", "content": "hi", "ai_id": "unauthorized"},
    )

    assert response.status_code == 403
    data = response.get_json()
    assert data["code"] == "INVALID_PARAMETERS"


def test_get_session_messages_invalid_limit(client, app):
    create = client.post("/sessions", json={"ai_ids": ["codex"]})
    session_id = create.get_json()["session"]["session_id"]

    response = client.get(f"/sessions/{session_id}/messages?limit=not-a-number")
    assert response.status_code == 400
    data = response.get_json()
    assert data["code"] == "INVALID_REQUEST"


def test_update_session_display_name(client, app):
    create = client.post("/sessions", json={"ai_ids": ["codex"]})
    session_id = create.get_json()["session"]["session_id"]

    patch = client.patch(f"/sessions/{session_id}", json={"display_name": "Renamed Session"})
    assert patch.status_code == 200
    updated = patch.get_json()["session"]
    assert updated["display_name"] == "Renamed Session"

    listing = client.get("/sessions")
    entry = next((s for s in listing.get_json()["sessions"] if s["session_id"] == session_id), None)
    assert entry is not None and entry["display_name"] == "Renamed Session"
