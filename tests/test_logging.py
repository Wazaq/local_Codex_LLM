import logging


def test_chat_logs_error_context(client, caplog):
    with caplog.at_level(logging.ERROR):
        client.post("/chat-with-codex", json={"session_id": "abc"})

    matching = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert matching, "Expected an error log entry"
    assert any(getattr(record, "endpoint", "") == "/chat-with-codex" for record in matching)
