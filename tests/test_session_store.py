from core.session_store import Session, approx_tokens, compact_session_if_needed


class DummyMetrics:
    def __init__(self):
        self.count = 0

    def inc_counter(self, *_args, **_kwargs):
        self.count += 1


def test_approx_tokens_estimation():
    assert approx_tokens("hello") > 0
    assert approx_tokens("") == 0
    assert approx_tokens("abcd") == 1


def test_compact_session_if_needed_reduces_tokens():
    session = Session(ai_ids=["codex"], max_tokens=2, display_name="Test")
    session.messages = [
        {"from": "user", "content": "long message one"},
        {"from": "ai", "content": "long message two"},
    ]
    session.token_usage = 20

    metrics = DummyMetrics()
    compact_session_if_needed(session, metrics)

    assert session.token_usage < 20
    assert len(session.messages) < 2
    assert session.display_name == "Test"
    assert metrics.count >= 1
    assert session.summary
