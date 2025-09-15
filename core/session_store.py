import time
import uuid
import json
import threading
from typing import Any, Dict

import config
from .utils import now_iso
from .metrics import PrometheusMetrics


class Session:
    def __init__(self, ai_ids=None, max_tokens=config.SESSION_TOKEN_LIMIT, ttl_seconds=config.SESSION_TTL_SECONDS):
        self.id = str(uuid.uuid4())
        self.created_at = now_iso()
        self.ai_ids = list(ai_ids or [])
        self.max_tokens = int(max_tokens or config.SESSION_TOKEN_LIMIT)
        self.ttl_seconds = int(ttl_seconds or config.SESSION_TTL_SECONDS)
        self.messages = []
        self.summary = ""
        self.token_usage = 0
        self.last_updated = time.time()

    def to_dict(self, include_messages=False):
        d = {
            "session_id": self.id,
            "created_at": self.created_at,
            "ai_ids": self.ai_ids,
            "max_tokens": self.max_tokens,
            "ttl_seconds": self.ttl_seconds,
            "token_usage": self.token_usage,
            "summary_len": len(self.summary or ""),
            "messages": (self.messages if include_messages else None)
        }
        if not include_messages:
            d.pop("messages")
        return d

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "ai_ids": list(self.ai_ids or []),
            "max_tokens": int(self.max_tokens),
            "ttl_seconds": int(self.ttl_seconds),
            "messages": list(self.messages or []),
            "summary": self.summary or "",
            "token_usage": int(self.token_usage),
            "last_updated": float(self.last_updated),
        }

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> 'Session':
        s = Session(ai_ids=obj.get('ai_ids', []), max_tokens=obj.get('max_tokens', config.SESSION_TOKEN_LIMIT), ttl_seconds=obj.get('ttl_seconds', config.SESSION_TTL_SECONDS))
        s.id = obj.get('id', s.id)
        s.created_at = obj.get('created_at', s.created_at)
        s.messages = obj.get('messages', [])
        s.summary = obj.get('summary', '')
        s.token_usage = obj.get('token_usage', 0)
        s.last_updated = obj.get('last_updated', time.time())
        return s


class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = {}

    def create(self, ai_ids=None, max_tokens=None, ttl_seconds=None):
        s = Session(ai_ids=ai_ids, max_tokens=max_tokens or config.SESSION_TOKEN_LIMIT, ttl_seconds=ttl_seconds if ttl_seconds is not None else config.SESSION_TTL_SECONDS)
        with self._lock:
            self._sessions[s.id] = s
        return s

    def get(self, session_id):
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id):
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_ids(self):
        with self._lock:
            return list(self._sessions.keys())

    def snapshot_stats(self):
        with self._lock:
            n = len(self._sessions)
            tok = sum((s.token_usage for s in self._sessions.values()), 0)
        return n, tok

    def _cleanup_expired(self):
        if config.SESSION_TTL_SECONDS <= 0:
            return
        now = time.time()
        cutoff = lambda s: (s.ttl_seconds > 0) and ((s.last_updated + s.ttl_seconds) < now)
        with self._lock:
            dead = [sid for sid, s in self._sessions.items() if cutoff(s)]
            for sid in dead:
                self._sessions.pop(sid, None)

    def save(self, session: 'Session') -> None:
        with self._lock:
            self._sessions[session.id] = session


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def compact_session_if_needed(sess: Session, metrics: PrometheusMetrics):
    if sess.token_usage <= sess.max_tokens:
        return
    removed = 0
    while sess.messages and sess.token_usage > sess.max_tokens:
        msg = sess.messages.pop(0)
        piece = f"[{msg.get('from')}] {msg.get('content')}\n"
        sess.summary += piece[:800]
        t = approx_tokens(msg.get('content') or '')
        sess.token_usage = max(0, sess.token_usage - t)
        removed += 1
    if removed:
        metrics.inc_counter('codex_sessions_compactions_total', {})


def make_session_store():
    if config.SESSION_BACKEND == 'redis':
        try:
            import redis  # type: ignore

            class _RedisWrap:
                def __init__(self, url: str):
                    self._redis = redis.from_url(url)
                    self._prefix = 'codex:sessions:'
                def _key(self, sid: str) -> str:
                    return f"{self._prefix}{sid}"
                def create(self, ai_ids=None, max_tokens=None, ttl_seconds=None):
                    s = Session(ai_ids=ai_ids, max_tokens=max_tokens or config.SESSION_TOKEN_LIMIT, ttl_seconds=ttl_seconds if ttl_seconds is not None else config.SESSION_TTL_SECONDS)
                    self.save(s)
                    return s
                def get(self, session_id: str):
                    raw = self._redis.get(self._key(session_id))
                    if not raw:
                        return None
                    try:
                        obj = json.loads(raw)
                        return Session.from_json(obj)
                    except Exception:
                        return None
                def save(self, session: 'Session'):
                    obj = session.to_json()
                    data = json.dumps(obj)
                    self._redis.set(self._key(session.id), data)
                    if session.ttl_seconds and session.ttl_seconds > 0:
                        try:
                            self._redis.expire(self._key(session.id), int(session.ttl_seconds))
                        except Exception:
                            pass
                def delete(self, session_id: str) -> bool:
                    return bool(self._redis.delete(self._key(session_id)))
                def list_ids(self):
                    keys = list(self._redis.scan_iter(match=f"{self._prefix}*"))
                    out = []
                    for k in keys:
                        try:
                            ks = k.decode('utf-8') if isinstance(k, (bytes, bytearray)) else str(k)
                            out.append(ks.split(':')[-1])
                        except Exception:
                            pass
                    return out
                def snapshot_stats(self):
                    n = 0
                    tok = 0
                    for sid in self.list_ids():
                        raw = self._redis.get(self._key(sid))
                        if not raw:
                            continue
                        try:
                            obj = json.loads(raw)
                            n += 1
                            tok += int(obj.get('token_usage', 0))
                        except Exception:
                            pass
                    return n, tok
            return _RedisWrap(config.REDIS_URL)
        except Exception:
            # Fallback to in-memory
            pass
    return SessionStore()

