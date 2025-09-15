import json
import random
import time
from typing import Any, Dict, List

import requests

import config


_cb_failures = 0
_cb_open_until = 0.0
_cb_lock = None


def _get_lock():
    global _cb_lock
    if _cb_lock is None:
        import threading
        _cb_lock = threading.Lock()
    return _cb_lock


def cb_record_success():
    global _cb_failures, _cb_open_until
    with _get_lock():
        _cb_failures = 0
        _cb_open_until = 0.0


def cb_can_call():
    now = time.time()
    with _get_lock():
        return not (now < _cb_open_until)


def cb_record_failure():
    global _cb_failures, _cb_open_until
    with _get_lock():
        _cb_failures += 1
        if _cb_failures >= config.CIRCUIT_BREAKER_THRESHOLD:
            _cb_open_until = time.time() + config.CIRCUIT_BREAKER_COOLDOWN


def now_iso():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def extract_results_from_mcp(mcp_response: Any) -> List[Any]:
    try:
        if not isinstance(mcp_response, dict):
            return []
        if 'error' in mcp_response and mcp_response['error']:
            return []
        obj = mcp_response.get('result', mcp_response)
        if isinstance(obj, dict) and isinstance(obj.get('results'), list):
            return obj['results']
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and any(k in obj for k in ('content', 'description', 'text', 'value')):
            return [obj]
        return []
    except Exception:
        return []


def to_text_list(items: List[Any], metrics=None, security_logger=None, content_filter=None) -> List[str]:
    out: List[str] = []
    from .security import sanitize_text, contains_blocked
    for it in items:
        if isinstance(it, dict):
            v = it.get('content') or it.get('description') or it.get('text') or it.get('value') or ''
            if isinstance(v, (dict, list)):
                try:
                    v = json.dumps(v)[:4000]
                except Exception:
                    v = str(v)
            if v:
                sv = sanitize_text(v)
                if content_filter and contains_blocked(sv):
                    if security_logger is not None:
                        try:
                            security_logger.info(json.dumps({
                                "timestamp": now_iso(),
                                "event": "memory_sanitized_drop",
                                "reason": "blocked_term",
                                "sample": sv[:120]
                            }))
                        except Exception:
                            pass
                    if metrics is not None:
                        try:
                            metrics.inc_counter('codex_sanitized_items_total', {"type": "memory"})
                        except Exception:
                            pass
                else:
                    out.append(sv)
        else:
            sv = sanitize_text(it)
            if content_filter and contains_blocked(sv):
                if security_logger is not None:
                    try:
                        security_logger.info(json.dumps({
                            "timestamp": now_iso(),
                            "event": "memory_sanitized_drop",
                            "reason": "blocked_term",
                            "sample": sv[:120]
                        }))
                    except Exception:
                        pass
                if metrics is not None:
                    try:
                        metrics.inc_counter('codex_sanitized_items_total', {"type": "memory"})
                    except Exception:
                        pass
            else:
                out.append(sv)
    return [s for s in out if s]


def http_ail_search(query, domain=None, limit=5, timeout=10):
    url = "https://neural-nexus-palace.wazaqglim.workers.dev/mobile/brain-ai-library-search"
    params = {"q": query, "limit": limit}
    if domain:
        params["domain"] = domain
    # Allow list enforcement
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        if host not in config.ALLOW_LIST_DOMAINS and config.ENABLE_CONTENT_FILTER:
            return {"results": []}
    except Exception:
        pass

    attempts = max(1, config.MCP_RETRY_ATTEMPTS)
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.ok:
                return resp.json()
        except Exception:
            pass
        if attempt >= attempts - 1:
            break
        backoff = config.MCP_BACKOFF_BASE * (2 ** attempt)
        jitter = backoff * random.uniform(0, 0.2)
        time.sleep(backoff + jitter)
    return {"results": []}

