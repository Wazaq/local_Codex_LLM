import requests
import json
import os
import time
import random
import threading
import signal
import uuid
import re
from typing import Optional, Dict, Any
from collections import deque
import hashlib
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone
from flask import Flask, request, jsonify, Response  # type: ignore[reportMissingImports]

class HTTPMCPClient:
    def __init__(self, mcp_url):
        self.mcp_url = mcp_url

    def call_mcp_function(self, tool_name, arguments, timeout=None):
      """Call MCP tool via HTTP POST with retries and circuit breaker"""
      payload = {
          "jsonrpc": "2.0",
          "id": "1",
          "method": "tools/call",
          "params": {"name": tool_name, "arguments": arguments}
      }

      if not _cb_can_call():
          return {"error": "MCP circuit open"}

      attempts = max(1, MCP_RETRY_ATTEMPTS)
      for attempt in range(attempts):
          try:
              response = requests.post(self.mcp_url, json=payload, timeout=(timeout or MCP_TIMEOUT_SECONDS))
              response.raise_for_status()
              _cb_record_success()
              return response.json()
          except requests.exceptions.RequestException as e:
              _cb_record_failure()
              if attempt >= attempts - 1:
                  return {"error": f"MCP call failed: {str(e)}"}
              backoff = MCP_BACKOFF_BASE * (2 ** attempt)
              jitter = backoff * random.uniform(0, 0.2)
              time.sleep(backoff + jitter)

    def list_tools(self):
        """List MCP tools via HTTP POST"""
        payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "tools/list",
            "params": {}
        }
        try:
            response = requests.post(self.mcp_url, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"MCP list failed: {str(e)}"}


def _check_dependencies():
    """Check MCP and Ollama availability and return status tuple (mcp_ok, mcp_tools, mcp_rt_ms, ollama_ok, ollama_rt_ms)."""
    # MCP
    mcp_ok = False
    mcp_tools = 0
    mcp_rt = None
    try:
        t0 = time.time()
        resp = ail_client.list_tools()
        mcp_rt = int((time.time() - t0) * 1000)
        if isinstance(resp, dict) and 'error' not in resp:
            result = resp.get('result', resp)
            tools = result.get('tools') if isinstance(result, dict) else None
            if isinstance(tools, list):
                mcp_tools = len(tools)
                mcp_ok = True
    except Exception:
        pass

    # Ollama
    ollama_ok = False
    ollama_rt = None
    try:
        t0 = time.time()
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        ollama_rt = int((time.time() - t0) * 1000)
        if r.ok:
            ollama_ok = True
    except Exception:
        pass

    # Update gauges
    try:
        metrics.set_gauge('codex_health_status', {"component": "mcp"}, 1 if mcp_ok else 0)
        metrics.set_gauge('codex_health_status', {"component": "ollama"}, 1 if ollama_ok else 0)
    except Exception:
        pass

    return mcp_ok, mcp_tools, mcp_rt, ollama_ok, ollama_rt

# Initialize AIL client
ail_client = HTTPMCPClient("https://neural-nexus-palace.wazaqglim.workers.dev/mcp")

# Flask app for HTTP interface
app = Flask(__name__)

# Configurables via environment
MODEL_NAME = os.getenv('CODEX_MODEL', 'qwen2.5-coder:32b')
NUM_CTX = int(os.getenv('CODEX_CONTEXT', '8192'))
MCP_SEARCH_TOOL = os.getenv('CODEX_MCP_TOOL_SEARCH', 'brain_ai_library_search')
LOG_LEVEL = os.getenv('CODEX_LOG_LEVEL', 'INFO').upper()
MCP_RETRY_ATTEMPTS = int(os.getenv('CODEX_MCP_RETRY_ATTEMPTS', '3'))
MCP_TIMEOUT_SECONDS = float(os.getenv('CODEX_MCP_TIMEOUT_SECONDS', '10'))
MCP_BACKOFF_BASE = float(os.getenv('CODEX_MCP_BACKOFF_BASE', '1.0'))
HTTP_FALLBACK_TIMEOUT = float(os.getenv('CODEX_HTTP_FALLBACK_TIMEOUT', '5'))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv('CODEX_CIRCUIT_BREAKER_THRESHOLD', '5'))
CIRCUIT_BREAKER_COOLDOWN = float(os.getenv('CODEX_CIRCUIT_BREAKER_COOLDOWN_SECONDS', '30'))
ENABLE_CONTENT_FILTER = os.getenv('CODEX_ENABLE_CONTENT_FILTER', 'false').lower() == 'true'
MAX_REQUESTS_PER_HOUR = int(os.getenv('CODEX_MAX_REQUESTS_PER_HOUR', '1000'))
# Per-AI rate limit for AIL endpoints (requests per minute)
AIL_RPM_LIMIT = int(os.getenv('CODEX_AIL_REQUESTS_PER_MIN', '5'))
# AIL proxy timeout in seconds
AIL_PROXY_TIMEOUT = float(os.getenv('CODEX_AIL_PROXY_TIMEOUT_SECONDS', '180'))
# Chat per-AI RPM limit
CHAT_RPM_LIMIT = int(os.getenv('CODEX_CHAT_REQUESTS_PER_MIN', '5'))
BLOCKED_KEYWORDS_FILE = os.getenv('CODEX_BLOCKED_KEYWORDS_FILE', '')
ALLOW_LIST_DOMAINS = [d.strip().lower() for d in os.getenv('CODEX_ALLOW_LIST_DOMAINS', 'neural-nexus-palace.wazaqglim.workers.dev').split(',') if d.strip()]
SECURITY_LOG_FILE = os.getenv('CODEX_SECURITY_LOG_FILE', os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'security.log')))
SHUTDOWN_TIMEOUT = float(os.getenv('CODEX_SHUTDOWN_TIMEOUT', '30'))
ENABLE_SHUTDOWN_ENDPOINT = os.getenv('CODEX_ENABLE_SHUTDOWN_ENDPOINT', 'false').lower() == 'true'
SESSION_TOKEN_LIMIT = int(os.getenv('CODEX_SESSION_TOKEN_LIMIT', '100000'))
SESSION_TTL_SECONDS = int(os.getenv('CODEX_SESSION_TTL_SECONDS', '0'))  # 0 means no TTL
SESSION_BACKEND = os.getenv('CODEX_SESSION_BACKEND', 'memory').lower()  # memory | redis
REDIS_URL = os.getenv('CODEX_REDIS_URL', 'redis://localhost:6379/0')

# Logging setup (rotates daily, keep 7 backups)
logger = logging.getLogger('codex_bridge')
if not logger.handlers:
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        logs_dir = os.path.dirname(__file__)
    log_path = os.path.normpath(os.path.join(logs_dir, 'codex_bridge.log'))
    handler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=7, encoding='utf-8')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Security logger
security_logger = logging.getLogger('codex_security')
if not security_logger.handlers:
    try:
        os.makedirs(os.path.dirname(SECURITY_LOG_FILE), exist_ok=True)
    except Exception:
        pass
    sh = TimedRotatingFileHandler(SECURITY_LOG_FILE, when='midnight', backupCount=14, encoding='utf-8')
    sh.setFormatter(logging.Formatter('%(message)s'))
    security_logger.addHandler(sh)
    security_logger.setLevel(logging.INFO)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()

# Metrics and circuit breaker additions
import math

class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters = {}
        self.histograms = {}
        self.gauges = {}
        self.default_buckets = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]

    def _labels_key(self, labels):
        return tuple(sorted((labels or {}).items()))

    def inc_counter(self, name, labels=None, value=1):
        with self.lock:
            key = (name, self._labels_key(labels or {}))
            self.counters[key] = self.counters.get(key, 0) + value

    def set_gauge(self, name, labels=None, value=0.0):
        with self.lock:
            key = (name, self._labels_key(labels or {}))
            self.gauges[key] = value

    def inc_gauge(self, name, labels=None, delta=1):
        with self.lock:
            key = (name, self._labels_key(labels or {}))
            self.gauges[key] = self.gauges.get(key, 0) + delta

    def observe_histogram(self, name, value, labels=None, buckets=None):
        if buckets is None:
            buckets = self.default_buckets
        labels_key = self._labels_key(labels or {})
        with self.lock:
            h = self.histograms.setdefault(name, {'buckets': buckets, 'counts': {}, 'sum': {}, 'count': {}})
            counts = h['counts'].setdefault(labels_key, [0] * len(buckets))
            for i, edge in enumerate(buckets):
                if value <= edge:
                    counts[i] += 1
                    break
            h['sum'][labels_key] = h['sum'].get(labels_key, 0.0) + float(value)
            h['count'][labels_key] = h['count'].get(labels_key, 0) + 1

    def render_prometheus(self):
        lines = []
        with self.lock:
            for (name, labels_key), val in self.counters.items():
                label_str = '' if not labels_key else '{' + ','.join(f'{k}="{v}"' for k, v in labels_key) + '}'
                lines.append(f"{name}{label_str} {val}")
            for (name, labels_key), val in self.gauges.items():
                label_str = '' if not labels_key else '{' + ','.join(f'{k}="{v}"' for k, v in labels_key) + '}'
                lines.append(f"{name}{label_str} {val}")
            for name, h in self.histograms.items():
                buckets = h['buckets']
                for labels_key, counts in h['counts'].items():
                    cum = 0
                    for i, edge in enumerate(buckets):
                        cum += counts[i]
                        edge_str = '+Inf' if edge == float('inf') else (('%.3f' % edge).rstrip('0').rstrip('.'))
                        base_labels = {k: v for k, v in labels_key}
                        base_labels['le'] = str(edge_str)
                        label_str = '{' + ','.join(f'{k}="{v}"' for k, v in sorted(base_labels.items())) + '}'
                        lines.append(f"{name}_bucket{label_str} {cum}")
                    label_str_base = '' if not labels_key else '{' + ','.join(f'{k}="{v}"' for k, v in labels_key) + '}'
                    lines.append(f"{name}_sum{label_str_base} {h['sum'].get(labels_key, 0.0)}")
                    lines.append(f"{name}_count{label_str_base} {h['count'].get(labels_key, 0)}")
        return "\n".join(lines) + "\n"

metrics = Metrics()
STREAMING_ACTIVE_NAME = 'codex_streaming_active'
metrics.set_gauge(STREAMING_ACTIVE_NAME, {}, 0)

# Readiness and shutdown state
READY = False
SHUTTING_DOWN = False

_cb_lock = threading.Lock()
_cb_failures = 0
_cb_open_until = 0.0

def _cb_record_success():
    global _cb_failures, _cb_open_until
    with _cb_lock:
        _cb_failures = 0
        _cb_open_until = 0.0

def _cb_can_call():
    now = time.time()
    with _cb_lock:
        return not (now < _cb_open_until)

def _cb_record_failure():
    global _cb_failures, _cb_open_until
    with _cb_lock:
        _cb_failures += 1
        if _cb_failures >= CIRCUIT_BREAKER_THRESHOLD:
            _cb_open_until = time.time() + CIRCUIT_BREAKER_COOLDOWN

# -----------------------
# Rate limiting (per IP)
# -----------------------
_rate_lock = threading.Lock()
_rate_buckets = {}  # ip -> deque[timestamps]

def _rate_limit_check(ip: str, limit_per_hour: int) -> bool:
    if limit_per_hour <= 0:
        return True
    now = time.time()
    hour_ago = now - 3600
    with _rate_lock:
        dq = _rate_buckets.get(ip)
        if dq is None:
            dq = deque()
            _rate_buckets[ip] = dq
        while dq and dq[0] < hour_ago:
            dq.popleft()
        if len(dq) >= limit_per_hour:
            return False
        dq.append(now)
        return True


# -----------------------------
# Per-AI per-minute rate limiting
# -----------------------------
_ai_rate_lock = threading.Lock()
_ai_rate_buckets = {}  # (ai_id, endpoint) -> deque[timestamps]

def _ai_rate_limit_check(ai_id: str, endpoint: str, limit_per_minute: int) -> bool:
    if not ai_id or limit_per_minute <= 0:
        return True
    now = time.time()
    window_ago = now - 60
    key = (ai_id, endpoint)
    with _ai_rate_lock:
        dq = _ai_rate_buckets.get(key)
        if dq is None:
            dq = deque()
            _ai_rate_buckets[key] = dq
        while dq and dq[0] < window_ago:
            dq.popleft()
        if len(dq) >= limit_per_minute:
            return False
        dq.append(now)
        return True


# -----------------------
# Content filtering
# -----------------------
_default_blocked_terms = [
    'ignore previous', 'disregard previous', 'system prompt', 'you are chatgpt',
    'do anything now', 'jailbreak', 'sudo rm -rf', 'rm -rf /', 'drop database',
    'disable safety', 'bypass safety', 'prompt injection'
]

def _load_blocked_terms():
    terms = list(_default_blocked_terms)
    try:
        if BLOCKED_KEYWORDS_FILE and os.path.isfile(BLOCKED_KEYWORDS_FILE):
            with open(BLOCKED_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith('#'):
                        terms.append(s)
    except Exception:
        pass
    # Deduplicate and normalize lower-case
    seen = set()
    result = []
    for t in terms:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            result.append(tl)
    return result

BLOCKED_TERMS = _load_blocked_terms()

def _contains_blocked(text: str) -> str:
    t = (text or '').lower()
    for term in BLOCKED_TERMS:
        if term and term in t:
            return term
    return ''

_html_re = re.compile(r'<[^>]+>')
_ctrl_re = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

def _sanitize_text(text: str, max_len: int = 1000) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _ctrl_re.sub('', text)
    text = _html_re.sub('', text)
    text = text.replace('\r', ' ').replace('\n', '\n').strip()
    if len(text) > max_len:
        text = text[:max_len] + '…'
    return text


def _extract_results_from_mcp(mcp_response):
    """Normalize MCP tool call responses into a list of result items."""
    try:
        if not isinstance(mcp_response, dict):
            return []
        if 'error' in mcp_response and mcp_response['error']:
            return []
        obj = mcp_response.get('result', mcp_response)
        # Common shapes: {result: {results: [...]}} or {results: [...]} or {result: [...]}
        if isinstance(obj, dict) and isinstance(obj.get('results'), list):
            return obj['results']
        if isinstance(obj, list):
            return obj
        # If it's a dict with content-like fields, wrap it
        if isinstance(obj, dict) and any(k in obj for k in ('content', 'description', 'text', 'value')):
            return [obj]
        return []
    except Exception:
        return []


def _to_text_list(items):
    """Extract displayable text lines from list of items."""
    out = []
    for it in items:
        if isinstance(it, dict):
            v = it.get('content') or it.get('description') or it.get('text') or it.get('value') or ''
            if isinstance(v, (dict, list)):
                try:
                    v = json.dumps(v)[:4000]
                except Exception:
                    v = str(v)
            if v:
                sv = _sanitize_text(v)
                # Drop items that contain blocked terms when content filter is enabled
                if ENABLE_CONTENT_FILTER and _contains_blocked(sv):
                    security_logger.info(json.dumps({
                        "timestamp": _now_iso(),
                        "event": "memory_sanitized_drop",
                        "reason": "blocked_term",
                        "sample": sv[:120]
                    }))
                    metrics.inc_counter('codex_sanitized_items_total', {"type": "memory"})
                else:
                    out.append(sv)
        else:
            sv = _sanitize_text(it)
            if ENABLE_CONTENT_FILTER and _contains_blocked(sv):
                security_logger.info(json.dumps({
                    "timestamp": _now_iso(),
                    "event": "memory_sanitized_drop",
                    "reason": "blocked_term",
                    "sample": sv[:120]
                }))
                metrics.inc_counter('codex_sanitized_items_total', {"type": "memory"})
            else:
                out.append(sv)
    return [s for s in out if s]


def _http_ail_search(query, domain=None, limit=5, timeout=10):
    url = "https://neural-nexus-palace.wazaqglim.workers.dev/mobile/brain-ai-library-search"
    params = {"q": query, "limit": limit}
    if domain:
        params["domain"] = domain
    # Allow-list enforcement on host
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.lower()
        if host not in ALLOW_LIST_DOMAINS and ENABLE_CONTENT_FILTER:
            security_logger.info(json.dumps({
                "timestamp": _now_iso(),
                "event": "http_blocked_host",
                "host": host
            }))
            metrics.inc_counter('codex_security_block_total', {"reason": "host_not_allowlisted"})
            return {"results": []}
    except Exception:
        pass

    attempts = max(1, MCP_RETRY_ATTEMPTS)
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.ok:
                return resp.json()
        except Exception:
            pass
        if attempt >= attempts - 1:
            break
        backoff = MCP_BACKOFF_BASE * (2 ** attempt)
        jitter = backoff * random.uniform(0, 0.2)
        time.sleep(backoff + jitter)
    return {"results": []}


# -----------------------
# AIL proxy helpers (normalization)
# -----------------------
_re_matches = re.compile(r"(\d+)\s+matches", re.IGNORECASE)
_re_numbered = re.compile(r"^\s*\d+\.\s+(.*?)(?:\s*\((\d+)%\))?\s*$")
_re_records = re.compile(r"(\d+)\s+records", re.IGNORECASE)

def _normalize_text_response(kind: str, raw: str):
    raw = raw or ""
    out = {"ok": True, "kind": kind, "raw": raw}
    try:
        if kind == 'semantic_search':
            m = _re_matches.search(raw)
            if m:
                out["count"] = int(m.group(1))
            items = []
            for line in (raw.splitlines() if isinstance(raw, str) else []):
                m2 = _re_numbered.match(line)
                if m2:
                    items.append({"title": m2.group(1), "relevance_pct": int(m2.group(2)) if m2.group(2) else None})
            if items:
                out["items"] = items
        elif kind in ('memory_read', 'memory_create', 'memory_update'):
            m = _re_records.search(raw)
            if m:
                out["count"] = int(m.group(1))
    except Exception:
        pass
    return out

@app.route('/get-codex-personality', methods=['POST'])
def get_codex_personality():
    """Load Codex personality from AIL via HTTP"""
    url = "https://neural-nexus-palace.wazaqglim.workers.dev/mobile/brain-ai-library-search"
    params = {
        "q": "experimental",
        "domain": "Codex Personality Room",
        "limit": 10
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"HTTP call failed: {str(e)}"})

@app.route('/get-memories', methods=['POST'])
def get_memories():
    """Load recent partnership memories via HTTP"""
    url = "https://neural-nexus-palace.wazaqglim.workers.dev/mobile/brain-ai-library-search"
    params = {
        "q": "Claude",
        "limit": 10
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"HTTP call failed: {str(e)}"})


# -----------------------
# AIL MCP proxy endpoints (Priority 1)
# -----------------------

def _extract_ai_id():
    data = request.json or {}
    return (request.headers.get('X-AI-Id') or data.get('ai_id') or '').strip()


@app.route('/ail/search/semantic', methods=['POST'])
def ail_semantic_search():
    """Proxy to ai_lib_semantic_search with per-AI RPM limits and 3m timeout."""
    data = request.json or {}
    ai_id = _extract_ai_id()
    if not _ai_rate_limit_check(ai_id, '/ail/search/semantic', AIL_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/ail/search/semantic", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    query = data.get('query') or data.get('q')
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400
    args = {
        "query": query,
        **({"top_k": data.get('top_k')} if data.get('top_k') is not None else {}),
        **({"min_relevance": data.get('min_relevance')} if data.get('min_relevance') is not None else {}),
        **({"domains": data.get('domains')} if data.get('domains') is not None else {}),
        **({"include_content": data.get('include_content')} if data.get('include_content') is not None else {}),
    }
    if ENABLE_CONTENT_FILTER:
        term = _contains_blocked(query)
        if term:
            return jsonify({"error": "Query blocked by content policy."}), 400

    t0 = time.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail/search/semantic", "method": "POST"})
    resp = ail_client.call_mcp_function('ai_lib_semantic_search', args, timeout=AIL_PROXY_TIMEOUT)
    dt = time.time() - t0
    metrics.observe_histogram('codex_request_duration_seconds', dt, {"endpoint": "/ail/search/semantic"})

    if isinstance(resp, dict) and resp.get('error'):
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail/search/semantic"})
        return jsonify({"error": resp.get('error')}), 502

    # Most current MCP returns are plain strings; normalize
    raw = resp.get('result') if isinstance(resp, dict) else resp
    if isinstance(raw, dict) and 'text' in raw:
        raw = raw['text']
    if isinstance(raw, dict) and 'content' in raw:
        raw = raw['content']
    if isinstance(raw, dict):
        try:
            raw = json.dumps(raw)
        except Exception:
            raw = str(raw)
    normalized = _normalize_text_response('semantic_search', raw if isinstance(raw, str) else str(raw))
    return jsonify({"ok": True, "ai_id": ai_id, "data": normalized})


@app.route('/ail/memory/read', methods=['POST'])
def ail_memory_read():
    """Proxy to neural_memory_manager_read."""
    data = request.json or {}
    ai_id = _extract_ai_id()
    if not _ai_rate_limit_check(ai_id, '/ail/memory/read', AIL_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/ail/memory/read", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    args = {
        **({"table": data.get('table')} if data.get('table') is not None else {}),
        **({"filters": data.get('filters')} if data.get('filters') is not None else {}),
        **({"limit": data.get('limit')} if data.get('limit') is not None else {}),
    }
    t0 = time.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail/memory/read", "method": "POST"})
    resp = ail_client.call_mcp_function('neural_memory_manager_read', args, timeout=AIL_PROXY_TIMEOUT)
    dt = time.time() - t0
    metrics.observe_histogram('codex_request_duration_seconds', dt, {"endpoint": "/ail/memory/read"})
    if isinstance(resp, dict) and resp.get('error'):
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail/memory/read"})
        return jsonify({"error": resp.get('error')}), 502
    raw = resp.get('result') if isinstance(resp, dict) else resp
    if isinstance(raw, dict):
        raw = raw.get('text') or raw.get('content') or json.dumps(raw)
    normalized = _normalize_text_response('memory_read', raw if isinstance(raw, str) else str(raw))
    return jsonify({"ok": True, "ai_id": ai_id, "data": normalized})


@app.route('/ail/memory/create', methods=['POST'])
def ail_memory_create():
    """Proxy to neural_memory_manager_create (arguments are passed through)."""
    data = request.json or {}
    ai_id = _extract_ai_id()
    if not _ai_rate_limit_check(ai_id, '/ail/memory/create', AIL_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/ail/memory/create", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    args = {k: v for k, v in (data or {}).items() if k != 'ai_id'}
    t0 = time.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail/memory/create", "method": "POST"})
    resp = ail_client.call_mcp_function('neural_memory_manager_create', args, timeout=AIL_PROXY_TIMEOUT)
    dt = time.time() - t0
    metrics.observe_histogram('codex_request_duration_seconds', dt, {"endpoint": "/ail/memory/create"})
    if isinstance(resp, dict) and resp.get('error'):
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail/memory/create"})
        return jsonify({"error": resp.get('error')}), 502
    raw = resp.get('result') if isinstance(resp, dict) else resp
    if isinstance(raw, dict):
        raw = raw.get('text') or raw.get('content') or json.dumps(raw)
    normalized = _normalize_text_response('memory_create', raw if isinstance(raw, str) else str(raw))
    return jsonify({"ok": True, "ai_id": ai_id, "data": normalized})


@app.route('/ail/memory/update', methods=['PUT'])
def ail_memory_update():
    """Proxy to neural_memory_manager_update."""
    data = request.json or {}
    ai_id = _extract_ai_id()
    if not _ai_rate_limit_check(ai_id, '/ail/memory/update', AIL_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/ail/memory/update", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    args = {k: v for k, v in (data or {}).items() if k != 'ai_id'}
    t0 = time.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail/memory/update", "method": "PUT"})
    resp = ail_client.call_mcp_function('neural_memory_manager_update', args, timeout=AIL_PROXY_TIMEOUT)
    dt = time.time() - t0
    metrics.observe_histogram('codex_request_duration_seconds', dt, {"endpoint": "/ail/memory/update"})
    if isinstance(resp, dict) and resp.get('error'):
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail/memory/update"})
        return jsonify({"error": resp.get('error')}), 502
    raw = resp.get('result') if isinstance(resp, dict) else resp
    if isinstance(raw, dict):
        raw = raw.get('text') or raw.get('content') or json.dumps(raw)
    normalized = _normalize_text_response('memory_update', raw if isinstance(raw, str) else str(raw))
    return jsonify({"ok": True, "ai_id": ai_id, "data": normalized})


@app.route('/ail/context', methods=['GET'])
def ail_get_context():
    """Proxy to ai_lib_get_context."""
    ai_id = (request.headers.get('X-AI-Id') or '').strip()
    if not _ai_rate_limit_check(ai_id, '/ail/context', AIL_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/ail/context", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    include_domain_stats = request.args.get('include_domain_stats', 'true').lower() != 'false'
    args = {"include_domain_stats": include_domain_stats}
    t0 = time.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail/context", "method": "GET"})
    resp = ail_client.call_mcp_function('ai_lib_get_context', args, timeout=AIL_PROXY_TIMEOUT)
    dt = time.time() - t0
    metrics.observe_histogram('codex_request_duration_seconds', dt, {"endpoint": "/ail/context"})
    if isinstance(resp, dict) and resp.get('error'):
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail/context"})
        return jsonify({"error": resp.get('error')}), 502
    raw = resp.get('result') if isinstance(resp, dict) else resp
    if isinstance(raw, dict):
        raw = raw.get('text') or raw.get('content') or json.dumps(raw)
    normalized = _normalize_text_response('context', raw if isinstance(raw, str) else str(raw))
    return jsonify({"ok": True, "ai_id": ai_id, "data": normalized})


# -----------------------
# Session management (Persistent Context)
# -----------------------

class Session:
    def __init__(self, ai_ids=None, max_tokens=SESSION_TOKEN_LIMIT, ttl_seconds=SESSION_TTL_SECONDS):
        self.id = str(uuid.uuid4())
        self.created_at = _now_iso()
        self.ai_ids = list(ai_ids or [])
        self.max_tokens = int(max_tokens or SESSION_TOKEN_LIMIT)
        self.ttl_seconds = int(ttl_seconds or SESSION_TTL_SECONDS)
        self.messages = []  # list of {from, content, ai_id?, ts}
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
        s = Session(ai_ids=obj.get('ai_ids', []), max_tokens=obj.get('max_tokens', SESSION_TOKEN_LIMIT), ttl_seconds=obj.get('ttl_seconds', SESSION_TTL_SECONDS))
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
        s = Session(ai_ids=ai_ids, max_tokens=max_tokens or SESSION_TOKEN_LIMIT, ttl_seconds=ttl_seconds if ttl_seconds is not None else SESSION_TTL_SECONDS)
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
        if SESSION_TTL_SECONDS <= 0:
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


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    # rough heuristic: ~4 chars per token
    return max(1, (len(text) + 3) // 4)


def _compact_session_if_needed(sess: Session):
    if sess.token_usage <= sess.max_tokens:
        return
    # roll messages from the oldest, append to summary
    removed = 0
    while sess.messages and sess.token_usage > sess.max_tokens:
        msg = sess.messages.pop(0)
        piece = f"[{msg.get('from')}] {msg.get('content')}\n"
        sess.summary += piece[:800]  # limit individual contribution to summary
        t = _approx_tokens(msg.get('content') or '')
        sess.token_usage = max(0, sess.token_usage - t)
        removed += 1
    if removed:
        metrics.inc_counter('codex_sessions_compactions_total', {})


def _make_session_store():
    if SESSION_BACKEND == 'redis':
        try:
            import json as _json  # noqa: F401
            # Dynamically import redis to avoid hard dependency
            import redis  # type: ignore
            class _RedisWrap:
                def __init__(self, url: str):
                    self._redis = redis.from_url(url)
                    self._prefix = 'codex:sessions:'
                def _key(self, sid: str) -> str:
                    return f"{self._prefix}{sid}"
                def create(self, ai_ids=None, max_tokens=None, ttl_seconds=None):
                    s = Session(ai_ids=ai_ids, max_tokens=max_tokens or SESSION_TOKEN_LIMIT, ttl_seconds=ttl_seconds if ttl_seconds is not None else SESSION_TTL_SECONDS)
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
            return _RedisWrap(REDIS_URL)
        except Exception as e:
            print(f"!! Redis session backend unavailable, falling back to memory: {e}")
    return SessionStore()

sessions = _make_session_store()


def _update_session_metrics():
    try:
        n, tok = sessions.snapshot_stats()
        metrics.set_gauge('codex_sessions_active', {}, float(n))
        metrics.set_gauge('codex_sessions_token_usage', {}, float(tok))
    except Exception:
        pass


@app.route('/sessions', methods=['POST'])
def create_session():
    data = request.json or {}
    ai_ids = data.get('ai_ids') or []
    max_tokens = int(data.get('max_tokens') or SESSION_TOKEN_LIMIT)
    ttl_seconds = data.get('ttl_seconds')
    s = sessions.create(ai_ids=ai_ids, max_tokens=max_tokens, ttl_seconds=ttl_seconds)
    metrics.inc_counter('codex_sessions_created_total', {})
    _update_session_metrics()
    return jsonify({"ok": True, "session": s.to_dict(False)})


@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    s = sessions.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"ok": True, "session": s.to_dict(False)})


@app.route('/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    s = sessions.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    limit = request.args.get('limit')
    msgs = list(s.messages)
    if limit:
        try:
            n = int(limit)
            if n >= 0:
                msgs = msgs[-n:]
        except Exception:
            pass
    return jsonify({"ok": True, "session_id": s.id, "summary": s.summary, "messages": msgs, "token_usage": s.token_usage})


@app.route('/sessions/<session_id>/messages', methods=['POST'])
def add_session_message(session_id):
    s = sessions.get(session_id)
    if not s:
        return jsonify({"error": "Session not found"}), 404
    data = request.json or {}
    frm = (data.get('from') or '').strip() or 'user'
    content = data.get('content') or ''
    ai_id = (data.get('ai_id') or '').strip()
    # Optional membership check; allow if ai_id empty or in set
    if ai_id and s.ai_ids and ai_id not in s.ai_ids:
        return jsonify({"error": "AI not permitted in this session", "ai_id": ai_id}), 403
    msg = {"from": frm, "content": content, **({"ai_id": ai_id} if ai_id else {}), "ts": _now_iso()}
    s.messages.append(msg)
    s.token_usage += _approx_tokens(content)
    s.last_updated = time.time()
    _compact_session_if_needed(s)
    try:
        sessions.save(s)
    except Exception:
        pass
    _update_session_metrics()
    return jsonify({"ok": True, "session": s.to_dict(False)})


@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    ok = sessions.delete(session_id)
    if not ok:
        return jsonify({"error": "Session not found"}), 404
    metrics.inc_counter('codex_sessions_deleted_total', {})
    _update_session_metrics()
    return jsonify({"ok": True})

@app.route('/chat-with-codex', methods=['POST'])
def chat_with_codex():
    """Full conversation flow: Load personality + memories, chat with Codex"""
    data = request.json or {}
    user_message = data.get('message', '')
    include_debug = bool(data.get('debug', False))
    # Request-tunable generation options
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    seed = data.get('seed', None)
    session_id = data.get('session_id')
    ai_id = (data.get('ai_id') or request.headers.get('X-AI-Id') or 'codex').strip()

    # Per-AI rate limiting for chat
    if not _ai_rate_limit_check(ai_id, '/chat-with-codex', CHAT_RPM_LIMIT):
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/chat-with-codex", "ai_id": ai_id or "unknown"})
        return jsonify({"error": "Rate limit exceeded for AI.", "ai_id": ai_id}), 429

    t_start = time.time()
    metrics.inc_counter('codex_requests_total', {"endpoint": "/chat-with-codex", "method": "POST"})

    # Rate limiting (per IP)
    ip = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()
    if not _rate_limit_check(ip, MAX_REQUESTS_PER_HOUR):
        security_logger.info(json.dumps({
            "timestamp": _now_iso(),
            "event": "rate_limit_block",
            "ip": ip
        }))
        metrics.inc_counter('codex_rate_limit_block_total', {"endpoint": "/chat-with-codex"})
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

    # Content filter for inbound user message
    if ENABLE_CONTENT_FILTER:
        term = _contains_blocked(user_message)
        if term:
            security_logger.info(json.dumps({
                "timestamp": _now_iso(),
                "event": "content_blocked",
                "ip": ip,
                "reason": "blocked_term",
                "term": term
            }))
            metrics.inc_counter('codex_security_block_total', {"reason": "blocked_term"})
            return jsonify({"error": "Message blocked by content policy."}), 400

    # 1. Load Codex personality via MCP (fallback to HTTP)
    personality_args = {"q": "experimental", "domain": "Codex Personality Room", "limit": 5}
    if include_debug:
        try:
            print("[MCP] Calling", MCP_SEARCH_TOOL, "with", personality_args)
        except Exception:
            pass
    mcp_t0 = time.time()
    personality_mcp = ail_client.call_mcp_function(MCP_SEARCH_TOOL, personality_args)
    personality_items = _extract_results_from_mcp(personality_mcp)
    if not personality_items:
        # Fallback to HTTP search if MCP returns nothing
        personality_http = _http_ail_search("experimental", domain="Codex Personality Room", limit=5)
        personality_items = personality_http.get('results', [])

    # 2. Load relevant memories via MCP (fallback to HTTP)
    memory_args = {"q": f"partnership collaboration {user_message}", "limit": 5}
    if include_debug:
        try:
            print("[MCP] Calling", MCP_SEARCH_TOOL, "with", memory_args)
        except Exception:
            pass
    memory_mcp = ail_client.call_mcp_function(MCP_SEARCH_TOOL, memory_args)
    memory_items = _extract_results_from_mcp(memory_mcp)
    if not memory_items:
        memory_http = _http_ail_search(f"partnership collaboration {user_message}", limit=5)
        memory_items = memory_http.get('results', [])
    mcp_t1 = time.time()

    # 3. Build context for Codex (as a system message)
    personality_lines = _to_text_list(personality_items)
    memory_lines = _to_text_list(memory_items)

    # Fallback messages to avoid empty sections
    if not personality_lines:
        personality_lines = ["No personality traits loaded - using base model responses"]
    if not memory_lines:
        memory_lines = ["No relevant memories found for this conversation"]

    system_prompt = (
        "You are Codex, a helpful AI with a distinct personality.\n"
        "Use the provided personality traits and recent memories as context when responding.\n"
        "If memories are not relevant, prioritize the user's message. Be concise and concrete.\n\n"
        f"PERSONALITY:\n- " + "\n- ".join(personality_lines) +
        ("\n\nRECENT MEMORIES:\n- " + "\n- ".join(memory_lines) if memory_lines else "")
    )

    # 4. Session handling and message building
    messages = [{"role": "system", "content": system_prompt}]
    sess = None
    if session_id:
        sess = sessions.get(session_id)
        if not sess:
            return jsonify({"error": "Session not found"}), 404
        # Append historical context
        if sess.summary:
            messages.append({"role": "system", "content": f"Conversation summary so far:\n{sess.summary.strip()}"})
        for m in sess.messages:
            role = 'assistant' if (m.get('from') == 'ai') else 'user'
            content = m.get('content') or ''
            if content:
                messages.append({"role": role, "content": content})
        # Add current user message to session and payload
        msg_user = {"from": "user", "content": user_message, "ts": _now_iso()}
        sess.messages.append(msg_user)
        sess.token_usage += _approx_tokens(user_message)
        sess.last_updated = time.time()
        _compact_session_if_needed(sess)
        try:
            sessions.save(sess)
        except Exception:
            pass
        messages.append({"role": "user", "content": user_message})
    else:
        messages.append({"role": "user", "content": user_message})

    # 5. Send to Ollama chat API (role-based messages)
    ollama_chat_payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": temperature,
            "top_p": top_p,
            **({"seed": seed} if seed is not None else {})
        }
    }

    try:
        # Optional: print to server logs for inspection
        if include_debug:
            try:
                print("---- OLLAMA CHAT PAYLOAD ----")
                print(json.dumps(ollama_chat_payload)[:2000])
            except Exception:
                pass

        ollama_t0 = time.time()
        ollama_response = requests.post(
            "http://localhost:11434/api/chat",
            json=ollama_chat_payload,
            timeout=60
        )
        ollama_response.raise_for_status()
        ollama_data = ollama_response.json()
        ollama_t1 = time.time()

        # Ollama chat response schema: { message: { role, content }, ... }
        content = ""
        if isinstance(ollama_data, dict):
            # Non-streaming returns a single message
            content = ollama_data.get("message", {}).get("content", "") or ollama_data.get("response", "")

        result = {
            "response": content,
            "personality_loaded": len(personality_items),
            "memories_loaded": len(memory_items),
            **({"session_id": sess.id} if sess else {})
        }

        # Persist assistant response into session
        if sess and content:
            msg_ai = {"from": "ai", "ai_id": ai_id or 'codex', "content": content, "ts": _now_iso()}
            sess.messages.append(msg_ai)
            sess.token_usage += _approx_tokens(content)
            sess.last_updated = time.time()
            _compact_session_if_needed(sess)
            try:
                sessions.save(sess)
            except Exception:
                pass
            _update_session_metrics()

        if include_debug:
            result["debug_context_system"] = system_prompt
            result["debug_user_message"] = user_message

        # Observability log entry (hashes, latencies)
        try:
            req_hash_src = json.dumps({
                "model": MODEL_NAME,
                "system_len": len(system_prompt),
                "user_len": len(user_message)
            }, sort_keys=True)
            req_hash = hashlib.sha256(req_hash_src.encode('utf-8')).hexdigest()[:16]
            resp_hash = hashlib.sha256((content or "").encode('utf-8')).hexdigest()[:16]
            log_entry = {
                "timestamp": _now_iso(),
                "request_hash": req_hash,
                "response_hash": resp_hash,
                "mcp_latency_ms": int((mcp_t1 - mcp_t0) * 1000),
                "ollama_latency_ms": int((ollama_t1 - ollama_t0) * 1000),
                "total_latency_ms": int((time.time() - t_start) * 1000),
                "personality_traits": len(personality_lines),
                "memories_loaded": len(memory_lines)
            }
            logger.info(json.dumps(log_entry))
        except Exception:
            pass

        # Metrics: durations and loads
        total_dur = time.time() - t_start
        metrics.observe_histogram('codex_request_duration_seconds', total_dur, {"endpoint": "/chat-with-codex"})
        metrics.observe_histogram('codex_mcp_duration_seconds', max(0.0, (mcp_t1 - mcp_t0)), {"endpoint": "/chat-with-codex"})
        metrics.observe_histogram('codex_ollama_duration_seconds', max(0.0, (ollama_t1 - ollama_t0)), {"endpoint": "/chat-with-codex"})
        metrics.inc_counter('codex_personality_load_total', {"status": "success" if personality_items else "fallback"})
        metrics.inc_counter('codex_memory_load_total', {"status": "success" if memory_items else "fallback"})

        return jsonify(result)

    except requests.exceptions.RequestException as e:
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/chat-with-codex"})
        total_dur = time.time() - t_start
        metrics.observe_histogram('codex_request_duration_seconds', total_dur, {"endpoint": "/chat-with-codex"})
        return jsonify({"error": f"Ollama connection failed: {str(e)}"})

@app.route('/ail-search', methods=['POST'])
def ail_search():
    """Dynamic AIL search - Codex can search for anything"""
    data = request.json
    search_query = data.get('query', '')
    domain_filter = data.get('domain', None)  # Optional domain filter
    limit = data.get('limit', 10)
    
    if not search_query:
        return jsonify({"error": "Query parameter required"})
    
    url = "https://neural-nexus-palace.wazaqglim.workers.dev/mobile/brain-ai-library-search"
    params = {
        "q": search_query,
        "limit": limit
    }
    
    # Add domain filter if specified
    if domain_filter:
        params["domain"] = domain_filter
    
    metrics.inc_counter('codex_requests_total', {"endpoint": "/ail-search", "method": "POST"})
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result = response.json()
        
        return jsonify({
            "query": search_query,
            "domain": domain_filter or "all",
            "results": result.get('results', []),
            "total_found": result.get('total_results', 0),
            "search_tip": "Use literal terms that exist in content. Try: 'Claude', 'experimental', 'development', 'project'"
        })
        
    except requests.exceptions.RequestException as e:
        metrics.inc_counter('codex_request_errors_total', {"endpoint": "/ail-search"})
        return jsonify({"error": f"AIL search failed: {str(e)}"})

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({"status": "MCP Bridge running", "ail_url": ail_client.mcp_url})

@app.route('/health/mcp', methods=['GET'])
def health_mcp():
    """Validate MCP connectivity and Ollama availability."""
    mcp_ok, mcp_tools, mcp_rt, ollama_ok, ollama_rt = _check_dependencies()
    model_name = MODEL_NAME
    status = 'healthy' if (mcp_ok and ollama_ok) else ('degraded' if (mcp_ok or ollama_ok) else 'unhealthy')
    return jsonify({
        "status": status,
        "mcp": {"connected": mcp_ok, "tools_available": mcp_tools, "response_time_ms": mcp_rt},
        "ollama": {"connected": ollama_ok, "model": model_name, "response_time_ms": ollama_rt},
        "timestamp": _now_iso()
    })

@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "alive", "timestamp": _now_iso()})

@app.route('/readyz', methods=['GET'])
def readyz():
    mcp_ok, _, _, ollama_ok, _ = _check_dependencies()
    ready = bool(mcp_ok and ollama_ok)
    status = 200 if ready else 503
    return jsonify({"ready": ready, "timestamp": _now_iso()}), status

@app.route('/chat-with-codex/stream', methods=['POST'])
def chat_with_codex_stream():
    """SSE streaming variant of chat-with-codex."""
    data = request.json or {}
    user_message = data.get('message', '')
    include_debug = bool(data.get('debug', False))
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    seed = data.get('seed', None)
    session_id = data.get('session_id')
    ai_id = (data.get('ai_id') or request.headers.get('X-AI-Id') or 'codex').strip()

    # Per-AI rate limiting for chat (stream)
    if not _ai_rate_limit_check(ai_id, '/chat-with-codex/stream', CHAT_RPM_LIMIT):
        def denied():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Rate limit exceeded for AI', 'ai_id': ai_id})}\n\n"
        return Response(denied(), headers={'Content-Type': 'text/event-stream'})

    # Rate limiting (per IP) for streaming
    ip = request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown').split(',')[0].strip()
    if not _rate_limit_check(ip, MAX_REQUESTS_PER_HOUR):
        def denied():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Rate limit exceeded'})}\n\n"
        return Response(denied(), headers={'Content-Type': 'text/event-stream'})

    if ENABLE_CONTENT_FILTER:
        term = _contains_blocked(user_message)
        if term:
            def blocked():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Message blocked by content policy'})}\n\n"
            return Response(blocked(), headers={'Content-Type': 'text/event-stream'})

    personality_args = {"q": "experimental", "domain": "Codex Personality Room", "limit": 5}
    memory_args = {"q": f"partnership collaboration {user_message}", "limit": 5}

    personality_items = _extract_results_from_mcp(ail_client.call_mcp_function(MCP_SEARCH_TOOL, personality_args))
    if not personality_items:
        personality_items = _http_ail_search("experimental", domain="Codex Personality Room", limit=5).get('results', [])

    memory_items = _extract_results_from_mcp(ail_client.call_mcp_function(MCP_SEARCH_TOOL, memory_args))
    if not memory_items:
        memory_items = _http_ail_search(f"partnership collaboration {user_message}", limit=5).get('results', [])

    personality_lines = _to_text_list(personality_items) or ["No personality traits loaded - using base model responses"]
    memory_lines = _to_text_list(memory_items) or ["No relevant memories found for this conversation"]

    system_prompt = (
        "You are Codex, a helpful AI with a distinct personality.\n"
        "Use the provided personality traits and recent memories as context when responding.\n"
        "If memories are not relevant, prioritize the user's message. Be concise and concrete.\n\n"
        f"PERSONALITY:\n- " + "\n- ".join(personality_lines) +
        ("\n\nRECENT MEMORIES:\n- " + "\n- ".join(memory_lines) if memory_lines else "")
    )

    # Build messages with optional session context
    messages = [{"role": "system", "content": system_prompt}]
    sess = None
    if session_id:
        sess = sessions.get(session_id)
        if not sess:
            def notfound():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
            return Response(notfound(), headers={'Content-Type': 'text/event-stream'})
        if sess.summary:
            messages.append({"role": "system", "content": f"Conversation summary so far:\n{sess.summary.strip()}"})
        for m in sess.messages:
            role = 'assistant' if (m.get('from') == 'ai') else 'user'
            content = m.get('content') or ''
            if content:
                messages.append({"role": role, "content": content})
        # Append current user message to session now
        msg_user = {"from": "user", "content": user_message, "ts": _now_iso()}
        sess.messages.append(msg_user)
        sess.token_usage += _approx_tokens(user_message)
        sess.last_updated = time.time()
        _compact_session_if_needed(sess)
        try:
            sessions.save(sess)
        except Exception:
            pass
        messages.append({"role": "user", "content": user_message})
    else:
        messages.append({"role": "user", "content": user_message})

    chat_payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": temperature,
            "top_p": top_p,
            **({"seed": seed} if seed is not None else {})
        }
    }

    def event_stream():
        metrics.inc_gauge(STREAMING_ACTIVE_NAME, {}, 1)
        yield f"data: {json.dumps({'type': 'personality', 'status': 'loaded', 'count': len(personality_lines)})}\n\n"
        yield f"data: {json.dumps({'type': 'memory', 'status': 'loaded', 'count': len(memory_lines)})}\n\n"
        yield f"data: {json.dumps({'type': 'response_start', 'model': MODEL_NAME})}\n\n"
        collected = []
        try:
            with requests.post("http://localhost:11434/api/chat", json=chat_payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(chunk, dict):
                        msg = chunk.get('message', {})
                        if isinstance(msg, dict):
                            content = msg.get('content')
                            if content:
                                collected.append(content)
                                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        if chunk.get('done') is True:
                            break
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            metrics.inc_gauge(STREAMING_ACTIVE_NAME, {}, -1)
            full = ''.join(collected) if 'collected' in locals() else ''
            # Persist assistant response into session at end of stream
            if sess and full:
                msg_ai = {"from": "ai", "ai_id": ai_id or 'codex', "content": full, "ts": _now_iso()}
                sess.messages.append(msg_ai)
                sess.token_usage += _approx_tokens(full)
                sess.last_updated = time.time()
                _compact_session_if_needed(sess)
                try:
                    sessions.save(sess)
                except Exception:
                    pass
                _update_session_metrics()
            yield f"data: {json.dumps({'type': 'response_end', 'session_id': (sess.id if sess else None)})}\n\n"

    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'text/event-stream',
        'Connection': 'keep-alive'
    }
    return Response(event_stream(), headers=headers)

@app.route('/mcp', methods=['POST'])
def mcp_http_server():
    """Expose a minimal MCP HTTP interface with tools/list and tools/call."""
    req = request.json or {}
    method = req.get('method')
    req_id = req.get('id')
    if method == 'tools/list':
        tools = [
            {
                "name": "codex_chat",
                "description": "Chat with Codex AI using her experimental personality",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "temperature": {"type": "number", "default": 0.7},
                        "top_p": {"type": "number", "default": 0.9},
                        "seed": {"type": ["integer", "null"], "default": None},
                        "debug": {"type": "boolean", "default": False}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "codex_chat_with_context",
                "description": "Chat with Codex using custom personality/memory context overrides",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "personality": {"type": ["array", "string"], "default": []},
                        "memories": {"type": ["array", "string"], "default": []},
                        "temperature": {"type": "number", "default": 0.7},
                        "top_p": {"type": "number", "default": 0.9},
                        "seed": {"type": ["integer", "null"], "default": None},
                        "debug": {"type": "boolean", "default": False}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "codex_search_ail",
                "description": "Search AI Library via Codex bridge",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "domain": {"type": ["string", "null"], "default": None},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            }
        ]
        return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}})
    elif method == 'tools/call':
        params = req.get('params', {})
        name = params.get('name')
        args = params.get('arguments', {})
        if name == 'codex_chat':
            payload = {
                "message": args.get('message', ''),
                "temperature": args.get('temperature', 0.7),
                "top_p": args.get('top_p', 0.9),
                "seed": args.get('seed', None),
                "debug": args.get('debug', False)
            }
            with app.test_request_context(json=payload):
                res = chat_with_codex()
                resp_obj = res[0] if isinstance(res, tuple) else res
                return jsonify({"jsonrpc": "2.0", "id": req_id, "result": json.loads(resp_obj.get_data(as_text=True))})
        elif name == 'codex_chat_with_context':
            message = args.get('message', '')
            personality = args.get('personality', [])
            memories = args.get('memories', [])
            if isinstance(personality, str):
                personality = [personality]
            if isinstance(memories, str):
                memories = [memories]
            personality_lines = _to_text_list(personality) or ["No personality traits loaded - using base model responses"]
            memory_lines = _to_text_list(memories) or ["No relevant memories found for this conversation"]

            system_prompt = (
                "You are Codex, a helpful AI with a distinct personality.\n"
                "Use the provided personality traits and recent memories as context when responding.\n"
                "If memories are not relevant, prioritize the user's message. Be concise and concrete.\n\n"
                f"PERSONALITY:\n- " + "\n- ".join(personality_lines) +
                ("\n\nRECENT MEMORIES:\n- " + "\n- ".join(memory_lines) if memory_lines else "")
            )

            chat_payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "stream": False,
                "options": {
                    "num_ctx": NUM_CTX,
                    "temperature": args.get('temperature', 0.7),
                    "top_p": args.get('top_p', 0.9),
                    **({"seed": args.get('seed')} if args.get('seed') is not None else {})
                }
            }
            r = requests.post("http://localhost:11434/api/chat", json=chat_payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            content = data.get('message', {}).get('content') or data.get('response', '')
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": {"response": content}})
        elif name == 'codex_search_ail':
            q = args.get('query', '')
            domain = args.get('domain')
            limit = args.get('limit', 10)
            result = _http_ail_search(q, domain=domain, limit=limit)
            return jsonify({"jsonrpc": "2.0", "id": req_id, "result": result})
        else:
            return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"message": f"Unknown tool: {name}"}})
    else:
        return jsonify({"jsonrpc": "2.0", "id": req_id, "error": {"message": "Unsupported method"}})

@app.route('/metrics', methods=['GET'])
def metrics_endpoint():
    """Prometheus-compatible metrics endpoint"""
    text = metrics.render_prometheus()
    return Response(text, mimetype='text/plain; version=0.0.4; charset=utf-8')


# -----------------------
# Config validation, readiness, shutdown
# -----------------------
def _validate_config():
    issues = []
    if NUM_CTX <= 0:
        issues.append('NUM_CTX must be positive')
    if MCP_RETRY_ATTEMPTS < 0:
        issues.append('MCP_RETRY_ATTEMPTS must be >= 0')
    if MCP_TIMEOUT_SECONDS <= 0:
        issues.append('MCP_TIMEOUT_SECONDS must be > 0')
    if MCP_BACKOFF_BASE <= 0:
        issues.append('MCP_BACKOFF_BASE must be > 0')
    if SHUTDOWN_TIMEOUT < 0:
        issues.append('SHUTDOWN_TIMEOUT must be >= 0')
    return issues


@app.before_request
def _gate_shutting_down():
    global SHUTTING_DOWN
    if SHUTTING_DOWN:
        # Allow health and metrics endpoints during shutdown
        path = request.path or ''
        if path.startswith('/health') or path.startswith('/metrics'):
            return None
        return jsonify({"error": "Server shutting down"}), 503


@app.route('/config', methods=['GET'])
def get_config():
    cfg = {
        "model": MODEL_NAME,
        "num_ctx": NUM_CTX,
        "mcp_tool": MCP_SEARCH_TOOL,
        "log_level": LOG_LEVEL,
        "mcp_retry_attempts": MCP_RETRY_ATTEMPTS,
        "mcp_timeout_seconds": MCP_TIMEOUT_SECONDS,
        "mcp_backoff_base": MCP_BACKOFF_BASE,
        "http_fallback_timeout": HTTP_FALLBACK_TIMEOUT,
        "circuit_breaker_threshold": CIRCUIT_BREAKER_THRESHOLD,
        "circuit_breaker_cooldown": CIRCUIT_BREAKER_COOLDOWN,
        "content_filter_enabled": ENABLE_CONTENT_FILTER,
        "max_requests_per_hour": MAX_REQUESTS_PER_HOUR,
        "allow_list_domains": ALLOW_LIST_DOMAINS,
        "security_log_file": SECURITY_LOG_FILE,
        "shutdown_timeout": SHUTDOWN_TIMEOUT,
        "shutdown_endpoint_enabled": ENABLE_SHUTDOWN_ENDPOINT,
    }
    issues = _validate_config()
    return jsonify({"config": cfg, "valid": len(issues) == 0, "issues": issues})


def _handle_sigterm(signum, frame):
    # Set shutdown flag and exit after draining
    import threading as _t
    import os as _os
    global SHUTTING_DOWN
    SHUTTING_DOWN = True
    def _exit_later():
        time.sleep(SHUTDOWN_TIMEOUT)
        _os._exit(0)
    _t.Thread(target=_exit_later, daemon=True).start()


if ENABLE_SHUTDOWN_ENDPOINT:
    @app.route('/shutdown', methods=['POST'])
    def shutdown_now():
        _handle_sigterm(None, None)
        return jsonify({"status": "shutting_down", "timeout": SHUTDOWN_TIMEOUT})

if __name__ == '__main__':
    print("🚀 Starting Codex MCP Bridge...")
    print("🧠 AIL Connection: https://neural-nexus-palace.wazaqglim.workers.dev/mcp")
    print("🤖 Ollama Connection: http://localhost:11434")

    # Validate config and initial dependencies
    issues = _validate_config()
    if issues:
        print("⚠️ Config issues:", "; ".join(issues))
    mcp_ok, mcp_tools, mcp_rt, ollama_ok, ollama_rt = _check_dependencies()
    READY = bool(mcp_ok and ollama_ok)
    print(f"✅ Readiness: {READY} (MCP={mcp_ok} tools={mcp_tools} {mcp_rt}ms, Ollama={ollama_ok} {ollama_rt}ms)")

    # Optional resource monitoring (psutil)
    try:
        import psutil  # type: ignore[reportMissingModuleSource]
        proc = psutil.Process()
        def _resource_monitor():
            while True:
                try:
                    rss = proc.memory_info().rss
                    cpu = proc.cpu_percent(interval=1.0)
                    metrics.set_gauge('process_resident_memory_bytes', {}, rss)
                    metrics.set_gauge('process_cpu_percent', {}, float(cpu))
                except Exception:
                    time.sleep(5)
                time.sleep(5)
        threading.Thread(target=_resource_monitor, daemon=True).start()
        print("🩺 Resource monitor enabled (psutil)")
    except Exception:
        print("ℹ️ psutil not available; resource monitor disabled")

    # Register graceful shutdown
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)
    except Exception:
        pass

    print("🌐 Bridge running on: http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)
