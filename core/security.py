import os
import re
from collections import deque
import threading
import time
from typing import List

import config
from .utils import now_iso


_rate_lock = threading.Lock()
_rate_buckets = {}  # ip -> deque[timestamps]


def rate_limit_check(ip: str, limit_per_hour: int) -> bool:
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


_ai_rate_lock = threading.Lock()
_ai_rate_buckets = {}  # (ai_id, endpoint) -> deque[timestamps]


def ai_rate_limit_check(ai_id: str, endpoint: str, limit_per_minute: int) -> bool:
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


_default_blocked_terms: List[str] = [
    'ignore previous', 'disregard previous', 'system prompt', 'you are chatgpt',
    'do anything now', 'jailbreak', 'sudo rm -rf', 'rm -rf /', 'drop database',
    'disable safety', 'bypass safety', 'prompt injection'
]


def _load_blocked_terms():
    terms = list(_default_blocked_terms)
    try:
        if config.BLOCKED_KEYWORDS_FILE and os.path.isfile(config.BLOCKED_KEYWORDS_FILE):
            with open(config.BLOCKED_KEYWORDS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith('#'):
                        terms.append(s)
    except Exception:
        pass
    seen = set()
    result = []
    for t in terms:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            result.append(tl)
    return result


BLOCKED_TERMS = _load_blocked_terms()


def contains_blocked(text: str) -> str:
    t = (text or '').lower()
    for term in BLOCKED_TERMS:
        if term and term in t:
            return term
    return ''


_html_re = re.compile(r'<[^>]+>')
_ctrl_re = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')


def sanitize_text(text: str, max_len: int = 1000) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _ctrl_re.sub('', text)
    text = _html_re.sub('', text)
    text = text.replace('\r', ' ').replace('\n', '\n').strip()
    if len(text) > max_len:
        text = text[:max_len] + '.'
    return text

