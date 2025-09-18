import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional


class PrometheusMetrics:
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


class MetricsCollector:
    """Collect in-memory health metrics for the admin dashboard."""

    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.lock = threading.RLock()

        # Metric primitives
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)

        # Time-series history
        self.request_history: Deque[Dict[str, Any]] = deque(maxlen=max_history_size)
        self.error_history: Deque[Dict[str, Any]] = deque(maxlen=max_history_size)
        self.session_history: Deque[Dict[str, Any]] = deque(maxlen=max_history_size)
        self.response_times: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))

        self.start_time = datetime.now(timezone.utc)

    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value

    def record_request(self, endpoint: str, method: str, status_code: int, duration: float) -> None:
        timestamp = self._now_iso()
        with self.lock:
            entry = {
                "timestamp": timestamp,
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 3),
                "success": status_code < 400,
            }
            self.request_history.append(entry)

            key = self._make_key("http_requests_total", {
                "endpoint": endpoint,
                "method": method,
                "status": str(status_code),
            })
            self.counters[key] += 1

            self.response_times[endpoint].append(duration)

    def record_error(self, error_code: str, endpoint: str, message: str) -> None:
        timestamp = self._now_iso()
        with self.lock:
            entry = {
                "timestamp": timestamp,
                "error_code": error_code,
                "endpoint": endpoint,
                "message": message,
            }
            self.error_history.append(entry)

            key = self._make_key("errors_total", {
                "error_code": error_code,
                "endpoint": endpoint,
            })
            self.counters[key] += 1

    def record_session_event(self, event_type: str, session_id: str, details: Optional[Dict[str, Any]] = None) -> None:
        timestamp = self._now_iso()
        with self.lock:
            entry = {
                "timestamp": timestamp,
                "event_type": event_type,
                "session_id": session_id,
                "details": details or {},
            }
            self.session_history.append(entry)

            key = self._make_key("session_events_total", {"event_type": event_type})
            self.counters[key] += 1

    def get_summary(self) -> Dict[str, Any]:
        with self.lock:
            uptime = datetime.now(timezone.utc) - self.start_time
            recent_requests = list(self.request_history)[-100:]
            errors = sum(1 for r in recent_requests if not r.get("success"))
            error_rate = (errors / len(recent_requests)) if recent_requests else 0.0

            avg_response_times: Dict[str, float] = {}
            for endpoint, durations in self.response_times.items():
                if durations:
                    avg_response_times[endpoint] = sum(durations) / len(durations)

            return {
                "uptime_seconds": uptime.total_seconds(),
                "uptime_formatted": str(uptime).split(".")[0],
                "total_requests": len(self.request_history),
                "total_errors": len(self.error_history),
                "error_rate_percent": round(error_rate * 100, 2),
                "average_response_times": avg_response_times,
                "active_counters": dict(self.counters),
                "current_gauges": dict(self.gauges),
            }

    def get_time_series_data(self, hours_back: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(hours_back, 0))
        with self.lock:
            requests = [r for r in self.request_history if self._parse_iso(r["timestamp"]) >= cutoff]
            errors = [e for e in self.error_history if self._parse_iso(e["timestamp"]) >= cutoff]
            sessions = [s for s in self.session_history if self._parse_iso(s["timestamp"]) >= cutoff]

        return {
            "requests": requests,
            "errors": errors,
            "sessions": sessions,
        }

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    @staticmethod
    def _parse_iso(timestamp: str) -> datetime:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_metrics_collector(max_history_size: int = 1000) -> MetricsCollector:
    return MetricsCollector(max_history_size=max_history_size)
