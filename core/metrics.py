import threading


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

