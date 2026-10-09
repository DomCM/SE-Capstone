import math
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import psutil


class PerformanceMetrics:
    def __init__(self, sample_limit=5000, window_seconds=300):
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._requests = deque(maxlen=sample_limit)
        self._timings = defaultdict(lambda: deque(maxlen=sample_limit))
        self._camera_open_attempts = 0
        self._camera_open_failures = 0
        self._camera_open_durations = deque(maxlen=sample_limit)
        self._process = psutil.Process()
        self._cpu_sample_ready = False

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return None
        ordered = sorted(values)
        index = max(0, math.ceil(percentile * len(ordered)) - 1)
        return round(ordered[index], 2)

    def record_request(self, endpoint, method, status_code, duration_ms):
        now = time.monotonic()
        with self._lock:
            self._requests.append((
                now,
                endpoint or 'unknown',
                method,
                status_code,
                duration_ms,
            ))

    def record_timing(self, name, duration_ms):
        now = time.monotonic()
        with self._lock:
            self._timings[name].append((now, duration_ms))

    def record_camera_open(self, duration_ms, succeeded):
        now = time.monotonic()
        with self._lock:
            self._camera_open_attempts += 1
            if not succeeded:
                self._camera_open_failures += 1
            self._camera_open_durations.append((now, duration_ms))

    def snapshot(self):
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            requests = [sample for sample in self._requests if sample[0] >= cutoff]
            timings = {
                name: [duration for recorded_at, duration in samples if recorded_at >= cutoff]
                for name, samples in self._timings.items()
            }
            attempts = self._camera_open_attempts
            failures = self._camera_open_failures
            open_durations = [
                duration for recorded_at, duration in self._camera_open_durations
                if recorded_at >= cutoff
            ]

        grouped_requests = defaultdict(list)
        for _, endpoint, method, status_code, duration_ms in requests:
            grouped_requests[(endpoint, method)].append((status_code, duration_ms))

        by_endpoint = {}
        for (endpoint, method), samples in sorted(grouped_requests.items()):
            durations = [duration for _, duration in samples]
            by_endpoint[f'{method} {endpoint}'] = {
                'count': len(samples),
                'errors_5xx': sum(status >= 500 for status, _ in samples),
                'p50_ms': self._percentile(durations, 0.50),
                'p95_ms': self._percentile(durations, 0.95),
                'p99_ms': self._percentile(durations, 0.99),
                'max_ms': round(max(durations), 2),
            }

        current_cpu = self._process.cpu_percent(interval=None)
        cpu_percent = current_cpu if self._cpu_sample_ready else None
        self._cpu_sample_ready = True
        memory_rss_bytes = self._process.memory_info().rss
        thread_count = self._process.num_threads()

        return {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'window_seconds': self.window_seconds,
            'process': {
                'cpu_percent': cpu_percent,
                'memory_rss_bytes': memory_rss_bytes,
                'thread_count': thread_count,
            },
            'requests': {
                'count': len(requests),
                'errors_5xx': sum(sample[3] >= 500 for sample in requests),
                'p50_ms': self._percentile([sample[4] for sample in requests], 0.50),
                'p95_ms': self._percentile([sample[4] for sample in requests], 0.95),
                'p99_ms': self._percentile([sample[4] for sample in requests], 0.99),
                'by_endpoint': by_endpoint,
            },
            'camera_connections': {
                'attempts_total': attempts,
                'failures_total': failures,
                'attempts_in_window': len(open_durations),
                'open_duration_p95_ms': self._percentile(open_durations, 0.95),
            },
            'vision_timings': {
                name: {
                    'count': len(durations),
                    'p50_ms': self._percentile(durations, 0.50),
                    'p95_ms': self._percentile(durations, 0.95),
                    'max_ms': round(max(durations), 2) if durations else None,
                }
                for name, durations in sorted(timings.items())
            },
        }
