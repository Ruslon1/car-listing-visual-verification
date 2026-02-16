from __future__ import annotations

import logging
from typing import Any


class _NullMetric:
    def labels(self, **_: Any) -> "_NullMetric":
        return self

    def inc(self, *_: Any, **__: Any) -> None:
        return None

    def observe(self, *_: Any, **__: Any) -> None:
        return None


class DromMetrics:
    def __init__(self) -> None:
        self.enabled = False
        self._server_started = False

        self.requests_total: Any = _NullMetric()
        self.request_latency_seconds: Any = _NullMetric()
        self.cache_hits_total: Any = _NullMetric()
        self.stage_rows_total: Any = _NullMetric()
        self.stage_errors_total: Any = _NullMetric()
        self.images_downloaded_total: Any = _NullMetric()

        self._init_prometheus_metrics()

    def _init_prometheus_metrics(self) -> None:
        try:
            from prometheus_client import Counter, Histogram
        except ImportError:
            return

        self.requests_total = Counter(
            "drom_requests_total",
            "HTTP requests executed by pipeline",
            ["host", "status"],
        )
        self.request_latency_seconds = Histogram(
            "drom_request_latency_seconds",
            "Latency of HTTP requests",
            ["host"],
            buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
        )
        self.cache_hits_total = Counter(
            "drom_cache_hits_total",
            "Cache hits for HTML/JSON responses",
        )
        self.stage_rows_total = Counter(
            "drom_stage_rows_total",
            "Rows produced by stage",
            ["stage"],
        )
        self.stage_errors_total = Counter(
            "drom_stage_errors_total",
            "Errors observed per stage",
            ["stage"],
        )
        self.images_downloaded_total = Counter(
            "drom_images_downloaded_total",
            "Images downloaded successfully",
        )
        self.enabled = True

    def maybe_start_server(self, port: int, logger: logging.Logger) -> None:
        if port <= 0:
            return

        if self._server_started:
            return

        try:
            from prometheus_client import start_http_server
        except ImportError:
            logger.warning(
                "prometheus_client not installed; metrics endpoint disabled",
                extra={"port": port},
            )
            return

        start_http_server(port)
        self._server_started = True
        logger.info("Prometheus metrics server started", extra={"port": port})

    def record_request(
        self,
        host: str,
        status: int,
        latency_seconds: float,
        from_cache: bool = False,
    ) -> None:
        self.requests_total.labels(host=host, status=str(status)).inc()
        self.request_latency_seconds.labels(host=host).observe(latency_seconds)
        if from_cache:
            self.cache_hits_total.inc()

    def record_stage_rows(self, stage: str, count: int) -> None:
        if count > 0:
            self.stage_rows_total.labels(stage=stage).inc(count)

    def record_stage_error(self, stage: str) -> None:
        self.stage_errors_total.labels(stage=stage).inc()

    def record_image_download(self) -> None:
        self.images_downloaded_total.inc()
