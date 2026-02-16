from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode, urlparse

from diskcache import Cache
import httpx

from car_listing_visual_verification.data.drom_metrics import DromMetrics
from car_listing_visual_verification.data.drom_types import SourceConfig
from car_listing_visual_verification.data.drom_utils import (
    jitter_sleep_seconds,
    stable_hash,
    utc_now_iso,
)

RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class RetryableStatusError(RuntimeError):
    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(f"retryable_status={status_code} {message}".strip())
        self.status_code = status_code


class CircuitOpenError(RuntimeError):
    pass


@dataclass(slots=True)
class HttpResult:
    status_code: int
    url: str
    headers: dict[str, str]
    body: bytes
    fetched_at: str
    from_cache: bool = False

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


@dataclass(slots=True)
class DownloadResult:
    url: str
    path: Path
    status_code: int
    bytes_written: int
    fetched_at: str
    skipped: bool = False
    error: str | None = None


class AsyncTokenBucket:
    def __init__(self, qps: float, burst: float) -> None:
        self._qps = max(qps, 0.01)
        self._capacity = max(burst, 1.0)
        self._tokens = self._capacity
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = max(0.0, now - self._updated_at)
                self._updated_at = now
                self._tokens = min(self._capacity, self._tokens + elapsed * self._qps)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_seconds = (1.0 - self._tokens) / self._qps
            await asyncio.sleep(wait_seconds)


class CircuitBreakerRegistry:
    def __init__(self, failures: int, cooldown_seconds: float) -> None:
        self._failure_threshold = max(failures, 1)
        self._cooldown_seconds = max(cooldown_seconds, 1.0)
        self._state: dict[str, dict[str, float]] = {}
        self._lock = asyncio.Lock()

    async def before_request(self, host: str) -> None:
        async with self._lock:
            state = self._state.get(host)
            if not state:
                return
            opened_until = state.get("opened_until", 0.0)
            if opened_until > time.monotonic():
                raise CircuitOpenError(f"circuit_open host={host} opened_until={opened_until}")

    async def on_success(self, host: str) -> None:
        async with self._lock:
            self._state[host] = {"consecutive_failures": 0.0, "opened_until": 0.0}

    async def on_failure(self, host: str) -> None:
        async with self._lock:
            state = self._state.setdefault(
                host, {"consecutive_failures": 0.0, "opened_until": 0.0}
            )
            state["consecutive_failures"] += 1.0
            if state["consecutive_failures"] >= self._failure_threshold:
                state["opened_until"] = time.monotonic() + self._cooldown_seconds
                state["consecutive_failures"] = 0.0


class DromHttpClient:
    def __init__(
        self,
        source: SourceConfig,
        cache_dir: Path,
        logger: logging.Logger,
        metrics: DromMetrics,
    ) -> None:
        self.source = source
        self.cache = Cache(directory=cache_dir.as_posix())
        self.logger = logger
        self.metrics = metrics

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(source.timeout_seconds),
            headers={
                "User-Agent": source.user_agent,
                "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
                **source.headers,
            },
            follow_redirects=True,
        )

        self._rate_limiters: dict[str, AsyncTokenBucket] = {}
        self._rate_limiter_lock = asyncio.Lock()
        self._circuit_breaker = CircuitBreakerRegistry(
            failures=source.circuit_breaker_failures,
            cooldown_seconds=source.circuit_breaker_cooldown_seconds,
        )

    async def __aenter__(self) -> "DromHttpClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()
        self.cache.close()

    async def get_text(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> HttpResult:
        return await self._request_bytes(
            method="GET",
            url=url,
            params=params,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )

    async def get_bytes(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = False,
        force_refresh: bool = False,
    ) -> HttpResult:
        return await self._request_bytes(
            method="GET",
            url=url,
            params=params,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )

    async def download_file(
        self,
        url: str,
        target: Path,
        overwrite: bool = False,
    ) -> DownloadResult:
        if target.exists() and not overwrite:
            return DownloadResult(
                url=url,
                path=target,
                status_code=200,
                bytes_written=target.stat().st_size,
                fetched_at=utc_now_iso(),
                skipped=True,
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_suffix(f"{target.suffix}.part")

        host = urlparse(url).netloc
        for attempt in range(1, self.source.retries + 1):
            started = time.perf_counter()
            try:
                await self._await_capacity(host)
                async with self._client.stream("GET", url) as response:
                    status = response.status_code
                    if status in RETRYABLE_STATUSES:
                        raise RetryableStatusError(status_code=status)

                    if status >= 400:
                        elapsed = time.perf_counter() - started
                        self.metrics.record_request(
                            host=host, status=status, latency_seconds=elapsed
                        )
                        await self._circuit_breaker.on_success(host)
                        return DownloadResult(
                            url=url,
                            path=target,
                            status_code=status,
                            bytes_written=0,
                            fetched_at=utc_now_iso(),
                            skipped=False,
                            error=f"http_status={status}",
                        )

                    bytes_written = 0
                    with temp_path.open("wb") as fp:
                        async for chunk in response.aiter_bytes():
                            fp.write(chunk)
                            bytes_written += len(chunk)

                    temp_path.replace(target)

                    elapsed = time.perf_counter() - started
                    self.metrics.record_request(host=host, status=status, latency_seconds=elapsed)
                    await self._circuit_breaker.on_success(host)
                    return DownloadResult(
                        url=url,
                        path=target,
                        status_code=status,
                        bytes_written=bytes_written,
                        fetched_at=utc_now_iso(),
                    )
            except Exception as exc:  # noqa: BLE001
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

                retryable = self._is_retryable_exception(exc)
                status_code = getattr(exc, "status_code", 0)
                elapsed = time.perf_counter() - started
                self.metrics.record_request(host=host, status=status_code, latency_seconds=elapsed)

                if retryable:
                    await self._circuit_breaker.on_failure(host)
                else:
                    await self._circuit_breaker.on_success(host)

                if not retryable or attempt >= self.source.retries:
                    return DownloadResult(
                        url=url,
                        path=target,
                        status_code=status_code,
                        bytes_written=0,
                        fetched_at=utc_now_iso(),
                        error=str(exc),
                    )

                sleep_seconds = jitter_sleep_seconds(
                    base_seconds=self.source.retry_backoff_base_seconds,
                    cap_seconds=self.source.retry_backoff_max_seconds,
                    attempt=attempt,
                )
                self.logger.warning(
                    "Retrying image download",
                    extra={
                        "url": url,
                        "attempt": attempt,
                        "sleep_seconds": round(sleep_seconds, 3),
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(sleep_seconds)

        return DownloadResult(
            url=url,
            path=target,
            status_code=0,
            bytes_written=0,
            fetched_at=utc_now_iso(),
            error="unexpected_retry_exit",
        )

    async def _request_bytes(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        use_cache: bool,
        force_refresh: bool,
    ) -> HttpResult:
        cache_key = self._cache_key(method=method, url=url, params=params)

        if use_cache and not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                result = HttpResult(
                    status_code=int(cached["status_code"]),
                    url=str(cached["url"]),
                    headers=dict(cached["headers"]),
                    body=bytes(cached["body"]),
                    fetched_at=str(cached["fetched_at"]),
                    from_cache=True,
                )
                host = urlparse(url).netloc
                self.metrics.record_request(
                    host=host,
                    status=result.status_code,
                    latency_seconds=0.0,
                    from_cache=True,
                )
                return result

        host = urlparse(url).netloc
        for attempt in range(1, self.source.retries + 1):
            started = time.perf_counter()
            try:
                await self._await_capacity(host)
                response = await self._client.request(method=method, url=url, params=params)
                status = response.status_code
                if status in RETRYABLE_STATUSES:
                    raise RetryableStatusError(status_code=status)

                elapsed = time.perf_counter() - started
                self.metrics.record_request(host=host, status=status, latency_seconds=elapsed)
                await self._circuit_breaker.on_success(host)

                result = HttpResult(
                    status_code=status,
                    url=str(response.url),
                    headers={k: v for k, v in response.headers.items()},
                    body=response.content,
                    fetched_at=utc_now_iso(),
                    from_cache=False,
                )

                if use_cache and status < 400:
                    self.cache.set(
                        cache_key,
                        {
                            "status_code": status,
                            "url": str(response.url),
                            "headers": dict(response.headers),
                            "body": response.content,
                            "fetched_at": result.fetched_at,
                        },
                        expire=self.source.cache_ttl_seconds,
                    )

                return result
            except Exception as exc:  # noqa: BLE001
                retryable = self._is_retryable_exception(exc)
                status_code = getattr(exc, "status_code", 0)
                elapsed = time.perf_counter() - started
                self.metrics.record_request(host=host, status=status_code, latency_seconds=elapsed)

                if retryable:
                    await self._circuit_breaker.on_failure(host)
                else:
                    await self._circuit_breaker.on_success(host)

                if not retryable or attempt >= self.source.retries:
                    raise

                sleep_seconds = jitter_sleep_seconds(
                    base_seconds=self.source.retry_backoff_base_seconds,
                    cap_seconds=self.source.retry_backoff_max_seconds,
                    attempt=attempt,
                )
                self.logger.warning(
                    "Retrying request",
                    extra={
                        "url": url,
                        "attempt": attempt,
                        "sleep_seconds": round(sleep_seconds, 3),
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(sleep_seconds)

        raise RuntimeError(f"request failed after retries: {url}")

    async def _await_capacity(self, host: str) -> None:
        await self._circuit_breaker.before_request(host)
        limiter = await self._get_limiter(host)
        await limiter.acquire()

    async def _get_limiter(self, host: str) -> AsyncTokenBucket:
        async with self._rate_limiter_lock:
            limiter = self._rate_limiters.get(host)
            if limiter is None:
                limiter = AsyncTokenBucket(qps=self.source.qps, burst=self.source.burst)
                self._rate_limiters[host] = limiter
            return limiter

    def _cache_key(self, method: str, url: str, params: dict[str, Any] | None) -> str:
        params_key = ""
        if params:
            params_key = urlencode(sorted(params.items()), doseq=True)
        payload = json.dumps(
            {
                "method": method,
                "url": url,
                "params": params_key,
            },
            sort_keys=True,
        )
        return stable_hash(payload)

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        if isinstance(exc, CircuitOpenError):
            return True
        if isinstance(exc, RetryableStatusError):
            return True
        if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError)):
            return True
        return False
