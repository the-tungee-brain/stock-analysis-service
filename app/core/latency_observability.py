from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("app.latency")

_request_metrics: ContextVar["RequestLatencyMetrics | None"] = ContextVar(
    "request_latency_metrics",
    default=None,
)


@dataclass
class DependencyTiming:
    count: int = 0
    total_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0

    def as_log_dict(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = {
            "count": self.count,
            "total_ms": round(self.total_ms, 2),
        }
        if self.cache_hits:
            payload["cache_hits"] = self.cache_hits
        if self.cache_misses:
            payload["cache_misses"] = self.cache_misses
        if self.errors:
            payload["errors"] = self.errors
        return payload


@dataclass
class RequestLatencyMetrics:
    route: str
    method: str
    started_at: float = field(default_factory=time.perf_counter)
    dependencies: dict[str, DependencyTiming] = field(default_factory=dict)
    attributes: dict[str, int] = field(default_factory=dict)

    def record_dependency(
        self,
        dependency: str,
        elapsed_ms: float,
        *,
        cache_status: str | None = None,
        error: bool = False,
    ) -> None:
        timing = self.dependencies.setdefault(dependency, DependencyTiming())
        timing.count += 1
        timing.total_ms += elapsed_ms
        if cache_status == "hit":
            timing.cache_hits += 1
        elif cache_status == "miss":
            timing.cache_misses += 1
        if error:
            timing.errors += 1

    def set_attribute(self, key: str, value: int | None) -> None:
        if value is None:
            return
        self.attributes[key] = int(value)

    def log_payload(self, *, status: int, route: str | None = None) -> dict[str, Any]:
        total_ms = (time.perf_counter() - self.started_at) * 1000
        payload: dict[str, Any] = {
            "event": "request_latency",
            "method": self.method,
            "route": route or self.route,
            "status": status,
            "total_ms": round(total_ms, 2),
        }
        if self.dependencies:
            payload["dependencies"] = {
                name: timing.as_log_dict()
                for name, timing in sorted(self.dependencies.items())
            }
        payload.update(self.attributes)
        return payload


class LatencyLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        route = _route_from_scope(scope) or request.url.path
        metrics = RequestLatencyMetrics(route=route, method=request.method)
        token = _request_metrics.set(metrics)
        status_code = 500

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            final_route = _route_from_scope(scope) or route
            _request_metrics.reset(token)
            if not _is_noise_route(final_route):
                logger.info(
                    json.dumps(
                        metrics.log_payload(status=status_code, route=final_route),
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )


def _route_from_scope(scope: Scope) -> str | None:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else None


def _is_noise_route(route: str) -> bool:
    return route in {"/", "/health", "/healthz", "/api/v1/product/health"}


def current_request_metrics() -> RequestLatencyMetrics | None:
    return _request_metrics.get()


def set_latency_attribute(key: str, value: int | None) -> None:
    metrics = current_request_metrics()
    if metrics is not None:
        metrics.set_attribute(key, value)


@contextmanager
def observe_dependency(
    dependency: str,
    *,
    cache_status: str | None = None,
) -> Iterator[None]:
    started_at = time.perf_counter()
    error = False
    try:
        yield
    except Exception:
        error = True
        raise
    finally:
        metrics = current_request_metrics()
        if metrics is not None:
            metrics.record_dependency(
                dependency,
                (time.perf_counter() - started_at) * 1000,
                cache_status=cache_status,
                error=error,
            )


def record_dependency_latency(
    dependency: str,
    elapsed_ms: float,
    *,
    cache_status: str | None = None,
    error: bool = False,
) -> None:
    metrics = current_request_metrics()
    if metrics is not None:
        metrics.record_dependency(
            dependency,
            elapsed_ms,
            cache_status=cache_status,
            error=error,
        )

