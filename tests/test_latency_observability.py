import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.latency_observability import (
    LatencyLoggingMiddleware,
    observe_dependency,
    record_dependency_latency,
    set_latency_attribute,
)


def _latency_records(caplog):
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "app.latency"
    ]


def test_latency_middleware_logs_route_totals_and_dependency_counters(caplog):
    app = FastAPI()
    app.add_middleware(LatencyLoggingMiddleware)

    @app.get("/items/{item_id}")
    def read_item(item_id: str):
        set_latency_attribute("symbol_count", 2)
        record_dependency_latency("redis", 1.25, cache_status="hit")
        with observe_dependency("schwab"):
            time.sleep(0.001)
        return {"id": item_id}

    with caplog.at_level(logging.INFO, logger="app.latency"):
        response = TestClient(app).get(
            "/items/secret-item",
            headers={"Authorization": "Bearer should-not-log"},
        )

    assert response.status_code == 200
    records = _latency_records(caplog)
    assert len(records) == 1
    payload = records[0]
    assert payload["event"] == "request_latency"
    assert payload["method"] == "GET"
    assert payload["route"] == "/items/{item_id}"
    assert payload["status"] == 200
    assert payload["total_ms"] >= 0
    assert payload["symbol_count"] == 2
    assert payload["dependencies"]["redis"] == {
        "count": 1,
        "total_ms": 1.25,
        "cache_hits": 1,
    }
    assert payload["dependencies"]["schwab"]["count"] == 1
    assert payload["dependencies"]["schwab"]["total_ms"] >= 0

    raw_log = caplog.text
    assert "should-not-log" not in raw_log
    assert "secret-item" not in raw_log


def test_latency_middleware_logs_errors_without_request_body(caplog):
    app = FastAPI()
    app.add_middleware(LatencyLoggingMiddleware)

    @app.post("/fail")
    async def fail(request: Request):
        await request.json()
        with observe_dependency("openai"):
            raise RuntimeError("model failed")

    with caplog.at_level(logging.INFO, logger="app.latency"):
        response = TestClient(app, raise_server_exceptions=False).post(
            "/fail",
            json={"prompt": "private prompt text"},
        )

    assert response.status_code == 500
    records = _latency_records(caplog)
    assert len(records) == 1
    payload = records[0]
    assert payload["route"] == "/fail"
    assert payload["status"] == 500
    assert payload["dependencies"]["openai"]["errors"] == 1
    assert "private prompt text" not in caplog.text
