import math

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.http.json_sanitizer import SanitizedJSONResponse, sanitize_json_value


def test_sanitize_json_value_replaces_non_finite_numbers() -> None:
    payload = sanitize_json_value(
        {
            "nan": float("nan"),
            "inf": float("inf"),
            "nested": [1.0, -float("inf")],
        }
    )

    assert payload == {"nan": None, "inf": None, "nested": [1.0, None]}


def test_sanitized_json_response_handles_fastapi_default_route_response() -> None:
    app = FastAPI(default_response_class=SanitizedJSONResponse)

    @app.get("/nan")
    def nan_payload():
        return {"value": math.nan, "items": [1.0, math.inf]}

    response = TestClient(app).get("/nan")

    assert response.status_code == 200
    assert response.json() == {"value": None, "items": [1.0, None]}
