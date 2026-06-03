"""Product portfolio v1 API mapping tests."""

from __future__ import annotations

from app.services.product_api_service import _parse_top_contributors_v1


def test_parse_top_contributors_v1_accepts_symbol_strings():
    raw = [
        {
            "symbol": "BRUN",
            "weight": 0.12,
            "expected_excess_return": 0.021,
            "contribution": 0.00252,
        },
        {
            "symbol": "PIII",
            "weight": 0.11,
            "expected_excess_return": 0.019,
            "contribution": 0.00209,
        },
    ]
    parsed = _parse_top_contributors_v1(raw)
    assert len(parsed) == 2
    assert parsed[0].symbol == "BRUN"
    assert parsed[0].contribution == 0.00252


def test_parse_top_contributors_v1_skips_invalid_rows():
    parsed = _parse_top_contributors_v1([{"symbol": "AAPL"}, {"no_symbol": 1}])
    assert parsed == []


def test_parse_top_contributors_v1_accepts_partial_metrics():
    parsed = _parse_top_contributors_v1(
        [{"symbol": "MSFT", "contribution": 0.01, "weight": 0.08}]
    )
    assert len(parsed) == 1
    assert parsed[0].symbol == "MSFT"
    assert parsed[0].contribution == 0.01
