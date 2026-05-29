from app.http.etag import json_weak_etag, normalize_if_none_match


def test_json_weak_etag_is_stable():
    payload = {"symbol": "AAPL", "price": 200.0}
    assert json_weak_etag(payload) == json_weak_etag(payload)


def test_normalize_if_none_match_strips_quotes():
    assert normalize_if_none_match('"abc123"') == "abc123"
    assert normalize_if_none_match("abc123") == "abc123"
