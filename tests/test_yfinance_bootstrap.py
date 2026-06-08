from unittest.mock import MagicMock, patch

import logging

from peewee import IntegrityError

from app.adapters.market.yfinance_bootstrap import (
    configure_yfinance,
    format_yahoo_finance_error,
    is_yahoo_permanent_unavailable,
)


def test_format_yahoo_finance_error_strips_html_body():
    exc = Exception(
        'HTTP Error 400: <!doctype html><html><head><title>Yahoo!</title></head></html>'
    )
    assert format_yahoo_finance_error(exc) == "Yahoo Finance HTTP 400"


def test_format_yahoo_finance_error_truncates_long_plain_message():
    exc = Exception("x" * 300)
    formatted = format_yahoo_finance_error(exc)
    assert len(formatted) <= 201
    assert formatted.endswith("…")


def test_format_yahoo_no_fundamentals_as_expected_unavailable():
    exc = Exception("HTTP Error 404: No fundamentals data found for symbol: BEAGR")

    assert format_yahoo_finance_error(exc) == (
        "Yahoo Finance fundamentals unavailable"
    )
    assert is_yahoo_permanent_unavailable(exc) is True


def test_configure_yfinance_suppresses_expected_no_fundamentals_logs(caplog):
    configure_yfinance()
    logger = logging.getLogger("yfinance.scrapers.quote")

    with caplog.at_level("ERROR", logger="yfinance.scrapers.quote"):
        logger.error(
            "HTTP Error 404: {\"quoteSummary\":{\"result\":null,"
            "\"error\":{\"description\":\"No fundamentals data found for symbol: SCHD\"}}}"
        )

    assert "No fundamentals data found" not in caplog.text


def test_configure_yfinance_tolerates_cookie_cache_integrity_error():
    configure_yfinance()

    from yfinance.cache import _CookieCache

    cache = _CookieCache()
    cache.dummy = False
    cache.initialised = 1
    cache.db = MagicMock()
    cache.get_db = MagicMock(return_value=cache.db)

    with patch("yfinance.cache._CookieSchema") as schema:
        schema.delete.return_value.where.return_value.execute = MagicMock()
        schema.insert.return_value.execute = MagicMock(
            side_effect=IntegrityError("UNIQUE constraint failed")
        )
        with patch.object(cache.db, "atomic"):
            cache.store("curlCffi", b"cookie-bytes")
