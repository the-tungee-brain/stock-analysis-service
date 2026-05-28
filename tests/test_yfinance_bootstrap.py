from unittest.mock import MagicMock, patch

from peewee import IntegrityError

from app.adapters.market.yfinance_bootstrap import configure_yfinance


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
