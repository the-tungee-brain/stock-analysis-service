"""Configure yfinance before any Ticker fetch (multi-worker / async safe)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

_configured = False
_config_lock = threading.Lock()

# Serialize Yahoo cookie/cache writes within a process (yfinance peewee race).
_yfinance_fetch_lock = threading.Lock()


def yfinance_fetch_lock() -> threading.Lock:
    return _yfinance_fetch_lock


def configure_yfinance() -> None:
    global _configured
    with _config_lock:
        if _configured:
            return

        from yfinance.cache import set_cache_location

        cache_root = os.environ.get("YFINANCE_CACHE_DIR")
        if cache_root:
            cache_dir = Path(cache_root)
        else:
            # Isolate SQLite cookie DB per process (gunicorn workers share UID otherwise).
            cache_dir = Path("/tmp") / "py-yfinance" / f"worker-{os.getpid()}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        set_cache_location(str(cache_dir))

        _patch_cookie_cache_store()
        _configured = True


def _patch_cookie_cache_store() -> None:
    from peewee import IntegrityError
    from yfinance.cache import _CookieCache

    original_store = _CookieCache.store

    def store_tolerant(self, strategy, cookie):  # noqa: ANN001
        try:
            original_store(self, strategy, cookie)
        except IntegrityError:
            # Concurrent requests can both delete+insert the same strategy row.
            pass

    _CookieCache.store = store_tolerant  # type: ignore[method-assign]
