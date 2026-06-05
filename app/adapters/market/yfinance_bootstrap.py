"""Configure yfinance before any Ticker fetch (multi-worker / async safe)."""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from types import TracebackType

_YAHOO_HTTP_ERROR_RE = re.compile(r"^HTTP Error (\d+):\s*", re.IGNORECASE)
_YAHOO_NO_FUNDAMENTALS_RE = re.compile(
    r"no fundamentals data found",
    re.IGNORECASE,
)

_configured = False
_config_lock = threading.Lock()

# Serialize Yahoo cookie/cache writes within a process (yfinance peewee race).
_yfinance_fetch_lock = threading.Lock()
_last_yahoo_fetch_at = 0.0


def _yahoo_min_interval_sec() -> float:
    raw = os.environ.get("YFINANCE_MIN_INTERVAL_SEC", "0.35")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.35


class _ThrottledYFinanceFetchLock:
    """Process-wide lock plus minimum gap between Yahoo requests (rate-limit friendly)."""

    def __enter__(self) -> _ThrottledYFinanceFetchLock:
        global _last_yahoo_fetch_at
        _yfinance_fetch_lock.acquire()
        gap = _yahoo_min_interval_sec()
        if gap > 0 and _last_yahoo_fetch_at > 0:
            wait = gap - (time.monotonic() - _last_yahoo_fetch_at)
            if wait > 0:
                time.sleep(wait)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        global _last_yahoo_fetch_at
        _last_yahoo_fetch_at = time.monotonic()
        _yfinance_fetch_lock.release()


def yfinance_fetch_lock() -> _ThrottledYFinanceFetchLock:
    return _ThrottledYFinanceFetchLock()


def format_yahoo_finance_error(exc: BaseException) -> str:
    """Short, log-safe summary (Yahoo often embeds HTML error pages in exceptions)."""
    raw = str(exc).strip()
    if not raw:
        return type(exc).__name__

    if _YAHOO_NO_FUNDAMENTALS_RE.search(raw):
        return "Yahoo Finance fundamentals unavailable"

    match = _YAHOO_HTTP_ERROR_RE.match(raw)
    if match:
        return f"Yahoo Finance HTTP {match.group(1)}"

    head = raw[:500].lower()
    if head.startswith("<!doctype") or "<html" in head:
        return "Yahoo Finance request rejected"

    first_line = raw.splitlines()[0].strip()
    if len(first_line) > 200:
        return f"{first_line[:200]}…"
    return first_line


def is_yahoo_permanent_unavailable(exc: BaseException) -> bool:
    raw = str(exc).strip()
    if not raw:
        return False
    if _YAHOO_NO_FUNDAMENTALS_RE.search(raw):
        return True
    match = _YAHOO_HTTP_ERROR_RE.match(raw)
    return bool(match and match.group(1) in {"400", "404"})


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
