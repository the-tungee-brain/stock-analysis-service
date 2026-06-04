"""Resolve Momentum Breakout persistence backend configuration."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_alert_store_mode() -> str:
    return os.getenv("MB_ALERT_STORE", "oracle").strip().lower()


def resolve_paper_trade_store_mode() -> str:
    return os.getenv("MB_PAPER_TRADE_STORE", "oracle").strip().lower()


def is_production_environment() -> bool:
    env = os.getenv("ENV", os.getenv("APP_ENV", "")).strip().lower()
    if env in {"prod", "production", "live"}:
        return True
    return os.getenv("MB_PRODUCTION", "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_alert_sqlite_path() -> str:
    return os.getenv(
        "MB_ALERT_SQLITE_PATH",
        str(Path("data") / "momentum_breakout_alerts.db"),
    )


def resolve_paper_trade_sqlite_path() -> str:
    return os.getenv(
        "MB_PAPER_TRADE_SQLITE_PATH",
        str(Path("data") / "momentum_breakout_paper_trades.db"),
    )
