"""Download US equity symbols from NASDAQ Trader symbol directories."""

from __future__ import annotations

import io
from urllib.request import urlopen

import pandas as pd

NASDAQ_LISTED_URL = "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


def _read_pipe_table(url: str) -> pd.DataFrame:
    with urlopen(url, timeout=60) as resp:  # noqa: S310
        text = resp.read().decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line and not line.startswith("File Creation")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|")


def fetch_nasdaq_symbols() -> list[str]:
    """NASDAQ-listed common stocks (excludes test issues and ETFs when flagged)."""
    df = _read_pipe_table(NASDAQ_LISTED_URL)
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] != "Y"]
    if "ETF" in df.columns:
        df = df[df["ETF"] != "Y"]
    symbols = df["Symbol"].astype(str).str.strip().str.upper()
    return _clean_symbols(symbols.tolist())


def fetch_other_exchange_symbols() -> list[str]:
    """NYSE / AMEX / ARCA symbols from otherlisted.txt."""
    df = _read_pipe_table(OTHER_LISTED_URL)
    # ACT Symbol is the tradable ticker on other exchanges
    col = "ACT Symbol" if "ACT Symbol" in df.columns else "NASDAQ Symbol"
    if "Test Issue" in df.columns:
        df = df[df["Test Issue"] != "Y"]
    if "ETF" in df.columns:
        df = df[df["ETF"] != "Y"]
    symbols = df[col].astype(str).str.strip().str.upper()
    return _clean_symbols(symbols.tolist())


def fetch_all_us_equity_symbols() -> list[str]:
    """Union of NASDAQ and other US listing files (no hardcoded ticker list)."""
    combined = set(fetch_nasdaq_symbols()) | set(fetch_other_exchange_symbols())
    return sorted(combined)


def _clean_symbols(symbols: list[str]) -> list[str]:
    out: list[str] = []
    for sym in symbols:
        if not sym or sym == "NAN":
            continue
        if sym.endswith("W") and len(sym) > 4:
            continue
        if "." in sym or "$" in sym or "^" in sym:
            continue
        if len(sym) > 5:
            continue
        out.append(sym)
    return out
