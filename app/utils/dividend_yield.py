from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

DividendYieldConvention = Literal["decimal_ratio", "percent_points"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizedDividendYield:
    raw_dividend_yield: float
    dividend_yield_pct: float | None
    raw_dividend_yield_source: str
    suspicious: bool = False
    warning: str | None = None


def normalize_dividend_yield_pct(
    value: object,
    *,
    asset_type: str | None = None,
    source: str = "provider",
    convention: DividendYieldConvention | None = None,
) -> NormalizedDividendYield | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(raw) or raw <= 0:
        return None

    kind = (asset_type or "").strip().upper()
    if convention == "decimal_ratio":
        pct = raw * 100.0
    elif convention == "percent_points":
        pct = raw
    elif kind == "ETF":
        if raw < 0.2:
            pct = raw * 100.0
        elif raw >= 1.0:
            pct = raw
        else:
            pct = None
    else:
        pct = raw * 100.0 if raw < 0.1 else raw

    warning = None
    suspicious = False
    if pct is None:
        suspicious = True
        warning = "dividend_yield_convention_ambiguous"
    elif kind not in {"ETF", "FUND", "MUTUAL_FUND"} and pct > 20.0:
        suspicious = True
        warning = "equity_dividend_yield_suspicious_gt_20_pct"

    return NormalizedDividendYield(
        raw_dividend_yield=raw,
        dividend_yield_pct=round(pct, 2) if pct is not None else None,
        raw_dividend_yield_source=source,
        suspicious=suspicious,
        warning=warning,
    )


def dividend_yield_pct_or_none(
    value: object,
    *,
    asset_type: str | None = None,
    source: str = "provider",
    convention: DividendYieldConvention | None = None,
    symbol: str | None = None,
) -> float | None:
    normalized = normalize_dividend_yield_pct(
        value,
        asset_type=asset_type,
        source=source,
        convention=convention,
    )
    if normalized is None:
        return None
    if normalized.suspicious:
        logger.warning(
            "Suspicious dividend yield normalization symbol=%s asset_type=%s "
            "raw=%s normalized_pct=%s source=%s warning=%s",
            symbol,
            asset_type,
            normalized.raw_dividend_yield,
            normalized.dividend_yield_pct,
            normalized.raw_dividend_yield_source,
            normalized.warning,
        )
    return normalized.dividend_yield_pct
