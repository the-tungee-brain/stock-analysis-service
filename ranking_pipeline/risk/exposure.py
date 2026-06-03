"""Sector and exposure aggregation."""

from __future__ import annotations

import pandas as pd


def sector_breakdown(
    weights: pd.Series,
    sector_by_symbol: dict[str, str],
) -> dict[str, float]:
    """Aggregate weight by sector label."""
    out: dict[str, float] = {}
    for sym, w in weights.items():
        sec = sector_by_symbol.get(sym, "Unknown")
        out[sec] = out.get(sec, 0.0) + float(w)
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def enforce_sector_limits(
    weights: pd.Series,
    sector_by_symbol: dict[str, str],
    max_sector_weight: float,
) -> pd.Series:
    """Cap sector weights; redistribute slack only to under-cap sectors."""
    if weights.empty or not sector_by_symbol:
        return weights
    w = weights.copy().astype("float64")
    for _ in range(25):
        sectors = sector_breakdown(w, sector_by_symbol)
        over = [s for s, total in sectors.items() if total > max_sector_weight + 1e-9]
        if not over:
            break
        capped_syms: list[str] = []
        for sec in over:
            syms = [s for s in w.index if sector_by_symbol.get(s, "Unknown") == sec]
            sec_total = w[syms].sum()
            if sec_total <= 0:
                continue
            w[syms] = w[syms] * (max_sector_weight / sec_total)
            capped_syms.extend(syms)
        capped_set = set(capped_syms)
        under_syms = [s for s in w.index if s not in capped_set]
        reserved = w[list(capped_set)].sum() if capped_set else 0.0
        slack = 1.0 - reserved
        if under_syms and slack > 0 and w[under_syms].sum() > 0:
            w[under_syms] = w[under_syms] / w[under_syms].sum() * slack
        elif w.sum() > 0:
            w = w / w.sum()
    return w[w > 1e-8]
