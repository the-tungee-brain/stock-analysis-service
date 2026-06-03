"""Portfolio weight constraints."""

from __future__ import annotations

import pandas as pd

from ranking_pipeline.portfolio.config import PortfolioConstraints


def filter_by_liquidity(
    weights: pd.Series,
    adv_by_symbol: dict[str, float | None],
    min_adv: float,
) -> pd.Series:
    """Drop symbols below ADV threshold and renormalize."""
    kept = {
        sym: w
        for sym, w in weights.items()
        if adv_by_symbol.get(sym, 0) is not None and adv_by_symbol.get(sym, 0) >= min_adv
    }
    if not kept:
        return weights
    out = pd.Series(kept, dtype="float64")
    total = out.sum()
    return out / total if total > 0 else out


def cap_position_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    """Iteratively cap weights and renormalize (feasible when N * max_weight >= 1)."""
    if weights.empty:
        return weights
    n = len(weights)
    if n * max_weight < 1.0:
        return pd.Series(1.0 / n, index=weights.index, dtype="float64")

    w = weights.copy().astype("float64")
    for _ in range(25):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        under = ~over
        if under.any() and w[under].sum() > 0:
            w[under] = w[under] + excess * (w[under] / w[under].sum())
        else:
            w[:] = 1.0 / n
            break
    total = w.sum()
    return w / total if total > 0 else w


def constrain_turnover(
    target: pd.Series,
    previous: pd.Series,
    max_turnover: float,
) -> pd.Series:
    """
    Scale move toward target so L1 turnover <= ``max_turnover``.

    Turnover = sum_i |w_i - w_prev_i| (one-way total).
    """
    if previous.empty:
        return target
    all_symbols = sorted(set(target.index) | set(previous.index))
    w_prev = previous.reindex(all_symbols).fillna(0.0)
    w_tgt = target.reindex(all_symbols).fillna(0.0)
    delta = w_tgt - w_prev
    turnover = float(delta.abs().sum())
    if turnover <= max_turnover or turnover <= 0:
        return w_tgt[w_tgt > 0]
    scale = max_turnover / turnover
    w_new = w_prev + scale * delta
    w_new = w_new.clip(lower=0.0)
    total = w_new.sum()
    if total > 0:
        w_new = w_new / total
    return w_new[w_new > 1e-8]


def apply_sector_neutral(
    weights: pd.Series,
    sector_by_symbol: dict[str, str],
    max_sector_weight: float,
) -> pd.Series:
    """Cap aggregate weight per sector and renormalize."""
    if weights.empty or not sector_by_symbol:
        return weights
    w = weights.copy()
    for _ in range(20):
        sector_sums: dict[str, float] = {}
        for sym, wt in w.items():
            sec = sector_by_symbol.get(sym, "Unknown")
            sector_sums[sec] = sector_sums.get(sec, 0.0) + wt
        over_sectors = {s for s, total in sector_sums.items() if total > max_sector_weight}
        if not over_sectors:
            break
        for sec in over_sectors:
            sec_syms = [s for s in w.index if sector_by_symbol.get(s, "Unknown") == sec]
            if not sec_syms:
                continue
            sec_total = w[sec_syms].sum()
            if sec_total <= 0:
                continue
            scale = max_sector_weight / sec_total
            w[sec_syms] = w[sec_syms] * scale
        total = w.sum()
        if total > 0:
            w = w / total
    return w


def apply_all_constraints(
    target: pd.Series,
    previous: pd.Series,
    *,
    config: PortfolioConstraints,
    adv_by_symbol: dict[str, float | None] | None = None,
    sector_by_symbol: dict[str, str] | None = None,
) -> pd.Series:
    w = target.copy()
    if adv_by_symbol:
        w = filter_by_liquidity(w, adv_by_symbol, config.min_adv_dollars)
    w = cap_position_weights(w, config.max_position_weight)
    if config.sector_neutral and sector_by_symbol:
        w = apply_sector_neutral(w, sector_by_symbol, config.max_sector_weight)
    w = constrain_turnover(w, previous, config.max_daily_turnover)
    total = w.sum()
    return w / total if total > 0 else w
