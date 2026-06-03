"""Portfolio beta vs benchmark."""

from __future__ import annotations

import pandas as pd


def symbol_betas(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """OLS beta per symbol vs benchmark daily returns."""
    spy = benchmark_returns.reindex(returns.index).astype("float64")
    spy_var = spy.var()
    betas: dict[str, float] = {}
    for col in returns.columns:
        sym_ret = returns[col].astype("float64")
        aligned = pd.concat([sym_ret, spy], axis=1).dropna()
        if len(aligned) < 10 or spy_var <= 0:
            betas[col] = 1.0
            continue
        cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        betas[col] = float(cov / spy_var)
    return pd.Series(betas, dtype="float64")


def portfolio_beta(weights: pd.Series, betas: pd.Series) -> float:
    """Weighted sum of symbol betas."""
    aligned = betas.reindex(weights.index).fillna(1.0)
    return float((weights * aligned).sum())


def enforce_beta_constraint(
    weights: pd.Series,
    betas: pd.Series,
    *,
    max_beta: float = 1.0,
    beta_neutral: bool = False,
    target_beta: float = 0.0,
) -> pd.Series:
    """
    Iteratively trim high-beta names until portfolio beta <= max_beta.

    If ``beta_neutral``, target is ``target_beta`` (typically 0).
    """
    if weights.empty:
        return weights
    goal = target_beta if beta_neutral else max_beta
    w = weights.copy().astype("float64")
    aligned = betas.reindex(w.index).fillna(1.0)

    for _ in range(100):
        pb = portfolio_beta(w, aligned)
        limit = goal if beta_neutral else max_beta
        if beta_neutral:
            if abs(pb - target_beta) <= 0.05:
                break
        elif pb <= max_beta:
            break
        contrib = (w * aligned).astype("float64")
        if contrib.sum() <= 0:
            break
        trim_sym = contrib.idxmax()
        w[trim_sym] *= 0.92
        total = w.sum()
        if total <= 0:
            break
        w = w / total

    return w[w > 1e-8]
