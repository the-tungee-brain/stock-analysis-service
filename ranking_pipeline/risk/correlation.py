"""Pairwise correlation and weight penalties."""

from __future__ import annotations

import pandas as pd


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample correlation matrix of daily returns."""
    if returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()
    return returns.corr()


def correlation_risk_score(weights: pd.Series, corr: pd.DataFrame) -> float:
    """
    Weighted average pairwise correlation (0 = diversifier, 1 = concentrated risk).

    Score = sum_{i<j} w_i * w_j * |rho_ij| / sum_{i<j} w_i * w_j (normalized).
    """
    symbols = [s for s in weights.index if s in corr.columns and weights[s] > 0]
    if len(symbols) < 2:
        return 0.0
    num = 0.0
    den = 0.0
    for i, si in enumerate(symbols):
        for sj in symbols[i + 1 :]:
            wi = float(weights[si])
            wj = float(weights[sj])
            rho = abs(float(corr.loc[si, sj])) if si in corr.index and sj in corr.columns else 0.0
            num += wi * wj * rho
            den += wi * wj
    return float(num / den) if den > 0 else 0.0


def apply_correlation_penalty(
    weights: pd.Series,
    corr: pd.DataFrame,
    *,
    threshold: float = 0.65,
    penalty_strength: float = 0.5,
) -> pd.Series:
    """
    Reduce weights on highly correlated pairs.

    For each pair with |rho| > threshold, scale both weights down proportionally.
    """
    if weights.empty or corr.empty:
        return weights
    w = weights.copy().astype("float64")
    symbols = list(w.index)
    for i, si in enumerate(symbols):
        if si not in corr.columns:
            continue
        for sj in symbols[i + 1 :]:
            if sj not in corr.columns:
                continue
            rho = float(corr.loc[si, sj])
            if abs(rho) <= threshold:
                continue
            excess = (abs(rho) - threshold) / max(1.0 - threshold, 1e-6)
            factor = max(0.1, 1.0 - penalty_strength * excess)
            w[si] *= factor
            w[sj] *= factor
    total = w.sum()
    return w / total if total > 0 else w
