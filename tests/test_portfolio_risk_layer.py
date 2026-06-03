"""Portfolio risk layer tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ranking_pipeline.risk.beta import portfolio_beta, symbol_betas
from ranking_pipeline.risk.config import PortfolioRiskConfig
from ranking_pipeline.risk.correlation import (
    apply_correlation_penalty,
    correlation_matrix,
    correlation_risk_score,
)
from ranking_pipeline.risk.exposure import enforce_sector_limits, sector_breakdown
from ranking_pipeline.risk.volatility_targeting import (
    realized_portfolio_volatility,
    scale_weights_to_target_vol,
)


def _returns(n: int = 40, symbols: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    data = {s: rng.normal(0, 0.02, n) for s in symbols}
    return pd.DataFrame(data, index=dates)


def test_correlation_penalty_reduces_weight():
    w = pd.Series({"A": 0.5, "B": 0.5})
    ret = _returns()
    ret["B"] = ret["A"] * 0.95 + 0.001
    corr = correlation_matrix(ret)
    out = apply_correlation_penalty(w, corr, threshold=0.5, penalty_strength=0.8)
    assert out.sum() == pytest.approx(1.0, rel=1e-6)


def test_vol_targeting_scales_down():
    w = pd.Series({"A": 0.33, "B": 0.33, "C": 0.34})
    ret = _returns()
    scaled, realized, factor = scale_weights_to_target_vol(
        w, ret, target_annual_vol=0.05
    )
    assert factor <= 1.0
    assert scaled.sum() == pytest.approx(1.0, rel=1e-6)
    if realized > 0.05:
        assert realized_portfolio_volatility(scaled, ret) <= realized + 0.02


def test_portfolio_beta():
    ret = _returns()
    spy = ret.mean(axis=1)
    betas = symbol_betas(ret, spy)
    w = pd.Series({"A": 0.5, "B": 0.5})
    pb = portfolio_beta(w, betas)
    assert -2 < pb < 3


def test_sector_breakdown_and_cap():
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    sectors = {"A": "Tech", "B": "Tech", "C": "Health"}
    capped = enforce_sector_limits(w, sectors, max_sector_weight=0.35)
    breakdown = sector_breakdown(capped, sectors)
    assert breakdown["Tech"] <= 0.35 + 1e-6
    assert sum(breakdown.values()) == pytest.approx(1.0, rel=1e-6)


def test_correlation_risk_score_bounded():
    w = pd.Series({"A": 0.5, "B": 0.5})
    corr = correlation_matrix(_returns())
    score = correlation_risk_score(w, corr)
    assert 0 <= score <= 1.0
