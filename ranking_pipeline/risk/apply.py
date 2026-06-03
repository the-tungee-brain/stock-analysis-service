"""Apply full portfolio risk pipeline to constructed weights."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ranking_pipeline.risk.beta import (
    enforce_beta_constraint,
    portfolio_beta,
    symbol_betas,
)
from ranking_pipeline.risk.config import PortfolioRiskConfig, default_risk_config
from ranking_pipeline.risk.correlation import (
    apply_correlation_penalty,
    correlation_matrix,
    correlation_risk_score,
)
from ranking_pipeline.risk.exposure import enforce_sector_limits, sector_breakdown
from ranking_pipeline.risk.returns_data import load_daily_returns
from ranking_pipeline.risk.volatility_targeting import (
    realized_portfolio_volatility,
    scale_weights_to_target_vol,
)


@dataclass(frozen=True)
class PortfolioRiskResult:
    weights: dict[str, float]
    weights_before_risk: dict[str, float]
    portfolio_beta: float
    portfolio_volatility: float
    realized_vol_before_scaling: float
    vol_scale_factor: float
    target_volatility: float
    correlation_risk_score: float
    sector_breakdown: dict[str, float]
    symbol_betas: dict[str, float]


def apply_portfolio_risk(
    weights: dict[str, float] | pd.Series,
    as_of: pd.Timestamp,
    *,
    sector_by_symbol: dict[str, str] | None = None,
    config: PortfolioRiskConfig | None = None,
) -> PortfolioRiskResult:
    """
    Transform constructed portfolio weights (downstream only).

    Order: correlation penalty → sector caps → beta → vol targeting.
    """
    cfg = config or default_risk_config()
    w0 = pd.Series(weights, dtype="float64")
    w0 = w0[w0 > cfg.min_weight_after_risk]
    if w0.empty:
        return _empty_result(cfg)

    total = w0.sum()
    w0 = w0 / total if total > 0 else w0

    symbols = list(w0.index)
    returns, spy_ret = load_daily_returns(
        symbols,
        as_of,
        benchmark_symbol=cfg.benchmark_symbol,
        lookback_days=cfg.return_lookback_days,
    )

    corr = correlation_matrix(returns)
    w1 = apply_correlation_penalty(
        w0,
        corr,
        threshold=cfg.correlation_threshold,
        penalty_strength=cfg.correlation_penalty_strength,
    )

    sectors = sector_by_symbol or {}
    if sectors:
        w1 = enforce_sector_limits(w1, sectors, cfg.max_sector_weight)

    betas = symbol_betas(returns, spy_ret) if not returns.empty else pd.Series(dtype="float64")
    w2 = enforce_beta_constraint(
        w1,
        betas,
        max_beta=cfg.max_portfolio_beta,
        beta_neutral=cfg.beta_neutral,
        target_beta=cfg.target_beta,
    )

    w3, realized_pre, vol_factor = scale_weights_to_target_vol(
        w2,
        returns,
        target_annual_vol=cfg.target_annual_volatility,
    )

    total = w3.sum()
    w3 = w3 / total if total > 0 else w3

    corr_score = correlation_risk_score(w3, corr) if not corr.empty else 0.0
    pb = portfolio_beta(w3, betas) if not betas.empty else 0.0
    port_vol = realized_portfolio_volatility(w3, returns) if not returns.empty else 0.0

    return PortfolioRiskResult(
        weights=w3.to_dict(),
        weights_before_risk=w0.to_dict(),
        portfolio_beta=pb,
        portfolio_volatility=port_vol,
        realized_vol_before_scaling=realized_pre,
        vol_scale_factor=vol_factor,
        target_volatility=cfg.target_annual_volatility,
        correlation_risk_score=corr_score,
        sector_breakdown=sector_breakdown(w3, sectors) if sectors else {},
        symbol_betas=betas.to_dict(),
    )


def _empty_result(cfg: PortfolioRiskConfig) -> PortfolioRiskResult:
    return PortfolioRiskResult(
        weights={},
        weights_before_risk={},
        portfolio_beta=0.0,
        portfolio_volatility=0.0,
        realized_vol_before_scaling=0.0,
        vol_scale_factor=1.0,
        target_volatility=cfg.target_annual_volatility,
        correlation_risk_score=0.0,
        sector_breakdown={},
        symbol_betas={},
    )
