"""Portfolio construction + risk layer (does not modify constructor.py)."""

from __future__ import annotations

import pandas as pd

from ranking_pipeline.portfolio.config import PortfolioConfig, default_portfolio_config
from ranking_pipeline.portfolio.constructor import construct_portfolio_from_run
from ranking_pipeline.risk.apply import apply_portfolio_risk
from ranking_pipeline.risk.config import PortfolioRiskConfig, default_risk_config
from ranking_pipeline.risk.persistence import open_risk_persistence


def construct_portfolio_with_risk(
    ranking_run_id: str | None = None,
    *,
    portfolio_config: PortfolioConfig | None = None,
    risk_config: PortfolioRiskConfig | None = None,
    sector_by_symbol: dict[str, str] | None = None,
) -> dict:
    """
    Run existing portfolio construction, then apply risk layer and persist.
    """
    base = construct_portfolio_from_run(
        ranking_run_id,
        portfolio_config=portfolio_config,
        sector_by_symbol=sector_by_symbol,
    )

    rcfg = risk_config or default_risk_config()
    as_of = pd.Timestamp(base["as_of_date"])
    risk = apply_portfolio_risk(
        base["weights"],
        as_of,
        sector_by_symbol=sector_by_symbol,
        config=rcfg,
    )

    rp = open_risk_persistence(rcfg)
    rp.apply_risk_to_portfolio(base["portfolio_id"], risk)

    return {
        **base,
        "weights": risk.weights,
        "weights_before_risk": risk.weights_before_risk,
        "risk_layer": {
            "portfolio_beta": risk.portfolio_beta,
            "portfolio_volatility": risk.portfolio_volatility,
            "target_volatility": risk.target_volatility,
            "correlation_risk_score": risk.correlation_risk_score,
            "sector_breakdown": risk.sector_breakdown,
            "vol_scale_factor": risk.vol_scale_factor,
        },
    }
