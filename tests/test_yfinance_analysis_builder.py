from unittest.mock import MagicMock, patch

import pandas as pd

from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder


def _sample_raw() -> dict:
    return {
        "price_targets": {
            "current": 100.0,
            "low": 90.0,
            "high": 120.0,
            "mean": 110.0,
            "median": 108.0,
        },
        "recommendations_summary": pd.DataFrame(
            [
                {
                    "period": "0m",
                    "strongBuy": 10,
                    "buy": 15,
                    "hold": 5,
                    "sell": 2,
                    "strongSell": 1,
                }
            ]
        ),
        "recommendations": pd.DataFrame(
            [
                {
                    "period": "-2m",
                    "strongBuy": 8,
                    "buy": 12,
                    "hold": 8,
                    "sell": 3,
                    "strongSell": 2,
                },
                {
                    "period": "0m",
                    "strongBuy": 12,
                    "buy": 18,
                    "hold": 4,
                    "sell": 2,
                    "strongSell": 1,
                },
            ]
        ),
        "earnings_estimate": {
            "0q": {
                "numberOfAnalysts": 20,
                "avg": 1.10,
                "low": 1.0,
                "high": 1.2,
                "growth": 0.15,
            },
            "+1q": {
                "numberOfAnalysts": 20,
                "avg": 1.25,
                "low": 1.1,
                "high": 1.4,
                "yearAgoEps": 1.0,
                "growth": 0.25,
            },
            "+1y": {
                "numberOfAnalysts": 22,
                "avg": 5.5,
                "low": 5.0,
                "high": 6.0,
                "growth": 0.18,
            },
        },
        "revenue_estimate": {
            "+1q": {
                "numberOfAnalysts": 18,
                "avg": 5_000_000_000,
                "low": 4_800_000_000,
                "high": 5_200_000_000,
                "yearAgoRevenue": 4_500_000_000,
                "growth": 0.11,
            }
        },
        "growth_estimates": {
            "+1y": {
                "stock": 0.22,
                "industry": 0.14,
                "sector": 0.16,
                "index": 0.10,
            }
        },
        "major_holders": pd.DataFrame(
            {
                "Value": ["0.08%", "68.50%"],
            },
            index=["% of Shares Held by All Insider", "% of Shares Held by Institutions"],
        ),
        "institutional_holders": pd.DataFrame(
            [
                {"Holder": "Vanguard", "Shares": 1000, "% Out": "8.5", "Value": 1_000_000},
                {"Holder": "BlackRock", "Shares": 900, "% Out": "7.2", "Value": 900_000},
            ]
        ),
        "insider_transactions": pd.DataFrame(
            [
                {
                    "Insider": "Jane Doe",
                    "Start Date": pd.Timestamp("2026-04-01"),
                    "Transaction": "Sale",
                    "Shares": 10000,
                    "Value": 250000,
                },
                {
                    "Insider": "John Smith",
                    "Start Date": pd.Timestamp("2026-03-15"),
                    "Transaction": "Purchase",
                    "Shares": 5000,
                    "Value": 125000,
                },
            ]
        ),
        "eps_revisions": {
            "+1q": {
                "upLast7days": 2,
                "upLast30days": 8,
                "downLast7days": 0,
                "downLast30days": 2,
            }
        },
        "eps_trend": {
            "+1q": {
                "current": 1.25,
                "7daysAgo": 1.24,
                "30daysAgo": 1.30,
                "60daysAgo": 1.28,
                "90daysAgo": 1.20,
            }
        },
        "upgrades_downgrades": pd.DataFrame(
            [
                {
                    "Firm": "Goldman Sachs",
                    "To Grade": "Buy",
                    "From Grade": "Neutral",
                    "Action": "up",
                },
                {
                    "Firm": "Morgan Stanley",
                    "To Grade": "Hold",
                    "From Grade": "Buy",
                    "Action": "down",
                },
            ],
            index=pd.to_datetime(["2026-05-20", "2026-05-10"]),
        ),
    }


@patch("app.builders.yfinance_analysis_builder.YFinanceAdapter")
def test_build_street_analysis_snapshot(mock_adapter_cls):
    adapter = MagicMock()
    adapter.get_street_analysis_raw.return_value = _sample_raw()
    adapter.get_ticker_info.return_value = {"currentPrice": 100.0}

    snapshot = YFinanceAnalysisBuilder(adapter).build("HOOD")

    assert snapshot is not None
    assert snapshot.consensus_label == "Mostly Buy"
    assert snapshot.price_targets is not None
    assert snapshot.price_targets.mean == 110.0
    assert snapshot.price_targets.upside_to_mean_pct == 10.0
    assert snapshot.next_quarter_eps is not None
    assert snapshot.next_quarter_eps.avg == 1.25
    assert snapshot.next_quarter_revenue is not None
    assert snapshot.estimate_revision_headline is not None
    assert "revised up 8×" in snapshot.estimate_revision_headline
    assert snapshot.estimate_drift_headline is not None
    assert "down" in snapshot.estimate_drift_headline
    assert len(snapshot.recent_rating_actions) == 2
    assert snapshot.recent_rating_actions[0].firm == "Goldman Sachs"
    assert len(snapshot.eps_estimates) == 3
    assert snapshot.rating_trend_headline is not None
    assert "more bullish" in snapshot.rating_trend_headline
    assert snapshot.growth_context_headline is not None
    assert "22.0%" in snapshot.growth_context_headline
    assert snapshot.ownership is not None
    assert snapshot.ownership.institutions_pct_held == 68.5
    assert len(snapshot.ownership.top_institutional) == 2


@patch("app.builders.yfinance_analysis_builder.YFinanceAdapter")
def test_insider_transactions_use_start_date_not_row_index(mock_adapter_cls):
    adapter = MagicMock()
    raw = _sample_raw()
    raw["insider_transactions"] = pd.DataFrame(
        [
            {
                "Insider": "HUANG JEN-HSUN",
                "Start Date": pd.Timestamp("2025-06-10"),
                "Transaction": "Sale at price 100.00 per share.",
                "Shares": 240_000,
            }
        ],
        index=[0],
    )
    adapter.get_street_analysis_raw.return_value = raw
    adapter.get_ticker_info.return_value = {"currentPrice": 100.0}

    snapshot = YFinanceAnalysisBuilder(adapter).build("NVDA")
    assert snapshot is not None
    assert snapshot.ownership is not None
    assert snapshot.ownership.recent_insider_transactions[0].date == "2025-06-10"


@patch("app.builders.yfinance_analysis_builder.YFinanceAdapter")
def test_build_returns_none_when_empty(mock_adapter_cls):
    adapter = MagicMock()
    adapter.get_street_analysis_raw.return_value = {
        "price_targets": None,
        "recommendations_summary": None,
        "recommendations": None,
        "earnings_estimate": None,
        "revenue_estimate": None,
        "eps_revisions": None,
        "eps_trend": None,
        "upgrades_downgrades": None,
        "growth_estimates": None,
        "institutional_holders": None,
        "insider_transactions": None,
        "major_holders": None,
    }
    adapter.get_ticker_info.return_value = {}

    assert YFinanceAnalysisBuilder(adapter).build("ZZZZ") is None
