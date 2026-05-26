import pytest

from app.models.company_research_models import EtfHoldingItem
from app.utils.etf_holdings_quality import (
    compute_quality_score,
    normalize_altman_z_score,
    normalize_piotroski_score,
    rank_etf_holdings_by_quality,
)


def _holding(
    ticker: str,
    *,
    weight_pct: float,
    piotroski_f: int | None = None,
    altman_z: float | None = None,
) -> EtfHoldingItem:
    return EtfHoldingItem(
        ticker=ticker,
        name=ticker,
        weight_pct=weight_pct,
        piotroski_f=piotroski_f,
        altman_z=altman_z,
        quality_score=compute_quality_score(piotroski_f, altman_z),
    )


def test_normalize_piotroski_score():
    assert normalize_piotroski_score(9) == pytest.approx(1.0)
    assert normalize_piotroski_score(0) == pytest.approx(0.0)
    assert normalize_piotroski_score(None) is None


def test_normalize_altman_z_score_buckets():
    assert normalize_altman_z_score(0.5) == pytest.approx(0.5 / 1.81 * 0.35)
    assert normalize_altman_z_score(2.4) == pytest.approx(
        0.35 + ((2.4 - 1.81) / (2.99 - 1.81)) * 0.35
    )
    assert normalize_altman_z_score(10.0) == pytest.approx(0.7 + ((10.0 - 2.99) / 10) * 0.3)


def test_rank_etf_holdings_by_quality():
    holdings = [
        _holding("WEAK", weight_pct=1.0, piotroski_f=2, altman_z=0.2),
        _holding("STRONG", weight_pct=2.0, piotroski_f=9, altman_z=12.0),
        _holding("MID", weight_pct=3.0, piotroski_f=6, altman_z=3.0),
        _holding("NOSCORE", weight_pct=4.0),
    ]

    strongest, weakest = rank_etf_holdings_by_quality(holdings, limit=2)

    assert [item.ticker for item in strongest] == ["STRONG", "MID"]
    assert [item.ticker for item in weakest] == ["WEAK", "MID"]
