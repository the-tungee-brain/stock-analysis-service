from app.builders.canonical_financial_metrics import CanonicalFinancialMetrics
from app.builders.financial_overview_generator import FinancialOverviewGenerator
from app.builders.financial_score_percentiles import rank_label_for_score
from app.builders.financial_sector_context import classify_archetype


def test_rank_labels_for_extremes():
    assert rank_label_for_score(100) == "Top 1%"
    assert rank_label_for_score(5) == "Bottom 5%"
    assert rank_label_for_score(12) == "Bottom 10%"


def test_bank_narrative_differs_from_hypergrowth():
    bank = FinancialOverviewGenerator().generate(
        "JPM",
        CanonicalFinancialMetrics(
            revenue_growth_yoy=8,
            net_margin_pct=28,
            debt_to_equity=1.1,
            current_ratio=1.2,
            free_cash_flow_latest=12_000_000_000,
            return_on_equity_pct=14,
        ),
        sector="Financial Services",
        industry="Banks - Diversified",
    )
    tech = FinancialOverviewGenerator().generate(
        "NVDA",
        CanonicalFinancialMetrics(
            revenue_growth_yoy=120,
            gross_margin_pct=75,
            net_margin_pct=55,
            debt_to_equity=0.2,
            free_cash_flow_latest=8_000_000_000,
        ),
        sector="Technology",
        industry="Semiconductors",
    )

    assert classify_archetype("Financial Services", "Banks") == classify_archetype(
        "Financial Services",
        "Banks - Diversified",
    )
    bank_text = " ".join([*bank.strengths, *bank.risks, bank.financial_verdict]).lower()
    tech_text = " ".join([*tech.strengths, *tech.risks, tech.financial_verdict]).lower()
    assert "loan-book" in bank_text or "capital" in bank_text or "spread" in bank_text
    assert "hypergrowth" in tech_text or "scaling" in tech_text or "exceptional" in tech_text
    assert bank.profile != tech.profile or bank.financial_verdict != tech.financial_verdict
