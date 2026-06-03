from app.builders.business_intelligence_validation import normalize_business_intelligence
from app.models.company_research_models import BusinessBlock


def test_normalize_filters_financial_overlap_bullets():
    raw = BusinessBlock(
        industry="Technology",
        primary_product="GPU Cloud",
        revenue_model="Infrastructure contracts",
        primary_customers=["AI labs"],
        business_model="Provides GPU compute to enterprises.",
        how_they_make_money=["Revenue from multi-year contracts."],
        advantages=["NVIDIA preferred partner"],
        challenges=["Analyst price target implies limited upside"],
        growth_drivers=["AI infrastructure demand"],
        business_risks=["Customer concentration"],
        dependencies=["NVIDIA hardware supply"],
    )
    result = normalize_business_intelligence(raw, fallback_industry="Technology")
    assert "P/E" not in " ".join(result.challenges)
    assert result.industry == "Technology"
    assert len(result.how_they_make_money) >= 1
