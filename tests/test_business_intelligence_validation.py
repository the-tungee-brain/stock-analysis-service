from app.builders.business_intelligence_validation import normalize_business_intelligence
from app.models.company_research_models import BusinessBlock


def test_normalize_filters_financial_and_filler_bullets():
    raw = BusinessBlock(
        industry="Technology",
        primary_product="GPU Cloud",
        revenue_model="Infrastructure contracts",
        primary_customers=["AI labs"],
        how_they_make_money=[
            "Revenue from multi-year GPU capacity contracts.",
            "Market expansion and increasing demand trends.",
        ],
        revenue_visibility=[
            "3–5 year contracts with annual renewals; high recurring base.",
            "Revenue recognized as compute is delivered and metered monthly.",
        ],
        advantages=["NVIDIA preferred partner"],
        challenges=["Analyst price target implies limited upside"],
        growth_drivers=["AI infrastructure demand", "Brand awareness campaigns"],
        business_risks=["Customer concentration"],
        dependencies=["NVIDIA hardware supply"],
    )
    result = normalize_business_intelligence(raw, fallback_industry="Technology")
    assert "price target" not in " ".join(result.challenges).lower()
    assert not any("increasing demand" in b.lower() for b in result.how_they_make_money)
    assert "brand awareness" not in " ".join(result.growth_drivers).lower()
    assert len(result.revenue_visibility) <= 2


def test_normalize_strips_filler_phrases():
    raw = BusinessBlock(
        industry="SaaS",
        primary_product="CRM platform",
        revenue_model="Per-seat subscriptions",
        primary_customers=["SMB sales teams"],
        how_they_make_money=["Per-seat monthly subscription fees."],
        revenue_visibility=["Annual contracts billed monthly.", "Recognized ratably over contract term."],
        advantages=["Disciplined execution across enterprise sales"],
        challenges=["Competes with Salesforce and HubSpot"],
        growth_drivers=["New enterprise contract wins"],
        business_risks=["Customer churn on seat downsizing"],
        dependencies=["Cloud hosting on AWS"],
    )
    result = normalize_business_intelligence(raw)
    assert not any("disciplined execution" in a.lower() for a in result.advantages)
