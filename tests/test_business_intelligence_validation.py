from app.builders.business_intelligence_validation import normalize_business_intelligence
from app.models.company_research_models import BusinessBlock


def test_normalize_filters_financial_filler_and_splits_growth():
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
            "Backlog is not fully recognized until clusters are deployed and utilized.",
            "Signed contracts do not guarantee revenue timing without capacity online.",
        ],
        advantages=["NVIDIA preferred partner status on H100 supply"],
        challenges=["Analyst price target implies limited upside"],
        revenue_drivers=["New GPU cluster deployments under contract"],
        constraints=["Ability to scale efficiently across regions"],
        business_risks=["Customer concentration"],
        dependencies=["NVIDIA hardware supply", "Hyperscaler demand", "Power permits", "Debt markets", "Brand"],
    )
    result = normalize_business_intelligence(raw, fallback_industry="Technology")
    assert "price target" not in " ".join(result.challenges).lower()
    assert not any("increasing demand" in b.lower() for b in result.how_they_make_money)
    assert not any("ability to" in b.lower() for b in result.constraints)
    assert len(result.dependencies) <= 4


def test_normalize_strips_filler_and_weak_challenges():
    raw = BusinessBlock(
        industry="SaaS",
        primary_product="CRM platform",
        revenue_model="Per-seat subscriptions",
        primary_customers=["SMB sales teams"],
        how_they_make_money=["Per-seat monthly subscription fees."],
        revenue_visibility=[
            "Annual contracts billed monthly; churn can delay recognized ARR.",
            "Revenue recognized ratably as seats are active each month.",
        ],
        advantages=["Direct sales force in mid-market"],
        challenges=["Competition"],
        revenue_drivers=["New enterprise contract wins"],
        constraints=["Limited by data center capacity for EU hosting"],
        business_risks=["Customer churn on seat downsizing"],
        dependencies=["Cloud hosting on AWS"],
    )
    result = normalize_business_intelligence(raw)
    assert not any("disciplined execution" in a.lower() for a in result.advantages)
    assert not any(c.lower() == "competition" for c in result.challenges)


def test_legacy_growth_drivers_migrate_to_revenue_drivers():
    raw = BusinessBlock.model_validate(
        {
            "industry": "Retail",
            "growthDrivers": ["Same-store traffic recovery"],
            "constraints": ["Shelf space at big-box partners"],
        }
    )
    assert raw.revenue_drivers == ["Same-store traffic recovery"]
