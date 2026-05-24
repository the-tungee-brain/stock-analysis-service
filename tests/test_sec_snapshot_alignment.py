from app.builders.sec_financials_builder import SecFinancialsBuilder
from app.models.sec_research_models import FinancialLineItem, FinancialObservation
from app.services.sec_research_service import SecResearchService


def test_value_at_period_returns_matching_observation():
    items = [
        FinancialLineItem(
            tag="Revenues",
            label="Revenues",
            unit="USD",
            observations=[
                FinancialObservation(
                    end="2024-09-28",
                    start="2023-09-30",
                    value=100.0,
                    fiscal_year=2024,
                    fiscal_period="FY",
                    form="10-K",
                    filed="2024-11-01",
                ),
                FinancialObservation(
                    end="2023-09-30",
                    start="2022-09-25",
                    value=80.0,
                    fiscal_year=2023,
                    fiscal_period="FY",
                    form="10-K",
                    filed="2023-11-03",
                ),
            ],
        )
    ]

    value = SecFinancialsBuilder.value_at_period(
        items,
        ["Revenues"],
        end="2024-09-28",
        fiscal_period="FY",
    )

    assert value == 100.0


def test_snapshot_metrics_are_consistent_rejects_impossible_margin():
    assert SecResearchService._snapshot_metrics_are_consistent(
        revenue=26_900_000_000,
        net_income=120_100_000_000,
    ) is False

    assert SecResearchService._snapshot_metrics_are_consistent(
        revenue=400_000_000_000,
        net_income=100_000_000_000,
    ) is True
