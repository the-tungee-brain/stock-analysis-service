"""Research dashboard builder from StrategyResearchReport."""

from __future__ import annotations

from trade_planner.research.models import ResearchDashboard, StrategyResearchReport


def build_research_dashboard(report: StrategyResearchReport) -> ResearchDashboard:
    return ResearchDashboard(
        setup_name=report.setup_name,
        symbols_tested=report.symbols_tested,
        start_date=report.start_date,
        end_date=report.end_date,
        overall=report.performance,
        by_year=report.yearly_performance,
        by_regime=report.regime_comparison.rows,
        walk_forward=report.walk_forward,
        top_conditions=report.top_conditions,
        worst_conditions=report.worst_conditions,
    )
