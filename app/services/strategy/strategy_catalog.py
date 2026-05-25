from __future__ import annotations

from app.models.strategy_models import (
    InvestmentStrategy,
    JourneyStep,
    JourneyStepStatus,
    StrategyCatalogItem,
)

JOURNEY_TEMPLATES: dict[InvestmentStrategy, list[tuple[str, str, str]]] = {
    InvestmentStrategy.WHEEL: [
        (
            "connect-schwab",
            "Connect Schwab",
            "Link your brokerage account so we can track your wheel progress.",
        ),
        (
            "learn-wheel-basics",
            "Learn the wheel",
            "Understand CSP → assignment → covered call → repeat.",
        ),
        (
            "pick-underlying",
            "Pick your underlying",
            "Choose stocks you're happy to own if assigned.",
        ),
        (
            "research-underlying",
            "Research before you sell",
            "Review valuation and risks before selling a cash-secured put.",
        ),
        (
            "sell-first-csp",
            "Sell your first CSP",
            "Open a cash-secured put on your chosen symbol.",
        ),
        (
            "monitor-or-roll",
            "Monitor or roll",
            "Watch delta and expiration; roll if needed before assignment.",
        ),
        (
            "sell-covered-call",
            "Sell a covered call",
            "If assigned shares, sell a call against your stock.",
        ),
        (
            "complete-cycle",
            "Complete a full cycle",
            "Finish CSP → shares → covered call → called away.",
        ),
    ],
    InvestmentStrategy.CSP_INCOME: [
        (
            "connect-schwab",
            "Connect Schwab",
            "Link your brokerage account to track CSP income.",
        ),
        (
            "learn-csp-basics",
            "Learn cash-secured puts",
            "Understand premium, assignment, and strike selection.",
        ),
        (
            "pick-underlying",
            "Pick underlyings",
            "Choose stocks you'd be comfortable owning at the strike.",
        ),
        (
            "sell-first-csp",
            "Sell your first CSP",
            "Collect premium on a cash-secured put.",
        ),
        (
            "track-income",
            "Track your income",
            "Review premium collected and win rate over time.",
        ),
    ],
    InvestmentStrategy.COVERED_CALL: [
        (
            "connect-schwab",
            "Connect Schwab",
            "Link your brokerage account to track covered calls.",
        ),
        (
            "learn-covered-calls",
            "Learn covered calls",
            "Understand upside cap, assignment, and roll timing.",
        ),
        (
            "confirm-share-count",
            "Confirm 100+ shares",
            "You need at least 100 shares to sell one covered call.",
        ),
        (
            "sell-first-call",
            "Sell your first covered call",
            "Collect premium against shares you already own.",
        ),
        (
            "monitor-assignment",
            "Monitor assignment risk",
            "Watch delta as expiration approaches.",
        ),
    ],
    InvestmentStrategy.DIVIDEND: [
        (
            "connect-schwab",
            "Connect Schwab",
            "Link your brokerage account to track dividend holdings.",
        ),
        (
            "set-income-preferences",
            "Set income preferences",
            "Define your yield target and risk tolerance.",
        ),
        (
            "pick-dividend-names",
            "Pick dividend names",
            "Choose 3–5 reliable income payers to research.",
        ),
        (
            "research-fundamentals",
            "Research fundamentals",
            "Review yield, payout ratio, and cash flow stability.",
        ),
        (
            "first-dividend-buy",
            "Make your first buy",
            "Start building your dividend portfolio.",
        ),
        (
            "monitor-income",
            "Monitor your income",
            "Track yield, concentration, and dividend health.",
        ),
    ],
    InvestmentStrategy.ETF_CORE: [
        (
            "connect-schwab",
            "Connect Schwab",
            "Link your brokerage account to track your allocation.",
        ),
        (
            "set-allocation",
            "Set target allocation",
            "Choose your broad market / bond mix.",
        ),
        (
            "first-etf-buy",
            "Make your first ETF buy",
            "Start with your core holding.",
        ),
        (
            "review-drift",
            "Review allocation drift",
            "Check whether you're still on target.",
        ),
        (
            "stay-the-course",
            "Stay the course",
            "Keep contributing and rebalance when drift exceeds your threshold.",
        ),
    ],
}

STRATEGY_CATALOG: list[StrategyCatalogItem] = [
    StrategyCatalogItem(
        id=InvestmentStrategy.WHEEL,
        title="Wheel strategy",
        subtitle="Income on stocks you'd own",
        description=(
            "Sell cash-secured puts, get assigned shares if needed, "
            "then sell covered calls — repeat the cycle for premium income."
        ),
        best_for=["Options income", "Stocks you want to own", "Active monitoring"],
        prerequisites=["Options approval", "Cash for CSP collateral"],
        step_count=len(JOURNEY_TEMPLATES[InvestmentStrategy.WHEEL]),
        requires_options=True,
    ),
    StrategyCatalogItem(
        id=InvestmentStrategy.CSP_INCOME,
        title="Cash-secured puts",
        subtitle="Premium without the full wheel",
        description=(
            "Focus on selling puts for income on names you're willing to buy at lower prices."
        ),
        best_for=["Income", "Buying dips with a discount"],
        prerequisites=["Options approval", "Cash reserves"],
        step_count=len(JOURNEY_TEMPLATES[InvestmentStrategy.CSP_INCOME]),
        requires_options=True,
    ),
    StrategyCatalogItem(
        id=InvestmentStrategy.COVERED_CALL,
        title="Covered calls",
        subtitle="Income on shares you hold",
        description="Generate premium on stock you already own by selling out-of-the-money calls.",
        best_for=["Existing stock positions", "Income on long holdings"],
        prerequisites=["100+ shares per contract", "Options approval"],
        step_count=len(JOURNEY_TEMPLATES[InvestmentStrategy.COVERED_CALL]),
        requires_options=True,
    ),
    StrategyCatalogItem(
        id=InvestmentStrategy.DIVIDEND,
        title="Dividend investing",
        subtitle="Reliable income payers",
        description=(
            "Build a portfolio of dividend stocks and hold for income and stability."
        ),
        best_for=["Long-term income", "Lower turnover"],
        prerequisites=["Patience", "Fundamental research"],
        step_count=len(JOURNEY_TEMPLATES[InvestmentStrategy.DIVIDEND]),
        requires_options=False,
    ),
    StrategyCatalogItem(
        id=InvestmentStrategy.ETF_CORE,
        title="ETF core portfolio",
        subtitle="Simple diversified indexing",
        description=(
            "Buy and hold broad ETFs with periodic rebalancing — set it and mostly forget it."
        ),
        best_for=["Beginners", "Passive investors", "Low maintenance"],
        prerequisites=["Regular contributions"],
        step_count=len(JOURNEY_TEMPLATES[InvestmentStrategy.ETF_CORE]),
        requires_options=False,
    ),
]


def build_initial_steps(strategy: InvestmentStrategy) -> list[JourneyStep]:
    template = JOURNEY_TEMPLATES[strategy]
    steps: list[JourneyStep] = []
    for index, (step_id, title, description) in enumerate(template):
        status = JourneyStepStatus.AVAILABLE if index == 0 else JourneyStepStatus.LOCKED
        steps.append(
            JourneyStep(
                step_id=step_id,
                title=title,
                description=description,
                status=status,
                order=index + 1,
            )
        )
    return steps


def catalog_item(strategy: InvestmentStrategy) -> StrategyCatalogItem:
    for item in STRATEGY_CATALOG:
        if item.id == strategy:
            return item
    raise KeyError(strategy)
