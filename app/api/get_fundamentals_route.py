from fastapi import APIRouter, Depends
from app.models.company_research_models import (
    FundamentalsBlock,
    FundamentalsOverview,
    FundamentalMetric,
)
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.llm_service import LLMService
from app.services.company_research_service import CompanyResearchService
from app.services.sec_research_service import SecResearchService
from app.dependencies.service_dependencies import (
    get_prompt_enrichment_service,
    get_llm_service,
    get_company_research_service,
    get_sec_research_service,
)

router = APIRouter()


def _merge_fundamental_metrics(
    *,
    sec_research_service: SecResearchService,
    symbol: str,
    yfinance_metrics: list[FundamentalMetric],
) -> list[FundamentalMetric]:
    merged: list[FundamentalMetric] = []
    seen_labels: set[str] = set()

    try:
        for item in sec_research_service.latest_snapshot_metrics(symbol=symbol):
            if not item.get("include") or not item.get("value"):
                continue
            label = str(item["label"])
            merged.append(
                FundamentalMetric(
                    label=label,
                    value=str(item["value"]),
                    note=str(item.get("note") or ""),
                )
            )
            seen_labels.add(label.lower())
    except Exception:
        pass

    for metric in yfinance_metrics:
        if metric.label.lower() in seen_labels:
            continue
        merged.append(metric)

    return merged


@router.get("/research/fundamentals", response_model=FundamentalsBlock)
async def get_fundamentals(
    symbol: str,
    company_research_service: CompanyResearchService = Depends(
        get_company_research_service
    ),
    sec_research_service: SecResearchService = Depends(get_sec_research_service),
    prompt_enrichment_service: PromptEnrichmentService = Depends(
        get_prompt_enrichment_service
    ),
    llm_service: LLMService = Depends(get_llm_service),
):
    ctx = company_research_service.build_context(symbol=symbol)
    metrics = _merge_fundamental_metrics(
        sec_research_service=sec_research_service,
        symbol=symbol,
        yfinance_metrics=ctx.fundamentals,
    )
    ctx.fundamentals = metrics
    prompts = prompt_enrichment_service.build_fundamentals_prompt(
        ctx=ctx, metrics=metrics
    )
    overview = await llm_service.generate_from_prompts(
        prompts=prompts, response_model=FundamentalsOverview
    )
    return FundamentalsBlock(
        overviewNote=overview.overviewNote,
        metrics=metrics,
    )
