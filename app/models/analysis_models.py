from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models.symbol_analysis_precomputed_models import SymbolAnalysisPrecomputed


class StructuredAnalysisActionLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    reason: str
    symbol: str = ""


class StructuredAnalysisSectionLLM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = ""
    title: str
    body: str = ""
    bullets: list[str] = Field(default_factory=list)


class PortfolioAnalysisV1LLMResponse(BaseModel):
    """Strict OpenAI json_schema output for portfolio_analysis_v1."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    recommendedAction: StructuredAnalysisActionLLM
    sections: list[StructuredAnalysisSectionLLM]


class SymbolAnalysisV1Envelope(BaseModel):
    """Symbol/portfolio analyze v1: LLM narrative plus optional server-side outcomes."""

    model_config = ConfigDict(populate_by_name=True)

    analysis: PortfolioAnalysisV1LLMResponse
    precomputed: SymbolAnalysisPrecomputed | None = None
