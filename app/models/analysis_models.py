from pydantic import BaseModel, ConfigDict, Field


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
