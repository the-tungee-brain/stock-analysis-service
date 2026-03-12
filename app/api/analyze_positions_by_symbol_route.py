from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.models.schwab_models import Position

router = APIRouter()


class AnalyzePositionsBySymbolRequest(BaseModel):
    positions: List[Position]
    prompt: str


@router.post("/analyze-positions-by-symbol")
def analyze_positions_by_symbol(request: AnalyzePositionsBySymbolRequest):
    pass
