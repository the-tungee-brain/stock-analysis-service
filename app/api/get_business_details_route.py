from fastapi import APIRouter, Depends
from app.models.company_research_models import BusinessBlock

router = APIRouter()


@router.get("/research/business", response_model=BusinessBlock)
def get_business_details(symbol: str):
    pass
