import asyncio

from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_portfolio_memory_service
from app.models.portfolio_optimization_models import PortfolioOptimizationResponse
from app.services.portfolio_memory_service import PortfolioMemoryService
from app.services.portfolio_optimization_service import PortfolioOptimizationService

router = APIRouter()


@router.get(
    "/portfolio/optimization",
    response_model=PortfolioOptimizationResponse,
    response_model_by_alias=True,
)
async def get_portfolio_optimization(
    user_id: str = Depends(get_current_user_id),
    portfolio_memory_service: PortfolioMemoryService = Depends(
        get_portfolio_memory_service
    ),
) -> PortfolioOptimizationResponse:
    snapshots = await asyncio.to_thread(
        portfolio_memory_service.portfolio_snapshot_adapter.list_recent,
        user_id,
        limit=1,
    )
    snapshot = snapshots[0] if snapshots else None

    return await asyncio.to_thread(
        PortfolioOptimizationService().build_from_snapshot,
        snapshot=snapshot,
    )
