"""Product-facing API layer for Web and iOS clients."""

from app.api.product.health_controller import router as health_router
from app.api.product.portfolio_controller import router as portfolio_router
from app.api.product.rankings_controller import router as rankings_router

__all__ = ["health_router", "portfolio_router", "rankings_router"]
