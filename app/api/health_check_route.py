from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.llm_config import settings

router = APIRouter()


@router.get("/")
def root():
    return {"message": "Hello from OCI!", "domain": "thetungeebrain.duckdns.org"}


@router.get("/health")
def health(request: Request):
    paid_pattern_enabled = bool(settings.PAID_USER_IDS or settings.PAID_USER_EMAILS)
    pattern_loaded = getattr(request.app.state, "pattern_loaded_model", None) is not None
    pattern_error = getattr(request.app.state, "pattern_model_error", None)

    payload = {
        "status": "ok",
        "patternModel": {
            "loaded": pattern_loaded,
            "artifactDir": _pattern_artifact_dir(),
            "error": pattern_error,
        },
    }

    if paid_pattern_enabled and not pattern_loaded:
        payload["status"] = "unhealthy"
        return JSONResponse(status_code=503, content=payload)

    return payload


def _pattern_artifact_dir() -> str:
    from models.artifact_store import resolve_artifact_dir

    return str(resolve_artifact_dir())
