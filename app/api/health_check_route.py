from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    return {"message": "Hello from OCI!", "domain": "thetungeebrain.duckdns.org"}


@router.get("/health")
def health():
    return {"status": "ok"}
