"""Health and readiness endpoints for orchestration probes."""

import time
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
_start_time = time.time()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/ready")
async def readiness():
    """Kubernetes readiness probe — checks models are loaded."""
    from app.services.detection.detector import DetectorService
    detector = DetectorService.get_instance()
    if detector._model is None:
        return {"status": "not_ready", "reason": "model_not_loaded"}, 503
    return {"status": "ready"}
