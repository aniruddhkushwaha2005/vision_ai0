"""
Vision Assistant for Blind People — FastAPI Entry Point
Production-grade startup with lifespan management, CORS, middleware, WebSocket support.
"""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from app.api.routes import detection, stream, health
from app.core.config import settings
from app.core.logger import setup_logging
from app.services.detection.detector import DetectorService
from app.services.depth.depth_estimator import DepthEstimatorService
from app.services.tts.tts_service import TTSService

# ── Observability ────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["endpoint"])

logger = logging.getLogger(__name__)


# ── Lifespan: warm-up / teardown ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy ML models once at startup; release on shutdown."""
    setup_logging()
    logger.info("🚀 Starting Vision Assistant API — loading models...")

    # Warm up singletons (loads YOLO + MiDaS + TTS into GPU/CPU memory)
    DetectorService.get_instance()
    DepthEstimatorService.get_instance()
    TTSService.get_instance()

    logger.info("✅ All models loaded. API ready.")
    yield

    logger.info("🛑 Shutting down — releasing resources.")
    DetectorService.get_instance().cleanup()
    DepthEstimatorService.get_instance().cleanup()


# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="Vision Assistant API",
        description="Real-time AI navigation assistance for visually impaired users",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Prometheus metrics endpoint ───────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def observe_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(detection.router, prefix="/api/v1", tags=["detection"])
    app.include_router(stream.router, prefix="/api/v1", tags=["stream"])

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1,           # Keep 1 worker — models are in-process singletons
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=10,
    )
