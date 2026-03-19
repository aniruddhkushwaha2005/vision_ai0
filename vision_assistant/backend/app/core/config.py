"""
Centralised configuration — all settings sourced from environment variables.
Never hardcode credentials. Use .env for local dev, secrets manager for prod.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "Vision Assistant API"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Model paths ───────────────────────────────────────────────────────────
    YOLO_MODEL_PATH: str = "models/yolov8n.pt"           # nano for speed; swap to yolov8s/m for accuracy
    MIDAS_MODEL_TYPE: str = "MiDaS_small"                # lightweight; use DPT_Large on GPU servers
    MIDAS_DEVICE: str = "cuda"                           # "cuda" | "cpu" | "mps" (Apple Silicon)

    # ── Navigation engine ────────────────────────────────────────────────────
    NAV_FRAME_WIDTH: int = 640
    NAV_FRAME_HEIGHT: int = 480
    NAV_LEFT_BOUNDARY: float = 0.33          # 33% = left zone boundary
    NAV_RIGHT_BOUNDARY: float = 0.67         # 67% = right zone boundary
    NAV_CENTER_BLOCK_THRESH: float = 0.45    # depth normalised threshold
    NAV_SMOOTHING_WINDOW: int = 5            # frames for temporal majority vote
    NAV_DANGER_DEPTH_THRESH: float = 0.70    # objects closer than 70% of frame considered dangerous
    NAV_PRIORITY_SCORES: dict = {
        "car": 10, "truck": 10, "bus": 9, "motorcycle": 8,
        "bicycle": 7, "person": 6, "dog": 5,
        "chair": 3, "table": 3, "potted plant": 2,
    }

    # ── TTS ───────────────────────────────────────────────────────────────────
    TTS_ENGINE: str = "gtts"                 # "gtts" | "pyttsx3" | "azure" | "google_cloud"
    TTS_DEFAULT_LANG: str = "en"
    TTS_HINDI_ENABLED: bool = True
    TTS_RATE_LIMIT_SECS: float = 2.0         # minimum seconds between repeated announcements

    # ── Redis (for state caching between frames) ──────────────────────────────
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_FRAME_TTL_SECS: int = 5

    # ── Celery ───────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://redis:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/2"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"                 # "json" for prod, "text" for dev

    # ── Cloud / deployment ───────────────────────────────────────────────────
    AWS_REGION: Optional[str] = None
    SENTRY_DSN: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
