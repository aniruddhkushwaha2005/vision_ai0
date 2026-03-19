"""
Pydantic data models — shared contracts between services and API layer.
All inter-service data passes through these typed structures.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class NavigationDecision(str, Enum):
    FORWARD = "FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
    DANGER = "DANGER"          # Urgent: fast-approaching object
    CLEAR = "CLEAR"            # Path fully clear


class FrameRegion(str, Enum):
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"


class ThreatLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ── Detection ─────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


class DetectedObject(BaseModel):
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    region: FrameRegion
    priority_score: int = Field(default=1, ge=1, le=10)
    estimated_distance_m: Optional[float] = None      # metres, if depth available
    depth_normalised: Optional[float] = None          # 0.0 (far) → 1.0 (very close)
    is_approaching: bool = False                       # motion analysis result
    threat_level: ThreatLevel = ThreatLevel.NONE


class DetectionResult(BaseModel):
    frame_id: str
    timestamp_ms: int
    objects: List[DetectedObject] = []
    frame_width: int
    frame_height: int
    inference_time_ms: float


# ── Depth ────────────────────────────────────────────────────────────────────

class DepthResult(BaseModel):
    frame_id: str
    depth_map_base64: Optional[str] = None    # serialised for WebSocket transport
    min_depth: float
    max_depth: float
    center_depth_normalised: float
    left_depth_normalised: float
    right_depth_normalised: float
    inference_time_ms: float


# ── Navigation ───────────────────────────────────────────────────────────────

class ZoneAnalysis(BaseModel):
    region: FrameRegion
    is_blocked: bool
    highest_priority_object: Optional[str] = None
    min_depth_normalised: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.NONE


class NavigationResult(BaseModel):
    frame_id: str
    decision: NavigationDecision
    confidence: float = Field(..., ge=0.0, le=1.0)
    zones: List[ZoneAnalysis]
    dominant_threat: Optional[DetectedObject] = None
    speech_text_en: str
    speech_text_hi: Optional[str] = None
    should_speak: bool = True          # False if same as last N decisions (dedup)
    raw_decision: NavigationDecision   # pre-smoothing decision
    smoothed_decision: NavigationDecision  # post-temporal-smoothing
    is_danger_alert: bool = False
    processing_time_ms: float


# ── WebSocket payloads ────────────────────────────────────────────────────────

class FramePayload(BaseModel):
    """Incoming frame from mobile client over WebSocket."""
    frame_id: str
    image_base64: str              # JPEG/PNG base64 encoded
    timestamp_ms: int
    session_id: str
    preferred_language: str = "en"


class StreamResponse(BaseModel):
    """Outgoing result streamed back to client."""
    frame_id: str
    session_id: str
    navigation: NavigationResult
    detections: DetectionResult
    depth: Optional[DepthResult] = None
    audio_base64: Optional[str] = None    # TTS audio bytes for offline playback
    server_processing_ms: float


# ── Scene description ─────────────────────────────────────────────────────────

class SceneDescription(BaseModel):
    frame_id: str
    description_en: str
    description_hi: Optional[str] = None
    detected_count: int
    dominant_objects: List[str]
    path_clear: bool
    ground_clearance: ThreatLevel
