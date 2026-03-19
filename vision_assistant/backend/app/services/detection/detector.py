"""
Detection Service — YOLOv8-based real-time object detection.

Design decisions:
  - Singleton pattern: model loaded once, reused per frame
  - Thread-safe via asyncio lock for concurrent requests
  - Normalised bounding boxes for resolution-independent downstream logic
  - Priority scoring injected from config (not hardcoded)
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.core.config import settings
from app.models.schemas import (
    BoundingBox, DetectedObject, DetectionResult, FrameRegion, ThreatLevel
)

logger = logging.getLogger(__name__)

# Lazy import: ultralytics may not be installed in all environments
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics not installed — using mock detector")


class DetectorService:
    """
    Singleton wrapping YOLOv8 for per-frame object detection.
    Outputs normalised DetectedObject list with region assignment.
    """
    _instance: Optional["DetectorService"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._model = None
        self._model_lock = asyncio.Lock()
        self._load_model()

    @classmethod
    def get_instance(cls) -> "DetectorService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Load YOLOv8 model; fallback to mock on failure."""
        if not YOLO_AVAILABLE:
            logger.warning("YOLO unavailable — DetectorService in mock mode")
            return

        model_path = Path(settings.YOLO_MODEL_PATH)
        if not model_path.exists():
            logger.info("Model not found at %s — downloading YOLOv8n...", model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            device = "cuda" if settings.MIDAS_DEVICE == "cuda" else "cpu"
            self._model = YOLO(str(model_path))
            self._model.to(device)
            logger.info("✅ YOLOv8 loaded on device=%s", device)
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            self._model = None

    async def detect(
        self,
        frame: np.ndarray,
        frame_id: Optional[str] = None,
        conf_threshold: float = 0.40,
    ) -> DetectionResult:
        """
        Run YOLOv8 inference on a single BGR numpy frame.

        Args:
            frame: BGR numpy array (H, W, 3)
            frame_id: optional trace ID; generated if None
            conf_threshold: minimum confidence to include a detection

        Returns:
            DetectionResult with all detected objects, regions, priority scores
        """
        if frame_id is None:
            frame_id = str(uuid.uuid4())

        frame_h, frame_w = frame.shape[:2]
        t0 = time.perf_counter()

        objects: List[DetectedObject] = []

        if self._model is None:
            # Mock: return a dummy person in center for integration testing
            objects = self._mock_detections(frame_w, frame_h)
        else:
            async with self._model_lock:
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._model.predict(
                        frame,
                        conf=conf_threshold,
                        verbose=False,
                        stream=False,
                    )
                )

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id].lower()
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Normalise coordinates to [0, 1]
                    bbox = BoundingBox(
                        x1=x1 / frame_w, y1=y1 / frame_h,
                        x2=x2 / frame_w, y2=y2 / frame_h,
                    )

                    region = self._assign_region(bbox.center_x)
                    priority = settings.NAV_PRIORITY_SCORES.get(class_name, 1)
                    threat = self._score_to_threat(priority)

                    objects.append(DetectedObject(
                        class_name=class_name,
                        confidence=conf,
                        bbox=bbox,
                        region=region,
                        priority_score=priority,
                        threat_level=threat,
                    ))

        # Sort by priority descending so consumers can take head
        objects.sort(key=lambda o: o.priority_score, reverse=True)

        inference_ms = (time.perf_counter() - t0) * 1000

        return DetectionResult(
            frame_id=frame_id,
            timestamp_ms=int(time.time() * 1000),
            objects=objects,
            frame_width=frame_w,
            frame_height=frame_h,
            inference_time_ms=round(inference_ms, 2),
        )

    @staticmethod
    def _assign_region(center_x_norm: float) -> FrameRegion:
        """Map normalised centre x to LEFT / CENTER / RIGHT zone."""
        if center_x_norm < settings.NAV_LEFT_BOUNDARY:
            return FrameRegion.LEFT
        if center_x_norm > settings.NAV_RIGHT_BOUNDARY:
            return FrameRegion.RIGHT
        return FrameRegion.CENTER

    @staticmethod
    def _score_to_threat(priority: int) -> ThreatLevel:
        if priority >= 9:
            return ThreatLevel.CRITICAL
        if priority >= 7:
            return ThreatLevel.HIGH
        if priority >= 5:
            return ThreatLevel.MEDIUM
        if priority >= 3:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    @staticmethod
    def _mock_detections(w: int, h: int) -> List[DetectedObject]:
        """Synthetic detections for testing without a real model."""
        return [
            DetectedObject(
                class_name="person",
                confidence=0.92,
                bbox=BoundingBox(x1=0.35, y1=0.2, x2=0.65, y2=0.95),
                region=FrameRegion.CENTER,
                priority_score=6,
                threat_level=ThreatLevel.MEDIUM,
            )
        ]

    def cleanup(self):
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        logger.info("DetectorService cleaned up")
