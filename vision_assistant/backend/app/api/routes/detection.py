"""
Detection REST Route — single-frame HTTP endpoint.

Used for:
  - One-shot scene descriptions (non-real-time)
  - Integration testing
  - Offline photo processing
"""

import logging

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.models.schemas import DetectionResult, NavigationResult, SceneDescription
from app.services.detection.detector import DetectorService
from app.services.depth.depth_estimator import DepthEstimatorService
from app.services.navigation.navigation_engine import NavigationEngine
from app.services.tts.tts_service import TTSService

logger = logging.getLogger(__name__)
router = APIRouter()


class AnalyseResponse(BaseModel):
    detection: DetectionResult
    navigation: NavigationResult
    scene: SceneDescription
    audio_base64: str | None = None


@router.post("/analyse", response_model=AnalyseResponse)
async def analyse_frame(
    file: UploadFile = File(..., description="JPEG or PNG image"),
    lang: str = "en",
    with_audio: bool = True,
):
    """
    Analyse a single uploaded image frame.
    Returns detection results, navigation decision, scene description, and optional TTS audio.
    """
    # Read and decode uploaded image
    contents = await file.read()
    np_arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    detector = DetectorService.get_instance()
    depth_svc = DepthEstimatorService.get_instance()
    tts_svc   = TTSService.get_instance()

    # Run detection + depth
    detection = await detector.detect(frame)
    depth = await depth_svc.estimate(frame, frame_id=detection.frame_id)

    # Navigation
    engine = NavigationEngine()
    nav = engine.process(detection, depth)

    # Scene description
    scene = _build_scene_description(detection, nav, depth)

    # TTS
    audio = None
    if with_audio:
        text = nav.speech_text_hi if lang == "hi" else nav.speech_text_en
        audio = await tts_svc.synthesize(text, lang=lang)

    return AnalyseResponse(
        detection=detection,
        navigation=nav,
        scene=scene,
        audio_base64=audio,
    )


def _build_scene_description(detection, nav, depth) -> SceneDescription:
    objects = detection.objects
    class_counts: dict = {}
    for o in objects:
        class_counts[o.class_name] = class_counts.get(o.class_name, 0) + 1

    dominant = sorted(class_counts, key=class_counts.get, reverse=True)[:3]

    count = len(objects)
    if count == 0:
        desc_en = "The path appears clear. No obstacles detected."
        desc_hi = "रास्ता साफ़ लग रहा है। कोई बाधा नहीं है।"
    else:
        parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in list(class_counts.items())[:3]]
        desc_en = f"Detected {count} object{'s' if count > 1 else ''}: {', '.join(parts)}."
        desc_hi = f"{count} वस्तुएं पाई गईं: {', '.join(dominant[:2])}।"

    return SceneDescription(
        frame_id=detection.frame_id,
        description_en=desc_en,
        description_hi=desc_hi,
        detected_count=count,
        dominant_objects=dominant,
        path_clear=nav.decision.value in ("FORWARD", "CLEAR"),
        ground_clearance=nav.zones[1].threat_level if len(nav.zones) > 1 else nav.zones[0].threat_level,
    )
