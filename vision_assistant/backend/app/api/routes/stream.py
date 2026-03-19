"""
WebSocket Stream Route — real-time frame-by-frame processing.

Client sends: JSON { frame_id, image_base64, timestamp_ms, session_id, preferred_language }
Server sends: JSON StreamResponse (navigation decision + detections + optional audio)

One NavigationEngine is maintained per session (WebSocket connection) to preserve
temporal state (smoother + motion detector) across frames.
"""

import asyncio
import base64
import logging
import time
from typing import Dict

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.models.schemas import FramePayload, StreamResponse
from app.services.detection.detector import DetectorService
from app.services.depth.depth_estimator import DepthEstimatorService
from app.services.navigation.navigation_engine import NavigationEngine
from app.services.tts.tts_service import TTSService

logger = logging.getLogger(__name__)
router = APIRouter()

# Active sessions: session_id → NavigationEngine
_sessions: Dict[str, NavigationEngine] = {}
_session_lock = asyncio.Lock()


async def _get_or_create_engine(session_id: str) -> NavigationEngine:
    async with _session_lock:
        if session_id not in _sessions:
            _sessions[session_id] = NavigationEngine()
            logger.info("Created NavigationEngine for session %s", session_id)
        return _sessions[session_id]


async def _cleanup_session(session_id: str):
    async with _session_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            logger.info("Cleaned up session %s (total active: %d)", session_id, len(_sessions))


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Primary WebSocket endpoint for real-time frame processing.

    Protocol:
      → Client sends base64 JPEG frames as JSON (FramePayload)
      ← Server responds with StreamResponse JSON per frame
    """
    await websocket.accept()
    session_id = None

    detector = DetectorService.get_instance()
    depth_svc = DepthEstimatorService.get_instance()
    tts_svc   = TTSService.get_instance()

    logger.info("WebSocket connected: %s", websocket.client)

    try:
        async for raw_message in websocket.iter_text():
            t_start = time.perf_counter()

            # ── Parse incoming frame ──────────────────────────────────────────
            try:
                payload = FramePayload.model_validate_json(raw_message)
            except (ValidationError, ValueError) as exc:
                await websocket.send_json({"error": f"Invalid payload: {exc}"})
                continue

            session_id = payload.session_id
            nav_engine = await _get_or_create_engine(session_id)

            # ── Decode image ──────────────────────────────────────────────────
            try:
                img_bytes = base64.b64decode(payload.image_base64)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("cv2.imdecode returned None")
            except Exception as exc:
                logger.warning("Frame decode error [%s]: %s", payload.frame_id, exc)
                await websocket.send_json({"error": "Frame decode failed"})
                continue

            # ── Parallel inference: detection + depth ─────────────────────────
            detection_task = asyncio.create_task(
                detector.detect(frame, frame_id=payload.frame_id)
            )
            depth_task = asyncio.create_task(
                depth_svc.estimate(frame, frame_id=payload.frame_id)
            )

            detection_result, depth_result = await asyncio.gather(
                detection_task, depth_task, return_exceptions=True
            )

            # Handle partial failures gracefully
            if isinstance(detection_result, Exception):
                logger.error("Detection failed: %s", detection_result)
                await websocket.send_json({"error": "Detection failed"})
                continue
            if isinstance(depth_result, Exception):
                logger.warning("Depth estimation failed: %s", depth_result)
                depth_result = None  # continue without depth

            # ── Navigation decision ───────────────────────────────────────────
            nav_result = nav_engine.process(detection_result, depth_result)

            # ── TTS synthesis (only if we should speak) ───────────────────────
            audio_b64 = None
            if nav_result.should_speak:
                lang = payload.preferred_language
                text = nav_result.speech_text_hi if lang == "hi" else nav_result.speech_text_en
                audio_b64 = await tts_svc.synthesize(text, lang=lang)

            # ── Build and send response ───────────────────────────────────────
            server_ms = (time.perf_counter() - t_start) * 1000

            response = StreamResponse(
                frame_id=payload.frame_id,
                session_id=session_id,
                navigation=nav_result,
                detections=detection_result,
                depth=depth_result,
                audio_base64=audio_b64,
                server_processing_ms=round(server_ms, 2),
            )

            await websocket.send_text(response.model_dump_json())

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
    except Exception as exc:
        logger.exception("Unexpected WebSocket error: %s", exc)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        if session_id:
            await _cleanup_session(session_id)
