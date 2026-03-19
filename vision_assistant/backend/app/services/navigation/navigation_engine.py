"""
Navigation Engine — Core Intelligence Module

This is the brain of the Vision Assistant. It takes detection + depth outputs
and produces a stable, noise-free navigation decision with appropriate voice text.

Architecture:
  1. Frame Segmentation  — L/C/R zone analysis
  2. Object Prioritizer  — risk-score-based threat ranking
  3. Path Clearance      — lower-frame walkable ground check
  4. Decision Logic      — deterministic state machine
  5. Temporal Smoother   — majority vote over last N frames
  6. Motion Detector     — velocity-based danger detection
  7. Feedback Generator  — deduped, localised speech text
"""

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.models.schemas import (
    DetectedObject, DetectionResult, DepthResult,
    FrameRegion, NavigationDecision, NavigationResult,
    ThreatLevel, ZoneAnalysis,
)

logger = logging.getLogger(__name__)


# ── Voice strings ─────────────────────────────────────────────────────────────
SPEECH_MAP_EN: Dict[NavigationDecision, str] = {
    NavigationDecision.FORWARD:    "Path is clear. Move forward.",
    NavigationDecision.TURN_LEFT:  "Obstacle ahead. Turn left.",
    NavigationDecision.TURN_RIGHT: "Obstacle ahead. Turn right.",
    NavigationDecision.STOP:       "Stop! Path is completely blocked.",
    NavigationDecision.DANGER:     "Danger! Fast-moving object approaching. Stop immediately.",
    NavigationDecision.CLEAR:      "All clear. You may proceed.",
}

SPEECH_MAP_HI: Dict[NavigationDecision, str] = {
    NavigationDecision.FORWARD:    "रास्ता साफ है। आगे बढ़ें।",
    NavigationDecision.TURN_LEFT:  "रास्ते में बाधा है। बाईं ओर मुड़ें।",
    NavigationDecision.TURN_RIGHT: "रास्ते में बाधा है। दाईं ओर मुड़ें।",
    NavigationDecision.STOP:       "रुकिए! रास्ता पूरी तरह बंद है।",
    NavigationDecision.DANGER:     "खतरा! तेज़ गति से वाहन आ रहा है। तुरंत रुकें।",
    NavigationDecision.CLEAR:      "रास्ता बिल्कुल साफ है। आगे जाएं।",
}

OBJECT_NAME_HI: Dict[str, str] = {
    "car": "गाड़ी", "truck": "ट्रक", "bus": "बस",
    "person": "व्यक्ति", "bicycle": "साइकिल",
    "motorcycle": "मोटरसाइकिल", "dog": "कुत्ता",
    "chair": "कुर्सी", "table": "मेज़",
}


class TemporalSmoother:
    """
    Majority-vote smoother over a sliding window of N frame decisions.
    Eliminates single-frame flickers (e.g. object briefly leaving/entering frame).
    """

    def __init__(self, window: int = 5):
        self._window = window
        self._history: deque = deque(maxlen=window)

    def update(self, decision: NavigationDecision) -> NavigationDecision:
        self._history.append(decision)
        if len(self._history) < 2:
            return decision
        # Majority vote — ties broken by most recent
        counts: Dict[NavigationDecision, int] = {}
        for d in self._history:
            counts[d] = counts.get(d, 0) + 1
        return max(counts, key=lambda k: (counts[k], list(self._history).index(k)))

    def reset(self):
        self._history.clear()


class MotionDetector:
    """
    Tracks bounding box centroids across frames to detect approaching objects.
    An object is flagged as approaching if its bbox area grows faster than
    the configured threshold between consecutive frames.
    """

    def __init__(self, velocity_threshold: float = 0.04):
        self._velocity_threshold = velocity_threshold
        self._prev_areas: Dict[str, float] = {}

    def update(self, objects: List[DetectedObject]) -> List[DetectedObject]:
        """Mutate objects in-place: set is_approaching flag."""
        current_areas: Dict[str, float] = {}

        for obj in objects:
            key = f"{obj.class_name}_{obj.region.value}"
            area = obj.bbox.area
            current_areas[key] = area

            if key in self._prev_areas:
                delta = area - self._prev_areas[key]
                if delta > self._velocity_threshold:
                    obj.is_approaching = True
                    logger.debug("Approaching: %s Δarea=%.4f", obj.class_name, delta)

        self._prev_areas = current_areas
        return objects


class NavigationEngine:
    """
    Stateful navigation engine — maintains temporal smoother and motion detector
    across consecutive frames for a single session.

    Usage (per session):
        engine = NavigationEngine()
        result = engine.process(detection_result, depth_result)
    """

    def __init__(self):
        self._smoother = TemporalSmoother(window=settings.NAV_SMOOTHING_WINDOW)
        self._motion_detector = MotionDetector()
        self._last_decision: Optional[NavigationDecision] = None
        self._last_spoken_at: float = 0.0
        self._frame_count: int = 0

    def process(
        self,
        detection: DetectionResult,
        depth: Optional[DepthResult] = None,
    ) -> NavigationResult:
        """
        Main entry point: process one frame's detection + depth into a navigation decision.

        Args:
            detection: output of DetectorService.detect()
            depth:     output of DepthEstimatorService.estimate() (optional)

        Returns:
            NavigationResult with decision, speech text, danger flags
        """
        t0 = time.perf_counter()
        self._frame_count += 1

        # Step 1: enrich objects with depth information
        objects = self._fuse_depth(detection.objects, depth)

        # Step 2: detect motion / approach vectors
        objects = self._motion_detector.update(objects)

        # Step 3: analyse each zone
        zones = self._analyse_zones(objects, depth)

        # Step 4: danger check (highest priority — overrides everything)
        is_danger, danger_obj = self._check_danger(objects)
        if is_danger:
            raw_decision = NavigationDecision.DANGER
        else:
            raw_decision = self._decide(zones)

        # Step 5: temporal smoothing
        smoothed_decision = self._smoother.update(raw_decision)

        # Step 6: dedup speech — only speak when decision changes
        dominant = self._dominant_threat(objects)
        speech_en = self._build_speech_en(smoothed_decision, dominant)
        speech_hi = self._build_speech_hi(smoothed_decision, dominant)
        should_speak = self._should_speak(smoothed_decision)

        if should_speak:
            self._last_decision = smoothed_decision
            self._last_spoken_at = time.time()

        confidence = self._compute_confidence(zones, objects)
        processing_ms = (time.perf_counter() - t0) * 1000

        return NavigationResult(
            frame_id=detection.frame_id,
            decision=smoothed_decision,
            confidence=round(confidence, 3),
            zones=zones,
            dominant_threat=dominant,
            speech_text_en=speech_en,
            speech_text_hi=speech_hi,
            should_speak=should_speak,
            raw_decision=raw_decision,
            smoothed_decision=smoothed_decision,
            is_danger_alert=is_danger,
            processing_time_ms=round(processing_ms, 3),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fuse_depth(
        objects: List[DetectedObject],
        depth: Optional[DepthResult],
    ) -> List[DetectedObject]:
        """
        If depth result available, look up depth at each object's bounding box centre.
        This approximation works well for MiDaS with normalised coordinates.
        """
        if depth is None:
            return objects

        # Use zone-level depth as proxy (full depth map not deserialised here for speed)
        zone_depths = {
            FrameRegion.LEFT:   depth.left_depth_normalised,
            FrameRegion.CENTER: depth.center_depth_normalised,
            FrameRegion.RIGHT:  depth.right_depth_normalised,
        }
        for obj in objects:
            obj.depth_normalised = zone_depths.get(obj.region, 0.0)
            # Rough distance estimate: assume depth_norm=1.0 → ~0.5m, 0.0 → >10m
            if obj.depth_normalised is not None:
                obj.estimated_distance_m = round(
                    max(0.3, (1.0 - obj.depth_normalised) * 10.0), 1
                )
        return objects

    @staticmethod
    def _analyse_zones(
        objects: List[DetectedObject],
        depth: Optional[DepthResult],
    ) -> List[ZoneAnalysis]:
        """Build zone-level summary for LEFT, CENTER, RIGHT."""
        zone_map: Dict[FrameRegion, List[DetectedObject]] = {
            FrameRegion.LEFT: [],
            FrameRegion.CENTER: [],
            FrameRegion.RIGHT: [],
        }
        for obj in objects:
            zone_map[obj.region].append(obj)

        zone_depths = {
            FrameRegion.LEFT:   depth.left_depth_normalised if depth else 0.0,
            FrameRegion.CENTER: depth.center_depth_normalised if depth else 0.0,
            FrameRegion.RIGHT:  depth.right_depth_normalised if depth else 0.0,
        }

        analyses = []
        for region, obs in zone_map.items():
            zone_depth = zone_depths[region]
            # A zone is blocked if it contains a high-priority object AND depth is close
            top = max(obs, key=lambda o: o.priority_score, default=None)
            if top is not None:
                depth_close = zone_depth > settings.NAV_CENTER_BLOCK_THRESH
                blocked = depth_close or top.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)
            else:
                blocked = zone_depth > settings.NAV_DANGER_DEPTH_THRESH

            analyses.append(ZoneAnalysis(
                region=region,
                is_blocked=blocked,
                highest_priority_object=top.class_name if top else None,
                min_depth_normalised=round(zone_depth, 4),
                threat_level=top.threat_level if top else ThreatLevel.NONE,
            ))
        return analyses

    @staticmethod
    def _check_danger(objects: List[DetectedObject]) -> Tuple[bool, Optional[DetectedObject]]:
        """Return (is_danger, object) if any critical fast-approaching object detected."""
        for obj in objects:
            if obj.is_approaching and obj.threat_level == ThreatLevel.CRITICAL:
                return True, obj
        return False, None

    @staticmethod
    def _decide(zones: List[ZoneAnalysis]) -> NavigationDecision:
        """
        Core decision state machine.

        Priority order:
          1. CENTER clear → FORWARD
          2. CENTER blocked, LEFT clear → TURN_LEFT
          3. CENTER blocked, RIGHT clear → TURN_RIGHT
          4. All blocked → STOP
        """
        zone_dict = {z.region: z for z in zones}
        c = zone_dict.get(FrameRegion.CENTER)
        l = zone_dict.get(FrameRegion.LEFT)
        r = zone_dict.get(FrameRegion.RIGHT)

        if c is None or not c.is_blocked:
            return NavigationDecision.FORWARD

        # Center blocked — find best alternative
        l_blocked = l.is_blocked if l else True
        r_blocked = r.is_blocked if r else True

        if not l_blocked and not r_blocked:
            # Both free — prefer lower-depth zone (more open)
            l_depth = l.min_depth_normalised if l else 1.0
            r_depth = r.min_depth_normalised if r else 1.0
            return NavigationDecision.TURN_LEFT if l_depth <= r_depth else NavigationDecision.TURN_RIGHT

        if not l_blocked:
            return NavigationDecision.TURN_LEFT
        if not r_blocked:
            return NavigationDecision.TURN_RIGHT

        return NavigationDecision.STOP

    def _should_speak(self, decision: NavigationDecision) -> bool:
        """Suppress speech if decision unchanged and within rate-limit window."""
        if decision == NavigationDecision.DANGER:
            return True   # always announce danger
        if decision != self._last_decision:
            return True
        elapsed = time.time() - self._last_spoken_at
        return elapsed >= settings.TTS_RATE_LIMIT_SECS

    @staticmethod
    def _dominant_threat(objects: List[DetectedObject]) -> Optional[DetectedObject]:
        """Return highest-priority detected object (for speech personalisation)."""
        critical = [o for o in objects if o.region == FrameRegion.CENTER]
        if not critical:
            critical = objects
        return critical[0] if critical else None

    @staticmethod
    def _build_speech_en(decision: NavigationDecision, threat: Optional[DetectedObject]) -> str:
        base = SPEECH_MAP_EN.get(decision, "Proceed with caution.")
        if threat and decision not in (NavigationDecision.FORWARD, NavigationDecision.CLEAR):
            base = base.replace("Obstacle", f"{threat.class_name.capitalize()}")
        return base

    @staticmethod
    def _build_speech_hi(decision: NavigationDecision, threat: Optional[DetectedObject]) -> str:
        if not settings.TTS_HINDI_ENABLED:
            return ""
        base = SPEECH_MAP_HI.get(decision, "सावधानी से चलें।")
        if threat and threat.class_name in OBJECT_NAME_HI:
            hi_name = OBJECT_NAME_HI[threat.class_name]
            base = base.replace("बाधा", hi_name)
        return base

    @staticmethod
    def _compute_confidence(
        zones: List[ZoneAnalysis],
        objects: List[DetectedObject],
    ) -> float:
        """Confidence metric: blend of detection confidences and zone clarity."""
        if not objects:
            return 0.95
        avg_conf = sum(o.confidence for o in objects) / len(objects)
        blocked_zones = sum(1 for z in zones if z.is_blocked)
        zone_clarity = 1.0 - (blocked_zones / max(len(zones), 1)) * 0.3
        return min(1.0, avg_conf * zone_clarity)

    def reset(self):
        """Reset state for a new navigation session."""
        self._smoother.reset()
        self._last_decision = None
        self._frame_count = 0
        logger.debug("NavigationEngine state reset")
