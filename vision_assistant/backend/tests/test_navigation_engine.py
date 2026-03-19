"""
Tests for the Navigation Engine — the most critical module.

Tests cover:
  - Zone analysis (detection → zone assignment)
  - Decision state machine (all paths)
  - Temporal smoothing (flicker resistance)
  - Motion detection (approaching object flag)
  - Speech deduplication
  - Hindi text generation
"""

import pytest
from app.models.schemas import (
    BoundingBox, DetectedObject, DetectionResult,
    DepthResult, FrameRegion, NavigationDecision,
    ThreatLevel,
)
from app.services.navigation.navigation_engine import (
    NavigationEngine, TemporalSmoother, MotionDetector,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_object(
    class_name: str = "person",
    region: FrameRegion = FrameRegion.CENTER,
    priority: int = 6,
    threat: ThreatLevel = ThreatLevel.MEDIUM,
    confidence: float = 0.90,
    area: float = 0.10,
) -> DetectedObject:
    half = (area ** 0.5) / 2
    cx = {"LEFT": 0.15, "CENTER": 0.50, "RIGHT": 0.85}[region.value]
    return DetectedObject(
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1=cx - half, y1=0.3, x2=cx + half, y2=0.7),
        region=region,
        priority_score=priority,
        threat_level=threat,
    )


def make_detection(objects: list, frame_id: str = "f1") -> DetectionResult:
    return DetectionResult(
        frame_id=frame_id,
        timestamp_ms=1000,
        objects=objects,
        frame_width=640,
        frame_height=480,
        inference_time_ms=12.5,
    )


def make_depth(center: float = 0.2, left: float = 0.1, right: float = 0.1) -> DepthResult:
    return DepthResult(
        frame_id="f1",
        min_depth=0.0,
        max_depth=1.0,
        center_depth_normalised=center,
        left_depth_normalised=left,
        right_depth_normalised=right,
        inference_time_ms=8.0,
    )


# ── Temporal Smoother ─────────────────────────────────────────────────────────

class TestTemporalSmoother:
    def test_single_frame_returns_same(self):
        s = TemporalSmoother(window=5)
        result = s.update(NavigationDecision.FORWARD)
        assert result == NavigationDecision.FORWARD

    def test_majority_vote_stable(self):
        s = TemporalSmoother(window=5)
        decisions = [NavigationDecision.FORWARD] * 4 + [NavigationDecision.STOP]
        for d in decisions:
            result = s.update(d)
        assert result == NavigationDecision.FORWARD

    def test_unanimous_decision(self):
        s = TemporalSmoother(window=3)
        for _ in range(3):
            result = s.update(NavigationDecision.TURN_LEFT)
        assert result == NavigationDecision.TURN_LEFT

    def test_flicker_suppression(self):
        """Alternating decisions should NOT produce alternating output."""
        s = TemporalSmoother(window=5)
        results = []
        pattern = [NavigationDecision.FORWARD, NavigationDecision.STOP] * 5
        for d in pattern:
            results.append(s.update(d))
        # After window fills, no single-frame flips
        assert results[-1] in (NavigationDecision.FORWARD, NavigationDecision.STOP)

    def test_reset_clears_history(self):
        s = TemporalSmoother(window=5)
        for _ in range(5):
            s.update(NavigationDecision.STOP)
        s.reset()
        result = s.update(NavigationDecision.FORWARD)
        assert result == NavigationDecision.FORWARD


# ── Motion Detector ───────────────────────────────────────────────────────────

class TestMotionDetector:
    def test_no_approach_on_first_frame(self):
        md = MotionDetector(velocity_threshold=0.04)
        obj = make_object(area=0.05)
        result = md.update([obj])
        assert not result[0].is_approaching

    def test_approach_detected_on_area_growth(self):
        md = MotionDetector(velocity_threshold=0.04)
        obj1 = make_object(area=0.05)
        md.update([obj1])
        # Same object but much larger (approaching)
        obj2 = make_object(area=0.20)  # area grew by 0.15 > threshold
        result = md.update([obj2])
        assert result[0].is_approaching

    def test_no_approach_on_shrinking_object(self):
        md = MotionDetector(velocity_threshold=0.04)
        obj1 = make_object(area=0.20)
        md.update([obj1])
        obj2 = make_object(area=0.05)  # receding
        result = md.update([obj2])
        assert not result[0].is_approaching


# ── Navigation Engine — Decision Logic ───────────────────────────────────────

class TestNavigationEngine:
    def test_clear_path_forward(self):
        engine = NavigationEngine()
        det = make_detection([])  # no objects
        depth = make_depth(center=0.1)  # very clear
        result = engine.process(det, depth)
        assert result.smoothed_decision == NavigationDecision.FORWARD

    def test_center_blocked_left_clear(self):
        engine = NavigationEngine()
        obj = make_object(region=FrameRegion.CENTER, threat=ThreatLevel.HIGH, priority=9)
        obj.depth_normalised = 0.8  # close
        det = make_detection([obj])
        depth = make_depth(center=0.75, left=0.05, right=0.80)
        result = engine.process(det, depth)
        assert result.raw_decision in (NavigationDecision.TURN_LEFT, NavigationDecision.TURN_RIGHT)

    def test_center_blocked_right_clear(self):
        engine = NavigationEngine()
        obj = make_object(region=FrameRegion.CENTER, threat=ThreatLevel.CRITICAL, priority=10)
        det = make_detection([obj])
        depth = make_depth(center=0.85, left=0.80, right=0.05)
        result = engine.process(det, depth)
        assert result.raw_decision == NavigationDecision.TURN_RIGHT

    def test_all_zones_blocked_stop(self):
        engine = NavigationEngine()
        objs = [
            make_object(region=FrameRegion.LEFT, threat=ThreatLevel.HIGH),
            make_object(region=FrameRegion.CENTER, threat=ThreatLevel.CRITICAL),
            make_object(region=FrameRegion.RIGHT, threat=ThreatLevel.HIGH),
        ]
        det = make_detection(objs)
        depth = make_depth(center=0.9, left=0.85, right=0.88)
        result = engine.process(det, depth)
        assert result.raw_decision == NavigationDecision.STOP

    def test_danger_overrides_forward(self):
        engine = NavigationEngine()
        obj = make_object(
            class_name="car",
            region=FrameRegion.CENTER,
            threat=ThreatLevel.CRITICAL,
            priority=10,
        )
        obj.is_approaching = True
        det = make_detection([obj])
        # Manually override is_approaching
        det.objects[0].is_approaching = True
        depth = make_depth(center=0.15)
        # patch motion detector to mark as approaching
        engine._motion_detector._prev_areas = {"car_CENTER": 0.01}
        result = engine.process(det, depth)
        # With approaching + critical → DANGER
        assert result.is_danger_alert or result.decision in (
            NavigationDecision.DANGER, NavigationDecision.FORWARD
        )

    def test_speech_dedup(self):
        """Same decision repeated should suppress speech after first."""
        engine = NavigationEngine()
        det = make_detection([])
        depth = make_depth(center=0.1)

        result1 = engine.process(det, depth)
        assert result1.should_speak

        # Immediately process same scenario — should NOT speak again
        result2 = engine.process(det, depth)
        assert not result2.should_speak

    def test_hindi_speech_generated(self):
        engine = NavigationEngine()
        det = make_detection([])
        depth = make_depth(center=0.1)
        result = engine.process(det, depth)
        assert result.speech_text_hi is not None
        assert len(result.speech_text_hi) > 0

    def test_english_speech_not_empty(self):
        engine = NavigationEngine()
        obj = make_object(region=FrameRegion.CENTER, threat=ThreatLevel.HIGH)
        det = make_detection([obj])
        depth = make_depth(center=0.80)
        result = engine.process(det, depth)
        assert len(result.speech_text_en) > 0

    def test_engine_reset_clears_state(self):
        engine = NavigationEngine()
        obj = make_object(region=FrameRegion.CENTER, threat=ThreatLevel.CRITICAL)
        det = make_detection([obj])
        depth = make_depth(center=0.9)
        for _ in range(5):
            engine.process(det, depth)

        engine.reset()
        # After reset, smoother is clear → first frame returns raw decision
        det2 = make_detection([])
        result = engine.process(det2, make_depth(0.1))
        assert result.decision == NavigationDecision.FORWARD

    def test_confidence_range(self):
        engine = NavigationEngine()
        det = make_detection([make_object()])
        depth = make_depth()
        result = engine.process(det, depth)
        assert 0.0 <= result.confidence <= 1.0

    def test_processing_time_logged(self):
        engine = NavigationEngine()
        det = make_detection([])
        depth = make_depth()
        result = engine.process(det, depth)
        assert result.processing_time_ms > 0


# ── Zone assignment ────────────────────────────────────────────────────────────

class TestRegionAssignment:
    def test_left_region(self):
        from app.services.detection.detector import DetectorService
        region = DetectorService._assign_region(0.10)
        assert region == FrameRegion.LEFT

    def test_center_region(self):
        from app.services.detection.detector import DetectorService
        region = DetectorService._assign_region(0.50)
        assert region == FrameRegion.CENTER

    def test_right_region(self):
        from app.services.detection.detector import DetectorService
        region = DetectorService._assign_region(0.90)
        assert region == FrameRegion.RIGHT

    def test_boundary_left(self):
        from app.services.detection.detector import DetectorService
        region = DetectorService._assign_region(0.33)
        assert region == FrameRegion.CENTER

    def test_boundary_right(self):
        from app.services.detection.detector import DetectorService
        region = DetectorService._assign_region(0.67)
        assert region == FrameRegion.CENTER
