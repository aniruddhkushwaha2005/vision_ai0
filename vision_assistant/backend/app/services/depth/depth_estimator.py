"""
Depth Estimation Service — MiDaS monocular depth.

Uses Intel's MiDaS (Mix Depth Supervision) to produce per-pixel relative depth.
Output is a normalised float map: 0.0 = very far, 1.0 = very close.

Note: MiDaS produces *relative* depth — we normalise it consistently per-frame
so downstream code can apply threshold comparisons.
"""

import asyncio
import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.models.schemas import DepthResult

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — DepthEstimatorService in mock mode")


class DepthEstimatorService:
    """
    Singleton MiDaS depth estimator.

    Returns per-frame DepthResult with:
      - Raw depth map (H, W) float32
      - Zone-averaged depths (left / center / right)
      - Normalised range [0.0 → 1.0] where 1.0 = closest
    """
    _instance: Optional["DepthEstimatorService"] = None

    def __init__(self):
        self._model = None
        self._transform = None
        self._device = None
        self._lock = asyncio.Lock()
        self._load_model()

    @classmethod
    def get_instance(cls) -> "DepthEstimatorService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        if not TORCH_AVAILABLE:
            return

        try:
            device_str = settings.MIDAS_DEVICE
            self._device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

            model_type = settings.MIDAS_MODEL_TYPE
            self._model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            self._model.to(self._device)
            self._model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if model_type in ("DPT_Large", "DPT_Hybrid"):
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform

            logger.info("✅ MiDaS loaded (%s) on %s", model_type, self._device)
        except Exception as exc:
            logger.error("Failed to load MiDaS: %s", exc)
            self._model = None

    async def estimate(
        self,
        frame: np.ndarray,
        frame_id: Optional[str] = None,
    ) -> DepthResult:
        """
        Estimate depth for a BGR numpy frame.

        Returns DepthResult with zone depths and optional base64 depth map.
        """
        t0 = time.perf_counter()

        if self._model is None or not TORCH_AVAILABLE:
            return self._mock_depth(frame_id or "mock", t0)

        async with self._lock:
            depth_map = await asyncio.get_event_loop().run_in_executor(
                None, self._infer, frame
            )

        # Normalise to [0,1]: higher = closer (invert MiDaS convention)
        depth_norm = self._normalise(depth_map)

        left, center, right = self._zone_averages(depth_norm)

        inference_ms = (time.perf_counter() - t0) * 1000

        return DepthResult(
            frame_id=frame_id or "unknown",
            depth_map_base64=None,          # skip serialisation for latency; enable when needed
            min_depth=float(depth_norm.min()),
            max_depth=float(depth_norm.max()),
            center_depth_normalised=round(center, 4),
            left_depth_normalised=round(left, 4),
            right_depth_normalised=round(right, 4),
            inference_time_ms=round(inference_ms, 2),
        )

    def _infer(self, frame: np.ndarray) -> np.ndarray:
        """Synchronous inference — called via executor to avoid blocking event loop."""
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()

    @staticmethod
    def _normalise(depth_map: np.ndarray) -> np.ndarray:
        """Normalise to [0,1]; invert so 1.0 = closest object."""
        d_min = depth_map.min()
        d_max = depth_map.max()
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth_map, dtype=np.float32)
        normed = (depth_map - d_min) / (d_max - d_min)
        return (1.0 - normed).astype(np.float32)   # invert: closer = higher value

    @staticmethod
    def _zone_averages(depth_norm: np.ndarray) -> Tuple[float, float, float]:
        """Return average depth in LEFT / CENTER / RIGHT horizontal thirds."""
        h, w = depth_norm.shape
        l_bound = int(w * settings.NAV_LEFT_BOUNDARY)
        r_bound = int(w * settings.NAV_RIGHT_BOUNDARY)

        left   = float(depth_norm[:, :l_bound].mean())
        center = float(depth_norm[:, l_bound:r_bound].mean())
        right  = float(depth_norm[:, r_bound:].mean())
        return left, center, right

    @staticmethod
    def _mock_depth(frame_id: str, t0: float) -> DepthResult:
        """Return plausible mock depth for unit testing."""
        return DepthResult(
            frame_id=frame_id,
            min_depth=0.1,
            max_depth=0.9,
            center_depth_normalised=0.35,
            left_depth_normalised=0.20,
            right_depth_normalised=0.15,
            inference_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    def cleanup(self):
        if self._model is not None:
            del self._model
            self._model = None
            if TORCH_AVAILABLE:
                import torch
                torch.cuda.empty_cache()
        logger.info("DepthEstimatorService cleaned up")
