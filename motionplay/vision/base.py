"""Common interface for every pose backend.

Two styles:
* top-down (RTMPose): ``detect()`` finds person boxes (expensive, run every
  N frames), ``estimate()`` runs the pose net on given boxes (cheap, every
  frame, boxes come from tracking).
* one-stage (RTMO, MediaPipe, synthetic): ``estimate_all()`` does both.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np

from ..types import Frame, PoseCandidate


class PoseBackend(ABC):
    name = "backend"
    one_stage = False
    provider = "cpu"

    def detect(self, frame: Frame) -> np.ndarray:
        """(N, 5) boxes: x0, y0, x1, y1, score (pixels)."""
        raise NotImplementedError

    def estimate(self, frame: Frame, boxes: np.ndarray) -> list[PoseCandidate]:
        raise NotImplementedError

    def estimate_all(self, frame: Frame) -> list[PoseCandidate]:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "one_stage": self.one_stage, "provider": self.provider}

    def close(self) -> None:
        pass


def boxes_from_keypoints(kpts: np.ndarray, scores: np.ndarray, thr: float = 0.3,
                         margin: float = 0.1) -> np.ndarray:
    """Tight box around confident keypoints, grown by ``margin`` per side."""
    ok = scores >= thr
    pts = kpts[ok] if ok.sum() >= 2 else kpts
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w, h = x1 - x0, y1 - y0
    return np.array([x0 - w * margin, y0 - h * margin, x1 + w * margin, y1 + h * margin * 0.5],
                    dtype=np.float32)
