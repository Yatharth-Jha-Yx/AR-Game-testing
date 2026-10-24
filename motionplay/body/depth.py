"""Distance to the camera without a depth sensor.

Apparent size shrinks in proportion to distance (pinhole camera), so

    depth = calibrated_distance * calibrated_size / current_size

for three cues: shoulder width, torso length (neck -> mid-hip) and head
height (head top -> neck). We take the *median* of the three so that
turning the shoulders (which shrinks only the shoulder width) does not read
as "stepped back".

Before calibration we fall back to typical adult sizes and the camera's
field of view, flagged ``calibrated = False`` (expect +-20-30%).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..config import PlayAreaConfig
from ..tracking.filters import OneEuroFilter
from .metrics import body_sizes

# Typical adult joint-to-joint sizes in metres (rough population averages;
# only used until the player calibrates).
DEFAULT_SIZES_M = {"shoulder": 0.36, "torso": 0.50, "head": 0.23}


@dataclass
class BodyCalibration:
    distance_m: float
    frame_height: int
    # Sizes in pixels at calibration, stored relative to frame height.
    sizes: dict[str, float]
    wingspan_torso: float = 3.2      # wrist-to-wrist in T-pose / torso length
    height_torso: float = 3.3        # head top to ankles / torso length
    hip_knee_torso: float = 0.85     # standing vertical hip->knee / torso length
    height_m: float = 0.0
    wingspan_m: float = 0.0
    floor_x: float = 0.5             # where they stood (0..1 of frame width)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BodyCalibration":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def focal_px(width: int, hfov_deg: float) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


class DepthEstimator:
    def __init__(self, play: PlayAreaConfig, frame_w: int, frame_h: int,
                 calibration: BodyCalibration | None = None):
        self.play = play
        self.w, self.h = frame_w, frame_h
        self.f = focal_px(frame_w, play.hfov_deg)
        self.calibration = calibration
        self._filter = OneEuroFilter((1,), min_cutoff=0.6, beta=0.4)
        self.depth = 0.0

    def estimate(self, kpts: np.ndarray, conf: np.ndarray, t: float) -> tuple[float, bool]:
        sizes = body_sizes(kpts, conf)
        raw = self.raw_depth(sizes)
        if raw is None:
            return self.depth, self.calibration is not None
        self.depth = float(self._filter(np.array([raw]), t)[0])
        return self.depth, self.calibration is not None

    def raw_depth(self, sizes: dict[str, float]) -> float | None:
        cal = self.calibration
        if cal is not None:
            ratios = [cal.sizes[k] * self.h / v for k, v in sizes.items() if k in cal.sizes]
            if ratios:
                return cal.distance_m * float(np.median(ratios))
        estimates = [self.f * DEFAULT_SIZES_M[k] / v for k, v in sizes.items()]
        return float(np.median(estimates)) if estimates else None

    def lateral_m(self, x_px: float, depth_m: float) -> float:
        """Sideways offset from the camera axis (+ = camera image right)."""
        return (x_px - self.w / 2) * depth_m / self.f
