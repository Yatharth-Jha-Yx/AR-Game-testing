"""Per-player joint smoothing, confidence gating and look-ahead prediction.

Units: filtering happens in "pixels at 1080p" so the One-Euro parameters
(min_cutoff ~1 Hz, beta ~0.01) mean the same thing at any camera
resolution.

Rules from the design plan:
* joint confidence < gate_conf (0.3): hold its last smoothed position for up
  to hold_frames (5) frames, then mark it invisible (confidence 0);
* predict one frame ahead with the filter's velocity to hide capture delay,
  but only for joints with confidence >= predict_min_conf (0.5), so a shaky
  joint doesn't overshoot.
"""

from __future__ import annotations

import numpy as np

from ..config import TrackingConfig
from ..skeleton import NUM_KEYPOINTS, J
from .filters import OneEuroFilter

# Multipliers on the base beta (speed responsiveness) per joint: wrists and
# feet move fast and need low lag; hips/torso are slow and should be calm.
_BETA_SCALE = np.ones(NUM_KEYPOINTS)
for _j, _s in {J.LEFT_HIP: 0.4, J.RIGHT_HIP: 0.4, J.HIP: 0.4, J.NECK: 0.6,
               J.LEFT_SHOULDER: 0.6, J.RIGHT_SHOULDER: 0.6,
               J.NOSE: 0.7, J.LEFT_EYE: 0.7, J.RIGHT_EYE: 0.7, J.LEFT_EAR: 0.7,
               J.RIGHT_EAR: 0.7, J.HEAD: 0.7, J.LEFT_KNEE: 0.8, J.RIGHT_KNEE: 0.8,
               J.LEFT_WRIST: 2.0, J.RIGHT_WRIST: 2.0, J.LEFT_ANKLE: 1.2, J.RIGHT_ANKLE: 1.2,
               J.LEFT_BIG_TOE: 1.2, J.RIGHT_BIG_TOE: 1.2, J.LEFT_SMALL_TOE: 1.2,
               J.RIGHT_SMALL_TOE: 1.2, J.LEFT_HEEL: 1.2, J.RIGHT_HEEL: 1.2}.items():
    _BETA_SCALE[_j] = _s
_CUTOFF_SCALE = np.ones(NUM_KEYPOINTS)
_CUTOFF_SCALE[[J.LEFT_HIP, J.RIGHT_HIP, J.HIP]] = 0.8


class JointSmoother:
    def __init__(self, cfg: TrackingConfig, frame_height: int):
        self.cfg = cfg
        self.to_1080 = 1080.0 / float(frame_height)
        self.filter = OneEuroFilter(
            (NUM_KEYPOINTS, 2),
            min_cutoff=(cfg.min_cutoff * _CUTOFF_SCALE)[:, None],
            beta=(cfg.beta * _BETA_SCALE)[:, None],
            d_cutoff=cfg.d_cutoff,
        )
        self.held = np.zeros(NUM_KEYPOINTS, np.int32)
        self.conf = np.zeros(NUM_KEYPOINTS, np.float32)
        self.smoothed: np.ndarray | None = None      # pixels, raw camera orientation
        self.t = 0.0

    def update(self, kpts: np.ndarray, scores: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        good = scores >= self.cfg.gate_conf
        if self.smoothed is None:
            # Never seen before: initialise everything, even weak joints.
            out = self.filter(kpts * self.to_1080, t) / self.to_1080
        else:
            out = self.filter(kpts * self.to_1080, t, mask=good[:, None]) / self.to_1080
        self.held = np.where(good, 0, self.held + 1)
        visible = good | (self.held <= self.cfg.hold_frames)
        # Held joints keep reporting a decaying version of their last confidence.
        self.conf = np.where(good, scores, np.where(visible, self.conf * 0.8, 0.0)).astype(np.float32)
        self.smoothed = out.astype(np.float32)
        self.t = t
        return self.smoothed, self.conf

    @property
    def velocity(self) -> np.ndarray:
        """Joint velocity in pixels/second (raw orientation)."""
        return (self.filter.dx / self.to_1080).astype(np.float32)

    def predicted(self, ahead_s: float) -> np.ndarray:
        """Smoothed joints pushed ``ahead_s`` seconds into the future."""
        assert self.smoothed is not None
        if not self.cfg.predict or ahead_s <= 0:
            return self.smoothed
        ok = (self.conf >= self.cfg.predict_min_conf)[:, None]
        return np.where(ok, self.smoothed + self.velocity * ahead_s, self.smoothed).astype(np.float32)

    def extrapolate(self, t: float) -> np.ndarray:
        """Position at time t without a new measurement (frames we skip)."""
        assert self.smoothed is not None
        return self.predicted(max(0.0, t - self.t))
