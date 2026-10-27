"""Plain data containers passed between engine stages.

Pixel coordinates inside the engine are always in the *raw* camera image
(not mirrored). Mirroring and 0..1 normalisation happen once, when a message
is built for games (see protocol.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame:
    image: np.ndarray            # BGR uint8, H x W x 3
    t_capture: float             # time.perf_counter() seconds when grabbed
    seq: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


@dataclass
class PoseCandidate:
    """One person as seen by a pose model on one frame (no identity yet)."""

    box: np.ndarray              # (4,) x0 y0 x1 y1 pixels
    kpts: np.ndarray             # (26, 2) pixels
    scores: np.ndarray           # (26,) 0..1
    det_score: float = 1.0
    z: np.ndarray | None = None  # (26,) relative depth if the model gives one
    # True when the pose came from a person-detector box (or a one-stage model),
    # False when it came from re-cropping an already tracked player. Only
    # detections may create new players.
    from_detection: bool = True
    in_frame: np.ndarray | None = None   # (26,) bool, joint inside the picture

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores))


@dataclass
class PlayerState:
    """A tracked, smoothed player as published to games."""

    id: int
    kpts: np.ndarray             # (26, 2) smoothed (+predicted) pixels
    conf: np.ndarray             # (26,) 0 = hidden
    box: np.ndarray              # (4,) pixels
    confidence: float
    torso_px: float
    depth_m: float
    depth_calibrated: bool
    position_m: tuple[float, float]   # (lateral x, depth z), metres, camera frame
    in_bounds: bool
    bounds_hint: str | None = None    # "step_back", "move_left", ... (screen terms)
    predicted: bool = False           # True = extrapolated, no pose run this frame
    reach_torso: float = 1.6          # half wingspan in torso lengths


@dataclass
class GestureEvent:
    player_id: int
    gesture: str                 # jump | squat | punch | swipe | arm_raise | lean
    phase: str                   # trigger | start | end
    strength: float              # 0..1
    t: float                     # engine time (s) of the frame that fired it
    side: str | None = None      # anatomical arm/leg: left | right | both
    direction: str | None = None # screen direction: left | right | up | down


@dataclass
class FrameResult:
    """Everything the engine learned from one camera frame."""

    frame: Frame
    players: list[PlayerState]
    gestures: list[GestureEvent] = field(default_factory=list)
    player_events: list[dict[str, Any]] = field(default_factory=list)
    # Ready-to-send JSON messages (calibration progress, camera lock report...).
    messages: list[dict[str, Any]] = field(default_factory=list)
    mask: np.ndarray | None = None       # uint8 person mask, raw orientation
    timings_ms: dict[str, float] = field(default_factory=dict)
    t_done: float = 0.0
