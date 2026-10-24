"""Lock webcam exposure / focus / white balance after calibration.

Why: auto-exposure keeps hunting as players move, which causes brightness
flicker (= keypoint jitter), and in dim rooms it silently stretches the
exposure until the camera drops from 60 to 30 fps and wrists/ankles blur.

OpenCV exposes UVC controls through ``cv2.VideoCapture.set`` but the units
differ per capture backend (OpenCV docs, videoio "VideoCaptureProperties"):

* V4L2 (Linux): AUTO_EXPOSURE 3 = auto, 1 = manual;
  EXPOSURE = exposure_time_absolute in 100 microsecond units.
* DirectShow / Media Foundation (Windows): AUTO_EXPOSURE 0.75 = auto,
  0.25 = manual; EXPOSURE = log2(seconds), e.g. -7 = 1/128 s.
* AVFoundation (macOS): not controllable through OpenCV.

The lock runs as a small state machine fed one frame at a time from the
capture thread, because VideoCapture must only be touched by one thread.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ExposureUnits:
    auto_on: float
    auto_off: float
    to_driver: Callable[[float], float]      # seconds -> driver units
    from_driver: Callable[[float], float]    # driver units -> seconds


V4L2_UNITS = ExposureUnits(3.0, 1.0, lambda s: round(s * 10000.0), lambda v: v / 10000.0)
WINDOWS_UNITS = ExposureUnits(
    0.75, 0.25, lambda s: math.floor(math.log2(s)), lambda v: 2.0 ** v
)


def units_for_api(api_name: str) -> ExposureUnits | None:
    if api_name in ("v4l2",):
        return V4L2_UNITS
    if api_name in ("dshow", "msmf"):
        return WINDOWS_UNITS
    return None


def frame_luma(image: np.ndarray) -> float:
    small = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


@dataclass
class ExposureLocker:
    """Feed it frames; it measures auto brightness, locks a fast shutter and
    raises gain until the picture is about as bright as before."""

    cap: Any
    api_name: str
    max_exposure_s: float
    on_done: Callable[[dict[str, Any]], None]
    settle_frames: int = 6
    measure_frames: int = 10
    max_gain_steps: int = 8
    state: str = "measure"
    _samples: list[float] = field(default_factory=list)
    _wait: int = 0
    _gain_steps: int = 0
    report: dict[str, Any] = field(default_factory=dict)

    def feed(self, image: np.ndarray) -> bool:
        """Returns True once finished."""
        if self.state == "done":
            return True
        if self._wait > 0:
            self._wait -= 1
            return False
        luma = frame_luma(image)
        if self.state == "measure":
            self._samples.append(luma)
            if len(self._samples) >= self.measure_frames:
                self.report["auto_luma"] = float(np.median(self._samples))
                self._apply_manual()
            return False
        if self.state == "gain":
            target = self.report["auto_luma"] * 0.85
            self.report["locked_luma"] = luma
            if luma >= target or self._gain_steps >= self.max_gain_steps or not self._bump_gain():
                self._finish()
                return True
            self._wait = self.settle_frames
        return False

    def _apply_manual(self) -> None:
        cap = self.cap
        units = units_for_api(self.api_name)
        if units is None:
            self.report.update(supported=False, reason=f"exposure lock unsupported on {self.api_name}")
            self._lock_focus_wb()
            self._finish()
            return
        before = cap.get(cv2.CAP_PROP_EXPOSURE)
        target = units.to_driver(self.max_exposure_s)
        # Keep the auto value when it is already fast enough.
        if before and units.from_driver(before) < self.max_exposure_s:
            target = before
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, units.auto_off)
        ok = cap.set(cv2.CAP_PROP_EXPOSURE, float(target))
        after = cap.get(cv2.CAP_PROP_EXPOSURE)
        self.report.update(
            supported=True,
            exposure_before=before,
            exposure_set=target,
            exposure_readback=after,
            exposure_seconds=units.from_driver(after) if after else None,
            exposure_set_ok=bool(ok),
            gain_before=cap.get(cv2.CAP_PROP_GAIN),
        )
        self._lock_focus_wb()
        self.state = "gain"
        self._wait = self.settle_frames

    def _bump_gain(self) -> bool:
        cap = self.cap
        gain = cap.get(cv2.CAP_PROP_GAIN)
        step = max(4.0, abs(gain) * 0.15)
        if not cap.set(cv2.CAP_PROP_GAIN, gain + step):
            return False
        new_gain = cap.get(cv2.CAP_PROP_GAIN)
        self._gain_steps += 1
        self.report["gain"] = new_gain
        return new_gain != gain

    def _lock_focus_wb(self) -> None:
        cap = self.cap
        focus = cap.get(cv2.CAP_PROP_FOCUS)
        self.report["autofocus_off"] = bool(cap.set(cv2.CAP_PROP_AUTOFOCUS, 0))
        if focus:
            cap.set(cv2.CAP_PROP_FOCUS, focus)
        wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
        self.report["auto_wb_off"] = bool(cap.set(cv2.CAP_PROP_AUTO_WB, 0))
        if wb:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        self.report["focus"] = focus
        self.report["wb_temperature"] = wb

    def _finish(self) -> None:
        self.state = "done"
        log.info("camera lock: %s", self.report)
        self.on_done(dict(self.report))


def unlock(cap: Any, api_name: str) -> None:
    units = units_for_api(api_name)
    if units is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, units.auto_on)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
