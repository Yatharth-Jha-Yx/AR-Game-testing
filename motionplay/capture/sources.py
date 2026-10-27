"""Frame sources: a real webcam, a video file, or a synthetic scene.

Every source hands out :class:`~motionplay.types.Frame` objects with a
capture timestamp. ``read()`` blocks until the next frame (or returns None
at the end / on failure).
"""

from __future__ import annotations

import logging
import platform
import queue
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..config import CameraConfig
from ..types import Frame
from . import uvc

log = logging.getLogger(__name__)

_APIS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "any": cv2.CAP_ANY,
}


def default_api() -> str:
    system = platform.system()
    if system == "Windows":
        # DirectShow opens fast and reliably honours MJPG 1080p60;
        # Media Foundation often negotiates YUY2 at 5-30 fps instead.
        return "dshow"
    if system == "Linux":
        return "v4l2"
    if system == "Darwin":
        return "avfoundation"
    return "any"


class FrameSource(ABC):
    name = "source"
    width = 0
    height = 0
    fps = 0.0

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> Frame | None: ...

    def close(self) -> None:
        pass

    # Camera-control hooks; only real cameras implement them.
    def lock_controls(self, on_done: Callable[[dict[str, Any]], None]) -> bool:
        on_done({"supported": False, "reason": f"{self.name} has no camera controls"})
        return False

    def unlock_controls(self) -> None:
        pass

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "width": self.width, "height": self.height, "fps": self.fps}


class CameraSource(FrameSource):
    """USB webcam via OpenCV, asking for MJPEG at the configured mode.

    1080p60 only fits through USB 2.0 as MJPEG (raw YUY2 1080p60 needs
    ~250 MB/s), so we request the MJPG FOURCC *before* width/height, which
    DirectShow requires for the mode switch to stick.
    """

    def __init__(self, index: int, cfg: CameraConfig):
        self.index = index
        self.cfg = cfg
        self.api_name = default_api() if cfg.api == "auto" else cfg.api
        self.name = f"camera:{index}"
        self.cap: cv2.VideoCapture | None = None
        self._seq = 0
        self._requests: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._locker: uvc.ExposureLocker | None = None
        self._last_t = 0.0
        self.measured_fps = 0.0
        self.mode = ""

    def open(self) -> None:
        from .negotiate import negotiate

        print(f"  Checking camera {self.index} modes (first time takes a few seconds)...", flush=True)
        result = negotiate(self.index, self.cfg)
        self.cap = result.cap
        self.api_name = result.attempt.api
        self.width, self.height = result.width, result.height
        self.fps = float(result.attempt.fps if result.ok else max(1.0, round(result.measured_fps)))
        self.mode = f"{result.width}x{result.height} {result.fourcc}"
        self.measured_fps = result.measured_fps
        log.info("camera %d: %s", self.index, result.describe())

    def read(self) -> Frame | None:
        assert self.cap is not None
        while True:
            try:
                self._requests.get_nowait()()
            except queue.Empty:
                break
        ok, image = self.cap.read()
        t = time.perf_counter()
        if not ok or image is None:
            return None
        if self._last_t:
            dt = t - self._last_t
            if dt > 0:
                inst = 1.0 / dt
                self.measured_fps = inst if not self.measured_fps else 0.9 * self.measured_fps + 0.1 * inst
        self._last_t = t
        if self._locker is not None and self._locker.feed(image):
            self._locker = None
        self._seq += 1
        return Frame(image=image, t_capture=t, seq=self._seq)

    def lock_controls(self, on_done: Callable[[dict[str, Any]], None]) -> bool:
        def start() -> None:
            self._locker = uvc.ExposureLocker(
                self.cap, self.api_name, self.cfg.max_exposure_s, on_done)
        self._requests.put(start)
        return True

    def unlock_controls(self) -> None:
        self._requests.put(lambda: uvc.unlock(self.cap, self.api_name))

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def info(self) -> dict[str, Any]:
        d = super().info()
        d.update(api=self.api_name, measured_fps=round(self.measured_fps, 1), mode=self.mode)
        return d


class VideoFileSource(FrameSource):
    """Plays a recorded clip as if it were a camera (great for repeatable tests)."""

    def __init__(self, path: str, cfg: CameraConfig):
        self.path = path
        self.cfg = cfg
        self.name = f"video:{Path(path).name}"
        self.cap: cv2.VideoCapture | None = None
        self._seq = 0
        self._t0 = 0.0

    def open(self) -> None:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video {self.path}")
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.cap = cap

    def read(self) -> Frame | None:
        assert self.cap is not None
        if not self._t0:
            self._t0 = time.perf_counter()    # clock starts at the first read
        ok, image = self.cap.read()
        if not ok:
            if not self.cfg.loop_video:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self.cap.read()
            if not ok:
                return None
        self._seq += 1
        t_virtual = self._t0 + self._seq / self.fps
        if self.cfg.realtime:
            delay = t_virtual - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            t = time.perf_counter()
        else:
            t = t_virtual
        return Frame(image=image, t_capture=t, seq=self._seq)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ImageSource(FrameSource):
    """A single still image repeated (handy for checking a model quickly)."""

    def __init__(self, path: str, cfg: CameraConfig):
        self.path = path
        self.cfg = cfg
        self.name = f"image:{Path(path).name}"
        self.fps = float(cfg.fps)
        self._img: np.ndarray | None = None
        self._seq = 0
        self._t0 = 0.0

    def open(self) -> None:
        self._img = cv2.imread(self.path)
        if self._img is None:
            raise RuntimeError(f"could not read image {self.path}")
        self.height, self.width = self._img.shape[:2]

    def read(self) -> Frame | None:
        assert self._img is not None
        if not self._t0:
            self._t0 = time.perf_counter()
        self._seq += 1
        t = self._t0 + self._seq / self.fps
        if self.cfg.realtime:
            delay = t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        return Frame(image=self._img, t_capture=t, seq=self._seq)


def open_source(cfg: CameraConfig) -> FrameSource:
    """Build a source from ``cfg.source`` (``camera:0``, ``video:clip.mp4``,
    ``image:photo.jpg`` or ``synthetic[:scenario]``)."""
    kind, _, arg = cfg.source.partition(":")
    if kind == "camera":
        src: FrameSource = CameraSource(int(arg or 0), cfg)
    elif kind == "video":
        src = VideoFileSource(arg, cfg)
    elif kind == "image":
        src = ImageSource(arg, cfg)
    elif kind == "synthetic":
        from .synthetic import SyntheticSource
        src = SyntheticSource(arg or "demo", cfg)
    else:
        raise ValueError(f"unknown source {cfg.source!r}")
    src.open()
    return src


def probe_cameras(max_index: int = 4) -> list[dict[str, Any]]:
    """For each camera that opens: the best mode it *really* delivers
    (timed frames, not the driver's claim)."""
    from .negotiate import negotiate

    found = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, _APIS[default_api()])
        opened = cap.isOpened()
        cap.release()
        if not opened:
            continue
        try:
            result = negotiate(index, CameraConfig(), use_cache=False)
        except RuntimeError as exc:
            found.append({"index": index, "error": str(exc)})
            continue
        found.append({"index": index, "best": result.describe(), "ok": result.ok, "api": result.attempt.api})
        result.cap.release()
    return found
