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

    def open(self) -> None:
        cap = cv2.VideoCapture(self.index, _APIS.get(self.api_name, cv2.CAP_ANY))
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {self.index} via {self.api_name}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(self.cfg.fps)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        got = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
        log.info("camera %d (%s): %dx%d @ %.1f fps, fourcc=%r",
                 self.index, self.api_name, self.width, self.height, self.fps, got)
        if (self.width, self.height) != (self.cfg.width, self.cfg.height) or self.fps < self.cfg.fps - 1:
            log.warning("camera gave %dx%d@%.0f instead of %dx%d@%d (try another USB port/cable, "
                        "or a camera that supports MJPEG)", self.width, self.height, self.fps,
                        self.cfg.width, self.cfg.height, self.cfg.fps)
        self.cap = cap

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
        d.update(api=self.api_name, measured_fps=round(self.measured_fps, 1))
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


def probe_cameras(max_index: int = 6) -> list[dict[str, Any]]:
    """List cameras that open, and whether 1080p60 MJPEG sticks on each."""
    found = []
    api = _APIS[default_api()]
    for index in range(max_index):
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            continue
        modes = []
        for w, h, fps in ((1920, 1080, 60), (1920, 1080, 30), (1280, 720, 60), (1280, 720, 30)):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   round(cap.get(cv2.CAP_PROP_FPS)))
            modes.append({"asked": [w, h, fps], "got": list(got), "ok": got == (w, h, fps)})
        found.append({"index": index, "api": default_api(), "modes": modes})
        cap.release()
    return found
