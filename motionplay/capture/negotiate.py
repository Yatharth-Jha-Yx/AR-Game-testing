"""Get the camera into a fast mode, and *measure* that it really is fast.

Why: on Windows a webcam only sends 1080p60 as MJPEG. If the driver ignores
the MJPEG request, it silently falls back to uncompressed YUY2, which at
1080p fits through USB 2.0 at about 5 fps, while CAP_PROP_FPS still claims
60. Whether the request sticks depends on the driver and on the order the
settings are sent (format before or after size), and DirectShow and Media
Foundation behave differently. So we try the known recipes, time real
frames for each, keep the first that delivers, and remember it.

Preference order: the configured mode (default 1080p60), then 720p at the
same fps (pose nets work at 256x192, so 720p loses nothing for full-body
play), then the configured size at 30 fps.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2

from ..config import CameraConfig, data_dir

log = logging.getLogger(__name__)

APIS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "any": cv2.CAP_ANY,
}
ORDERS = ("fourcc_first", "size_first")
GOOD_ENOUGH = 0.85      # measured fps >= 85% of the asked fps counts as success


@dataclass(frozen=True)
class Attempt:
    api: str
    order: str
    width: int
    height: int
    fps: int


@dataclass
class Negotiated:
    attempt: Attempt
    cap: Any
    width: int
    height: int
    fourcc: str
    claimed_fps: float
    measured_fps: float

    @property
    def ok(self) -> bool:
        a = self.attempt
        return (self.width, self.height) == (a.width, a.height) and self.measured_fps >= GOOD_ENOUGH * a.fps

    def describe(self) -> str:
        return (f"{self.width}x{self.height} {self.fourcc} at {self.measured_fps:.0f} fps "
                f"(asked {self.attempt.fps}, driver claims {self.claimed_fps:.0f}) via {self.attempt.api}")


def platform_apis(cfg: CameraConfig) -> list[str]:
    if cfg.api != "auto":
        return [cfg.api]
    system = platform.system()
    if system == "Windows":
        return ["dshow", "msmf"]
    if system == "Linux":
        return ["v4l2"]
    if system == "Darwin":
        return ["avfoundation"]
    return ["any"]


def attempts(cfg: CameraConfig, apis: list[str], cached: Attempt | None = None) -> Iterator[Attempt]:
    modes = [(cfg.width, cfg.height, cfg.fps)]
    if cfg.height > 720:
        modes.append((1280, 720, cfg.fps))
    if cfg.fps > 30:
        modes.append((cfg.width, cfg.height, 30))
    if cached is not None:
        yield cached
    for w, h, fps in modes:
        for api in apis:
            for order in ORDERS:
                a = Attempt(api, order, w, h, fps)
                if a != cached:
                    yield a


def fourcc_str(value: float) -> str:
    v = int(value)
    s = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    return s if s.isprintable() and s.strip() else f"0x{v:08x}"


def configure(cap: Any, a: Attempt, fourcc: str) -> None:
    code = cv2.VideoWriter_fourcc(*fourcc)
    if a.order == "fourcc_first":
        cap.set(cv2.CAP_PROP_FOURCC, code)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_FPS, a.fps)
    if a.order == "size_first":
        cap.set(cv2.CAP_PROP_FOURCC, code)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def measure_fps(cap: Any, clock: Callable[[], float] = time.perf_counter, seconds: float = 1.0,
                max_frames: int = 45, warmup: int = 3) -> float:
    """Real delivered frame rate. The first frames after a mode switch are
    often slow, so they are skipped."""
    for _ in range(warmup):
        ok, _img = cap.read()
        if not ok:
            return 0.0
    t0 = clock()
    n = 0
    while n < max_frames:
        ok, _img = cap.read()
        if not ok:
            break
        n += 1
        if clock() - t0 >= seconds:
            break
    dt = clock() - t0
    return n / dt if dt > 0 else 0.0


def cache_path() -> Path:
    return data_dir() / "camera.json"


def _cache_key(index: int, cfg: CameraConfig) -> str:
    return f"{index}:{cfg.width}x{cfg.height}@{cfg.fps}:{cfg.fourcc}:{cfg.api}"


def load_cached(index: int, cfg: CameraConfig) -> Attempt | None:
    try:
        entry = json.loads(cache_path().read_text())[_cache_key(index, cfg)]
        return Attempt(**entry)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_cached(index: int, cfg: CameraConfig, a: Attempt) -> None:
    try:
        data = json.loads(cache_path().read_text()) if cache_path().exists() else {}
    except (OSError, ValueError):
        data = {}
    data[_cache_key(index, cfg)] = asdict(a)
    try:
        cache_path().write_text(json.dumps(data, indent=2))
    except OSError:
        pass


def negotiate(index: int, cfg: CameraConfig,
              opener: Callable[[int, int], Any] = lambda i, api: cv2.VideoCapture(i, api),
              clock: Callable[[], float] = time.perf_counter,
              report: Callable[[str], None] = print,
              use_cache: bool = True) -> Negotiated:
    """Open the camera in the best mode that really delivers. Raises
    RuntimeError if the camera can't be opened at all."""
    cached = load_cached(index, cfg) if use_cache else None
    best: Negotiated | None = None
    tried = 0
    for a in attempts(cfg, platform_apis(cfg), cached):
        cap = opener(index, APIS.get(a.api, cv2.CAP_ANY))
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            continue
        tried += 1
        configure(cap, a, cfg.fourcc)
        result = Negotiated(a, cap, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)), float(cap.get(cv2.CAP_PROP_FPS) or 0),
                            measure_fps(cap, clock))
        report(f"  camera check {tried}: {result.describe()}{'  <- OK' if result.ok else ''}")
        if result.ok:
            if best is not None:
                best.cap.release()
            if use_cache:
                save_cached(index, cfg, a)
            return result
        if best is None or (result.measured_fps, result.width * result.height) > \
                (best.measured_fps, best.width * best.height):
            if best is not None:
                best.cap.release()
            best = result
        else:
            cap.release()
    if best is None:
        raise RuntimeError(f"could not open camera {index} (is another app using it?)")
    report(f"  WARNING: no mode reached {cfg.fps} fps; using the fastest found: {best.describe()}.\n"
           "  Try another USB port (USB 3, blue), a different cable, or more light in the room.")
    return best
