"""Camera mode negotiation against fake drivers (no webcam needed).

Real report this reproduces: a 1080p60-capable webcam on Windows delivered
5 fps because the driver fell back to uncompressed YUY2 while still
claiming 60 fps."""

import cv2
import pytest

from motionplay.capture import negotiate as neg
from motionplay.config import CameraConfig

MJPG = cv2.VideoWriter_fourcc(*"MJPG")
YUY2 = cv2.VideoWriter_fourcc(*"YUY2")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeCam:
    """A driver whose real frame rate depends on the format it ends up in."""

    def __init__(self, api, clock, mjpg_needs_size_first, msmf_works=True, max_mjpg_fps=60):
        self.api, self.clock = api, clock
        self.mjpg_needs_size_first = mjpg_needs_size_first
        self.msmf_works = msmf_works
        self.max_mjpg_fps = max_mjpg_fps
        self.props = {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480,
                      cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FOURCC: YUY2}
        self.size_set = False
        self.released = False

    def isOpened(self):
        return self.api != cv2.CAP_MSMF or self.msmf_works

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_FOURCC:
            # The quirk: on this "DirectShow" driver MJPG only sticks when it
            # is requested after the size.
            if self.api == cv2.CAP_DSHOW and self.mjpg_needs_size_first and not self.size_set:
                return False
            self.props[prop] = value
            return True
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            self.size_set = True
            self.props[prop] = value
            return True
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def real_fps(self):
        w, h = self.props[cv2.CAP_PROP_FRAME_WIDTH], self.props[cv2.CAP_PROP_FRAME_HEIGHT]
        asked = self.props[cv2.CAP_PROP_FPS]
        if self.props[cv2.CAP_PROP_FOURCC] == MJPG:
            return min(asked, self.max_mjpg_fps)
        return 5 if w * h >= 1920 * 1080 else (10 if w * h >= 1280 * 720 else 30)

    def read(self):
        self.clock.t += 1.0 / self.real_fps()
        return True, None

    def release(self):
        self.released = True


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(neg.platform, "system", lambda: "Windows")


def run(cfg, **cam_kwargs):
    clock = Clock()
    opened = []

    def opener(index, api):
        cam = FakeCam(api, clock, **cam_kwargs)
        opened.append(cam)
        return cam
    result = neg.negotiate(0, cfg, opener=opener, clock=clock, report=lambda s: None)
    return result, opened


def test_finds_1080p60_when_driver_wants_format_after_size(windows):
    result, opened = run(CameraConfig(), mjpg_needs_size_first=True)
    assert result.ok and (result.width, result.height) == (1920, 1080)
    assert result.measured_fps == pytest.approx(60, rel=0.05)
    assert result.fourcc == "MJPG" and result.attempt.order == "size_first"
    assert all(c.released for c in opened if c is not result.cap)    # no leaked cameras


def test_remembers_the_winning_recipe(windows):
    cfg = CameraConfig()
    run(cfg, mjpg_needs_size_first=True)
    result, opened = run(cfg, mjpg_needs_size_first=True)
    assert result.ok and len(opened) == 1                           # cached recipe first


def test_falls_back_to_720p60_when_1080p60_is_impossible(windows):
    # e.g. on a USB 2 port that can only carry MJPEG 1080p at 30 fps.
    class Slow1080(FakeCam):
        def real_fps(self):
            fps = super().real_fps()
            return min(fps, 30) if self.props[cv2.CAP_PROP_FRAME_WIDTH] >= 1920 else fps

    clock = Clock()
    result = neg.negotiate(0, CameraConfig(), opener=lambda i, api: Slow1080(api, clock, False),
                           clock=clock, report=lambda s: None, use_cache=False)
    assert result.ok and (result.width, result.height, round(result.measured_fps)) == (1280, 720, 60)


def test_reports_best_effort_when_nothing_is_fast(windows):
    lines = []
    clock = Clock()
    result = neg.negotiate(0, CameraConfig(), opener=lambda i, api: FakeCam(api, clock, True, max_mjpg_fps=15),
                           clock=clock, report=lines.append, use_cache=False)
    assert not result.ok and round(result.measured_fps) == 15
    assert any("WARNING" in line for line in lines)
