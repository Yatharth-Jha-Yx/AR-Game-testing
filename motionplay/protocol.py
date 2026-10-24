"""Wire protocol between the engine and games (see docs/PROTOCOL.md).

Games never touch the camera. They get:
  * JSON text messages: hello, frame (BodyState per player), gesture, player,
    calibration, status, settings.
  * Optional binary messages: a 16-byte header + JPEG preview or PNG mask.

All coordinates sent to games are normalised 0..1 of the camera image and
already mirrored when ``camera.mirror`` is on, so a game can draw them
straight onto its canvas.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from . import __version__
from .skeleton import EDGES, KEYPOINTS, J
from .types import FrameResult, GestureEvent, PlayerState

PROTOCOL_VERSION = 1

# Binary message kinds.
BIN_VIDEO_JPEG = 1
BIN_MASK_PNG = 2
_BIN_HEADER = struct.Struct("<B3xId")   # kind, pad, seq, t_capture seconds
BIN_HEADER_SIZE = _BIN_HEADER.size      # 16


def pack_binary(kind: int, seq: int, t: float, payload: bytes) -> bytes:
    return _BIN_HEADER.pack(kind, seq & 0xFFFFFFFF, t) + payload


def unpack_binary(data: bytes) -> tuple[int, int, float, bytes]:
    kind, seq, t = _BIN_HEADER.unpack_from(data)
    return kind, seq, t, data[BIN_HEADER_SIZE:]


def hello_message(width: int, height: int, fps: float, mirror: bool,
                  backend: str, settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "hello",
        "protocol": PROTOCOL_VERSION,
        "engine": "motionplay",
        "version": __version__,
        "skeleton": {
            "name": "halpe26",
            "keypoints": list(KEYPOINTS),
            "edges": [[int(a), int(b)] for a, b in EDGES],
        },
        "camera": {"width": width, "height": height, "fps": fps, "mirrored": mirror},
        "backend": backend,
        "settings": settings,
    }


class Normaliser:
    """Turns raw pixel coordinates into mirrored 0..1 screen coordinates."""

    def __init__(self, width: int, height: int, mirror: bool):
        self.w = float(width)
        self.h = float(height)
        self.mirror = mirror

    def xy(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts, dtype=np.float32)
        out[..., 0] = pts[..., 0] / self.w
        out[..., 1] = pts[..., 1] / self.h
        if self.mirror:
            out[..., 0] = 1.0 - out[..., 0]
        return out

    def box(self, box: np.ndarray) -> list[float]:
        x0, y0, x1, y1 = (float(v) for v in box)
        if self.mirror:
            x0, x1 = self.w - x1, self.w - x0
        return [_r(x0 / self.w), _r(y0 / self.h), _r(x1 / self.w), _r(y1 / self.h)]

    def dx_sign(self) -> float:
        return -1.0 if self.mirror else 1.0


def body_state(p: PlayerState, norm: Normaliser) -> dict[str, Any]:
    joints = norm.xy(p.kpts)
    flat = np.empty((len(joints), 3), dtype=np.float32)
    flat[:, :2] = joints
    flat[:, 2] = p.conf
    # Play space: origin at mid-hip, 1 unit = torso length (neck->mid-hip),
    # y up, x towards screen-right. Same numbers for kids and adults.
    torso = max(p.torso_px, 1e-3)
    play = (p.kpts - p.kpts[J.HIP]) / torso
    play[:, 1] *= -1.0
    play[:, 0] *= norm.dx_sign()
    return {
        "id": p.id,
        "confidence": _r(p.confidence),
        "bbox": norm.box(p.box),
        "joints": np.round(flat, 4).ravel().tolist(),
        "play": np.round(play, 3).ravel().tolist(),
        "depth_m": _r(p.depth_m, 3),
        "depth_calibrated": p.depth_calibrated,
        "position_m": [_r(p.position_m[0] * norm.dx_sign(), 3), _r(p.position_m[1], 3)],
        "in_bounds": p.in_bounds,
        "bounds_hint": p.bounds_hint,
        "reach_torso": _r(p.reach_torso, 3),
        "predicted": p.predicted,
    }


def frame_message(result: FrameResult, norm: Normaliser) -> dict[str, Any]:
    return {
        "type": "frame",
        "seq": result.frame.seq,
        "t": round(result.frame.t_capture, 5),
        "latency_ms": round((result.t_done - result.frame.t_capture) * 1000.0, 2),
        "players": [body_state(p, norm) for p in result.players],
    }


def gesture_message(ev: GestureEvent) -> dict[str, Any]:
    return {
        "type": "gesture",
        "player_id": ev.player_id,
        "gesture": ev.gesture,
        "phase": ev.phase,
        "strength": _r(ev.strength, 3),
        "side": ev.side,
        "direction": ev.direction,
        "t": round(ev.t, 5),
    }


def _r(v: float, nd: int = 4) -> float:
    return round(float(v), nd)
