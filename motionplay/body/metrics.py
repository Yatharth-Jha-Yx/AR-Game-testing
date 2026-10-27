"""Small geometry helpers on Halpe26 skeletons (pixel coordinates)."""

from __future__ import annotations

import math

import numpy as np

from ..skeleton import J


def dist(kpts: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(kpts[a] - kpts[b]))


def seen(conf: np.ndarray, *joints: int, thr: float = 0.5) -> bool:
    return bool(np.all(conf[list(joints)] >= thr))


def torso_length(kpts: np.ndarray) -> float:
    """Neck -> mid-hip. The body unit used for all normalisation."""
    return dist(kpts, J.NECK, J.HIP)


def shoulder_width(kpts: np.ndarray) -> float:
    return dist(kpts, J.LEFT_SHOULDER, J.RIGHT_SHOULDER)


def head_height(kpts: np.ndarray) -> float:
    return dist(kpts, J.HEAD, J.NECK)


def angle(kpts: np.ndarray, a: int, b: int, c: int) -> float:
    """Angle ABC in degrees (e.g. elbow angle = shoulder, elbow, wrist)."""
    v1 = kpts[a] - kpts[b]
    v2 = kpts[c] - kpts[b]
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return 180.0
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def body_sizes(kpts: np.ndarray, conf: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    """The three size cues used for depth, each only if its joints are confident."""
    out: dict[str, float] = {}
    if seen(conf, J.LEFT_SHOULDER, J.RIGHT_SHOULDER, thr=thr):
        out["shoulder"] = shoulder_width(kpts)
    if seen(conf, J.NECK, J.HIP, thr=thr):
        out["torso"] = torso_length(kpts)
    if seen(conf, J.HEAD, J.NECK, thr=thr):
        out["head"] = head_height(kpts)
    return {k: v for k, v in out.items() if v > 1.0}
