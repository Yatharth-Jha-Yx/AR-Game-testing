"""A stand-in "model" that reads ground truth from synthetic frames.

It behaves like a real backend (noise, shuffled order, misses on bad boxes)
so the tracker, smoothing and gesture code get exercised for real, without
needing model weights or a camera.
"""

from __future__ import annotations

import numpy as np

from ..skeleton import NUM_KEYPOINTS
from ..types import Frame, PoseCandidate
from .base import PoseBackend


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _center_offset(a: np.ndarray, b: np.ndarray) -> float:
    """Centre distance, relative to box size (0 = same centre and size)."""
    ca = np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
    cb = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
    size = max(b[3] - b[1], 1.0)
    return float(np.linalg.norm(ca - cb) / size + abs((a[3] - a[1]) - (b[3] - b[1])) / size)


class SyntheticPoseBackend(PoseBackend):
    def __init__(self, one_stage: bool = False, noise_px_at_1080: float = 2.0, seed: int = 0):
        self.one_stage = one_stage
        self.name = "synthetic-" + ("onestage" if one_stage else "topdown")
        self.provider = "none"
        self.noise = noise_px_at_1080
        self.rng = np.random.default_rng(seed)

    def _gt(self, frame: Frame) -> list[dict]:
        gt = frame.meta.get("gt")
        if gt is None:
            raise RuntimeError("synthetic backend needs a synthetic source (frame.meta['gt'])")
        return [p for p in gt if p["vis"].sum() >= 6]

    def _candidate(self, frame: Frame, person: dict, box: np.ndarray | None = None) -> PoseCandidate:
        sigma = self.noise * frame.image.shape[0] / 1080.0
        kpts = person["kpts"] + self.rng.normal(0, sigma, size=person["kpts"].shape).astype(np.float32)
        scores = np.clip(person["vis"] * 0.92 + self.rng.normal(0, 0.03, NUM_KEYPOINTS), 0, 1)
        return PoseCandidate(box=(person["box"] if box is None else box[:4]).astype(np.float32),
                             kpts=kpts.astype(np.float32), scores=scores.astype(np.float32), det_score=0.9)

    def detect(self, frame: Frame) -> np.ndarray:
        people = self._gt(frame)
        out = np.zeros((len(people), 5), np.float32)
        for i, p in enumerate(people):
            jitter = self.rng.normal(0, 4, 4) * frame.image.shape[0] / 1080.0
            out[i, :4] = p["box"] + jitter
            out[i, 4] = 0.85
        return out[self.rng.permutation(len(out))]

    def estimate(self, frame: Frame, boxes: np.ndarray) -> list[PoseCandidate]:
        people = self._gt(frame)
        out = []
        for box in boxes:
            # A top-down net reports the person centred in its crop.
            near = [p for p in people if _iou(p["box"], box[:4]) >= 0.2]
            best = min(near, key=lambda p: _center_offset(p["box"], box[:4]), default=None)
            if best is None:
                # A top-down net on an empty crop returns junk with low scores.
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                kpts = np.tile([[cx, cy]], (NUM_KEYPOINTS, 1)).astype(np.float32)
                out.append(PoseCandidate(box=box[:4].astype(np.float32), kpts=kpts,
                                         scores=np.full(NUM_KEYPOINTS, 0.05, np.float32),
                                         det_score=float(box[4]) if len(box) > 4 else 0.5))
            else:
                out.append(self._candidate(frame, best, box))
        return out

    def estimate_all(self, frame: Frame) -> list[PoseCandidate]:
        people = self._gt(frame)
        order = self.rng.permutation(len(people))
        return [self._candidate(frame, people[i]) for i in order]
