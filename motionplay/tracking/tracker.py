"""Stable player IDs.

ByteTrack-style two-stage matching (Zhang et al., "ByteTrack: Multi-Object
Tracking by Associating Every Detection Box", ECCV 2022):
  1. confident poses <-> tracks, Hungarian on a cost that mixes box IoU with
     Object Keypoint Similarity (OKS, the COCO keypoint metric). When two
     tracks' boxes overlap (two kids hugging), OKS dominates, because whole
     skeletons tell people apart better than overlapping boxes;
  2. weak poses <-> leftover tracks, IoU only.
Unmatched confident poses are compared to recently lost players by an HSV
histogram of their torso (shirt colour). Within ``reattach_s`` (2 s) a match
re-attaches the old ID; otherwise the slot is freed and the game is told.

IDs are small slot numbers 1..max_players so games can say "Player 2".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import Config
from ..skeleton import OKS_SIGMAS, J
from ..types import PoseCandidate
from ..vision.base import boxes_from_keypoints
from .filters import BoxKalman
from .smoothing import JointSmoother

_INF = 1e6
_CORE = [J.NOSE, J.LEFT_SHOULDER, J.RIGHT_SHOULDER, J.LEFT_ELBOW, J.RIGHT_ELBOW,
         J.LEFT_WRIST, J.RIGHT_WRIST, J.LEFT_HIP, J.RIGHT_HIP, J.LEFT_KNEE, J.RIGHT_KNEE,
         J.NECK, J.HIP]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def oks(k1: np.ndarray, s1: np.ndarray, k2: np.ndarray, s2: np.ndarray, area: float,
        thr: float = 0.3) -> float:
    """COCO Object Keypoint Similarity between two skeletons (0..1)."""
    both = (s1 >= thr) & (s2 >= thr)
    if both.sum() < 3 or area <= 0:
        return 0.0
    d2 = np.sum((k1[both] - k2[both]) ** 2, axis=1)
    var = (2 * OKS_SIGMAS[both]) ** 2
    return float(np.mean(np.exp(-d2 / (2 * area * var))))


def quality(scores: np.ndarray) -> float:
    """Pose quality = mean confidence of the core body joints."""
    return float(np.mean(scores[_CORE]))


def torso_histogram(image: np.ndarray, kpts: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
    pts = [J.LEFT_SHOULDER, J.RIGHT_SHOULDER, J.LEFT_HIP, J.RIGHT_HIP]
    if np.any(scores[pts] < 0.4):
        return None
    xs, ys = kpts[pts, 0], kpts[pts, 1]
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # Shrink 15% so we sample shirt, not background.
    dx, dy = (x1 - x0) * 0.15, (y1 - y0) * 0.15
    h, w = image.shape[:2]
    x0, x1 = int(max(0, x0 + dx)), int(min(w, x1 - dx))
    y0, y1 = int(max(0, y0 + dy)), int(min(h, y1 - dy))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


@dataclass
class Track:
    id: int
    kf: BoxKalman
    smoother: JointSmoother
    kpts: np.ndarray                 # last raw measurement
    scores: np.ndarray
    box: np.ndarray
    t_first: float
    t_last: float
    quality: float
    z: np.ndarray | None = None
    hist: np.ndarray | None = None
    state: str = "active"            # active | lost
    misses: int = 0
    hits: int = 1
    matched_now: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


class PlayerTracker:
    def __init__(self, cfg: Config, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.w, self.h = frame_w, frame_h
        self.tracks: dict[int, Track] = {}
        self.max_players = cfg.vision.max_players

    # --- what the engine asks before running the pose net -------------------
    def active(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.state == "active"]

    def crop_boxes(self, t: float) -> np.ndarray:
        """Boxes to run the top-down pose net on when we skip detection."""
        out = []
        for tr in self.tracks.values():
            if tr.state == "active" or tr.misses <= 3:
                b = tr.kf.predict_box(t)
                out.append([*b, 0.6])
        return np.array(out, np.float32).reshape(-1, 5)

    def wants_detection(self, frame_index: int, interval: int) -> bool:
        if not self.tracks:
            return frame_index % 2 == 0     # nobody yet: look often
        if any(tr.state == "lost" for tr in self.tracks.values()):
            return frame_index % 3 == 0     # someone missing: look more often
        if any(tr.quality < 0.4 for tr in self.tracks.values()):
            return True
        return frame_index % max(1, interval) == 0

    # --- association ---------------------------------------------------------
    def update(self, cands: list[PoseCandidate], image: np.ndarray, t: float) -> list[dict[str, Any]]:
        """Associate this frame's poses to players. Returns player events."""
        events: list[dict[str, Any]] = []
        vcfg = self.cfg.vision
        cands = self._dedupe([c for c in cands if quality(c.scores) >= vcfg.min_pose_score * 0.5])
        for c in cands:
            # Occluded joints still carry a rough position; use them for the box
            # so a half-hidden player keeps a full-size box.
            c.box = boxes_from_keypoints(c.kpts, c.scores, thr=0.1)
        high = [i for i, c in enumerate(cands) if quality(c.scores) >= vcfg.min_pose_score]
        low = [i for i, c in enumerate(cands) if i not in high]
        for tr in self.tracks.values():
            tr.matched_now = False

        recent = [tr for tr in self.tracks.values() if tr.state == "active" or tr.misses <= 5]
        pred = {tr.id: tr.kf.predict_box(t) for tr in recent}
        ambiguous = {tr.id: any(iou(pred[tr.id], pred[o.id]) > 0.3 for o in recent if o.id != tr.id)
                     for tr in recent}

        # Stage 1: confident poses, IoU + OKS.
        matches, rest_tracks, rest_high = self._hungarian(
            recent, [cands[i] for i in high], high,
            lambda tr, c: self._cost(tr, c, pred[tr.id], ambiguous[tr.id], t), 0.85)
        # Stage 2: weak poses vs leftover tracks, IoU only.
        m2, rest_tracks, _ = self._hungarian(
            rest_tracks, [cands[i] for i in low], low,
            lambda tr, c: 1 - iou(pred[tr.id], c.box) if iou(pred[tr.id], c.box) >= 0.3 else _INF, 0.7)
        matches += m2
        for tr, ci in matches:
            self._apply(tr, cands[ci], image, t, others=[c for j, c in enumerate(cands) if j != ci])

        # Unmatched confident poses: re-identify a missing player or add a new one.
        for ci in rest_high:
            c = cands[ci]
            hist = torso_histogram(image, c.kpts, c.scores)
            best = self._reidentify(c, hist, t)
            if best is not None:
                was_lost = best.state == "lost"
                self._apply(best, c, image, t, others=[])
                if was_lost:
                    events.append({"event": "rejoined", "player_id": best.id})
                continue
            new_id = self._free_id()
            if new_id is None:
                continue             # all slots taken: treat as a spectator
            tr = Track(new_id, BoxKalman(c.box, t), JointSmoother(self.cfg.tracking, self.h),
                       c.kpts, c.scores, c.box, t, t, quality(c.scores), z=c.z, hist=hist)
            tr.smoother.update(c.kpts, c.scores, t)
            self.tracks[new_id] = tr
            events.append({"event": "joined", "player_id": new_id})

        # Tracks nobody matched.
        for tr in list(self.tracks.values()):
            if tr.matched_now:
                continue
            tr.misses += 1
            if tr.state == "active" and (tr.misses >= 3 and t - tr.t_last > 0.08):
                tr.state = "lost"
                events.append({"event": "lost", "player_id": tr.id})
            elif tr.state == "lost" and t - tr.t_last > self.cfg.tracking.reattach_s:
                del self.tracks[tr.id]
                events.append({"event": "left", "player_id": tr.id})
        return events

    def _reidentify(self, c: PoseCandidate, hist: np.ndarray | None, t: float) -> Track | None:
        """Best unmatched player for this pose: same shirt colours, or still
        close to where they were when we lost them a moment ago."""
        best, best_score = None, 1e9
        for tr in self.tracks.values():
            if tr.matched_now:
                continue
            pbox = tr.kf.predict_box(t) if t - tr.t_last < 0.5 else tr.box
            height = max(pbox[3] - pbox[1], 1.0)
            center_d = np.hypot((pbox[0] + pbox[2] - c.box[0] - c.box[2]) / 2,
                                (pbox[1] + pbox[3] - c.box[1] - c.box[3]) / 2) / height
            hist_d = hist_distance(hist, tr.hist) if hist is not None and tr.hist is not None else None
            colour_ok = hist_d is not None and hist_d < self.cfg.tracking.reid_max_distance
            nearby_ok = center_d < 0.6 and (hist_d is None or hist_d < 0.6)
            if not (colour_ok or nearby_ok):
                continue
            score = (hist_d if hist_d is not None else 0.4) + 0.5 * center_d
            if score < best_score:
                best, best_score = tr, score
        return best

    def _cost(self, tr: Track, c: PoseCandidate, pbox: np.ndarray, ambiguous: bool, t: float) -> float:
        i = iou(pbox, c.box)
        area = max((pbox[2] - pbox[0]) * (pbox[3] - pbox[1]), 1.0)
        prev = tr.smoother.extrapolate(t) if tr.smoother.smoothed is not None else tr.kpts
        o = oks(prev, tr.smoother.conf, c.kpts, c.scores, area, thr=0.1)
        if i < 0.1 and o < 0.2:
            return _INF
        return 1 - (0.3 * i + 0.7 * o) if ambiguous else 1 - (0.7 * i + 0.3 * o)

    @staticmethod
    def _hungarian(tracks, cands, cand_idx, cost_fn, max_cost):
        if not tracks or not cands:
            return [], list(tracks), list(cand_idx)
        cost = np.array([[cost_fn(tr, c) for c in cands] for tr in tracks])
        rows, cols = linear_sum_assignment(cost)
        matches, used_t, used_c = [], set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] <= max_cost:
                matches.append((tracks[r], cand_idx[c]))
                used_t.add(r)
                used_c.add(c)
        rest_t = [tr for i, tr in enumerate(tracks) if i not in used_t]
        rest_c = [ci for j, ci in enumerate(cand_idx) if j not in used_c]
        return matches, rest_t, rest_c

    def _apply(self, tr: Track, c: PoseCandidate, image: np.ndarray, t: float,
               others: list[PoseCandidate]) -> None:
        tr.kf.update(c.box, t)
        tr.smoother.update(c.kpts, c.scores, t)
        tr.kpts, tr.scores, tr.box, tr.z = c.kpts, c.scores, c.box, c.z
        tr.quality = quality(c.scores)
        tr.t_last = t
        tr.misses = 0
        tr.hits += 1
        tr.matched_now = True
        if tr.state == "lost":
            tr.state = "active"
        # Update the colour signature only when nobody overlaps this player.
        if tr.quality >= 0.5 and all(iou(c.box, o.box) < 0.15 for o in others):
            hist = torso_histogram(image, c.kpts, c.scores)
            if hist is not None:
                tr.hist = hist if tr.hist is None else (0.9 * tr.hist + 0.1 * hist).astype(np.float32)

    def _dedupe(self, cands: list[PoseCandidate]) -> list[PoseCandidate]:
        """Drop duplicate skeletons (same person found by detector and tracker).

        Duplicates are near-identical (OKS ~0.95+). Two different people who
        overlap can still reach OKS ~0.6, so the bar is high and the boxes
        must agree too."""
        cands = sorted(cands, key=lambda c: -quality(c.scores))
        kept: list[tuple[PoseCandidate, np.ndarray]] = []
        for c in cands:
            box = boxes_from_keypoints(c.kpts, c.scores)
            area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
            if all(oks(c.kpts, c.scores, k.kpts, k.scores, area) < 0.85 or iou(box, kb) < 0.7
                   for k, kb in kept):
                kept.append((c, box))
        return [c for c, _ in kept]

    def _free_id(self) -> int | None:
        for i in range(1, self.max_players + 1):
            if i not in self.tracks:
                return i
        return None
