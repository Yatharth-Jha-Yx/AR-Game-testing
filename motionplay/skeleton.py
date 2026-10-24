"""Halpe26 keypoint layout used inside the engine and on the wire.

Halpe26 = the 17 COCO body points + head top, neck, mid-hip and 6 foot points
(big toe, small toe, heel per foot). Feet matter for step/dance games, which is
why we prefer it over plain COCO-17.

Source of names, order, links and OKS sigmas: MMPose dataset config
``configs/_base_/datasets/halpe26.py`` (identical in rtmlib's
``visualization/skeleton/halpe26.py``).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

KEYPOINTS: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
)
NUM_KEYPOINTS = len(KEYPOINTS)


class J(IntEnum):
    """Joint indices, so code reads ``kp[J.LEFT_WRIST]`` instead of ``kp[9]``."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    HEAD = 17
    NECK = 18
    HIP = 19
    LEFT_BIG_TOE = 20
    RIGHT_BIG_TOE = 21
    LEFT_SMALL_TOE = 22
    RIGHT_SMALL_TOE = 23
    LEFT_HEEL = 24
    RIGHT_HEEL = 25


EDGES: tuple[tuple[int, int], ...] = (
    (J.LEFT_ANKLE, J.LEFT_KNEE), (J.LEFT_KNEE, J.LEFT_HIP), (J.LEFT_HIP, J.HIP),
    (J.RIGHT_ANKLE, J.RIGHT_KNEE), (J.RIGHT_KNEE, J.RIGHT_HIP), (J.RIGHT_HIP, J.HIP),
    (J.HEAD, J.NECK), (J.NECK, J.HIP),
    (J.NECK, J.LEFT_SHOULDER), (J.LEFT_SHOULDER, J.LEFT_ELBOW), (J.LEFT_ELBOW, J.LEFT_WRIST),
    (J.NECK, J.RIGHT_SHOULDER), (J.RIGHT_SHOULDER, J.RIGHT_ELBOW), (J.RIGHT_ELBOW, J.RIGHT_WRIST),
    (J.LEFT_EYE, J.RIGHT_EYE), (J.NOSE, J.LEFT_EYE), (J.NOSE, J.RIGHT_EYE),
    (J.LEFT_EYE, J.LEFT_EAR), (J.RIGHT_EYE, J.RIGHT_EAR),
    (J.LEFT_EAR, J.LEFT_SHOULDER), (J.RIGHT_EAR, J.RIGHT_SHOULDER),
    (J.LEFT_ANKLE, J.LEFT_BIG_TOE), (J.LEFT_ANKLE, J.LEFT_SMALL_TOE), (J.LEFT_ANKLE, J.LEFT_HEEL),
    (J.RIGHT_ANKLE, J.RIGHT_BIG_TOE), (J.RIGHT_ANKLE, J.RIGHT_SMALL_TOE), (J.RIGHT_ANKLE, J.RIGHT_HEEL),
)

# Per-keypoint OKS sigmas (how "loose" a match is allowed to be per joint).
OKS_SIGMAS = np.array(
    [0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
     0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.026, 0.026, 0.066,
     0.079, 0.079, 0.079, 0.079, 0.079, 0.079],
    dtype=np.float32,
)

# Mapping of each joint to its left/right twin; used when mirroring.
FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (J.LEFT_EYE, J.RIGHT_EYE), (J.LEFT_EAR, J.RIGHT_EAR),
    (J.LEFT_SHOULDER, J.RIGHT_SHOULDER), (J.LEFT_ELBOW, J.RIGHT_ELBOW),
    (J.LEFT_WRIST, J.RIGHT_WRIST), (J.LEFT_HIP, J.RIGHT_HIP),
    (J.LEFT_KNEE, J.RIGHT_KNEE), (J.LEFT_ANKLE, J.RIGHT_ANKLE),
    (J.LEFT_BIG_TOE, J.RIGHT_BIG_TOE), (J.LEFT_SMALL_TOE, J.RIGHT_SMALL_TOE),
    (J.LEFT_HEEL, J.RIGHT_HEEL),
)

# Joints that must be visible for a player to count as "head to ankle in view".
FULL_BODY_JOINTS: tuple[int, ...] = (
    J.NOSE, J.LEFT_SHOULDER, J.RIGHT_SHOULDER, J.LEFT_HIP, J.RIGHT_HIP,
    J.LEFT_ANKLE, J.RIGHT_ANKLE,
)

# COCO-17 is a prefix of Halpe26; one-stage models (RTMO) only give these.
COCO17 = tuple(range(17))


def complete_from_coco17(kpts: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand COCO-17 arrays (N,17,2)/(N,17) to Halpe26 by synthesising
    neck, mid-hip and head-top. Foot points are left at confidence 0."""
    n = kpts.shape[0]
    out_k = np.zeros((n, NUM_KEYPOINTS, 2), dtype=np.float32)
    out_s = np.zeros((n, NUM_KEYPOINTS), dtype=np.float32)
    out_k[:, :17] = kpts[:, :17]
    out_s[:, :17] = scores[:, :17]
    synthesize_center_joints(out_k, out_s)
    return out_k, out_s


def synthesize_center_joints(kpts: np.ndarray, scores: np.ndarray) -> None:
    """Fill neck / mid-hip / head-top in place from their neighbours.

    Neck = shoulder midpoint, hip = hip midpoint, head-top = ear (or eye)
    midpoint pushed up by the ear-to-neck distance * 0.45. Used for models
    that don't predict these points directly (RTMO, MediaPipe)."""
    ls, rs = J.LEFT_SHOULDER, J.RIGHT_SHOULDER
    lh, rh = J.LEFT_HIP, J.RIGHT_HIP
    kpts[:, J.NECK] = (kpts[:, ls] + kpts[:, rs]) * 0.5
    scores[:, J.NECK] = np.minimum(scores[:, ls], scores[:, rs])
    kpts[:, J.HIP] = (kpts[:, lh] + kpts[:, rh]) * 0.5
    scores[:, J.HIP] = np.minimum(scores[:, lh], scores[:, rh])
    ear_mid = (kpts[:, J.LEFT_EAR] + kpts[:, J.RIGHT_EAR]) * 0.5
    ear_s = np.minimum(scores[:, J.LEFT_EAR], scores[:, J.RIGHT_EAR])
    eye_mid = (kpts[:, J.LEFT_EYE] + kpts[:, J.RIGHT_EYE]) * 0.5
    eye_s = np.minimum(scores[:, J.LEFT_EYE], scores[:, J.RIGHT_EYE])
    use_ear = (ear_s >= eye_s)[:, None]
    base = np.where(use_ear, ear_mid, eye_mid)
    base_s = np.maximum(ear_s, eye_s)
    up = base - kpts[:, J.NECK]
    kpts[:, J.HEAD] = base + up * 0.45
    scores[:, J.HEAD] = np.minimum(base_s, scores[:, J.NECK]) * 0.9
