"""MediaPipe lite path: Pose Landmarker (33 BlazePose points) + selfie segmenter.

Runs on CPU via XNNPACK with no GPU needed; good for 1 player on weak PCs
and for the segmentation mask. Uses the MediaPipe Tasks API
(``mediapipe.tasks.python.vision``), the only API in mediapipe >= 0.10.30.

BlazePose landmark indices come from the MediaPipe Pose Landmarker model
card; we map them onto Halpe26 so the rest of the engine sees one layout.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..skeleton import NUM_KEYPOINTS, synthesize_center_joints
from ..types import Frame, PoseCandidate
from .base import PoseBackend, boxes_from_keypoints

# halpe26 index -> blazepose33 index (feet: big toe = foot_index; no small
# toe in BlazePose, so we reuse foot_index at half confidence).
_MAP = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
        11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
        20: 31, 21: 32, 22: 31, 23: 32, 24: 29, 25: 30}
_HALF_CONF = (22, 23)
_HALPE_IDX = np.array(list(_MAP.keys()))
_BLAZE_IDX = np.array(list(_MAP.values()))


def blaze_to_halpe(xyzv: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(33, 4) normalised x, y, z, visibility -> Halpe26 pixels, scores, z."""
    k = np.zeros((1, NUM_KEYPOINTS, 2), np.float32)
    s = np.zeros((1, NUM_KEYPOINTS), np.float32)
    z = np.zeros(NUM_KEYPOINTS, np.float32)
    src = xyzv[_BLAZE_IDX]
    k[0, _HALPE_IDX, 0] = src[:, 0] * width
    k[0, _HALPE_IDX, 1] = src[:, 1] * height
    # Points MediaPipe extrapolates outside the image get no confidence.
    inside = (src[:, 0] > -0.02) & (src[:, 0] < 1.02) & (src[:, 1] > -0.02) & (src[:, 1] < 1.02)
    s[0, _HALPE_IDX] = np.where(inside, src[:, 3], 0.0)
    s[0, list(_HALF_CONF)] *= 0.5
    z[_HALPE_IDX] = src[:, 2] * width     # same scale as x, in pixels
    synthesize_center_joints(k, s)
    return k[0], s[0], z


class MediaPipePoseBackend(PoseBackend):
    one_stage = True

    def __init__(self, model_path: str, num_poses: int = 1, name: str = "mediapipe"):
        from mediapipe.tasks.python import BaseOptions, vision

        self.name = name
        self.provider = "cpu (xnnpack)"
        self._landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))
        self._last_ts = -1

    def estimate_all(self, frame: Frame) -> list[PoseCandidate]:
        import mediapipe as mp

        h, w = frame.image.shape[:2]
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        ts = max(int(frame.t_capture * 1000), self._last_ts + 1)   # must strictly increase
        self._last_ts = ts
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        out = []
        for landmarks in result.pose_landmarks:
            arr = np.array([[lm.x, lm.y, lm.z, lm.visibility if lm.visibility is not None else 1.0]
                            for lm in landmarks], np.float32)
            kpts, scores, z = blaze_to_halpe(arr, w, h)
            out.append(PoseCandidate(box=boxes_from_keypoints(kpts, scores), kpts=kpts, scores=scores,
                                     det_score=float(np.mean(scores)), z=z))
        return out

    def close(self) -> None:
        self._landmarker.close()


class Segmenter:
    """Person mask from a MediaPipe selfie segmenter.

    Runs on a downscaled frame (the models are 144-256 px internally) and
    returns a small uint8 mask, 255 = person; games upscale it on the GPU.
    The single-class "selfie" model outputs person confidence directly; the
    multiclass model's channel 0 is background, so person = 1 - channel 0."""

    def __init__(self, model_path: str, height: int = 144):
        from mediapipe.tasks.python import BaseOptions, vision

        self.height = height
        self._seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False,
        ))
        labels = [str(label).lower() for label in (getattr(self._seg, "labels", None) or [])]
        self._channel0_is_background = bool(labels) and labels[0] == "background"

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import mediapipe as mp

        h, w = image.shape[:2]
        small_w = max(1, int(round(w * self.height / h)))
        small = cv2.resize(image, (small_w, self.height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = self._seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        channel0 = np.squeeze(res.confidence_masks[0].numpy_view())
        person = 1.0 - channel0 if self._channel0_is_background else channel0
        return (np.clip(person, 0, 1) * 255).astype(np.uint8)

    def close(self) -> None:
        self._seg.close()

    def info(self) -> dict[str, Any]:
        return {"height": self.height, "channel0_is_background": self._channel0_is_background}
