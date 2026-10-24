"""RTMDet / YOLOX person detector, RTMPose top-down and RTMO one-stage.

Pre/post-processing follows rtmlib (github.com/Tau-J/rtmlib, Apache-2.0):
letterbox resize + 114 grey padding for detectors, 1.25x padded affine crop
for RTMPose, and SimCC arg-max decoding (``simcc_split_ratio`` = 2).

Differences from rtmlib on purpose:
* all person crops go through the pose net as one batch when the ONNX model
  has a dynamic batch dimension (one GPU launch instead of N);
* BGR->RGB is applied when the export says so (pipeline.json ``to_rgb``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..skeleton import NUM_KEYPOINTS, complete_from_coco17
from ..types import Frame, PoseCandidate
from .base import PoseBackend
from .models import ModelSpec
from .runtime import active_provider, create_session, session_input


def letterbox(image: np.ndarray, size: tuple[int, int], pad_val: int = 114) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio into a (w, h) canvas, padding bottom/right."""
    w, h = size
    ih, iw = image.shape[:2]
    ratio = min(h / ih, w / iw)
    nw, nh = int(iw * ratio), int(ih * ratio)
    out = np.full((h, w, 3), pad_val, dtype=np.uint8)
    out[:nh, :nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, ratio


def to_tensor(batch_hwc: np.ndarray, mean, std, to_rgb: bool) -> np.ndarray:
    """(N,H,W,3) uint8 BGR -> (N,3,H,W) float32 normalised."""
    x = batch_hwc.astype(np.float32)
    if to_rgb:
        x = x[..., ::-1]
    if mean is not None:
        x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


class PersonDetector:
    """mmdeploy "end2end" detector with NMS baked in: outputs dets (1,N,5)
    [x0,y0,x1,y1,score] and optionally labels (1,N)."""

    def __init__(self, model_path: Path, spec: ModelSpec, providers: Sequence[str], fp16: bool):
        self.spec = spec
        self.sess = create_session(model_path, providers, fp16=fp16, max_batch=1)
        self.input = session_input(self.sess)
        hw = self.input.hw
        self.size = (hw[1], hw[0]) if hw else spec.input_size
        self.pad_val = int(spec.extra.get("pad_val", 114))
        self.provider = active_provider(self.sess)

    def __call__(self, image: np.ndarray, score_thr: float = 0.15) -> np.ndarray:
        canvas, ratio = letterbox(image, self.size, self.pad_val)
        x = to_tensor(canvas[None], self.spec.mean, self.spec.std, self.spec.to_rgb)
        outputs = self.sess.run(None, {self.input.name: x})
        return parse_detections(outputs, ratio, image.shape[1], image.shape[0], score_thr)


def parse_detections(outputs: list[np.ndarray], ratio: float, width: int, height: int,
                     score_thr: float) -> np.ndarray:
    dets = np.asarray(outputs[0])
    if dets.ndim == 3:
        dets = dets[0]
    if dets.shape[-1] != 5:
        raise ValueError(f"expected detector output (N,5) with NMS baked in, got {dets.shape}")
    keep = dets[:, 4] >= score_thr
    if len(outputs) > 1:
        labels = np.asarray(outputs[1]).reshape(-1)[: len(dets)]
        keep &= labels == 0          # class 0 = person
    out = dets[keep].astype(np.float32)
    out[:, :4] /= ratio
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width - 1)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height - 1)
    return out


def crop_transform(box: np.ndarray, input_size: tuple[int, int], padding: float = 1.25):
    """Centre/scale of a box grown by ``padding`` and widened to the model's
    aspect ratio, plus the 2x3 affine matrix that maps it to the model input."""
    w, h = input_size
    x0, y0, x1, y1 = box[:4]
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
    bw, bh = (x1 - x0) * padding, (y1 - y0) * padding
    aspect = w / h
    if bw > bh * aspect:
        bh = bw / aspect
    else:
        bw = bh * aspect
    scale = np.array([bw, bh], np.float32)
    k = w / bw                                   # uniform scale (bw/bh == w/h)
    mat = np.array([[k, 0, w / 2 - center[0] * k],
                    [0, k, h / 2 - center[1] * k]], np.float32)
    return center, scale, mat


def simcc_decode(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = 2.0):
    """Arg-max SimCC decoding -> (N,K,2) locations in model-input pixels, (N,K) scores."""
    n, k, _ = simcc_x.shape
    sx = simcc_x.reshape(n * k, -1)
    sy = simcc_y.reshape(n * k, -1)
    locs = np.stack([sx.argmax(1), sy.argmax(1)], axis=-1).astype(np.float32)
    vals = 0.5 * (sx.max(1) + sy.max(1))
    locs[vals <= 0] = -1
    return locs.reshape(n, k, 2) / split_ratio, vals.reshape(n, k)


class TopDownPose:
    """RTMPose on person crops. Batched when the model allows it."""

    def __init__(self, model_path: Path, spec: ModelSpec, providers: Sequence[str], fp16: bool,
                 max_batch: int = 4):
        self.spec = spec
        self.sess = create_session(model_path, providers, fp16=fp16, max_batch=max_batch)
        self.input = session_input(self.sess)
        hw = self.input.hw
        self.size = (hw[1], hw[0]) if hw else spec.input_size        # (w, h)
        self.batched = self.input.dynamic_batch
        self.max_batch = max_batch
        self.split = float(spec.extra.get("simcc_split_ratio", 2.0))
        self.provider = active_provider(self.sess)

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return np.zeros((0, NUM_KEYPOINTS, 2), np.float32), np.zeros((0, NUM_KEYPOINTS), np.float32)
        w, h = self.size
        crops, centers, scales = [], [], []
        for box in boxes:
            c, s, mat = crop_transform(box, self.size)
            crops.append(cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR))
            centers.append(c)
            scales.append(s)
        x = to_tensor(np.stack(crops), self.spec.mean, self.spec.std, self.spec.to_rgb)
        step = self.max_batch if self.batched else 1
        xs, ys = [], []
        for i in range(0, len(x), step):
            sx, sy = self.sess.run(None, {self.input.name: x[i:i + step]})[:2]
            xs.append(sx)
            ys.append(sy)
        locs, scores = simcc_decode(np.concatenate(xs), np.concatenate(ys), self.split)
        centers_a = np.stack(centers)[:, None, :]
        scales_a = np.stack(scales)[:, None, :]
        kpts = locs / np.array([w, h], np.float32) * scales_a + centers_a - scales_a / 2
        return kpts.astype(np.float32), np.clip(scores, 0.0, 1.0).astype(np.float32)


class RTMDetPoseBackend(PoseBackend):
    """Detector every N frames + RTMPose every frame (the default backend)."""

    one_stage = False

    def __init__(self, detector: PersonDetector, pose: TopDownPose, name: str):
        self.detector = detector
        self.pose = pose
        self.name = name
        self.provider = pose.provider

    def detect(self, frame: Frame) -> np.ndarray:
        return self.detector(frame.image)

    def estimate(self, frame: Frame, boxes: np.ndarray) -> list[PoseCandidate]:
        kpts, scores = self.pose(frame.image, boxes)
        det_scores = boxes[:, 4] if boxes.shape[1] > 4 else np.ones(len(boxes), np.float32)
        return [PoseCandidate(box=b[:4].astype(np.float32), kpts=k, scores=s, det_score=float(d))
                for b, k, s, d in zip(boxes, kpts, scores, det_scores)]


class RTMOBackend(PoseBackend):
    """One-stage RTMO: boxes + COCO-17 keypoints for everyone in one pass.
    Outputs: dets (1,N,5) and keypoints (1,N,17,3); needs NMS afterwards."""

    one_stage = True

    def __init__(self, model_path: Path, spec: ModelSpec, providers: Sequence[str], fp16: bool,
                 score_thr: float = 0.5, nms_thr: float = 0.45):
        self.spec = spec
        self.name = spec.name
        self.sess = create_session(model_path, providers, fp16=fp16, max_batch=1)
        self.input = session_input(self.sess)
        hw = self.input.hw
        self.size = (hw[1], hw[0]) if hw else spec.input_size
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.provider = active_provider(self.sess)

    def estimate_all(self, frame: Frame) -> list[PoseCandidate]:
        canvas, ratio = letterbox(frame.image, self.size)
        x = to_tensor(canvas[None], self.spec.mean, self.spec.std, self.spec.to_rgb)
        dets, kpts = self.sess.run(None, {self.input.name: x})[:2]
        boxes = dets[0, :, :4] / ratio
        scores = dets[0, :, 4]
        keep_mask = scores >= self.score_thr
        idx = np.flatnonzero(keep_mask)
        if len(idx) == 0:
            return []
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes[idx]]
        kept = cv2.dnn.NMSBoxes(xywh, scores[idx].tolist(), self.score_thr, self.nms_thr)
        kept = idx[np.asarray(kept, dtype=int).reshape(-1)]
        k17 = kpts[0, kept, :, :2] / ratio
        s17 = np.clip(kpts[0, kept, :, 2], 0, 1)
        k26, s26 = complete_from_coco17(k17, s17)
        return [PoseCandidate(box=boxes[i].astype(np.float32), kpts=k, scores=s, det_score=float(scores[i]))
                for i, k, s in zip(kept, k26, s26)]
