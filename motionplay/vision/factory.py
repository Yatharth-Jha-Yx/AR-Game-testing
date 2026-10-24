"""Pick and build the pose backend for this machine.

Tiers (pose model + detector), chosen by the first-launch benchmark
(``motionplay bench``) or, before that, by whether a GPU provider exists:

  full      RTMPose-m Halpe26 + RTMDet-nano   (discrete GPU, 60 Hz, up to 4 players)
  balanced  RTMPose-s Halpe26 + RTMDet-nano   (weaker GPU / fast CPU)
  lite      RTMPose-t Halpe26 + RTMDet-nano   (CPU / integrated GPU, 1-2 players)
"""

from __future__ import annotations

import importlib.util
import json
import logging
from typing import Any

from ..config import Config, data_dir
from . import models
from .base import PoseBackend
from .runtime import available_providers

log = logging.getLogger(__name__)

TIERS: dict[str, dict[str, str]] = {
    "full": {"pose": "rtmpose-m-halpe26", "detector": "rtmdet-nano", "onestage": "rtmo-m",
             "mediapipe": "mediapipe-pose-full"},
    "balanced": {"pose": "rtmpose-s-halpe26", "detector": "rtmdet-nano", "onestage": "rtmo-s",
                 "mediapipe": "mediapipe-pose-full"},
    "lite": {"pose": "rtmpose-t-halpe26", "detector": "rtmdet-nano", "onestage": "rtmo-s",
             "mediapipe": "mediapipe-pose-lite"},
}
_GPU = {"tensorrt", "cuda", "directml", "coreml"}


def tier_file():
    return data_dir() / "tier.json"


def resolve_tier(cfg: Config) -> str:
    if cfg.vision.tier != "auto":
        return cfg.vision.tier
    path = tier_file()
    if path.exists():
        try:
            return json.loads(path.read_text())["tier"]
        except (ValueError, KeyError):
            pass
    usable = set(available_providers()) & set(cfg.vision.providers)
    return "full" if usable & _GPU else "lite"


def build_backend(cfg: Config, source_name: str = "") -> PoseBackend:
    kind = cfg.vision.backend
    if kind == "auto":
        kind = "synthetic" if source_name.startswith("synthetic") else "rtmpose"
    tier = resolve_tier(cfg)
    names = TIERS[tier]
    v = cfg.vision
    if kind == "synthetic":
        from .synthetic_backend import SyntheticPoseBackend
        return SyntheticPoseBackend(one_stage=False)
    if kind == "rtmpose":
        try:
            from .rtm import PersonDetector, RTMDetPoseBackend, TopDownPose
            det_path, det_spec = models.ensure(names["detector"])
            pose_path, pose_spec = models.ensure(names["pose"])
            det = PersonDetector(det_path, det_spec, v.providers, v.fp16)
            pose = TopDownPose(pose_path, pose_spec, v.providers, v.fp16, max_batch=v.max_players)
            return RTMDetPoseBackend(det, pose, name=f"{names['pose']} + {names['detector']} ({tier})")
        except Exception as exc:  # noqa: BLE001 - fall back to MediaPipe, say why
            if cfg.vision.backend == "rtmpose":
                raise
            log.warning("RTMPose unavailable (%s); falling back to MediaPipe", exc)
            rtmpose_error = exc
            kind = "mediapipe"
            if importlib.util.find_spec("mediapipe") is None:
                raise RuntimeError(
                    f"no pose model available. RTMPose failed ({rtmpose_error}) and MediaPipe is not "
                    "installed. Fix one: `motionplay models download` (needs download.openmmlab.com) "
                    "or `pip install mediapipe`.") from None
    if kind == "rtmo":
        from .rtm import RTMOBackend
        path, spec = models.ensure(names["onestage"])
        return RTMOBackend(path, spec, v.providers, v.fp16)
    if kind == "mediapipe":
        from .mediapipe_backend import MediaPipePoseBackend
        path, _ = models.ensure(names["mediapipe"])
        return MediaPipePoseBackend(str(path), num_poses=min(v.max_players, 2), name=names["mediapipe"])
    raise ValueError(f"unknown backend {kind!r}")


def build_segmenter(cfg: Config) -> Any | None:
    if not cfg.vision.segmentation:
        return None
    try:
        from .mediapipe_backend import Segmenter
        path, spec = models.ensure(cfg.vision.segmentation_model)
        return Segmenter(str(path), height=spec.input_size[1] or 144)
    except Exception as exc:  # noqa: BLE001
        log.warning("segmentation disabled: %s", exc)
        return None
