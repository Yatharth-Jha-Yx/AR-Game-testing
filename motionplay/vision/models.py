"""Model registry + downloader.

URLs come from MMPose's ``projects/rtmpose/README.md`` and rtmlib's
``tools/solution/*.py`` (OpenMMLab models, Apache-2.0), and Google's
MediaPipe model cards (Apache-2.0). Files are cached in
``~/.motionplay/models/<name>/``.

OpenMMLab "onnx_sdk" zips contain ``end2end.onnx`` plus mmdeploy's
``pipeline.json``. When present we read the preprocessing (mean/std,
BGR->RGB, input size) from pipeline.json so it always matches the export.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..config import data_dir

log = logging.getLogger(__name__)

_MMPOSE = "https://download.openmmlab.com/mmpose/v1/projects"
_MP = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                    # detector | pose_topdown | pose_onestage | mediapipe_pose | segmenter
    url: str
    member: str = "*end2end.onnx"           # file to pull out of a zip (glob)
    input_size: tuple[int, int] = (0, 0)    # (width, height)
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    to_rgb: bool = False
    keypoints: str = "halpe26"
    license: str = "Apache-2.0"
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


_RTMPOSE_NORM = dict(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), to_rgb=True)
_RTMDET_NORM = dict(mean=(103.53, 116.28, 123.675), std=(57.375, 57.12, 58.395), to_rgb=False)
_README = "MMPose projects/rtmpose/README.md"

REGISTRY: dict[str, ModelSpec] = {s.name: s for s in [
    # Person detectors (boxes only; run every ~10 frames).
    ModelSpec("rtmdet-nano", "detector", f"{_MMPOSE}/rtmpose/rtmpose-cpu.zip",
              member="*rtmdet-nano*/end2end.onnx", input_size=(320, 320), **_RTMDET_NORM,
              source=_README + " (SDK quick start: rtmpose-cpu.zip)"),
    ModelSpec("yolox-tiny", "detector",
              f"{_MMPOSE}/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
              input_size=(416, 416), source="rtmlib tools/solution/body.py (lightweight)"),
    # Top-down body pose, Halpe26 (17 body + head/neck/hip + 6 foot points).
    ModelSpec("rtmpose-t-halpe26", "pose_topdown",
              f"{_MMPOSE}/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7-halpe26_700e-256x192-6020f8a6_20230605.zip",
              input_size=(192, 256), **_RTMPOSE_NORM, source=_README),
    ModelSpec("rtmpose-s-halpe26", "pose_topdown",
              f"{_MMPOSE}/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
              input_size=(192, 256), **_RTMPOSE_NORM, source=_README),
    ModelSpec("rtmpose-m-halpe26", "pose_topdown",
              f"{_MMPOSE}/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
              input_size=(192, 256), **_RTMPOSE_NORM, source=_README),
    ModelSpec("rtmpose-l-halpe26", "pose_topdown",
              f"{_MMPOSE}/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip",
              input_size=(192, 256), **_RTMPOSE_NORM, source=_README),
    # One-stage multi-person (COCO-17, no feet). Cost doesn't grow with people.
    ModelSpec("rtmo-s", "pose_onestage",
              "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",
              input_size=(640, 640), keypoints="coco17", source="rtmlib tools/solution/body.py"),
    ModelSpec("rtmo-m", "pose_onestage",
              "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
              input_size=(640, 640), keypoints="coco17", source="rtmlib tools/solution/body.py"),
    # MediaPipe lite path (33 BlazePose landmarks, mapped to Halpe26).
    ModelSpec("mediapipe-pose-lite", "mediapipe_pose",
              f"{_MP}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
              member="", keypoints="blazepose33", source="MediaPipe Pose Landmarker model card"),
    ModelSpec("mediapipe-pose-full", "mediapipe_pose",
              f"{_MP}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
              member="", keypoints="blazepose33", source="MediaPipe Pose Landmarker model card"),
    ModelSpec("mediapipe-pose-heavy", "mediapipe_pose",
              f"{_MP}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
              member="", keypoints="blazepose33", source="MediaPipe Pose Landmarker model card"),
    # Person segmentation. "landscape" is tiny (250 KB, ~3 ms on CPU) and the
    # default; "multiclass" (background/hair/skin/clothes/..) is sharper but
    # ~70 ms on a 4-core CPU, so only worth it with a GPU delegate.
    ModelSpec("selfie-landscape", "segmenter",
              f"{_MP}/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
              member="", input_size=(256, 144), source="MediaPipe Image Segmenter model card"),
    ModelSpec("selfie-multiclass", "segmenter",
              f"{_MP}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
              member="", input_size=(256, 256), source="MediaPipe Image Segmenter model card"),
]}


def models_dir() -> Path:
    path = data_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def local_path(spec: ModelSpec) -> Path:
    folder = models_dir() / spec.name
    if spec.url.endswith(".zip"):
        return folder / "end2end.onnx"
    return folder / spec.url.rsplit("/", 1)[-1]


def is_downloaded(name: str) -> bool:
    return local_path(REGISTRY[name]).exists()


def ensure(name: str, allow_download: bool = True) -> tuple[Path, ModelSpec]:
    """Return (path, spec) for a model, downloading it on first use."""
    spec = REGISTRY[name]
    path = local_path(spec)
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"model {name} not downloaded; run: motionplay models download {name}")
        download(spec)
    return path, _with_pipeline(spec, path.parent / "pipeline.json")


def download(spec: ModelSpec) -> Path:
    path = local_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / ("download.zip" if spec.url.endswith(".zip") else path.name + ".part")
    log.info("downloading %s from %s", spec.name, spec.url)
    _fetch(spec.url, tmp)
    if spec.url.endswith(".zip"):
        _extract(tmp, spec.member, path)
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(path)
    (path.parent / "SOURCE.json").write_text(json.dumps(
        {"name": spec.name, "url": spec.url, "license": spec.license, "documented_in": spec.source}, indent=2))
    return path


def _fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "motionplay"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total and sys.stderr.isatty():
                print(f"\r  {dest.parent.name}: {done * 100 // total}% of {total >> 20} MB",
                      end="", file=sys.stderr)
    if total and sys.stderr.isatty():
        print(file=sys.stderr)


def _extract(zip_path: Path, member_glob: str, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        matches = [n for n in names if fnmatch.fnmatch(n, member_glob)]
        if not matches:
            raise RuntimeError(f"{member_glob} not found in {zip_path.name}: {names[:20]}")
        member = sorted(matches, key=len)[0]
        with zf.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        # Keep the deploy metadata that sits next to the model.
        folder = member.rsplit("/", 1)[0] + "/" if "/" in member else ""
        for extra in ("pipeline.json", "deploy.json", "detail.json"):
            if folder + extra in names:
                with zf.open(folder + extra) as src, open(dest.parent / extra, "wb") as out:
                    shutil.copyfileobj(src, out)


def _with_pipeline(spec: ModelSpec, pipeline_json: Path) -> ModelSpec:
    """Override preprocessing with what mmdeploy recorded at export time."""
    if not pipeline_json.exists():
        return spec
    try:
        data = json.loads(pipeline_json.read_text())
        found = parse_pipeline(data)
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("could not parse %s: %s", pipeline_json, exc)
        return spec
    return replace(spec, **found) if found else spec


def pad_scalar(pad: Any, default: int = 114) -> int:
    """mmdeploy writes the padding colour as 114, [114, 114, 114] or
    {"img": [114, 114, 114]}; we only need one grey level."""
    if isinstance(pad, dict):
        pad = pad.get("img", default)
    if isinstance(pad, (list, tuple)):
        pad = pad[0] if pad else default
    try:
        return int(pad)
    except (TypeError, ValueError):
        return default


def parse_pipeline(data: dict[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for task in data.get("pipeline", {}).get("tasks", []):
        for tr in task.get("transforms", []) or []:
            kind = tr.get("type")
            if kind == "Normalize":
                found["mean"] = tuple(float(v) for v in tr["mean"])
                found["std"] = tuple(float(v) for v in tr["std"])
                found["to_rgb"] = bool(tr.get("to_rgb", False))
            elif kind == "TopDownAffine" and "input_size" in tr:
                w, h = tr["input_size"]
                found["input_size"] = (int(w), int(h))
            elif kind == "Pad" and tr.get("pad_val") is not None:
                extra["pad_val"] = pad_scalar(tr["pad_val"])
        params = task.get("params") or {}
        if "simcc_split_ratio" in params:
            extra["simcc_split_ratio"] = float(params["simcc_split_ratio"])
    if extra:
        found["extra"] = extra
    return found
