"""ONNX Runtime session factory with GPU provider priority.

Order (first available wins, CPU is always the last fallback):
  TensorRT (FP16, engine cache) > CUDA > DirectML > CoreML > CPU

TensorRT through ONNX Runtime gives the same FP16 engine you'd get from
``trtexec`` but builds and caches it automatically on first run
(``~/.motionplay/trt_cache``). The first launch on a new GPU therefore takes
a minute or two; later launches load the cached engine in seconds.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..config import data_dir

log = logging.getLogger(__name__)

PROVIDERS = {
    "tensorrt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def available_providers() -> list[str]:
    """Short names of usable providers; [] when no ONNX Runtime is installed."""
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    reverse = {v: k for k, v in PROVIDERS.items()}
    return [reverse[p] for p in ort.get_available_providers() if p in reverse]


@dataclass
class InputInfo:
    name: str
    shape: list[Any]          # e.g. ['batch', 3, 256, 192] or [1, 3, 256, 192]

    @property
    def dynamic_batch(self) -> bool:
        b = self.shape[0] if self.shape else 1
        return not isinstance(b, int) or b <= 0

    @property
    def hw(self) -> tuple[int, int] | None:
        if len(self.shape) == 4 and all(isinstance(v, int) and v > 0 for v in self.shape[2:]):
            return int(self.shape[2]), int(self.shape[3])
        return None


def create_session(model_path: str | Path, prefer: Sequence[str] = ("tensorrt", "cuda", "directml", "coreml", "cpu"),
                   fp16: bool = True, max_batch: int = 4, threads: int | None = None,
                   fixed_shape: Sequence[int] | None = None):
    """Create an InferenceSession using the best available provider.

    ``fixed_shape`` (e.g. (1, 3, 256, 192)) pins every symbolic input
    dimension when DirectML is used: ONNX Runtime's DirectML docs say it
    performs best with static shapes, and the OpenMMLab exports have a
    free batch (and sometimes height/width) dimension."""
    import onnxruntime as ort

    model_path = str(model_path)
    available = set(ort.get_available_providers())
    if available & {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        _preload_cuda(ort)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = threads

    chosen = next((k for k in prefer if PROVIDERS.get(k) in available), "cpu")
    if chosen == "directml" and fixed_shape is not None:
        info = read_input(model_path)
        for dim, value in zip(info.shape, fixed_shape):
            if isinstance(dim, str) and dim:
                so.add_free_dimension_override_by_name(dim, int(value))

    providers: list[Any] = []
    for key in prefer:
        ep = PROVIDERS.get(key)
        if ep is None or ep not in available or ep == "CPUExecutionProvider":
            continue
        if ep == "TensorrtExecutionProvider":
            cache = data_dir() / "trt_cache"
            cache.mkdir(exist_ok=True)
            opts: dict[str, Any] = {
                "trt_fp16_enable": bool(fp16),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": str(cache),
                "trt_max_workspace_size": str(1 << 30),
            }
            info = read_input(model_path)
            if info.dynamic_batch and info.hw:
                # One engine for batch 1..max_batch instead of a rebuild per batch size.
                c = info.shape[1] if isinstance(info.shape[1], int) else 3
                h, w = info.hw
                opts["trt_profile_min_shapes"] = f"{info.name}:1x{c}x{h}x{w}"
                opts["trt_profile_opt_shapes"] = f"{info.name}:2x{c}x{h}x{w}"
                opts["trt_profile_max_shapes"] = f"{info.name}:{max_batch}x{c}x{h}x{w}"
            providers.append((ep, opts))
        elif ep == "CUDAExecutionProvider":
            providers.append((ep, {"cudnn_conv_algo_search": "HEURISTIC"}))
        elif ep == "DmlExecutionProvider":
            # DirectML requires these two settings.
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            providers.append((ep, {}))
        else:
            providers.append((ep, {}))
    providers.append(("CPUExecutionProvider", {}))
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    log.info("loaded %s on %s", os.path.basename(model_path), sess.get_providers()[0])
    return sess


_preloaded = False


def _preload_cuda(ort) -> None:
    """Let onnxruntime-gpu find CUDA/cuDNN installed as pip packages
    (``pip install onnxruntime-gpu[cuda,cudnn]``), e.g. on Windows."""
    global _preloaded
    if _preloaded or not hasattr(ort, "preload_dlls"):
        return
    _preloaded = True
    try:
        ort.preload_dlls()
    except Exception as exc:  # noqa: BLE001 - system-wide CUDA may still work
        log.debug("preload_dlls: %s", exc)


def read_input(model_path: str | Path) -> InputInfo:
    """Input name/shape via a throwaway CPU session (cheap, no onnx dependency)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    return InputInfo(inp.name, list(inp.shape))


def session_input(sess) -> InputInfo:
    inp = sess.get_inputs()[0]
    return InputInfo(inp.name, list(inp.shape))


def active_provider(sess) -> str:
    reverse = {v: k for k, v in PROVIDERS.items()}
    return reverse.get(sess.get_providers()[0], sess.get_providers()[0])
