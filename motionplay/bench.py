"""First-launch benchmark: time each model on this machine and pick a tier.

Budget (from the design plan): at 60 fps a frame is 16.7 ms and the same GPU
also renders the game, so vision gets ~40% => ~10 ms for 2 players:
    detector/10 (it runs every ~10 frames) + pose(batch of 2) <= 10 ms.
If no tier fits 60 Hz, the best tier that fits 30 Hz (~20 ms) is chosen.

The numbers printed are measurements on *this* machine; the plan's numbers
were estimates to verify, which is exactly what this command is for.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import numpy as np

from .capture.synthetic import SyntheticSource
from .config import CameraConfig, Config
from .vision import models
from .vision.factory import TIERS, tier_file

BUDGET_60_MS = 10.0
BUDGET_30_MS = 20.0


def _time(fn: Callable[[], Any], runs: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.median(samples))


def test_frame(width: int = 1920, height: int = 1080) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic 1080p frame with 4 people, and their boxes."""
    src = SyntheticSource("quad", CameraConfig(width=width, height=height, realtime=False))
    src.open()
    frame = src.read()
    assert frame is not None
    boxes = np.array([[*g["box"], 0.9] for g in frame.meta["gt"]], np.float32)
    return frame.image, boxes


def bench_rtmpose(cfg: Config, runs: int = 30, log: Callable[[str], None] = print) -> dict[str, Any]:
    from .vision.rtm import PersonDetector, TopDownPose

    image, boxes = test_frame()
    results: dict[str, Any] = {}
    det_cache: dict[str, float] = {}
    failed: set[str] = set()
    for tier in ("full", "balanced", "lite"):
        names = TIERS[tier]
        if failed & {names["detector"], names["pose"]}:
            log(f"  {tier}: skipped (download failed above)")
            continue
        try:
            det_path, det_spec = models.ensure(names["detector"])
        except Exception as exc:  # noqa: BLE001
            failed.add(names["detector"])
            log(f"  {tier}: detector unavailable ({exc})")
            continue
        try:
            pose_path, pose_spec = models.ensure(names["pose"])
        except Exception as exc:  # noqa: BLE001
            failed.add(names["pose"])
            log(f"  {tier}: pose model unavailable ({exc})")
            continue
        try:
            if names["detector"] not in det_cache:
                det = PersonDetector(det_path, det_spec, cfg.vision.providers, cfg.vision.fp16)
                det_cache[names["detector"]] = _time(lambda: det(image), runs)
                log(f"  detector {names['detector']} on {det.provider}: {det_cache[names['detector']]:.2f} ms")
            pose = TopDownPose(pose_path, pose_spec, cfg.vision.providers, cfg.vision.fp16, max_batch=4)
            per_batch = {n: _time(lambda n=n: pose(image, boxes[:n]), runs) for n in (1, 2, 4)}
        except Exception as exc:  # noqa: BLE001 - report and try the next tier
            log(f"  {tier}: failed to run ({type(exc).__name__}: {exc})")
            continue
        det_ms = det_cache[names["detector"]]
        two_players = det_ms / cfg.vision.detect_interval + per_batch[2]
        results[tier] = {"pose_model": names["pose"], "provider": pose.provider, "detector_ms": det_ms,
                         "pose_ms": per_batch, "two_player_ms": round(two_players, 2)}
        log(f"  {tier:8s} {names['pose']} on {pose.provider}: 1p {per_batch[1]:.2f} ms, "
            f"2p {per_batch[2]:.2f} ms, 4p {per_batch[4]:.2f} ms -> 2-player frame cost {two_players:.2f} ms")
    return results


def choose_tier(results: dict[str, Any]) -> tuple[str, int]:
    for budget, hz in ((BUDGET_60_MS, 60), (BUDGET_30_MS, 30)):
        for tier in ("full", "balanced", "lite"):
            if tier in results and results[tier]["two_player_ms"] <= budget:
                return tier, hz
    return "lite", 30


def run(cfg: Config, save: bool = True, log: Callable[[str], None] = print) -> dict[str, Any]:
    from .vision.runtime import available_providers

    providers = available_providers()
    if not providers:
        log("no ONNX Runtime installed. Install one: pip install -e \".[directml]\" (Windows GPU), "
            "\".[gpu]\" (NVIDIA CUDA/TensorRT) or \".[cpu]\"")
        return {}
    log(f"execution providers available: {', '.join(providers)}")
    results = bench_rtmpose(cfg, log=log)
    if not results:
        log("no RTMPose models could be loaded; the engine will use MediaPipe (lite path)")
        return {}
    tier, hz = choose_tier(results)
    summary = {"tier": tier, "recommended_fps": hz, "results": results, "measured_at": time.time()}
    log(f"\nrecommended tier: {tier} at {hz} fps (budget {BUDGET_60_MS if hz == 60 else BUDGET_30_MS} ms "
        f"for 2 players)")
    if save:
        tier_file().write_text(json.dumps(summary, indent=2))
        log(f"saved to {tier_file()}")
    return summary
