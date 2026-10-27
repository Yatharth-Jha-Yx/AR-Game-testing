"""All tunable settings in one place, with the defaults from the design plan.

Settings load from ``~/.motionplay/config.json`` (override the folder with the
``MOTIONPLAY_HOME`` env var) and can then be overridden from the CLI.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    path = Path(os.environ.get("MOTIONPLAY_HOME", Path.home() / ".motionplay"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class CameraConfig:
    # camera:<index> | video:<path> | synthetic[:<scenario>]
    source: str = "camera:0"
    # 1080p60 MJPEG is the default: pose nets downscale to 256x192 anyway, and
    # 60 fps halves per-frame latency vs 30 fps.
    width: int = 1920
    height: int = 1080
    fps: int = 60
    fourcc: str = "MJPG"
    # auto | dshow | msmf | v4l2 | avfoundation
    api: str = "auto"
    # Mirror the view (coordinates + preview) like a mirror, which is what
    # players expect when they see themselves on screen.
    mirror: bool = True
    # Exposure cap when locking: 1/120 s or faster stops wrist/ankle blur.
    max_exposure_s: float = 1 / 120
    loop_video: bool = True
    # Pace video files / synthetic scenes to their fps (False = as fast as possible).
    realtime: bool = True


@dataclass
class VisionConfig:
    # auto | rtmpose | rtmo | mediapipe | synthetic
    backend: str = "auto"
    # auto | full | balanced | lite   (auto = pick from the first-launch benchmark)
    tier: str = "auto"
    # Execution provider priority. Missing ones are skipped.
    providers: list[str] = field(
        default_factory=lambda: ["tensorrt", "cuda", "directml", "coreml", "cpu"]
    )
    fp16: bool = True
    # Run the person detector every N frames; tracked boxes fill the gaps.
    detect_interval: int = 10
    det_score_low: float = 0.15
    det_score_high: float = 0.5
    # Mean keypoint confidence below this = "not a real person".
    min_pose_score: float = 0.3
    max_players: int = 4
    # With 3+ players over budget, run pose every Nth frame and predict between.
    crowded_pose_divisor: int = 2
    # Segmentation mask (person vs background) for AR occlusion.
    segmentation: bool = False
    segmentation_hz: float = 30.0
    # selfie-landscape (~3 ms CPU) | selfie-multiclass (~70 ms CPU, sharper)
    segmentation_model: str = "selfie-landscape"


@dataclass
class TrackingConfig:
    # Re-attach a player who vanished within this window, else free the slot.
    reattach_s: float = 2.0
    # One-Euro filter (units: pixels at 1080p, seconds). Per-joint betas
    # are scaled from this base in tracking/smoothing.py.
    min_cutoff: float = 1.0
    beta: float = 0.01
    d_cutoff: float = 1.0
    # Predict this far ahead to hide capture delay (None = one frame).
    predict: bool = True
    predict_ms: float | None = None
    predict_min_conf: float = 0.5
    # Joint under this confidence is held for hold_frames, then hidden.
    gate_conf: float = 0.3
    hold_frames: int = 5
    # Bhattacharyya distance under which a torso colour histogram is "same person".
    reid_max_distance: float = 0.35


@dataclass
class PlayAreaConfig:
    # Distance of the floor tape mark used during T-pose calibration.
    calib_distance_m: float = 2.0
    min_depth_m: float = 1.4
    max_depth_m: float = 3.6
    half_width_m: float = 1.6
    # Horizontal field of view used for metric estimates before calibration
    # (typical 1080p webcam: ~70 deg horizontal / ~78 deg diagonal).
    hfov_deg: float = 70.0


@dataclass
class PrivacyConfig:
    # Parental switch. When False the engine never sends camera pixels to any
    # client. It can only be changed in config.json or on the CLI, never from
    # a game, so a child can't turn it back on from the TV.
    video_enabled: bool = True
    # Display preference (games draw avatars/skeletons instead of video).
    # Changeable from the launcher UI.
    skeleton_only: bool = False
    cooldown_minutes: float = 20.0


@dataclass
class ServerConfig:
    # Loopback only: no video ever leaves the machine.
    host: str = "127.0.0.1"
    port: int = 8765
    preview_width: int = 960
    preview_fps: float = 30.0
    jpeg_quality: int = 70


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    play_area: PlayAreaConfig = field(default_factory=PlayAreaConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        cfg = cls()
        _merge(cfg, data)
        return cfg

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or data_dir() / "config.json"
        if path.exists():
            return cls.from_dict(json.loads(path.read_text()))
        return cls()

    def save(self, path: Path | None = None) -> Path:
        path = path or data_dir() / "config.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def set(self, dotted_key: str, value: Any) -> None:
        """Set e.g. ``"vision.detect_interval"`` with type coercion."""
        section, _, name = dotted_key.partition(".")
        target = getattr(self, section)
        if not name or not hasattr(target, name):
            raise KeyError(dotted_key)
        current = getattr(target, name)
        setattr(target, name, _coerce(value, current))


def _merge(obj: Any, data: dict[str, Any]) -> None:
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(obj, key, _coerce(value, current))


def _coerce(value: Any, current: Any) -> Any:
    if isinstance(value, str) and not isinstance(current, str):
        if isinstance(current, bool):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(current, int):
            return int(value)
        if isinstance(current, float) or current is None:
            try:
                return float(value)
            except ValueError:
                return value
        if isinstance(current, list):
            return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value
