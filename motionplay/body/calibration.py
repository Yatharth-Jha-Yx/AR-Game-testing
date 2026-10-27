"""Guided calibration for one player.

Steps:
  1. placement - head to ankles visible, at a good distance, light OK,
                 held for 0.7 s;
  2. tpose     - arms straight out to the sides, held still for 1.5 s; we
                 record shoulder width, torso length, head height, wingspan
                 and height (median over the hold);
  3. done      - results go to the depth estimator, gesture baselines, the
                 profile file, and the camera gets exposure/focus/WB locked.

The player stands on a floor mark at ``play_area.calib_distance_m`` (2 m by
default) so pixel sizes can be turned into metres.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..config import PlayAreaConfig, data_dir
from ..skeleton import J
from .checks import MESSAGES, lighting_check, placement_check
from .depth import BodyCalibration, focal_px
from .metrics import angle, body_sizes, dist, torso_length

PLACEMENT_HOLD_S = 0.7
TPOSE_HOLD_S = 1.5
TIMEOUT_S = 45.0


def is_tpose(kpts: np.ndarray, conf: np.ndarray) -> bool:
    joints = [J.LEFT_SHOULDER, J.RIGHT_SHOULDER, J.LEFT_ELBOW, J.RIGHT_ELBOW,
              J.LEFT_WRIST, J.RIGHT_WRIST, J.NECK, J.HIP]
    if np.any(conf[joints] < 0.4):
        return False
    torso = torso_length(kpts)
    if torso < 1:
        return False
    for sh, el, wr in ((J.LEFT_SHOULDER, J.LEFT_ELBOW, J.LEFT_WRIST),
                       (J.RIGHT_SHOULDER, J.RIGHT_ELBOW, J.RIGHT_WRIST)):
        if abs(kpts[wr][1] - kpts[sh][1]) > 0.3 * torso:       # wrist at shoulder height
            return False
        if abs(kpts[wr][0] - kpts[sh][0]) < 0.9 * torso:       # arm reaching out sideways
            return False
        if angle(kpts, sh, el, wr) < 150:                       # elbow straight
            return False
    return True


class CalibrationSession:
    def __init__(self, player_id: int, play: PlayAreaConfig, frame_w: int, frame_h: int, t: float,
                 distance_m: float | None = None):
        self.player_id = player_id
        self.play = play
        self.w, self.h = frame_w, frame_h
        self.distance_m = distance_m or play.calib_distance_m
        self.state = "placement"
        self.t_start = t
        self.t_state = t
        self.t_ok: float | None = None
        self.samples: list[dict[str, float]] = []
        self.result: BodyCalibration | None = None
        self.last_wrists: np.ndarray | None = None

    def update(self, kpts: np.ndarray | None, conf: np.ndarray | None, box: np.ndarray | None,
               image: np.ndarray, depth_m: float | None, lateral_screen_m: float, t: float) -> dict[str, Any]:
        light = lighting_check(image, box)
        place = placement_check(kpts, conf, self.w, self.h, depth_m, self.play, lateral_screen_m)
        msg: dict[str, Any] = {
            "type": "calibration", "player_id": self.player_id, "state": self.state,
            "checks": {"full_body": place["full_body"], "issues": place["issues"],
                       "lighting": light["status"], "light": light},
            "progress": 0.0, "message": "",
        }
        if self.state in ("done", "failed"):
            msg["state"] = self.state
            if self.result:
                msg["result"] = self.result.to_dict()
            return msg
        if t - self.t_start > TIMEOUT_S:
            self.state = "failed"
            msg.update(state="failed", message="Calibration timed out - press calibrate to try again")
            return msg
        problems = place["issues"] + ([light["status"]] if light["status"] != "ok" else [])

        if self.state == "placement":
            if problems or not place["ok"]:
                self.t_ok = None
                msg["message"] = MESSAGES.get(problems[0], "Stand still, facing the camera") if problems \
                    else "Stand still, facing the camera"
            else:
                self.t_ok = self.t_ok or t
                msg["progress"] = min(1.0, (t - self.t_ok) / PLACEMENT_HOLD_S)
                msg["message"] = "Great - hold still"
                if t - self.t_ok >= PLACEMENT_HOLD_S:
                    self.state, self.t_state, self.t_ok = "tpose", t, None
                    msg["state"] = "tpose"
                    msg["message"] = "Now stretch your arms out to the sides like a T"
            return msg

        # T-pose step.
        assert kpts is not None and conf is not None
        still = True
        wrists = kpts[[J.LEFT_WRIST, J.RIGHT_WRIST]]
        torso = max(torso_length(kpts), 1.0)
        if self.last_wrists is not None:
            still = float(np.max(np.linalg.norm(wrists - self.last_wrists, axis=1))) < 0.08 * torso
        self.last_wrists = wrists.copy()
        if is_tpose(kpts, conf) and still and not place["issues"]:
            self.t_ok = self.t_ok or t
            self.samples.append(self._measure(kpts, conf))
            msg["progress"] = min(1.0, (t - self.t_ok) / TPOSE_HOLD_S)
            msg["message"] = "Hold it..."
            if t - self.t_ok >= TPOSE_HOLD_S and len(self.samples) >= 5:
                self.result = self._finish()
                self.state = "done"
                msg.update(state="done", progress=1.0, message="All set!", result=self.result.to_dict())
        else:
            self.t_ok = None
            self.samples.clear()
            msg["message"] = MESSAGES.get(problems[0]) if place["issues"] else \
                "Arms straight out to the sides, at shoulder height"
        return msg

    def _measure(self, kpts: np.ndarray, conf: np.ndarray) -> dict[str, float]:
        m = {k: v for k, v in body_sizes(kpts, conf, thr=0.4).items()}
        torso = torso_length(kpts)
        ankles_y = (kpts[J.LEFT_ANKLE][1] + kpts[J.RIGHT_ANKLE][1]) / 2
        m["wingspan"] = dist(kpts, J.LEFT_WRIST, J.RIGHT_WRIST)
        m["height"] = ankles_y - kpts[J.HEAD][1]
        m["hip_knee"] = ((kpts[J.LEFT_KNEE][1] - kpts[J.LEFT_HIP][1]) +
                         (kpts[J.RIGHT_KNEE][1] - kpts[J.RIGHT_HIP][1])) / 2
        m["torso_px"] = torso
        m["floor_x"] = float(kpts[J.HIP][0] / self.w)
        return m

    def _finish(self) -> BodyCalibration:
        med = {k: float(np.median([s[k] for s in self.samples if k in s]))
               for k in self.samples[0].keys()}
        torso = med["torso_px"]
        f = focal_px(self.w, self.play.hfov_deg)
        to_m = self.distance_m / f
        # Wrists sit a hand-length short of the fingertips; +0.08 m ankle->floor.
        return BodyCalibration(
            distance_m=self.distance_m,
            frame_height=self.h,
            sizes={k: med[k] / self.h for k in ("shoulder", "torso", "head") if k in med},
            wingspan_torso=med["wingspan"] / torso,
            height_torso=med["height"] / torso,
            hip_knee_torso=med["hip_knee"] / torso,
            height_m=round(med["height"] * to_m + 0.08, 2),
            wingspan_m=round(med["wingspan"] * to_m + 0.36, 2),
            floor_x=med["floor_x"],
        )


def load_profiles() -> dict[int, BodyCalibration]:
    path = data_dir() / "calibration.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        return {int(k): BodyCalibration.from_dict(v) for k, v in raw.items()}
    except (ValueError, TypeError):
        return {}


def save_profile(player_id: int, cal: BodyCalibration) -> None:
    profiles = load_profiles()
    profiles[player_id] = cal
    path = data_dir() / "calibration.json"
    path.write_text(json.dumps({str(k): v.to_dict() for k, v in profiles.items()}, indent=2))
