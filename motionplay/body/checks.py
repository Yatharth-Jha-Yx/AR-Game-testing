"""Setup checks: is the player fully in view, centred, at a good distance,
and is the lighting usable? Used by calibration and the in-game warnings."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..config import PlayAreaConfig
from ..skeleton import FULL_BODY_JOINTS, J

MESSAGES = {
    "no_player": "Stand in front of the camera",
    "feet": "Step back until your feet are visible",
    "head": "Step back so your whole head is visible",
    "too_close": "Step back a little",
    "too_far": "Come a bit closer",
    "move_left": "Move a little to your left",
    "move_right": "Move a little to your right",
    "dark": "The room is too dark - turn on a light",
    "backlit": "Bright light behind you - close the curtains or turn around the camera",
    "bright": "The picture is washed out - reduce direct light on the camera",
}


def lighting_check(image: np.ndarray, box: np.ndarray | None = None) -> dict[str, Any]:
    """Classify lighting as ok / dark / bright / backlit from a 160x90 thumbnail."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    frame_mean = float(gray.mean())
    clipped = float((gray >= 250).mean())
    person_mean = bg_mean = frame_mean
    if box is not None:
        sx, sy = 160 / w, 90 / h
        x0, y0 = int(max(0, box[0] * sx)), int(max(0, box[1] * sy))
        x1, y1 = int(min(160, box[2] * sx)), int(min(90, box[3] * sy))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            mask = np.zeros_like(gray, bool)
            mask[y0:y1, x0:x1] = True
            person_mean = float(gray[mask].mean())
            if (~mask).any():
                bg_mean = float(gray[~mask].mean())
    if frame_mean < 45 or person_mean < 40:
        status = "dark"
    elif clipped > 0.12:
        status = "bright"
    elif bg_mean > 150 and person_mean < 0.55 * bg_mean:
        status = "backlit"
    else:
        status = "ok"
    return {"status": status, "frame_luma": round(frame_mean, 1), "person_luma": round(person_mean, 1),
            "background_luma": round(bg_mean, 1), "clipped": round(clipped, 3)}


def placement_check(kpts: np.ndarray | None, conf: np.ndarray | None, frame_w: int, frame_h: int,
                    depth_m: float | None, play: PlayAreaConfig, lateral_screen_m: float = 0.0,
                    thr: float = 0.5) -> dict[str, Any]:
    """Everything a player must fix before calibrating, in priority order."""
    if kpts is None or conf is None:
        return {"ok": False, "issues": ["no_player"], "full_body": False}
    issues = []
    head_ok = (conf[J.HEAD] >= thr or conf[J.NOSE] >= thr) and kpts[J.HEAD][1] > 0.01 * frame_h
    feet_ok = all(conf[j] >= thr for j in (J.LEFT_ANKLE, J.RIGHT_ANKLE)) and \
        max(kpts[J.LEFT_ANKLE][1], kpts[J.RIGHT_ANKLE][1]) < 0.99 * frame_h
    body_ok = all(conf[j] >= thr for j in FULL_BODY_JOINTS)
    if not feet_ok:
        issues.append("feet")
    if not head_ok:
        issues.append("head")
    if depth_m:
        if depth_m < play.min_depth_m:
            issues.append("too_close")
        elif depth_m > play.max_depth_m:
            issues.append("too_far")
    if lateral_screen_m > play.half_width_m * 0.5:
        issues.append("move_left")
    elif lateral_screen_m < -play.half_width_m * 0.5:
        issues.append("move_right")
    return {"ok": not issues and body_ok, "issues": issues, "full_body": head_ok and feet_ok and body_ok}
