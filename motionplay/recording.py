"""Record / replay *keypoints only* (never video) for tuning gestures.

  motionplay run --record-keypoints session.jsonl     # play normally, joints get saved
  motionplay replay session.jsonl --threshold swipe_speed=5

Replay feeds the saved smoothed joints through the gesture detectors with the
thresholds you pass, so you can tune on real players without a camera and
without keeping any images. One JSON object per frame:
  {"t": 12.345, "w": 1920, "h": 1080, "mirror": true,
   "players": [{"id": 1, "kpts": [[x, y], ...], "conf": [...]}],
   "gestures": [{"player_id": 1, "gesture": "jump", ...}]}
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .gestures.engine import GestureEngine, GestureThresholds


class KeypointRecorder:
    def __init__(self, path: str | Path, width: int, height: int, mirror: bool):
        self.path = Path(path)
        self.file: TextIO = open(self.path, "w", encoding="utf-8")
        self.meta = {"w": width, "h": height, "mirror": mirror}

    def write(self, t: float, players: list[tuple[int, np.ndarray, np.ndarray]],
              gestures: list[dict[str, Any]]) -> None:
        row = {"t": round(t, 4), **self.meta,
               "players": [{"id": pid, "kpts": np.round(k, 1).tolist(), "conf": np.round(c, 3).tolist()}
                           for pid, k, c in players],
               "gestures": gestures}
        self.file.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self.file.close()


def parse_thresholds(items: list[str] | None) -> GestureThresholds:
    th = GestureThresholds()
    names = {f.name for f in fields(GestureThresholds)}
    for item in items or []:
        key, _, value = item.partition("=")
        if key not in names:
            raise ValueError(f"unknown threshold {key!r}; choose from {sorted(names)}")
        setattr(th, key, float(value))
    return th


def replay(path: str | Path, thresholds: GestureThresholds | None = None) -> dict[str, Any]:
    """Re-run gesture detection over a recording. Returns new vs recorded events."""
    engines: dict[int, GestureEngine] = {}
    new, recorded = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            recorded += row.get("gestures", [])
            for p in row["players"]:
                ge = engines.get(p["id"])
                if ge is None:
                    ge = engines[p["id"]] = GestureEngine(p["id"], mirror=row.get("mirror", True),
                                                          thresholds=thresholds)
                for ev in ge.update(np.array(p["kpts"], np.float32), np.array(p["conf"], np.float32), row["t"]):
                    new.append({"t": ev.t, "player_id": ev.player_id, "gesture": ev.gesture,
                                "phase": ev.phase, "side": ev.side, "direction": ev.direction,
                                "strength": round(ev.strength, 3)})
    return {"events": new, "recorded": recorded}
