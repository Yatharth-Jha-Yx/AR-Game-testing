import json
import os

import numpy as np
import pytest
from conftest import make_engine, run_frames

from motionplay.body.checks import lighting_check
from motionplay.body.depth import BodyCalibration, DepthEstimator
from motionplay.config import PlayAreaConfig
from motionplay.skeleton import J


def test_uncalibrated_depth_is_in_the_right_ballpark():
    engine, src = make_engine("demo")
    results = run_frames(engine, src, 1.0)
    depth = results[-1].players[0].depth_m
    assert results[-1].players[0].depth_calibrated is False
    assert depth == pytest.approx(3.0, rel=0.15)      # actor stands at 3.0 m


def test_calibration_flow_measures_body_and_fixes_depth():
    engine, src = make_engine("demo")
    engine.start_calibration(distance_m=3.0)
    msgs = []
    results = run_frames(engine, src, 5.0, lambda r: msgs.extend(r.messages))
    states = [m.get("state") for m in msgs if m["type"] == "calibration"]
    assert states[0] == "placement" and "tpose" in states and states[-1] == "done"
    done = next(m for m in msgs if m.get("state") == "done")
    assert done["result"]["height_m"] == pytest.approx(1.70, abs=0.05)
    p = results[-1].players[0]
    assert p.depth_calibrated and p.depth_m == pytest.approx(3.0, abs=0.06)
    # Saved for next time, and the camera was asked to lock.
    saved = json.loads((__import__("pathlib").Path(os.environ["MOTIONPLAY_HOME"]) / "calibration.json").read_text())
    assert "1" in saved
    assert any(m["type"] == "camera_lock" for m in msgs)


def test_depth_median_ignores_turned_shoulders():
    """Turning sideways shrinks shoulder width only; median of 3 cues holds."""
    cal = BodyCalibration(distance_m=2.0, frame_height=1080,
                          sizes={"shoulder": 200 / 1080, "torso": 300 / 1080, "head": 130 / 1080})
    est = DepthEstimator(PlayAreaConfig(), 1920, 1080, cal)
    assert est.raw_depth({"shoulder": 200, "torso": 300, "head": 130}) == pytest.approx(2.0)
    assert est.raw_depth({"shoulder": 90, "torso": 300, "head": 130}) == pytest.approx(2.0)
    assert est.raw_depth({"shoulder": 100, "torso": 150, "head": 65}) == pytest.approx(4.0)


def test_lighting_classifier():
    dark = np.full((720, 1280, 3), 20, np.uint8)
    assert lighting_check(dark)["status"] == "dark"
    backlit = np.full((720, 1280, 3), 230, np.uint8)
    backlit[200:700, 540:740] = 60
    assert lighting_check(backlit, np.array([540, 200, 740, 700]))["status"] == "backlit"
    ok = np.full((720, 1280, 3), 130, np.uint8)
    assert lighting_check(ok, np.array([540, 200, 740, 700]))["status"] == "ok"


def test_bounds_hint_when_feet_out_of_frame():
    engine, src = make_engine("demo")
    for actor in src.actors:
        actor.z = 2.2                                   # too close: feet cut off at 70 deg FOV
    r = run_frames(engine, src, 0.5)[-1]
    assert r.players[0].bounds_hint == "step_back"
    assert r.players[0].kpts[J.HEAD][1] > 0
