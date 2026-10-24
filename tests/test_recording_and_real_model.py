import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import make_cfg, make_engine, run_frames, scenario_period

from motionplay.recording import KeypointRecorder, parse_thresholds, replay


def test_record_then_replay_reproduces_live_gestures(tmp_path):
    engine, src = make_engine("demo")
    path = tmp_path / "session.jsonl"
    engine.recorder = KeypointRecorder(path, src.width, src.height, mirror=True)
    run_frames(engine, src, scenario_period(src))
    engine.recorder.close()
    text = path.read_text()
    assert '"kpts"' in text and "jpg" not in text.lower()       # joints only, no images
    out = replay(path)
    key = lambda e: (e["gesture"], e["phase"], e["side"], e["direction"])  # noqa: E731
    assert [key(e) for e in out["events"]] == [key(e) for e in out["recorded"]]
    # Raising the swipe threshold above the stroke speed removes the swipes.
    stricter = replay(path, parse_thresholds(["swipe_speed=100"]))
    assert not [e for e in stricter["events"] if e["gesture"] == "swipe"]
    with pytest.raises(ValueError):
        parse_thresholds(["nope=1"])


def _mediapipe_model() -> Path | None:
    """A cached pose_landmarker_lite.task, if this machine has one."""
    if importlib.util.find_spec("mediapipe") is None:
        return None
    candidates = [os.environ.get("MOTIONPLAY_TEST_MP_MODEL", ""),
                  str(Path.home() / ".motionplay/models/mediapipe-pose-lite/pose_landmarker_lite.task")]
    return next((Path(c) for c in candidates if c and Path(c).is_file()), None)


@pytest.mark.skipif(_mediapipe_model() is None, reason="MediaPipe lite model not downloaded "
                    "(motionplay models download mediapipe-pose-lite)")
def test_real_mediapipe_model_tracks_and_detects_big_moves():
    """A real neural network on the synthetic actor: tracking should be
    continuous and the whole-body gestures should still come through."""
    from motionplay.vision.mediapipe_backend import MediaPipePoseBackend

    cfg = make_cfg(fps=30)
    engine, src = make_engine("demo", cfg=cfg)
    engine.backend = MediaPipePoseBackend(str(_mediapipe_model()))
    tracked, events = [], []
    try:
        run_frames(engine, src, scenario_period(src), lambda r: (
            tracked.append(len(r.players)), events.extend((r.frame.meta["t_scene"], g) for g in r.gestures)))
    finally:
        engine.backend.close()
    assert np.mean(np.array(tracked) == 1) > 0.95
    got = {(g.gesture, g.phase, g.direction) for _, g in events}
    for expected in [("jump", "trigger", None), ("squat", "start", None), ("squat", "end", None),
                     ("lean", "start", "left"), ("lean", "start", "right")]:
        assert expected in got, f"missing {expected}"
    jumps = [t for t, g in events if g.gesture == "jump"]
    assert len(jumps) == 1 and 4.3 < jumps[0] < 4.6
