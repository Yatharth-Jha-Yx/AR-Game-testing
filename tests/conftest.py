import numpy as np
import pytest

from motionplay.capture.synthetic import SyntheticSource
from motionplay.config import Config
from motionplay.engine import Engine
from motionplay.vision.synthetic_backend import SyntheticPoseBackend


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real ~/.motionplay from tests."""
    monkeypatch.setenv("MOTIONPLAY_HOME", str(tmp_path / "home"))


def make_cfg(width=1280, height=720, fps=60) -> Config:
    cfg = Config()
    cfg.camera.width, cfg.camera.height, cfg.camera.fps = width, height, fps
    cfg.camera.realtime = False
    return cfg


def make_engine(scenario: str, one_stage: bool = False, cfg: Config | None = None):
    cfg = cfg or make_cfg()
    cfg.camera.source = f"synthetic:{scenario}"
    src = SyntheticSource(scenario, cfg.camera)
    src.open()
    return Engine(cfg, src, SyntheticPoseBackend(one_stage=one_stage)), src


def scenario_period(src: SyntheticSource) -> float:
    return max(sum(s.duration for s in a.segments) for a in src.actors)


def run_frames(engine, src, seconds: float, on_result=None):
    results = []
    for _ in range(int(seconds * src.fps)):
        r = engine.process(src.read())
        if on_result:
            on_result(r)
        results.append(r)
    return results


@pytest.fixture
def rng():
    return np.random.default_rng(0)
