import json

import numpy as np
import pytest

from motionplay import protocol
from motionplay.config import Config
from motionplay.skeleton import J, NUM_KEYPOINTS
from motionplay.types import PlayerState


def _player():
    k = np.zeros((NUM_KEYPOINTS, 2), np.float32)
    k[J.NECK] = (300, 200)
    k[J.HIP] = (300, 400)
    k[J.LEFT_WRIST] = (500, 200)       # camera image right = player's left hand
    return PlayerState(id=1, kpts=k, conf=np.ones(NUM_KEYPOINTS, np.float32),
                       box=np.array([100, 50, 600, 700], np.float32), confidence=0.9, torso_px=200,
                       depth_m=2.5, depth_calibrated=False, position_m=(0.4, 2.5), in_bounds=True)


def test_body_state_is_mirrored_and_normalised():
    norm = protocol.Normaliser(1000, 800, mirror=True)
    b = protocol.body_state(_player(), norm)
    joints = np.array(b["joints"]).reshape(-1, 3)
    assert joints[J.LEFT_WRIST][0] == pytest.approx(1 - 0.5)     # mirrored
    assert joints[J.LEFT_WRIST][1] == pytest.approx(0.25)
    play = np.array(b["play"]).reshape(-1, 2)
    assert play[J.NECK].tolist() == pytest.approx([0, 1])        # 1 torso above mid-hip
    assert play[J.LEFT_WRIST][0] == pytest.approx(-1.0)          # screen-left in mirror view
    assert b["bbox"] == pytest.approx([0.4, 0.0625, 0.9, 0.875])
    assert b["position_m"][0] == pytest.approx(-0.4)
    json.dumps(b)


def test_binary_header_roundtrip():
    pkt = protocol.pack_binary(protocol.BIN_VIDEO_JPEG, 42, 12.5, b"\xff\xd8jpeg")
    assert len(pkt) == protocol.BIN_HEADER_SIZE + 6
    assert protocol.unpack_binary(pkt) == (protocol.BIN_VIDEO_JPEG, 42, 12.5, b"\xff\xd8jpeg")


def test_config_set_coerces_types_and_persists(tmp_path):
    cfg = Config()
    cfg.set("vision.detect_interval", "8")
    cfg.set("camera.mirror", "false")
    cfg.set("vision.providers", "cuda,cpu")
    cfg.set("tracking.predict_ms", "12.5")
    assert cfg.vision.detect_interval == 8 and cfg.camera.mirror is False
    assert cfg.vision.providers == ["cuda", "cpu"] and cfg.tracking.predict_ms == 12.5
    path = cfg.save(tmp_path / "c.json")
    again = Config.load(path)
    assert again.to_dict() == cfg.to_dict()
    with pytest.raises(KeyError):
        cfg.set("vision.nope", "1")


def test_benchmark_tier_choice():
    from motionplay.bench import choose_tier

    res = {"full": {"two_player_ms": 14.0}, "balanced": {"two_player_ms": 8.0}, "lite": {"two_player_ms": 4.0}}
    assert choose_tier(res) == ("balanced", 60)          # best tier inside the 10 ms budget
    assert choose_tier({"full": {"two_player_ms": 18.0}}) == ("full", 30)
    assert choose_tier({"lite": {"two_player_ms": 40.0}}) == ("lite", 30)
