import asyncio
import json

import pytest
from conftest import make_cfg

from motionplay import protocol
from motionplay.capture.synthetic import SyntheticSource
from motionplay.engine import Engine
from motionplay.server import Server
from motionplay.vision.synthetic_backend import SyntheticPoseBackend


@pytest.fixture
async def setup(aiohttp_client):
    async def make(**privacy):
        cfg = make_cfg(width=640, height=360, fps=30)
        cfg.camera.realtime = True
        for k, v in privacy.items():
            setattr(cfg.privacy, k, v)
        src = SyntheticSource("demo", cfg.camera)
        src.open()
        engine = Engine(cfg, src, SyntheticPoseBackend())
        server = Server(engine, cfg)
        client = await aiohttp_client(server.app)
        engine.start()
        made.append(engine)
        return client, engine, server
    made = []
    yield make
    for e in made:
        e.stop()


async def _collect(ws, until, timeout=5.0):
    """Read messages until until(text_msgs, binary_msgs) is true."""
    texts, bins = [], []
    async def loop():
        while not until(texts, bins):
            msg = await ws.receive()
            if msg.type.name == "TEXT":
                texts.append(json.loads(msg.data))
            elif msg.type.name == "BINARY":
                bins.append(msg.data)
            else:
                break
    await asyncio.wait_for(loop(), timeout)
    return texts, bins


async def test_hello_frames_and_status(setup):
    client, engine, _ = await setup()
    ws = await client.ws_connect("/ws")
    texts, _ = await _collect(ws, lambda t, b: any(m["type"] == "status" for m in t)
                              and sum(m["type"] == "frame" for m in t) >= 5)
    hello = texts[0]
    assert hello["type"] == "hello" and hello["protocol"] == protocol.PROTOCOL_VERSION
    assert len(hello["skeleton"]["keypoints"]) == 26 and hello["camera"]["mirrored"] is True
    frame = next(m for m in texts if m["type"] == "frame" and m["players"])
    p = frame["players"][0]
    assert len(p["joints"]) == 26 * 3 and len(p["play"]) == 26 * 2
    assert any(m["type"] == "player" and m["event"] == "joined" for m in texts)
    await ws.close()


async def test_video_subscription_and_privacy(setup):
    client, *_ = await setup()
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "subscribe", "video": True})
    _, bins = await _collect(ws, lambda t, b: len(b) >= 2)
    kind, seq, t, payload = protocol.unpack_binary(bins[0])
    assert kind == protocol.BIN_VIDEO_JPEG and payload[:2] == b"\xff\xd8"   # JPEG magic
    # Skeleton-only from a game: video stops, settings broadcast.
    await ws.send_json({"type": "settings", "skeleton_only": True})
    texts, _ = await _collect(ws, lambda t, b: any(m["type"] == "settings" for m in t))
    assert texts[-1]["settings"]["skeleton_only"] is True
    await asyncio.sleep(0.3)
    _, later = await _collect(ws, lambda t, b: sum(m["type"] == "frame" for m in t) >= 10)
    assert later == []
    await ws.close()


async def test_parental_switch_blocks_video_and_cannot_be_flipped_by_a_game(setup):
    client, *_ = await setup(video_enabled=False)
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "subscribe", "video": True})
    await ws.send_json({"type": "settings", "video_enabled": True, "skeleton_only": False})
    texts, bins = await _collect(ws, lambda t, b: sum(m["type"] == "frame" for m in t) >= 20)
    assert bins == []
    settings = [m for m in texts if m["type"] == "settings"][-1]["settings"]
    assert settings["video_enabled"] is False
    await ws.close()


async def test_calibrate_command_streams_progress(setup):
    client, *_ = await setup()
    ws = await client.ws_connect("/ws")
    await ws.send_json({"type": "calibrate", "action": "start", "distance_m": 3.0})
    texts, _ = await _collect(ws, lambda t, b: any(m["type"] == "calibration" for m in t))
    cal = next(m for m in texts if m["type"] == "calibration")
    assert cal["state"] in ("placement", "tpose") and "checks" in cal
    await ws.send_json({"type": "calibrate", "action": "cancel"})
    texts, _ = await _collect(ws, lambda t, b: any(m.get("state") == "cancelled" for m in t))
    await ws.close()


async def test_static_files_and_traversal(setup):
    client, *_ = await setup()
    assert (await client.get("/")).status == 200
    assert (await client.get("/games/fruit-slice/")).status == 200
    assert (await client.get("/sdk/motionplay.js")).status == 200
    assert (await client.get("/nope.html")).status == 404
    status = await (await client.get("/api/status")).json()
    assert status["type"] == "status" and status["backend"]["name"].startswith("synthetic")
