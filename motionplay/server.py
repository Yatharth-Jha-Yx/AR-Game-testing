"""Local web server: WebSocket API for games + static files for the bundled
web pages (launcher, calibration, debug viewer, games).

Binds to 127.0.0.1 by default, so camera pixels never leave the machine.
One aiohttp process can feed any number of games/overlays at once.

Client -> server messages (JSON):
  {"type": "subscribe", "frames": true, "video": false, "mask": false}
  {"type": "calibrate", "action": "start" | "cancel", "player_id": 1, "distance_m": 2.0}
  {"type": "camera", "action": "lock" | "unlock"}
  {"type": "settings", "skeleton_only": true}
  {"type": "ping", "t": <client ms>}
See docs/PROTOCOL.md for everything the server sends.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from . import protocol
from .config import Config
from .engine import Engine
from .types import FrameResult

log = logging.getLogger(__name__)

WEB_ROOT = Path(__file__).parent / "web"


@dataclass(eq=False)   # identity hashing, so clients can live in a set
class Client:
    ws: web.WebSocketResponse
    frames: bool = True
    video: bool = False
    mask: bool = False
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    pending_frames: int = 0

    def offer(self, msg: str | bytes, droppable: bool = False) -> None:
        """Queue a message. Frames/video are dropped for slow clients
        (stale body data is useless); events are never dropped."""
        if droppable and self.pending_frames >= 2:
            return
        try:
            self.queue.put_nowait((msg, droppable))
            if droppable:
                self.pending_frames += 1
        except asyncio.QueueFull:
            pass


class Server:
    def __init__(self, engine: Engine, cfg: Config):
        self.engine = engine
        self.cfg = cfg
        self.clients: set[Client] = set()
        self.norm = protocol.Normaliser(engine.w, engine.h, cfg.camera.mirror)
        self.loop: asyncio.AbstractEventLoop | None = None
        self._last_preview = 0.0
        self._encoding = False
        self.app = web.Application()
        self.app.router.add_get("/ws", self.ws_handler)
        self.app.router.add_get("/api/status", self.status_handler)
        self.app.router.add_get("/{path:.*}", self.static_handler)
        self.app.on_startup.append(self._on_startup)
        self.app.on_shutdown.append(self._on_shutdown)

    # --- lifecycle -----------------------------------------------------------
    async def _on_startup(self, app: web.Application) -> None:
        self.loop = asyncio.get_running_loop()
        self.engine.listeners.append(self._from_engine_thread)
        self._status_task = asyncio.create_task(self._status_loop())

    async def _on_shutdown(self, app: web.Application) -> None:
        self._status_task.cancel()
        for c in list(self.clients):
            await c.ws.close()

    def _from_engine_thread(self, result: FrameResult) -> None:
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.publish, result)

    def settings(self) -> dict[str, Any]:
        p = self.cfg.privacy
        return {"video_enabled": p.video_enabled, "skeleton_only": p.skeleton_only,
                "cooldown_minutes": p.cooldown_minutes, "segmentation": self.engine.segmenter is not None}

    def video_allowed(self) -> bool:
        return self.cfg.privacy.video_enabled and not self.cfg.privacy.skeleton_only

    # --- engine -> clients -------------------------------------------------
    def publish(self, result: FrameResult) -> None:
        if not self.clients:
            return
        frame_json = json.dumps(protocol.frame_message(result, self.norm), separators=(",", ":"))
        events = [json.dumps(protocol.gesture_message(g)) for g in result.gestures]
        events += [json.dumps({"type": "player", **e}) for e in result.player_events]
        events += [json.dumps(m) for m in result.messages]
        for c in self.clients:
            for e in events:
                c.offer(e)
            if c.frames:
                c.offer(frame_json, droppable=True)
        now = time.perf_counter()
        wants_video = self.video_allowed() and any(c.video for c in self.clients)
        if wants_video and not self._encoding and now - self._last_preview >= 1.0 / self.cfg.server.preview_fps:
            self._last_preview = now
            self._encoding = True
            asyncio.ensure_future(self._send_preview(result))
        if result.mask is not None and self.video_allowed():
            asyncio.ensure_future(self._send_mask(result))

    async def _send_preview(self, result: FrameResult) -> None:
        try:
            data = await asyncio.to_thread(self._encode_preview, result.frame.image)
            packet = protocol.pack_binary(protocol.BIN_VIDEO_JPEG, result.frame.seq, result.frame.t_capture, data)
            for c in self.clients:
                if c.video:
                    c.offer(packet, droppable=True)
        finally:
            self._encoding = False

    def _encode_preview(self, image: np.ndarray) -> bytes:
        h, w = image.shape[:2]
        pw = min(self.cfg.server.preview_width, w)
        small = cv2.resize(image, (pw, int(h * pw / w)), interpolation=cv2.INTER_AREA)
        if self.cfg.camera.mirror:
            small = cv2.flip(small, 1)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.server.jpeg_quality])
        return buf.tobytes() if ok else b""

    async def _send_mask(self, result: FrameResult) -> None:
        mask = result.mask
        if self.cfg.camera.mirror:
            mask = cv2.flip(mask, 1)
        ok, buf = await asyncio.to_thread(cv2.imencode, ".png", mask)
        if not ok:
            return
        packet = protocol.pack_binary(protocol.BIN_MASK_PNG, result.frame.seq, result.frame.t_capture, buf.tobytes())
        for c in self.clients:
            if c.mask:
                c.offer(packet, droppable=True)

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            if self.clients:
                msg = json.dumps(self.engine.status())
                for c in self.clients:
                    c.offer(msg)

    # --- HTTP / WebSocket handlers ------------------------------------------
    async def ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=1 << 20)
        await ws.prepare(request)
        client = Client(ws)
        self.clients.add(client)
        src = self.engine.source
        client.offer(json.dumps(protocol.hello_message(
            src.width, src.height, self.engine.fps, self.cfg.camera.mirror,
            self.engine.backend.name, self.settings())))
        sender = asyncio.create_task(self._sender(client))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        self._handle(client, json.loads(msg.data))
                    except (ValueError, KeyError, TypeError) as exc:
                        client.offer(json.dumps({"type": "error", "message": str(exc)}))
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self.clients.discard(client)
            sender.cancel()
            self._update_mask_interest()
        return ws

    async def _sender(self, client: Client) -> None:
        try:
            while True:
                msg, droppable = await client.queue.get()
                if droppable:
                    client.pending_frames -= 1
                if isinstance(msg, bytes):
                    await client.ws.send_bytes(msg)
                else:
                    await client.ws.send_str(msg)
        except (ConnectionResetError, RuntimeError, asyncio.CancelledError):
            pass

    def _handle(self, client: Client, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "subscribe":
            client.frames = bool(msg.get("frames", client.frames))
            client.video = bool(msg.get("video", client.video))
            client.mask = bool(msg.get("mask", client.mask))
            self._update_mask_interest()
        elif kind == "calibrate":
            if msg.get("action", "start") == "start":
                self.engine.start_calibration(msg.get("player_id"), msg.get("distance_m"))
            else:
                self.engine.cancel_calibration()
        elif kind == "camera":
            if msg.get("action") == "unlock":
                self.engine.unlock_camera()
            else:
                self.engine.lock_camera()
        elif kind == "settings":
            # Only display preferences can change from a client. The parental
            # video switch lives in config.json / CLI on purpose.
            if "skeleton_only" in msg:
                self.cfg.privacy.skeleton_only = bool(msg["skeleton_only"])
                try:
                    self.cfg.save()
                except OSError:
                    pass
            out = json.dumps({"type": "settings", "settings": self.settings()})
            for c in self.clients:
                c.offer(out)
        elif kind == "ping":
            client.offer(json.dumps({"type": "pong", "t": msg.get("t"), "server_t": time.perf_counter()}))
        else:
            raise ValueError(f"unknown message type {kind!r}")

    def _update_mask_interest(self) -> None:
        self.engine.mask_wanted = any(c.mask for c in self.clients)

    async def status_handler(self, request: web.Request) -> web.Response:
        return web.json_response(self.engine.status())

    async def static_handler(self, request: web.Request) -> web.StreamResponse:
        rel = request.match_info["path"]
        root = WEB_ROOT.resolve()
        path = (root / rel).resolve()
        if root not in path.parents and path != root:
            raise web.HTTPForbidden()
        if path.is_dir():
            if rel and not rel.endswith("/"):
                raise web.HTTPFound(f"/{rel}/")
            path = path / "index.html"
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "no-cache"})


def health_line(st: dict[str, Any]) -> str:
    """One line that says where any lag comes from: the camera, the AI, or neither."""
    cam = st["source"].get("measured_fps") or st["source"].get("fps")
    tm = st["timings_ms"]
    ai = tm.get("pose", 0) + tm.get("detect", 0) / 10
    return (f"camera {cam:.0f} fps | tracking {st['processed_fps']} fps | pose {ai:.1f} ms | "
            f"delay {st['latency_ms']['mean']:.0f} ms (p95 {st['latency_ms']['p95']:.0f}) | "
            f"players {sum(p['state'] == 'active' for p in st['players'])}")


async def _health_loop(engine: Engine, every_s: float = 5.0) -> None:
    while True:
        await asyncio.sleep(every_s)
        try:
            print("  " + health_line(engine.status()), flush=True)
        except Exception:  # noqa: BLE001 - diagnostics must never stop the engine
            log.debug("health line failed", exc_info=True)


def run_server(engine: Engine, cfg: Config, open_browser: bool = False) -> None:
    server = Server(engine, cfg)
    url = f"http://{'localhost' if cfg.server.host in ('127.0.0.1', '0.0.0.0') else cfg.server.host}:{cfg.server.port}/"
    if cfg.server.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("listening on %s: other devices on your network can connect and see the "
                    "camera preview", cfg.server.host)

    async def on_start(app: web.Application) -> None:
        engine.start()
        print(f"\n  MotionPlay engine running - open {url}\n  backend: {engine.backend.name} "
              f"({engine.backend.provider}), source: {engine.source.name} "
              f"{engine.w}x{engine.h}@{engine.fps:.0f}\n", flush=True)
        if engine.backend.name.startswith("mediapipe") and cfg.vision.backend != "mediapipe":
            print("  NOTE: running the CPU fallback (MediaPipe) because RTMPose could not start -\n"
                  "  see the warning above. Expect more lag and only one player.\n", flush=True)
        app["health_task"] = asyncio.create_task(_health_loop(engine))
        if open_browser:
            import webbrowser
            webbrowser.open(url)

    async def on_cleanup(app: web.Application) -> None:
        app["health_task"].cancel()
        engine.stop()

    server.app.on_startup.append(on_start)
    server.app.on_cleanup.append(on_cleanup)
    web.run_app(server.app, host=cfg.server.host, port=cfg.server.port, print=None)
