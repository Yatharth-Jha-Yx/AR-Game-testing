"""The tracker service: capture -> inference -> publish.

Two worker threads plus the caller's thread (the web server's asyncio loop):

  capture thread   reads + decodes camera frames (MJPEG decode happens inside
                   OpenCV here, overlapped with inference) into a "newest
                   wins" slot;
  inference thread takes the newest frame, runs detector/pose, tracking,
                   smoothing, depth, gestures and calibration, then hands a
                   FrameResult to listeners (the server).

``Engine.process(frame)`` is plain synchronous code so tests can drive it
frame by frame without threads.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .body.calibration import CalibrationSession, load_profiles, save_profile
from .body.depth import BodyCalibration, DepthEstimator
from .capture.buffer import LatestSlot
from .capture.sources import FrameSource
from .config import Config
from .gestures.engine import GestureEngine
from .skeleton import J
from .tracking.tracker import PlayerTracker, Track, iou
from .types import Frame, FrameResult, PlayerState
from .vision.base import PoseBackend

log = logging.getLogger(__name__)


@dataclass
class PlayerContext:
    depth: DepthEstimator
    gestures: GestureEngine
    calibration: BodyCalibration | None = None


class Engine:
    def __init__(self, cfg: Config, source: FrameSource, backend: PoseBackend, segmenter: Any = None):
        self.cfg = cfg
        self.source = source
        self.backend = backend
        self.segmenter = segmenter
        self.w, self.h = source.width, source.height
        self.fps = source.fps or float(cfg.camera.fps)
        self.tracker = PlayerTracker(cfg, self.w, self.h)
        self.players: dict[int, PlayerContext] = {}
        self.profiles = load_profiles()
        self.calibration: CalibrationSession | None = None
        self.frame_index = 0
        self._t_now = 0.0            # capture time of the frame being processed
        self.mask_wanted = False
        self._last_mask_t = -1e9
        self._last_mask: np.ndarray | None = None
        self._commands: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._outbox: deque[dict[str, Any]] = deque()
        self.listeners: list[Callable[[FrameResult], None]] = []
        self.recorder: Any = None    # recording.KeypointRecorder (joints only, never video)
        self.frames: LatestSlot[Frame] = LatestSlot()
        self._threads: list[threading.Thread] = []
        self._running = False
        self.ended = False
        # Stats (exponential moving averages).
        self.stats: dict[str, Any] = {"timings_ms": {}}
        self._latencies: deque[float] = deque(maxlen=240)
        self._done_times: deque[float] = deque(maxlen=600)
        self._pose_ms = 0.0

    # --- control (thread-safe: runs inside the inference thread) ------------
    def submit(self, fn: Callable[[], None]) -> None:
        self._commands.put(fn)

    def start_calibration(self, player_id: int | None = None, distance_m: float | None = None) -> None:
        def go() -> None:
            pid = player_id
            if pid is None:
                active = self.tracker.active()
                # Default: the biggest (closest) player, or slot 1 if nobody yet.
                pid = max(active, key=lambda tr: tr.box[3] - tr.box[1]).id if active else 1
            self.calibration = CalibrationSession(pid, self.cfg.play_area, self.w, self.h,
                                                  self._t_now, distance_m)
        self.submit(go)

    def cancel_calibration(self) -> None:
        def go() -> None:
            if self.calibration is not None:
                self._outbox.append({"type": "calibration", "player_id": self.calibration.player_id,
                                     "state": "cancelled", "progress": 0, "message": "Calibration cancelled"})
            self.calibration = None
        self.submit(go)

    def lock_camera(self) -> None:
        def done(report: dict[str, Any]) -> None:
            self._outbox.append({"type": "camera_lock", "report": report})
        self.source.lock_controls(done)

    def unlock_camera(self) -> None:
        self.source.unlock_controls()
        self._outbox.append({"type": "camera_lock", "report": {"unlocked": True}})

    # --- one frame ---------------------------------------------------------
    def process(self, frame: Frame) -> FrameResult:
        self._t_now = frame.t_capture
        while True:
            try:
                self._commands.get_nowait()()
            except queue.Empty:
                break
        t = frame.t_capture
        timings: dict[str, float] = {}
        self.frame_index += 1
        p0 = time.perf_counter()

        crowded = len(self.tracker.active()) >= 3
        budget_ms = 1000.0 / self.fps * 0.4
        skip = (crowded and self._pose_ms > budget_ms and
                self.frame_index % max(1, self.cfg.vision.crowded_pose_divisor) != 0)
        events: list[dict[str, Any]] = []
        if not skip:
            if self.backend.one_stage:
                cands = self.backend.estimate_all(frame)
                timings["pose"] = (time.perf_counter() - p0) * 1000
            else:
                boxes = self.tracker.crop_boxes(t)
                if self.tracker.wants_detection(self.frame_index, self.cfg.vision.detect_interval):
                    d0 = time.perf_counter()
                    dets = self.backend.detect(frame)
                    timings["detect"] = (time.perf_counter() - d0) * 1000
                    dets = dets[dets[:, 4] >= self.cfg.vision.det_score_low] if len(dets) else dets
                    keep = [b for b in boxes if all(iou(b, d) < 0.5 for d in dets)]
                    boxes = np.vstack([dets.reshape(-1, 5), np.array(keep, np.float32).reshape(-1, 5)])
                    n_det = len(dets)
                else:
                    n_det = 0
                p1 = time.perf_counter()
                cands = self.backend.estimate(frame, boxes) if len(boxes) else []
                for i, c in enumerate(cands):
                    c.from_detection = i < n_det
                timings["pose"] = (time.perf_counter() - p1) * 1000
            self._pose_ms = 0.9 * self._pose_ms + 0.1 * (timings["pose"] + timings.get("detect", 0) / 10)
            p2 = time.perf_counter()
            events = self.tracker.update(cands, frame.image, t)
            timings["track"] = (time.perf_counter() - p2) * 1000

        p3 = time.perf_counter()
        for ev in events:
            if ev["event"] == "joined":
                self._new_context(ev["player_id"])
            elif ev["event"] == "left":
                self.players.pop(ev["player_id"], None)
        players, gestures = [], []
        ahead = (self.cfg.tracking.predict_ms / 1000.0) if self.cfg.tracking.predict_ms else 1.0 / self.fps
        for tr in self.tracker.active():
            ctx = self.players.get(tr.id) or self._new_context(tr.id)
            if skip:
                kpts = tr.smoother.extrapolate(t + ahead)
                conf = tr.smoother.conf
            else:
                smoothed, conf = tr.smoother.smoothed, tr.smoother.conf
                gestures += ctx.gestures.update(smoothed, conf, t)
                kpts = tr.smoother.predicted(ahead)
            players.append(self._player_state(tr, ctx, kpts, conf, t, predicted=skip))
        timings["body"] = (time.perf_counter() - p3) * 1000
        if self.recorder is not None and not skip:
            self.recorder.write(t, [(tr.id, tr.smoother.smoothed, tr.smoother.conf) for tr in self.tracker.active()],
                                [{"player_id": g.player_id, "gesture": g.gesture, "phase": g.phase,
                                  "side": g.side, "direction": g.direction, "t": g.t} for g in gestures])

        messages = list(self._drain_outbox())
        if self.calibration is not None:
            messages.append(self._calibrate(frame, players))

        mask = None
        if self.segmenter is not None and self.mask_wanted:
            if t - self._last_mask_t >= 1.0 / max(self.cfg.vision.segmentation_hz, 1.0):
                s0 = time.perf_counter()
                self._last_mask = self.segmenter(frame.image)
                self._last_mask_t = t
                timings["segment"] = (time.perf_counter() - s0) * 1000
                mask = self._last_mask

        result = FrameResult(frame=frame, players=players, gestures=gestures, player_events=events,
                             messages=messages, mask=mask, timings_ms=timings)
        result.t_done = time.perf_counter() if self._running else t + (time.perf_counter() - p0)
        self._update_stats(result)
        return result

    def _new_context(self, pid: int) -> PlayerContext:
        cal = self.profiles.get(pid)
        if cal is not None and cal.frame_height != self.h:
            cal = None   # different camera mode: sizes no longer comparable
        ctx = PlayerContext(DepthEstimator(self.cfg.play_area, self.w, self.h, cal),
                            GestureEngine(pid, mirror=self.cfg.camera.mirror), cal)
        if cal is not None:
            ctx.gestures.set_calibration(cal.hip_knee_torso)
        self.players[pid] = ctx
        return ctx

    def _player_state(self, tr: Track, ctx: PlayerContext, kpts: np.ndarray, conf: np.ndarray,
                      t: float, predicted: bool) -> PlayerState:
        depth, calibrated = ctx.depth.estimate(tr.smoother.smoothed, conf, t)
        hip_x = float(kpts[J.HIP][0])
        lateral = ctx.depth.lateral_m(hip_x, depth) if depth else 0.0
        play = self.cfg.play_area
        screen_lateral = -lateral if self.cfg.camera.mirror else lateral
        feet_cut = max(kpts[J.LEFT_ANKLE][1], kpts[J.RIGHT_ANKLE][1]) > 0.99 * self.h or \
            min(conf[J.LEFT_ANKLE], conf[J.RIGHT_ANKLE]) < 0.3
        hint = None
        if depth and depth < play.min_depth_m or feet_cut:
            hint = "step_back"
        elif depth and depth > play.max_depth_m:
            hint = "step_forward"
        elif screen_lateral > play.half_width_m:
            hint = "move_left"
        elif screen_lateral < -play.half_width_m:
            hint = "move_right"
        torso = float(np.linalg.norm(kpts[J.NECK] - kpts[J.HIP]))
        reach = ctx.calibration.wingspan_torso / 2 if ctx.calibration else 1.6
        return PlayerState(id=tr.id, kpts=kpts, conf=conf, box=tr.box, confidence=tr.quality,
                           torso_px=torso, depth_m=depth, depth_calibrated=calibrated,
                           position_m=(lateral, depth), in_bounds=hint is None, bounds_hint=hint,
                           predicted=predicted, reach_torso=reach)

    def _calibrate(self, frame: Frame, players: list[PlayerState]) -> dict[str, Any]:
        cal = self.calibration
        assert cal is not None
        p = next((pl for pl in players if pl.id == cal.player_id), None)
        tr = self.tracker.tracks.get(cal.player_id)
        kpts = tr.smoother.smoothed if (p and tr) else None
        lateral_screen = (-p.position_m[0] if self.cfg.camera.mirror else p.position_m[0]) if p else 0.0
        msg = cal.update(kpts, p.conf if p else None, p.box if p else None, frame.image,
                         p.depth_m if p else None, lateral_screen, frame.t_capture)
        if cal.state == "done" and cal.result is not None:
            ctx = self.players.get(cal.player_id) or self._new_context(cal.player_id)
            ctx.calibration = cal.result
            ctx.depth.calibration = cal.result
            ctx.gestures.set_calibration(cal.result.hip_knee_torso)
            self.profiles[cal.player_id] = cal.result
            try:
                save_profile(cal.player_id, cal.result)
            except OSError as exc:
                log.warning("could not save calibration: %s", exc)
            self.lock_camera()
            self.calibration = None
        elif cal.state == "failed":
            self.calibration = None
        return msg

    def _drain_outbox(self):
        while self._outbox:
            yield self._outbox.popleft()

    def _update_stats(self, r: FrameResult) -> None:
        self._done_times.append(r.t_done)
        self._latencies.append((r.t_done - r.frame.t_capture) * 1000)
        tm = self.stats["timings_ms"]
        for k, v in r.timings_ms.items():
            tm[k] = round(0.9 * tm.get(k, v) + 0.1 * v, 2)

    def status(self) -> dict[str, Any]:
        # Called from the server thread while the inference thread keeps
        # writing: take C-level copies first (list()/dict() hold the GIL).
        latencies = list(self._latencies)
        done = list(self._done_times)
        tracks = list(self.tracker.tracks.values())
        calibration = self.calibration
        lat = np.array(latencies) if latencies else np.zeros(1)
        # Frames finished during the last second (robust to bursts).
        fps = sum(1 for t in done if done[-1] - t < 1.0) if done else 0
        return {
            "type": "status",
            "source": self.source.info(),
            "backend": self.backend.info(),
            "processed_fps": fps,
            "latency_ms": {"mean": round(float(lat.mean()), 1), "p95": round(float(np.percentile(lat, 95)), 1)},
            "timings_ms": dict(self.stats["timings_ms"]),
            "players": [{"id": tr.id, "state": tr.state} for tr in tracks],
            "dropped_frames": self.frames.dropped,
            "segmentation": self.segmenter is not None,
            "calibrating": calibration.player_id if calibration else None,
        }

    # --- threads -------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._threads = [threading.Thread(target=self._capture_loop, name="capture", daemon=True),
                         threading.Thread(target=self._infer_loop, name="inference", daemon=True)]
        for th in self._threads:
            th.start()

    def stop(self) -> None:
        self._running = False
        self.frames.close()
        for th in self._threads:
            th.join(timeout=2)
        self.source.close()
        self.backend.close()
        if self.segmenter is not None:
            self.segmenter.close()
        if self.recorder is not None:
            self.recorder.close()

    def _capture_loop(self) -> None:
        failures = 0
        while self._running:
            try:
                frame = self.source.read()
            except Exception:  # noqa: BLE001
                log.exception("capture failed")
                frame = None
            if frame is None:
                failures += 1
                if failures > 30 or not self.source.name.startswith("camera"):
                    log.warning("source %s ended", self.source.name)
                    self.ended = True
                    self.frames.close()
                    return
                time.sleep(0.05)
                continue
            failures = 0
            self.frames.put(frame)

    def _infer_loop(self) -> None:
        while self._running:
            frame = self.frames.get(timeout=0.5)
            if frame is None:
                if self.ended:
                    return
                continue
            try:
                result = self.process(frame)
            except Exception:  # noqa: BLE001
                log.exception("processing failed")
                continue
            for listener in list(self.listeners):
                try:
                    listener(result)
                except Exception:  # noqa: BLE001
                    log.exception("listener failed")
