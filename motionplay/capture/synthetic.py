"""Synthetic players for testing without a webcam or model weights.

Each "actor" is a simple 3D stick body (1.70 m tall by default) standing in
front of a virtual pinhole camera 1 m above the floor. A scripted timeline
makes it jump, squat, punch, swipe, raise arms, lean and walk around. Every
frame is rendered with OpenCV and carries exact ground truth in
``frame.meta["gt"]`` so tests can check tracking and gestures, and the
synthetic pose backend can stand in for a real model.

Body proportions are rough adult averages (e.g. upper arm 0.30 m, forearm
0.27 m, thigh/shin 0.43 m), good enough to exercise the maths.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import CameraConfig
from ..skeleton import NUM_KEYPOINTS, J
from ..types import Frame
from .sources import FrameSource

CAMERA_HEIGHT_M = 1.0
UPPER_ARM, FOREARM, THIGH, SHIN, ANKLE_Y = 0.30, 0.27, 0.43, 0.43, 0.09

# Rest positions for a 1.70 m person: x = towards the person's LEFT,
# y = up from the floor, z = towards the camera.
_REST = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
for _j, _p in {
    J.HEAD: (0, 1.70, 0), J.NOSE: (0, 1.58, 0.10),
    J.LEFT_EYE: (0.033, 1.62, 0.08), J.RIGHT_EYE: (-0.033, 1.62, 0.08),
    J.LEFT_EAR: (0.075, 1.60, 0), J.RIGHT_EAR: (-0.075, 1.60, 0),
    J.NECK: (0, 1.46, 0),
    J.LEFT_SHOULDER: (0.19, 1.43, 0), J.RIGHT_SHOULDER: (-0.19, 1.43, 0),
    J.LEFT_HIP: (0.11, 0.95, 0), J.RIGHT_HIP: (-0.11, 0.95, 0), J.HIP: (0, 0.95, 0),
    J.LEFT_ANKLE: (0.11, ANKLE_Y, 0), J.RIGHT_ANKLE: (-0.11, ANKLE_Y, 0),
    J.LEFT_HEEL: (0.11, 0.03, -0.05), J.RIGHT_HEEL: (-0.11, 0.03, -0.05),
    J.LEFT_BIG_TOE: (0.12, 0.02, 0.15), J.RIGHT_BIG_TOE: (-0.12, 0.02, 0.15),
    J.LEFT_SMALL_TOE: (0.17, 0.02, 0.11), J.RIGHT_SMALL_TOE: (-0.17, 0.02, 0.11),
}.items():
    _REST[_j] = _p

_UPPER_BODY = [J.HEAD, J.NOSE, J.LEFT_EYE, J.RIGHT_EYE, J.LEFT_EAR, J.RIGHT_EAR,
               J.NECK, J.LEFT_SHOULDER, J.RIGHT_SHOULDER]

ArmAngles = tuple[float, float, float, float]   # upper (abd, flex), forearm (abd, flex), degrees
ARM_DOWN: ArmAngles = (10, 0, 8, 0)
ARM_T: ArmAngles = (90, 0, 90, 0)
ARM_UP: ArmAngles = (172, 0, 176, 0)
ARM_GUARD: ArmAngles = (22, 35, -158, 20)
ARM_PUNCH: ArmAngles = (75, 55, 75, 55)
ARM_SWIPE_START: ArmAngles = (-40, 20, -95, 10)
ARM_SWIPE_END: ArmAngles = (75, 20, 95, 10)


@dataclass
class Pose:
    arm_l: ArmAngles = ARM_DOWN
    arm_r: ArmAngles = ARM_DOWN
    squat: float = 0.0     # 0..1
    jump_h: float = 0.0    # metres
    lean_deg: float = 0.0  # + = towards the person's left
    x: float = 0.0         # lateral metres (+ = camera image right)
    z: float = 3.0         # distance from camera, metres
    hidden: bool = False


@dataclass
class Segment:
    action: str
    duration: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Actor:
    pid: int
    x: float
    z: float
    segments: list[Segment]
    height: float = 1.70
    shirt: tuple[int, int, int] = (200, 90, 40)   # BGR
    pants: tuple[int, int, int] = (60, 50, 40)
    skin: tuple[int, int, int] = (140, 170, 210)

    def pose_at(self, t: float) -> Pose:
        total = sum(s.duration for s in self.segments)
        t = t % total if total > 0 else 0.0
        x, z = self.x, self.z
        start = 0.0
        for seg in self.segments:
            end = start + seg.duration
            if t < end:
                return _apply(seg, (t - start) / seg.duration, t - start, x, z, t)
            if seg.action == "walk":
                x = seg.params.get("to_x", x)
                z = seg.params.get("to_z", z)
            start = end
        return Pose(x=x, z=z)


def _smooth(p: float) -> float:
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def _envelope(elapsed: float, duration: float, ramp: float) -> float:
    """0 -> 1 over `ramp` seconds, hold, back to 0 over the last `ramp`."""
    up = _smooth(elapsed / ramp)
    down = _smooth((duration - elapsed) / ramp)
    return min(up, down)


def _lerp_arm(a: ArmAngles, b: ArmAngles, p: float) -> ArmAngles:
    return tuple(av + (bv - av) * p for av, bv in zip(a, b))  # type: ignore[return-value]


def _screen_to_person_side(direction: str) -> str:
    # The view is a mirror, so screen-right is the player's own right.
    return "right" if direction == "right" else "left"


def _apply(seg: Segment, p: float, elapsed: float, x: float, z: float, t_abs: float) -> Pose:
    a, prm = seg.action, seg.params
    pose = Pose(x=x, z=z)
    sway = 3.0 * math.sin(2 * math.pi * 0.5 * t_abs)
    pose.arm_l = (ARM_DOWN[0] + sway, 0, ARM_DOWN[2] + sway, 0)
    pose.arm_r = (ARM_DOWN[0] - sway, 0, ARM_DOWN[2] - sway, 0)
    if a == "idle":
        pass
    elif a == "hidden":
        pose.hidden = True
    elif a == "tpose":
        e = _envelope(elapsed, seg.duration, 0.3)
        pose.arm_l = _lerp_arm(pose.arm_l, ARM_T, e)
        pose.arm_r = _lerp_arm(pose.arm_r, ARM_T, e)
    elif a == "raise":
        e = _envelope(elapsed, seg.duration, 0.25)
        side = prm.get("side", "both")
        if side in ("both", "left"):
            pose.arm_l = _lerp_arm(pose.arm_l, ARM_UP, e)
        if side in ("both", "right"):
            pose.arm_r = _lerp_arm(pose.arm_r, ARM_UP, e)
    elif a == "jump":
        air = 0.5
        h = prm.get("height", 0.32)
        if elapsed < air:
            q = elapsed / air
            pose.jump_h = 4 * h * q * (1 - q)
            lift = math.sin(math.pi * q)
            pose.arm_l = _lerp_arm(pose.arm_l, (45, 20, 50, 20), lift)
            pose.arm_r = _lerp_arm(pose.arm_r, (45, 20, 50, 20), lift)
    elif a == "squat":
        pose.squat = prm.get("depth", 1.0) * _envelope(elapsed, seg.duration, 0.35)
        reach = (60, 70, 60, 70)
        pose.arm_l = _lerp_arm(pose.arm_l, reach, pose.squat)
        pose.arm_r = _lerp_arm(pose.arm_r, reach, pose.squat)
    elif a == "punch":
        side = prm.get("side", "right")
        guard_in, extend_end, retract_end = 0.15, 0.27, 0.6
        if elapsed < guard_in:
            arm = _lerp_arm(ARM_DOWN, ARM_GUARD, _smooth(elapsed / guard_in))
        elif elapsed < extend_end:
            arm = _lerp_arm(ARM_GUARD, ARM_PUNCH, _smooth((elapsed - guard_in) / (extend_end - guard_in)))
        elif elapsed < extend_end + 0.08:
            arm = ARM_PUNCH
        elif elapsed < retract_end:
            arm = _lerp_arm(ARM_PUNCH, ARM_DOWN, _smooth((elapsed - extend_end - 0.08) / (retract_end - extend_end - 0.08)))
        else:
            arm = ARM_DOWN
        if side == "left":
            pose.arm_l = arm
        else:
            pose.arm_r = arm
    elif a == "swipe":
        # Screen direction; the hand on that side sweeps across the body.
        side = _screen_to_person_side(prm.get("direction", "right"))
        # Deliberate wind-up (0.6 s), fast stroke (0.22 s), relaxed return.
        wind, sweep_end = 0.6, 0.82
        if elapsed < wind:
            arm = _lerp_arm(ARM_DOWN, ARM_SWIPE_START, _smooth(elapsed / wind))
        elif elapsed < sweep_end:
            arm = _lerp_arm(ARM_SWIPE_START, ARM_SWIPE_END, _smooth((elapsed - wind) / (sweep_end - wind)))
        else:
            arm = _lerp_arm(ARM_SWIPE_END, ARM_DOWN, _smooth((elapsed - sweep_end) / max(seg.duration - sweep_end, 1e-3)))
        if side == "left":
            pose.arm_l = arm
        else:
            pose.arm_r = arm
    elif a == "lean":
        side = _screen_to_person_side(prm.get("direction", "left"))
        angle = prm.get("angle", 20.0) * _envelope(elapsed, seg.duration, 0.3)
        pose.lean_deg = angle if side == "left" else -angle
    elif a == "walk":
        q = _smooth(p)
        pose.x = x + (prm.get("to_x", x) - x) * q
        pose.z = z + (prm.get("to_z", z) - z) * q
        swing = 12 * math.sin(2 * math.pi * 1.6 * elapsed)
        pose.arm_l = (10 + swing, swing, 10 + swing, swing)
        pose.arm_r = (10 - swing, -swing, 10 - swing, -swing)
    else:
        raise ValueError(f"unknown action {a!r}")
    return pose


def _dir(side: float, abd: float, flex: float) -> np.ndarray:
    ab, fl = math.radians(abd), math.radians(flex)
    return np.array([side * math.sin(ab) * math.cos(fl), -math.cos(ab) * math.cos(fl), math.sin(fl)],
                    dtype=np.float32)


def body_points(pose: Pose, height: float) -> np.ndarray:
    """Local 3D joint positions (26, 3) for a pose."""
    # Work in the 1.70 m rest frame, scale to the actor's height at the end.
    s = height / 1.70
    pts = _REST.copy()
    # Legs (squat bends knees forward, hips drop).
    theta = math.radians(120.0 * pose.squat)
    half = theta / 2
    for side, hip_j, knee_j, ankle_j in ((1, J.LEFT_HIP, J.LEFT_KNEE, J.LEFT_ANKLE),
                                         (-1, J.RIGHT_HIP, J.RIGHT_KNEE, J.RIGHT_ANKLE)):
        ankle = _REST[ankle_j]
        knee = ankle + SHIN * np.array([side * 0.06 * pose.squat, math.cos(half), math.sin(half)])
        hip = knee + THIGH * np.array([0.0, math.cos(half), -math.sin(half)])
        pts[knee_j] = knee
        pts[hip_j] = np.array([_REST[hip_j][0], hip[1], hip[2]])
    pts[J.HIP] = (pts[J.LEFT_HIP] + pts[J.RIGHT_HIP]) / 2
    drop = pts[J.HIP][1] - _REST[J.HIP][1]
    pts[_UPPER_BODY] = _REST[_UPPER_BODY]
    pts[_UPPER_BODY, 1] += drop
    pts[_UPPER_BODY, 2] += pts[J.HIP][2]
    # Arms.
    for side, arm, sh, el, wr in ((1, pose.arm_l, J.LEFT_SHOULDER, J.LEFT_ELBOW, J.LEFT_WRIST),
                                  (-1, pose.arm_r, J.RIGHT_SHOULDER, J.RIGHT_ELBOW, J.RIGHT_WRIST)):
        pts[el] = pts[sh] + UPPER_ARM * _dir(side, arm[0], arm[1])
        pts[wr] = pts[el] + FOREARM * _dir(side, arm[2], arm[3])
    # Lean: rotate everything above the hips around the mid-hip (frontal plane).
    if pose.lean_deg:
        upper = _UPPER_BODY + [J.LEFT_ELBOW, J.RIGHT_ELBOW, J.LEFT_WRIST, J.RIGHT_WRIST]
        c, sn = math.cos(math.radians(pose.lean_deg)), math.sin(math.radians(pose.lean_deg))
        rel = pts[upper] - pts[J.HIP]
        # Positive angle tips the top towards +x (the person's left).
        x = rel[:, 0] * c + rel[:, 1] * sn
        y = -rel[:, 0] * sn + rel[:, 1] * c
        pts[upper, 0] = pts[J.HIP][0] + x
        pts[upper, 1] = pts[J.HIP][1] + y
    pts *= s
    pts[:, 1] += pose.jump_h
    return pts


class PinholeCamera:
    def __init__(self, width: int, height: int, hfov_deg: float):
        self.w, self.h = width, height
        self.f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.cx, self.cy = width / 2, height / 2

    def project(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """world (N,3) with X right (image), Y up, Z away from camera."""
        z = np.maximum(world[:, 2], 0.1)
        u = self.cx + self.f * world[:, 0] / z
        v = self.cy - self.f * (world[:, 1] - CAMERA_HEIGHT_M) / z
        return np.stack([u, v], axis=1).astype(np.float32), z


def scenario(name: str) -> list[Actor]:
    S = Segment
    if name == "demo":
        return [Actor(1, 0.0, 3.0, [
            S("idle", 1.5), S("tpose", 2.0), S("idle", 0.8),
            S("jump", 0.9), S("idle", 0.6), S("squat", 1.5), S("idle", 0.6),
            S("punch", 0.7, {"side": "left"}), S("punch", 0.7, {"side": "right"}), S("idle", 0.5),
            S("swipe", 1.5, {"direction": "right"}), S("swipe", 1.5, {"direction": "left"}),
            S("idle", 0.5), S("raise", 1.3), S("idle", 0.6),
            S("lean", 1.3, {"direction": "left"}), S("lean", 1.3, {"direction": "right"}),
            S("idle", 1.0),
        ])]
    if name == "fruit":
        return [Actor(1, 0.0, 2.9, [
            S("idle", 0.6), S("swipe", 1.4, {"direction": "right"}), S("swipe", 1.4, {"direction": "left"}),
            S("raise", 0.9), S("swipe", 1.4, {"direction": "left"}), S("walk", 0.8, {"to_x": 0.4}),
            S("swipe", 1.4, {"direction": "right"}), S("walk", 0.8, {"to_x": 0.0}),
        ])]
    if name == "duo":
        return [
            Actor(1, -0.7, 3.1, [S("idle", 1.0), S("jump", 0.9), S("swipe", 1.4, {"direction": "right"}),
                                 S("raise", 1.0), S("idle", 0.8)], shirt=(200, 90, 40)),
            Actor(2, 0.7, 3.1, [S("idle", 0.6), S("squat", 1.4), S("punch", 0.7, {"side": "left"}),
                                S("idle", 1.0), S("swipe", 1.4, {"direction": "left"})], shirt=(40, 60, 210)),
        ]
    if name == "cross":
        return [
            Actor(1, -0.9, 2.8, [S("idle", 0.5), S("walk", 2.5, {"to_x": 0.9}), S("idle", 0.5),
                                 S("walk", 2.5, {"to_x": -0.9})], shirt=(200, 90, 40)),
            Actor(2, 0.9, 3.4, [S("idle", 0.5), S("walk", 2.5, {"to_x": -0.9}), S("idle", 0.5),
                                S("walk", 2.5, {"to_x": 0.9})], shirt=(40, 60, 210), height=1.62),
        ]
    if name == "leave_return":
        return [Actor(1, 0.0, 3.0, [S("idle", 1.0), S("hidden", 1.2), S("idle", 1.5), S("hidden", 3.0),
                                    S("idle", 1.5)], shirt=(60, 160, 60))]
    if name == "quad":
        cols = [(200, 90, 40), (40, 60, 210), (60, 160, 60), (160, 60, 160)]
        return [Actor(i + 1, x, 3.4, [S("idle", 0.4 * i + 0.2), S("jump", 0.9), S("swipe", 1.4, {"direction": "right"}),
                                      S("idle", 1.0)], shirt=cols[i], height=1.55 + 0.06 * i)
                for i, x in enumerate((-1.2, -0.4, 0.4, 1.2))]
    raise ValueError(f"unknown synthetic scenario {name!r}")


SCENARIOS = ("demo", "fruit", "duo", "cross", "leave_return", "quad")


class SyntheticSource(FrameSource):
    def __init__(self, scenario_name: str, cfg: CameraConfig, hfov_deg: float = 70.0,
                 noise_seed: int = 0):
        self.cfg = cfg
        self.scenario_name = scenario_name
        self.name = f"synthetic:{scenario_name}"
        self.actors = scenario(scenario_name)
        self.width, self.height = cfg.width, cfg.height
        self.fps = float(cfg.fps)
        self.camera = PinholeCamera(self.width, self.height, hfov_deg)
        self._bg = self._background()
        self._seq = 0
        self._t0 = 0.0
        self.rng = np.random.default_rng(noise_seed)

    def open(self) -> None:
        self._t0 = 0.0    # the clock starts at the first read, like a camera

    def _background(self) -> np.ndarray:
        h, w = self.height, self.width
        bg = np.zeros((h, w, 3), np.uint8)
        horizon = int(h * 0.62)
        grad = np.linspace(170, 120, horizon, dtype=np.float32)[:, None]
        bg[:horizon] = np.dstack([grad + 10, grad + 5, grad]).astype(np.uint8)
        bg[horizon:] = (70, 85, 95)
        cv2.rectangle(bg, (int(w * 0.05), int(h * 0.25)), (int(w * 0.2), horizon), (90, 110, 130), -1)
        cv2.rectangle(bg, (int(w * 0.75), int(h * 0.15)), (int(w * 0.9), int(h * 0.4)), (150, 130, 100), -1)
        return bg

    def scene_at(self, t: float) -> list[dict[str, Any]]:
        """Ground truth for all visible actors at scene time t (seconds)."""
        people = []
        for actor in self.actors:
            pose = actor.pose_at(t)
            if pose.hidden:
                continue
            local = body_points(pose, actor.height)
            # The actor faces the camera, so their LEFT (+local x) is on the
            # camera image's RIGHT (+world X); "towards camera" is -Z.
            world = np.stack([pose.x + local[:, 0], local[:, 1], pose.z - local[:, 2]], axis=1)
            uv, z = self.camera.project(world)
            inside = ((uv[:, 0] >= 0) & (uv[:, 0] < self.width) &
                      (uv[:, 1] >= 0) & (uv[:, 1] < self.height)).astype(np.float32)
            if inside.sum() < 5:
                continue
            pad = self.camera.f * 0.12 / pose.z
            box = np.array([uv[:, 0].min() - pad, uv[:, 1].min() - pad,
                            uv[:, 0].max() + pad, uv[:, 1].max() + pad * 0.5], np.float32)
            people.append({"pid": actor.pid, "kpts": uv, "vis": inside, "z": z,
                           "box": box, "depth": pose.z, "actor": actor, "pose": pose})
        # Simple occlusion: a joint of a farther person inside a nearer person's
        # torso box becomes low-visibility.
        for a in people:
            for b in people:
                if b["depth"] < a["depth"] - 0.1:
                    x0, y0, x1, y1 = b["box"]
                    w, h = x1 - x0, y1 - y0
                    inside_b = ((a["kpts"][:, 0] > x0 + 0.2 * w) & (a["kpts"][:, 0] < x1 - 0.2 * w) &
                                (a["kpts"][:, 1] > y0 + 0.1 * h) & (a["kpts"][:, 1] < y1 - 0.1 * h))
                    a["vis"] = np.where(inside_b, np.minimum(a["vis"], 0.25), a["vis"])
        return people

    def render(self, people: list[dict[str, Any]]) -> np.ndarray:
        img = self._bg.copy()
        for person in sorted(people, key=lambda p: -p["depth"]):
            _draw_person(img, person, self.camera.f)
        return img

    def read(self) -> Frame | None:
        if not self._t0:
            self._t0 = time.perf_counter()
        self._seq += 1
        t_scene = self._seq / self.fps
        t = self._t0 + t_scene
        if self.cfg.realtime:
            delay = t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        people = self.scene_at(t_scene)
        image = self.render(people)
        gt = [{k: p[k] for k in ("pid", "kpts", "vis", "z", "box", "depth")} for p in people]
        return Frame(image=image, t_capture=t, seq=self._seq, meta={"gt": gt, "t_scene": t_scene})


def _draw_person(img: np.ndarray, person: dict[str, Any], f: float) -> None:
    k = person["kpts"]
    actor: Actor = person["actor"]
    z = person["depth"]
    thick = max(2, int(f * 0.09 / z))

    def P(j: int) -> tuple[int, int]:
        return int(k[j][0]), int(k[j][1])

    for a, b in ((J.LEFT_HIP, J.LEFT_KNEE), (J.LEFT_KNEE, J.LEFT_ANKLE),
                 (J.RIGHT_HIP, J.RIGHT_KNEE), (J.RIGHT_KNEE, J.RIGHT_ANKLE)):
        cv2.line(img, P(a), P(b), actor.pants, int(thick * 1.3), cv2.LINE_AA)
    for a, b in ((J.LEFT_ANKLE, J.LEFT_BIG_TOE), (J.RIGHT_ANKLE, J.RIGHT_BIG_TOE),
                 (J.LEFT_HEEL, J.LEFT_SMALL_TOE), (J.RIGHT_HEEL, J.RIGHT_SMALL_TOE)):
        cv2.line(img, P(a), P(b), (30, 30, 30), max(2, thick // 2), cv2.LINE_AA)
    torso = np.array([P(J.LEFT_SHOULDER), P(J.RIGHT_SHOULDER), P(J.RIGHT_HIP), P(J.LEFT_HIP)], np.int32)
    cv2.fillConvexPoly(img, torso, actor.shirt, cv2.LINE_AA)
    cv2.polylines(img, [torso], True, actor.shirt, thick, cv2.LINE_AA)
    for a, b in ((J.LEFT_SHOULDER, J.LEFT_ELBOW), (J.RIGHT_SHOULDER, J.RIGHT_ELBOW)):
        cv2.line(img, P(a), P(b), actor.shirt, thick, cv2.LINE_AA)
    for a, b in ((J.LEFT_ELBOW, J.LEFT_WRIST), (J.RIGHT_ELBOW, J.RIGHT_WRIST)):
        cv2.line(img, P(a), P(b), actor.skin, max(2, int(thick * 0.8)), cv2.LINE_AA)
    cv2.line(img, P(J.NECK), P(J.NOSE), actor.skin, thick, cv2.LINE_AA)
    head_c = (k[J.HEAD] + k[J.NECK]) / 2
    radius = max(3, int(np.linalg.norm(k[J.HEAD] - k[J.NECK]) * 0.55))
    cv2.circle(img, (int(head_c[0]), int(head_c[1])), radius, actor.skin, -1, cv2.LINE_AA)
    for j in (J.LEFT_EYE, J.RIGHT_EYE):
        cv2.circle(img, P(j), max(1, thick // 4), (40, 40, 40), -1, cv2.LINE_AA)
