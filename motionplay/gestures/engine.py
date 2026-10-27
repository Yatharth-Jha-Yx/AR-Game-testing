"""Hand-coded gesture detectors -> GestureEvent stream.

All maths happens in *body units*: joints divided by the player's torso
length (neck -> mid-hip), x towards screen-right, y up. A child and an
adult doing the same move produce the same numbers.

Each detector uses hysteresis (a higher threshold to start than to stop) and
a refractory period (cool-down) so one jump never fires twice.

Detectors (thresholds from the design plan, tuned on the synthetic actors):
  jump      both ankles rise > 0.25 torso within ~150 ms, hips rising too
  squat     vertical hip->knee distance < 0.8x standing baseline AND hips
            dropped > 0.3 torso (start/end). The plan said 0.6x, but with the
            camera near hip height the knees come towards the lens in a squat
            and project lower, so the 2D gap only shrinks to ~0.72x for a 120
            degree knee bend (measured on the synthetic body); hip drop is the
            stronger cue.
  punch     fast wrist move *away from the shoulder* with the elbow
            straightening (per arm)
  swipe     fast horizontal wrist stroke (>0.6 torso of travel) with the elbow
            trailing, not mirrored by the other arm (that's arms opening),
            not right after a punch (per hand, reports screen direction)
  arm_raise wrist above the head (left / right / both, start/end)
  lean      shoulders shifted sideways over the hips (start/end)

Games subscribe to these events and never see raw joints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..skeleton import J
from ..types import GestureEvent
from ..body.metrics import angle


@dataclass
class GestureThresholds:
    jump_rise: float = 0.25
    jump_hip_rise: float = 0.15
    jump_window_s: float = 0.15
    jump_land: float = 0.10
    jump_refractory_s: float = 0.25
    squat_enter: float = 0.80
    squat_exit: float = 0.90
    squat_drop: float = 0.30          # torso lengths the hips must drop
    squat_drop_exit: float = 0.15
    punch_speed: float = 2.5          # torso lengths / s, wrist relative to shoulder
    punch_radial: float = 1.8         # torso lengths / s away from the shoulder
    punch_elbow_deg: float = 120.0
    punch_refractory_s: float = 0.35
    swipe_speed: float = 4.0          # torso lengths / s horizontally
    swipe_travel: float = 0.6         # torso lengths covered in the last 0.2 s
    swipe_rearm: float = 1.0
    swipe_refractory_s: float = 0.20
    swipe_after_punch_s: float = 0.45
    raise_margin: float = 0.10        # torso lengths above head top
    raise_hold_s: float = 0.20
    lean_enter: float = 0.25
    lean_exit: float = 0.15
    arm_conf: float = 0.55            # elbow/wrist confidence needed for arm gestures


class GestureEngine:
    """One per player. Feed it smoothed (not predicted) joints every frame."""

    def __init__(self, player_id: int, mirror: bool = True,
                 thresholds: GestureThresholds | None = None):
        self.id = player_id
        self.sx = -1.0 if mirror else 1.0
        self.th = thresholds or GestureThresholds()
        self.hist: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=90)
        self.torso: float | None = None
        # jump
        self.floor: float | None = None
        self.hip_base: float | None = None
        self.airborne = False
        self.t_land = -1e9
        # squat
        self.hk_base: float | None = None
        self.standing_hip: float | None = None
        self.squatting = False
        self.t_squat_end = -1e9
        # punch / swipe per side (0 = left, 1 = right)
        self.t_punch = [-1e9, -1e9]
        self.t_swipe = [-1e9, -1e9]
        self.swipe_armed = [True, True]
        # arm raise / lean
        self.raise_state = "none"
        self.raise_pending: tuple[str, float] | None = None
        self.lean_state: str | None = None

    def set_calibration(self, hip_knee_torso: float | None) -> None:
        if hip_knee_torso:
            self.hk_base = hip_knee_torso

    def update(self, kpts: np.ndarray, conf: np.ndarray, t: float) -> list[GestureEvent]:
        ok = conf >= 0.4
        if not (ok[J.NECK] and ok[J.HIP]):
            return []
        torso = float(np.linalg.norm(kpts[J.NECK] - kpts[J.HIP]))
        if torso < 1:
            return []
        self.torso = torso if self.torso is None else 0.95 * self.torso + 0.05 * torso
        p = np.empty_like(kpts, dtype=np.float64)
        p[:, 0] = self.sx * kpts[:, 0] / self.torso
        p[:, 1] = -kpts[:, 1] / self.torso
        self.hist.append((t, p, ok))
        events: list[GestureEvent] = []
        events += self._jump(p, ok, t)
        events += self._squat(p, ok, t)
        arm_ok = conf >= self.th.arm_conf
        for side in (0, 1):
            events += self._punch(side, p, arm_ok, t, kpts)
            events += self._swipe(side, p, arm_ok, t)
        events += self._arm_raise(p, ok, t)
        events += self._lean(p, ok, t)
        return events

    # -- helpers ----------------------------------------------------------------
    def _past(self, seconds: float) -> tuple[float, np.ndarray, np.ndarray] | None:
        """The newest history sample at least `seconds` old."""
        t_now = self.hist[-1][0]
        for item in reversed(self.hist):
            if t_now - item[0] >= seconds:
                return item
        return None

    def _velocity(self, joint: int, window: float = 0.05) -> np.ndarray | None:
        old = self._past(window)
        if old is None:
            return None
        t1, p1, ok1 = self.hist[-1]
        t0, p0, ok0 = old
        if not (ok1[joint] and ok0[joint]) or t1 <= t0:
            return None
        return (p1[joint] - p0[joint]) / (t1 - t0)

    def _ev(self, gesture: str, phase: str, strength: float, t: float, side: str | None = None,
            direction: str | None = None) -> GestureEvent:
        return GestureEvent(self.id, gesture, phase, float(np.clip(strength, 0, 1)), t, side, direction)

    # -- detectors --------------------------------------------------------------
    def _jump(self, p, ok, t) -> list[GestureEvent]:
        if not (ok[J.LEFT_ANKLE] and ok[J.RIGHT_ANKLE] and ok[J.HIP]):
            return []
        ankles = np.array([p[J.LEFT_ANKLE][1], p[J.RIGHT_ANKLE][1]])
        hip = p[J.HIP][1]
        if self.floor is None:
            self.floor, self.hip_base = float(ankles.mean()), hip
        rise = ankles - self.floor
        th = self.th
        out = []
        if not self.airborne:
            hip_rise = hip - self.hip_base
            # The whole body lifts as one piece: standing up from a squat lifts
            # the hips but not the ankles, so it must not count.
            rigid = abs(float(rise.mean()) - hip_rise) < 0.5 * float(rise.mean())
            if (np.all(rise >= th.jump_rise) and hip_rise >= th.jump_hip_rise and rigid
                    and not self.squatting and t - self.t_squat_end >= 0.4
                    and t - self.t_land >= th.jump_refractory_s and self._rose_quickly(t)):
                self.airborne = True
                v = self._velocity(J.LEFT_ANKLE)
                speed = float(v[1]) if v is not None else 3.0
                out.append(self._ev("jump", "trigger", speed / 6.0, t))
            else:
                # Feet on the floor: floor follows quickly down, slowly up
                # (so walking towards / away from the camera re-bases it).
                m = float(ankles.mean())
                a = 0.3 if m < self.floor else 0.02
                self.floor += a * (m - self.floor)
                self.hip_base += 0.05 * (hip - self.hip_base)
        elif np.all(rise < th.jump_land):
            self.airborne = False
            self.t_land = t
        return out

    def _rose_quickly(self, t: float) -> bool:
        """Were both ankles still near the floor within the jump window?"""
        window = self.th.jump_window_s + 0.034          # + ~1 frame of slack at 30 fps
        for ts, p, ok in reversed(self.hist):
            if t - ts > window:
                return False
            if ok[J.LEFT_ANKLE] and ok[J.RIGHT_ANKLE]:
                r = max(p[J.LEFT_ANKLE][1], p[J.RIGHT_ANKLE][1]) - self.floor
                if r < 0.08:
                    return True
        return False

    def _squat(self, p, ok, t) -> list[GestureEvent]:
        pairs = [(J.LEFT_HIP, J.LEFT_KNEE), (J.RIGHT_HIP, J.RIGHT_KNEE)]
        vals = [p[h][1] - p[k][1] for h, k in pairs if ok[h] and ok[k]]
        if not vals or self.airborne or not ok[J.HIP]:
            return []
        hk = float(np.mean(vals))
        hip = float(p[J.HIP][1])
        if self.hk_base is None:
            self.hk_base = hk
        if self.standing_hip is None:
            self.standing_hip = hip
        ratio = hk / max(self.hk_base, 1e-3)
        drop = self.standing_hip - hip
        th = self.th
        out = []
        if not self.squatting and ratio < th.squat_enter and drop >= th.squat_drop:
            self.squatting = True
            out.append(self._ev("squat", "start", drop / 0.7, t))
        elif self.squatting and (ratio > th.squat_exit or drop < th.squat_drop_exit):
            self.squatting = False
            self.t_squat_end = t
            out.append(self._ev("squat", "end", 0.0, t))
        elif not self.squatting and ratio > 0.9:
            # Standing: refine baselines (follows walking towards/away slowly).
            self.hk_base += 0.02 * (hk - self.hk_base)
            self.standing_hip += 0.05 * (hip - self.standing_hip)
        return out

    def _punch(self, side, p, ok, t, kpts_px) -> list[GestureEvent]:
        sh, el, wr = ((J.LEFT_SHOULDER, J.LEFT_ELBOW, J.LEFT_WRIST) if side == 0
                      else (J.RIGHT_SHOULDER, J.RIGHT_ELBOW, J.RIGHT_WRIST))
        if not (ok[sh] and ok[el] and ok[wr]) or t - self.t_punch[side] < self.th.punch_refractory_s:
            return []
        old = self._past(0.05)
        if old is None:
            return []
        t0, p0, ok0 = old
        if not (ok0[sh] and ok0[wr] and ok0[el]):
            return []
        dt = self.hist[-1][0] - t0
        rel_now, rel_old = p[wr] - p[sh], p0[wr] - p0[sh]
        v_rel = (rel_now - rel_old) / dt
        speed = float(np.linalg.norm(v_rel))
        radial = (np.linalg.norm(rel_now) - np.linalg.norm(rel_old)) / dt
        elbow_now = angle(p, sh, el, wr)
        elbow_old = angle(p0, sh, el, wr)
        th = self.th
        if (speed >= th.punch_speed and radial >= th.punch_radial and radial >= 0.6 * speed
                and elbow_now >= th.punch_elbow_deg and elbow_now > elbow_old + 5
                and p[wr][1] > p[J.HIP][1]):
            self.t_punch[side] = t
            return [self._ev("punch", "trigger", speed / 8.0, t, side=("left", "right")[side])]
        return []

    def _swipe(self, side, p, ok, t) -> list[GestureEvent]:
        el, wr = (J.LEFT_ELBOW, J.LEFT_WRIST) if side == 0 else (J.RIGHT_ELBOW, J.RIGHT_WRIST)
        other = J.RIGHT_WRIST if side == 0 else J.LEFT_WRIST
        vw = self._velocity(wr)
        if vw is None:
            return []
        vx, vy = float(vw[0]), float(vw[1])
        th = self.th
        if abs(vx) < th.swipe_rearm:
            self.swipe_armed[side] = True
        if (not self.swipe_armed[side] or t - self.t_swipe[side] < th.swipe_refractory_s
                or t - self.t_punch[side] < th.swipe_after_punch_s):
            return []
        if abs(vx) < th.swipe_speed or abs(vx) < 2 * abs(vy):
            return []
        ve = self._velocity(el)
        if ve is not None and abs(float(ve[0])) >= abs(vx):
            return []            # whole arm/body moving, elbow not trailing
        vo = self._velocity(other)
        if vo is not None and float(vo[0]) * vx < 0 and abs(float(vo[0])) > 0.5 * abs(vx):
            return []            # both arms opening/closing symmetrically
        old = self._past(0.2) or self.hist[0]
        if not old[2][wr] or abs(p[wr][0] - old[1][wr][0]) < th.swipe_travel:
            return []
        self.swipe_armed[side] = False
        self.t_swipe[side] = t
        return [self._ev("swipe", "trigger", abs(vx) / 10.0, t, side=("left", "right")[side],
                         direction="right" if vx > 0 else "left")]

    def _arm_raise(self, p, ok, t) -> list[GestureEvent]:
        top = p[J.HEAD][1] if ok[J.HEAD] else (p[J.NOSE][1] + 0.35 if ok[J.NOSE] else None)
        if top is None:
            return []
        m = self.th.raise_margin
        up = []
        for wr, name in ((J.LEFT_WRIST, "left"), (J.RIGHT_WRIST, "right")):
            if ok[wr]:
                was_up = self.raise_state in (name, "both")
                # Hysteresis: needs to clear the head to go up, drop below it to go down.
                if p[wr][1] > top + m or (was_up and p[wr][1] > top - 0.05):
                    up.append(name)
        state = "both" if len(up) == 2 else (up[0] if up else "none")
        if state == self.raise_state:
            self.raise_pending = None
            return []
        if self.raise_pending is None or self.raise_pending[0] != state:
            self.raise_pending = (state, t)
            return []
        if t - self.raise_pending[1] < self.th.raise_hold_s:
            return []
        out = []
        if self.raise_state != "none":
            out.append(self._ev("arm_raise", "end", 0.0, t, side=self.raise_state))
        if state != "none":
            out.append(self._ev("arm_raise", "start", 1.0, t, side=state))
        self.raise_state = state
        self.raise_pending = None
        return out

    def _lean(self, p, ok, t) -> list[GestureEvent]:
        need = [J.LEFT_SHOULDER, J.RIGHT_SHOULDER, J.LEFT_HIP, J.RIGHT_HIP]
        if not all(ok[j] for j in need):
            return []
        sh = (p[J.LEFT_SHOULDER][0] + p[J.RIGHT_SHOULDER][0]) / 2
        hp = (p[J.LEFT_HIP][0] + p[J.RIGHT_HIP][0]) / 2
        ratio = float(sh - hp)
        out = []
        if self.lean_state is None and abs(ratio) >= self.th.lean_enter:
            self.lean_state = "right" if ratio > 0 else "left"
            out.append(self._ev("lean", "start", abs(ratio) / 0.6, t, direction=self.lean_state))
        elif self.lean_state is not None and abs(ratio) < self.th.lean_exit:
            out.append(self._ev("lean", "end", 0.0, t, direction=self.lean_state))
            self.lean_state = None
        return out
