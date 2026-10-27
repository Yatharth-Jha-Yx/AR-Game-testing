"""Smoothing / prediction filters.

* OneEuroFilter: Casiez, Roussel & Vogel, "1 Euro Filter: A Simple
  Speed-based Low-pass Filter for Noisy Input in Interactive Systems",
  CHI 2012. Slow movement -> low cutoff (kills jitter); fast movement ->
  cutoff rises with speed (kills lag). Vectorised: one call filters all
  26 joints x 2 coordinates.
* BoxKalman: constant-velocity Kalman filter on (cx, cy, w, h), noise
  scaled by box height like SORT/ByteTrack's filter.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, shape: tuple[int, ...], min_cutoff: np.ndarray | float = 1.0,
                 beta: np.ndarray | float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = np.broadcast_to(np.asarray(min_cutoff, np.float64), shape).copy()
        self.beta = np.broadcast_to(np.asarray(beta, np.float64), shape).copy()
        self.d_cutoff = d_cutoff
        self.shape = shape
        self.x: np.ndarray | None = None
        self.dx = np.zeros(shape, np.float64)
        self.t: float | None = None

    def reset(self) -> None:
        self.x = None
        self.dx[:] = 0.0
        self.t = None

    def __call__(self, x: np.ndarray, t: float, mask: np.ndarray | None = None) -> np.ndarray:
        """Filter ``x`` taken at time ``t`` (s). Elements where ``mask`` is
        False keep their previous state (used for low-confidence joints)."""
        x = np.asarray(x, np.float64)
        if self.x is None or self.t is None:
            self.x = x.copy()
            self.t = t
            return self.x.copy()
        dt = t - self.t
        if dt <= 1e-6:
            dt = 1e-3
        dx = (x - self.x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x
        if mask is not None:
            m = np.broadcast_to(mask, self.shape)
            x_hat = np.where(m, x_hat, self.x)
            dx_hat = np.where(m, dx_hat, self.dx * 0.5)
        self.x, self.dx, self.t = x_hat, dx_hat, t
        return x_hat.copy()


class BoxKalman:
    """State [cx, cy, w, h, vcx, vcy, vw, vh]."""

    def __init__(self, box: np.ndarray, t: float):
        x0, y0, x1, y1 = box[:4]
        self.x = np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0, 0, 0, 0], np.float64)
        h = max(self.x[3], 1.0)
        self.P = np.diag([(h / 10) ** 2] * 4 + [(h / 2) ** 2] * 4)
        self.t = t
        self.H = np.hstack([np.eye(4), np.zeros((4, 4))])

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        h = max(self.x[3], 1.0)
        pos = (h / 20) ** 2 * dt * 30
        vel = (h / 6) ** 2 * dt * 30
        return np.diag([pos] * 4 + [vel] * 4)

    def predict_box(self, t: float) -> np.ndarray:
        """Where the box will be at time t (does not change the state)."""
        x = self._F(max(t - self.t, 0.0)) @ self.x
        cx, cy, w, h = x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)

    def update(self, box: np.ndarray, t: float) -> None:
        dt = max(t - self.t, 1e-3)
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        x0, y0, x1, y1 = box[:4]
        z = np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
        h = max(z[3], 1.0)
        R = np.diag([(h / 20) ** 2] * 4)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.t = t
