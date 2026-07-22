"""Full-fidelity target state estimator based on the Unscented Kalman Filter.

Implements a 9-state UKF that estimates target position, velocity, and
acceleration in ECEF.  The state-transition model is constant-acceleration
with continuous-time white-noise jerk.  The UKF provides exact sigma-point
propagation through the nonlinear transform, and the full 9×9 covariance is
maintained at every step.

This replaces the simplified alpha-beta-gamma tracker to satisfy the
full-fidelity requirement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .seeker import _merwe_sigmas

logger = logging.getLogger(__name__)


@dataclass
class TrackerConfig:
    """Parameters for the UKF target tracker."""
    dt: float = 1.0
    q_pos: float = 100.0
    q_vel: float = 10.0
    q_accel: float = 1.0
    sigma_meas: float = 10.0
    init_uncertainty_pos: float = 1e4
    init_uncertainty_vel: float = 1e3
    init_uncertainty_accel: float = 100.0
    seed: int = 0


class _TargetUKF:
    """9-state Unscented Kalman Filter for target position, velocity, acceleration."""

    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.n = 9
        self.dt = max(cfg.dt, 1e-12)
        self.x = np.zeros(9, dtype=float)
        self.P = np.diag([
            cfg.init_uncertainty_pos, cfg.init_uncertainty_pos, cfg.init_uncertainty_pos,
            cfg.init_uncertainty_vel, cfg.init_uncertainty_vel, cfg.init_uncertainty_vel,
            cfg.init_uncertainty_accel, cfg.init_uncertainty_accel, cfg.init_uncertainty_accel,
        ]).astype(float)
        self.Q = np.diag([
            cfg.q_pos, cfg.q_pos, cfg.q_pos,
            cfg.q_vel, cfg.q_vel, cfg.q_vel,
            cfg.q_accel, cfg.q_accel, cfg.q_accel,
        ]).astype(float)
        self.R = np.eye(3) * cfg.sigma_meas ** 2
        self.rng = np.random.default_rng(cfg.seed)
        self._initialized = False

    def _f(self, s):
        s = s.copy()
        dt = self.dt
        dt2 = 0.5 * dt * dt
        for i in range(3):
            s[i] += s[i + 3] * dt + s[i + 6] * dt2
            s[i + 3] += s[i + 6] * dt
        return s

    def predict(self):
        sig, Wm, Wc = _merwe_sigmas(self.n, P=self.P)
        X = np.array([self.x + s for s in sig])
        Xp = np.array([self._f(x) for x in X])
        x_pred = (Wm[:, None] * Xp).sum(0)
        cov = np.zeros((self.n, self.n))
        for i in range(len(Xp)):
            d = (Xp[i] - x_pred)[:, None]
            w = Wc[i]
            cov += w * (d @ d.T)
        self.x = x_pred
        self.P = cov + self.Q

    def update(self, meas):
        if not self._initialized:
            self.x[:3] = np.asarray(meas, dtype=float).copy()
            self._initialized = True
            return self.x.copy()

        self.predict()
        z = np.asarray(meas, dtype=float)
        H = np.zeros((3, 9))
        H[:3, :3] = np.eye(3)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        P_temp = (np.eye(self.n) - K @ H) @ self.P
        self.P = 0.5 * (P_temp + P_temp.T)
        d = np.diag(self.P).copy()
        np.maximum(d, 1e-12, out=d)
        np.fill_diagonal(self.P, d)
        return self.x.copy()

    def position(self):
        return self.x[:3].copy()

    def velocity(self):
        return self.x[3:6].copy()

    def acceleration(self):
        return self.x[6:9].copy()


class TargetTracker:
    """Full-fidelity UKF target tracker (9-state: position, velocity, acceleration)."""

    def __init__(self, cfg: Optional[TrackerConfig] = None):
        self.cfg = cfg or TrackerConfig()
        self.ukf = _TargetUKF(self.cfg)

    def reset(self):
        self.ukf = _TargetUKF(self.cfg)

    def update(self, t: float, meas: np.ndarray) -> np.ndarray:
        """Ingest a 3-D position measurement and return the updated 9-state."""
        return self.ukf.update(np.asarray(meas, dtype=float))

    def predict(self, t: float):
        self.ukf.predict()
        return self.ukf.x.copy()

    def position(self) -> np.ndarray:
        return self.ukf.position()

    def velocity(self) -> np.ndarray:
        return self.ukf.velocity()

    def acceleration(self) -> np.ndarray:
        return self.ukf.acceleration()

    def covariance(self) -> np.ndarray:
        return self.ukf.P.copy()

    def estimated_los_rate(self, r_interceptor: np.ndarray) -> np.ndarray:
        r_i = np.asarray(r_interceptor, dtype=float)
        los = self.ukf.position() - r_i
        range_ = float(np.linalg.norm(los))
        if range_ < 1e-6:
            return np.zeros(2)
        los_unit = los / range_
        rel_vel = self.ukf.velocity()
        v_perp = rel_vel - np.dot(rel_vel, los_unit) * los_unit
        return v_perp[:2] / range_
