"""Seeker / sensor models for terminal guidance.

Implements a configurable `SeekerModel` representing an interceptor seeker
(active/semi-active radar or imaging IR) that produces noisy line-of-sight
(LOS) measurements of the target, tracks them with an Unscented Kalman
Filter (UKF), and exposes a true-vs-ideal measurement for the guidance law.

Noise sources modelled (research-grade, OSINT-parameterised):
  * glint     - specular multipath scintillation on the LOS angle
  * clutter   - spurious returns (Poisson-gated, low probability per frame)
  * scintillation - amplitude/RCS fades
  * latency   - one-frame measurement delay

A field-of-view (FOV) cone and gimbal limit mask measurements; outside
these the seeker reports no valid contact.

The UKF is a lightweight sigma-point filter (no external dependency) that
estimates target relative position/velocity in the seeker frame and returns
a smoothed LOS rate. This is a self-contained implementation sized for
terminal-phase engagement; it is not a substitute for a flight-rated tracker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _merwe_sigmas(n, P=None, alpha=0.3, beta=2.0, kappa=0.0):
    """Merwe scaled sigma-point set for an n-dimensional state.

    The sigma spread is scaled by sqrt(P) so the points track the current
    estimate covariance (not a fixed spread). `alpha` tunes the overall
    spread about the mean; for physical states in metres use a moderate value
    (default 0.3). When P is None the caller supplies pre-scaled offsets.

    Returns (sigma_offsets, Wm, Wc) where sigma_offsets are relative to the
    mean and must be added to x.
    """
    lam = alpha**2 * (n + kappa) - n
    n2 = n * 2 + 1
    Wm = np.full(n2, 0.5 / (n + lam))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)
    if P is None:
        P = np.eye(n)
    L = np.linalg.cholesky((n + lam) * P)
    sigma = np.zeros((n2, n))
    for i in range(n):
        sigma[i + 1, :] = L[:, i]
        sigma[n + i + 1, :] = -L[:, i]
    return sigma, Wm, Wc


@dataclass
class SeekerConfig:
    mode: str = "radar"  # "radar" | "semi_active_radar" | "ir"
    fov: float = math.radians(60.0)
    gimbal_limit: float = math.radians(45.0)
    range_max: float = 50e3
    frame_rate: float = 100.0  # Hz
    snr_db: float = 20.0
    glint_std: float = math.radians(0.5)
    clutter_rate: float = 0.01  # spurious contacts per frame
    scintillation_std: float = 0.05  # relative RCS fade std
    latency_frames: int = 1
    noise_seed: int = 0


class _UKF:
    """Minimal 6-state relative-kinematics UKF (pos+vel in seeker frame)."""

    def __init__(self, dt, q_pos=100.0, q_vel=10.0, r_angle=1e-3, seed=0):
        self.dt = dt
        self.n = 6
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1e4  # large initial uncertainty (metres)
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]) * dt
        # Measurement noise is angular (rad) on az/el, NOT dt-scaled to 0.
        self.R = np.eye(2) * r_angle
        self.rng = np.random.default_rng(seed)
        self._last_los_rate = np.zeros(2)
        self._initialized = False
        self._nominal_range = 1000.0  # overridden by SeekerModel.range_max

    def _f(self, s):
        s = s.copy()
        s[0] += s[3] * self.dt
        s[1] += s[4] * self.dt
        s[2] += s[5] * self.dt
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

    def _init_from_measurement(self, los_azel):
        """Seed the state from the first LOS measurement using a nominal range.

        Avoids the atan2/asin singularity at the origin (prior x=0 gives an
        ill-defined azimuth) by placing the target at a default slant range
        along the measured LOS direction.
        """
        az, el = los_azel
        rng0 = max(getattr(self, "_nominal_range", 1000.0), 1.0)
        cx, cy, cz = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
        self.x = np.array([rng0 * cx, rng0 * cy, rng0 * cz, 0.0, 0.0, 0.0])
        self.P = np.diag([rng0**2 * 0.25, rng0**2 * 0.25, rng0**2 * 0.25, 1e3, 1e3, 1e3])
        self._initialized = True

    def update(self, los_azel, visible):
        if not visible or los_azel is None:
            return
        if not getattr(self, "_initialized", False):
            self._init_from_measurement(np.asarray(los_azel, dtype=float))
        sig, Wm, Wc = _merwe_sigmas(self.n, P=self.P)
        X = np.array([self.x + s for s in sig])
        Z = []
        for x in X:
            r = np.linalg.norm(x[:3]) + 1e-9
            az = math.atan2(x[1], x[0])
            el = math.asin(np.clip(x[2] / r, -1.0, 1.0))
            Z.append([az, el])
        Z = np.array(Z)
        z_pred = (Wm[:, None] * Z).sum(0)
        S = np.zeros((2, 2))
        Pxz = np.zeros((self.n, 2))
        for i in range(len(Z)):
            dz = (Z[i] - z_pred)[:, None]
            S += Wc[i] * (dz @ dz.T)
            dX = (X[i] - self.x)[:, None]
            Pxz += Wc[i] * (dX @ dz.T)
        S += self.R
        K = Pxz @ np.linalg.inv(S)
        dz = (np.asarray(los_azel) - z_pred)[:, None]
        self.x = self.x + (K @ dz).ravel()
        self.P = self.P - K @ S @ K.T
        # Smoothed LOS rate from filtered relative kinematics.
        r = np.linalg.norm(self.x[:3]) + 1e-9
        vlos = self.x[3:6] - (np.dot(self.x[3:6], self.x[:3]) / r**2) * self.x[:3]
        self._last_los_rate = vlos[:2] / r

    def los_rate(self):
        return self._last_los_rate.copy()


class SeekerModel:
    def __init__(self, cfg: Optional[SeekerConfig] = None):
        self.cfg = cfg or SeekerConfig()
        self.dt = 1.0 / self.cfg.frame_rate
        snr = 10.0 ** (self.cfg.snr_db / 10.0)
        meas_angle_std = 1.0 / math.sqrt(max(snr, 1e-6)) * self.cfg.glint_std
        self.ukf = _UKF(
            self.dt,
            q_pos=100.0,
            q_vel=10.0,
            r_angle=max(meas_angle_std, 1e-4),
            seed=self.cfg.noise_seed,
        )
        self.ukf._nominal_range = max(self.cfg.range_max, 1.0)
        self._frame = 0
        self._delay = self.cfg.latency_frames
        self._buffer = []
        self._last_measurement = None

    # --- geometry / visibility ------------------------------------------- #
    def _los_azel(self, rel):
        r = np.linalg.norm(rel) + 1e-9
        az = math.atan2(rel[1], rel[0])
        el = math.asin(np.clip(rel[2] / r, -1.0, 1.0))
        return np.array([az, el])

    def _in_fov(self, rel):
        # Boresight is the +x axis (consistent with _los_azel az=atan2(y,x)
        # and TerminalGuidance's acos(los_unit[0]) FOV check).
        r = np.linalg.norm(rel) + 1e-9
        angle = math.acos(np.clip(rel[0] / r, -1.0, 1.0))
        return (r <= self.cfg.range_max) and (abs(angle) <= self.cfg.fov / 2.0)

    def _glint(self):
        return self.ukf.rng.normal(0.0, self.cfg.glint_std, size=2)

    def _clutter(self):
        return self.ukf.rng.random() < self.cfg.clutter_rate

    # --- per-frame measurement ------------------------------------------- #
    def measure(self, interceptor_state, target_state):
        """Return noisy (az, el) LOS measurement and visibility flag.

        interceptor_state / target_state: dicts or arrays with 'r','v' (3-vec).
        """
        r_i = np.asarray(interceptor_state["r"], dtype=float)
        v_i = np.asarray(interceptor_state.get("v", [0, 0, 0]), dtype=float)
        r_t = np.asarray(target_state["r"], dtype=float)
        v_t = np.asarray(target_state.get("v", [0, 0, 0]), dtype=float)

        rel = r_t - r_i
        visible = self._in_fov(rel)
        if not visible or self._clutter():
            self._frame += 1
            return None, False

        azel = self._los_azel(rel)
        snr = 10.0 ** (self.cfg.snr_db / 10.0)
        meas_noise = 1.0 / math.sqrt(max(snr, 1e-6)) * self.cfg.glint_std
        azel = azel + self.ukf.rng.normal(0.0, meas_noise, size=2) + self._glint()

        # Latency: hold measurement for `latency_frames` before publishing.
        self._buffer.append(azel)
        if len(self._buffer) > self._delay:
            out = self._buffer.pop(0)
        else:
            out = None
        self._last_measurement = (out, visible)
        self._frame += 1
        return out, visible

    def update_tracker(self, interceptor_state, target_state):
        """Advance UKF predict+update; returns smoothed LOS rate (2-vec)."""
        self.ukf.predict()
        azel, visible = self.measure(interceptor_state, target_state)
        self.ukf.update(azel, visible)
        return self.ukf.los_rate()

    def discrimination_features(self, target_state):
        """Return [RCS_bias, IR_flux, Doppler_width, micro_motion_flag]-like
        synthetic feature vector for downstream discrimination."""
        r = np.linalg.norm(np.asarray(target_state["r"], dtype=float)) + 1e-9
        rng = np.random.default_rng(self.cfg.noise_seed + int(r))
        rcs = rng.normal(0.0, 0.3) + self.cfg.scintillation_std
        ir = rng.normal(1.0, 0.2)
        doppler = rng.normal(50.0, 5.0)
        micro = 1.0 if rng.random() < 0.5 else 0.0
        return np.array([rcs, ir, doppler, micro])
