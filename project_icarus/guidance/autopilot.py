"""Full-fidelity guidance autopilot with actuator and body-rate dynamics.

Implements a second-order autopilot loop with body-rate feedback and
gimbal/divert-nozzle angle states.  The model replaces the simplified
first-order lag with a proper second-order+damped system whose states are
integrated by the simulation runner alongside the 6-DOF EOM.

States
------
a_real     (3) realized acceleration in inertial frame [m/s^2]
a_real_dot (3) first derivative of a_real [m/s^3]
gimbal_az  (1) gimbal azimuth [rad]
gimbal_el  (1) gimbal elevation [rad]

Inputs
------
a_cmd      (3) guidance acceleration command in inertial frame [m/s^2]
omega_body (3) body angular rates [rad/s] from the 6-DOF EOM
dt         (1) integration time step [s]

Outputs
-------
a_real     (3) realized acceleration in inertial frame [m/s^2]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AutopilotConfig:
    """Parameters for the second-order autopilot with body-rate feedback."""
    omega_n: float = 100.0
    damping: float = 0.7
    accel_rate_limit: float = 2000.0
    accel_limit: float = 150.0
    gimbal_limit_deg: float = 30.0
    gimbal_rate_limit_deg_s: float = 200.0
    use_gimbal_limit: bool = True
    rate_gain: float = 0.5
    cross_coupling_gain: float = 0.1


class Autopilot:
    """Second-order autopilot with body-rate feedback and gimbal constraints.

    The autopilot dynamics are
        a_real_ddot + 2*zeta*omega_n*a_real_dot + omega_n^2*a_real = omega_n^2*a_cmd - rate_gain*omega_body

    body-rate feedback (`rate_gain`) damps the autopilot resonance and
    improves robustness to structural flex.  Cross-coupling (`cross_coupling_gain`)
    couples the gimbal azimuth command into the elevation channel and vice-versa,
    representing a simplified gimbal-coupling matrix.

    Gimbal constraints are enforced in the body frame after transforming the
    desired inertial acceleration through the current DCM.
    """

    def __init__(self, cfg: Optional[AutopilotConfig] = None):
        self.cfg = cfg or AutopilotConfig()
        self.omega_n = max(self.cfg.omega_n, 1e-6)
        self.damping = max(min(self.cfg.damping, 0.999), 0.0)
        self.accel_rate_limit = max(self.cfg.accel_rate_limit, 1e-6)
        self.accel_limit = max(self.cfg.accel_limit, 1e-9)
        self.gimbal_limit = np.radians(self.cfg.gimbal_limit_deg)
        self.gimbal_rate_limit = np.radians(self.cfg.gimbal_rate_limit_deg_s)
        self.use_gimbal_limit = self.cfg.use_gimbal_limit
        self.rate_gain = self.cfg.rate_gain
        self.cross_coupling_gain = self.cfg.cross_coupling_gain
        self._a_real = np.zeros(3, dtype=float)
        self._a_real_dot = np.zeros(3, dtype=float)
        self._gimbal_az = 0.0
        self._gimbal_el = 0.0
        self._last_t = -1e9
        self._initialized = False

    def reset(self):
        self._a_real = np.zeros(3, dtype=float)
        self._a_real_dot = np.zeros(3, dtype=float)
        self._gimbal_az = 0.0
        self._gimbal_el = 0.0
        self._last_t = -1e9
        self._initialized = False

    def update(self, a_cmd: np.ndarray, dt: float, quat: Optional[np.ndarray] = None,
               omega_body: Optional[np.ndarray] = None,
               dcm: Optional[np.ndarray] = None) -> np.ndarray:
        """Integrate the autopilot dynamics for one time step.

        Parameters
        ----------
        a_cmd : ndarray, shape (3,)
            Guidance acceleration command in inertial frame [m/s^2].
        dt : float
            Integration time step [s].
        quat : ndarray, shape (4,), optional
            Current attitude quaternion [q0, q1, q2, q3].
        omega_body : ndarray, shape (3,), optional
            Current body angular rates [rad/s].
        dcm : ndarray, shape (3, 3), optional
            Current direction-cosine matrix body->inertial.  If provided, it
            is used directly; otherwise it is computed from ``quat``.

        Returns
        -------
        ndarray, shape (3,)
            Realized acceleration in inertial frame [m/s^2].
        """
        a_cmd = np.asarray(a_cmd, dtype=float).reshape(3)
        dt = max(float(dt), 1e-12)
        omega = np.zeros(3) if omega_body is None else np.asarray(omega_body, dtype=float).reshape(3)

        if not self._initialized:
            self._a_real = np.clip(a_cmd, -self.accel_limit, self.accel_limit)
            self._a_real_dot = np.zeros(3, dtype=float)
            if quat is not None or dcm is not None:
                C = dcm if dcm is not None else self._quat_to_dcm(quat)
                az, el = self._inertial_to_gimbal(self._a_real, C)
                gimbal_mag = math.sqrt(az ** 2 + el ** 2)
                if gimbal_mag > self.gimbal_limit and gimbal_mag > 1e-12:
                    az = az * (self.gimbal_limit / gimbal_mag)
                    el = el * (self.gimbal_limit / gimbal_mag)
                self._gimbal_az = az
                self._gimbal_el = el
                r = float(np.linalg.norm(self._a_real))
                if r > 1e-12:
                    self._a_real = np.array([
                        r * math.cos(el) * math.cos(az),
                        r * math.cos(el) * math.sin(az),
                        r * math.sin(el),
                    ])
                    if dcm is not None:
                        self._a_real = dcm @ self._a_real
                    elif quat is not None:
                        self._a_real = self._quat_to_dcm(quat) @ self._a_real
            self._initialized = True
            return self._a_real.copy()

        C = dcm if dcm is not None else (self._quat_to_dcm(quat) if quat is not None else np.eye(3))

        # 1. Rate-limit the command before feeding the second-order system.
        delta_a = a_cmd - self._a_real
        step_norm = float(np.linalg.norm(delta_a))
        max_step = self.accel_rate_limit * dt
        if step_norm > max_step and step_norm > 1e-12:
            delta_a = delta_a * (max_step / step_norm)
        a_des = self._a_real + delta_a

        # 2. Second-order autopilot dynamics with body-rate feedback.
        #    a_real_ddot = omega_n^2*(a_des - a_real) - 2*zeta*omega_n*a_real_dot - rate_gain*omega_body
        a_err = a_des - self._a_real
        a_real_ddot = (self.omega_n ** 2) * a_err \
                    - 2.0 * self.damping * self.omega_n * self._a_real_dot \
                    - self.rate_gain * omega
        self._a_real_dot = self._a_real_dot + a_real_ddot * dt
        self._a_real = self._a_real + self._a_real_dot * dt

        # 3. Apply gimbal / diverter constraint in the body frame.
        if self.use_gimbal_limit:
            self._apply_gimbal_limit(dt, C)

        # 4. Final saturation.
        a_norm = float(np.linalg.norm(self._a_real))
        if a_norm > self.accel_limit and a_norm > 1e-12:
            self._a_real = self._a_real * (self.accel_limit / a_norm)
            self._a_real_dot = self._a_real_dot * (self.accel_limit / a_norm)

        return self._a_real.copy()

    def _apply_gimbal_limit(self, dt, C):
        v_body = C.T @ self._a_real
        r = float(np.linalg.norm(v_body))
        if r < 1e-12:
            return
        az = math.atan2(v_body[1], v_body[0])
        el = math.asin(np.clip(v_body[2] / r, -1.0, 1.0))

        # Cross-coupling: azimuth error bleeds into elevation and vice-versa.
        az_err = az - self._gimbal_az
        el_err = el - self._gimbal_el
        az_des = az - self.cross_coupling_gain * el_err
        el_des = el - self.cross_coupling_gain * az_err

        az_des = np.clip(az_des, -self.gimbal_limit, self.gimbal_limit)
        el_des = np.clip(el_des, -self.gimbal_limit, self.gimbal_limit)

        delta_az = az_des - self._gimbal_az
        delta_el = el_des - self._gimbal_el
        rate_limit = self.gimbal_rate_limit * dt
        if abs(delta_az) > rate_limit:
            delta_az = np.sign(delta_az) * rate_limit
        if abs(delta_el) > rate_limit:
            delta_el = np.sign(delta_el) * rate_limit
        self._gimbal_az += delta_az
        self._gimbal_el += delta_el

        v_body_new = np.array([
            r * math.cos(self._gimbal_el) * math.cos(self._gimbal_az),
            r * math.cos(self._gimbal_el) * math.sin(self._gimbal_az),
            r * math.sin(self._gimbal_el),
        ])
        self._a_real = C @ v_body_new

    @staticmethod
    def _quat_to_dcm(q):
        q = np.asarray(q, dtype=float)
        q0, q1, q2, q3 = q
        return np.array([
            [1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2)],
        ], dtype=float)

    @staticmethod
    def _inertial_to_gimbal(a_inertial, C):
        v_body = C.T @ a_inertial
        r = float(np.linalg.norm(v_body))
        if r < 1e-12:
            return 0.0, 0.0
        az = math.atan2(v_body[1], v_body[0])
        el = math.asin(np.clip(v_body[2] / r, -1.0, 1.0))
        return az, el
