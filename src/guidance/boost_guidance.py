import numpy as np


class BoostGuidance:
    def __init__(self, pitch_over_q=15000.0, pitch_over_angle=np.radians(5.0), gimbal_limits=np.radians(15)):
        self.pitch_over_q = pitch_over_q
        self.pitch_over_angle = pitch_over_angle
        self.gimbal_limits = gimbal_limits
        self.pitched = False

    def commanded_gimbal(self, t, state, rho, v_inertial):
        q_dyn = 0.5 * rho * np.dot(v_inertial, v_inertial)
        if not self.pitched and q_dyn > self.pitch_over_q:
            self.pitched = True
        if self.pitched:
            angle = self.pitch_over_angle
        else:
            angle = 0.0
        return np.clip([angle, 0.0], -self.gimbal_limits, self.gimbal_limits)


class MidcourseGuidance:
    def __init__(self, N=5.0, accel_limit=50.0, update_interval=2.0):
        self.N = N
        self.accel_limit = accel_limit
        self.update_interval = update_interval
        self.target_state = None

    def update_target(self, target_state):
        self.target_state = np.asarray(target_state, dtype=float)

    def commanded_accel(self, t, interceptor_state, los_rate, range_):
        if self.target_state is None:
            return np.zeros(3)
        r = interceptor_state["r"]
        v = interceptor_state["v"]
        los = self.target_state[:3] - r
        los_unit = los / max(np.linalg.norm(los), 1e-6)
        rel_vel = self.target_state[3:6] - v
        closing = -np.dot(rel_vel, los_unit)
        accel = self.N * closing * np.cross(rel_vel, los_unit)
        accel = np.clip(accel, -self.accel_limit, self.accel_limit)
        return accel
