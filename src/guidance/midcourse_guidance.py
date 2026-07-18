import numpy as np


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
