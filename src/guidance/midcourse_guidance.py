import numpy as np


class MidcourseGuidance:
    def __init__(self, N=5.0, accel_limit=50.0, update_interval=2.0,
                 sigma_pos=100.0, sigma_vel=10.0):
        self.N = N
        self.accel_limit = accel_limit
        self.update_interval = update_interval
        self.sigma_pos = sigma_pos
        self.sigma_vel = sigma_vel
        self.target_state = None
        self._last_update = -1e9

    def update_target(self, target_state, t=None):
        if t is None or (t - self._last_update) >= self.update_interval:
            self.target_state = np.asarray(target_state, dtype=float)
            self._last_update = t if t is not None else self._last_update

    def commanded_accel(self, t, interceptor_state, los_rate=None, range_=None):
        if self.target_state is None:
            return np.zeros(3)
        r = np.asarray(interceptor_state["r"], dtype=float)
        v = np.asarray(interceptor_state["v"], dtype=float)
        los = self.target_state[:3] - r
        range_ = np.linalg.norm(los)
        los_unit = los / max(range_, 1e-6)
        rel_vel = self.target_state[3:6] - v
        # True proportional navigation.
        los_dot = (rel_vel - np.dot(rel_vel, los_unit) * los_unit) / max(range_, 1e-6)
        Vc = -np.dot(rel_vel, los_unit)
        a_c = self.N * Vc * los_dot
        return np.clip(a_c, -self.accel_limit, self.accel_limit)

    def measurement_noise(self, rng=None):
        if self.target_state is None:
            return None
        state = self.target_state.copy()
        noise = np.array([
            np.random.normal(0, self.sigma_pos),
            np.random.normal(0, self.sigma_pos),
            np.random.normal(0, self.sigma_pos),
            np.random.normal(0, self.sigma_vel),
            np.random.normal(0, self.sigma_vel),
            np.random.normal(0, self.sigma_vel),
        ])
        return state + noise
