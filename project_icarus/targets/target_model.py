import numpy as np

MU_EARTH = 3.986004418e14
R_EARTH = 6371e3


class TargetModel:
    def __init__(self, mass=500.0, area=0.2, cd=0.3, initial_position=None, initial_velocity=None,
                 use_j2=True):
        self.mass = mass
        self.area = area
        self.cd = cd
        self.r = np.asarray(initial_position if initial_position is not None else [0.0, 0.0, 0.0], dtype=float)
        self.v = np.asarray(initial_velocity if initial_velocity is not None else [0.0, 0.0, 0.0], dtype=float)
        self.use_j2 = use_j2
        self.midcourse_accel_limit = 5.0

    def _gravity(self, r):
        r = np.asarray(r, dtype=float)
        rm = np.linalg.norm(r)
        if rm < 1e-6:
            return np.zeros(3)
        a = -MU_EARTH / rm**3 * r
        if self.use_j2 and rm - R_EARTH > 50e3:
            z = r[2]
            factor = -MU_EARTH / (rm**5) * (
                1.0 + 1.5 * 1.08263e-3 * (R_EARTH / rm)**2 * (5.0 * (z / rm)**2 - 1.0)
            )
            a = factor * r
        return a

    def ballistic(self, t, dt=0.1):
        r = self.r.copy()
        v = self.v.copy()
        n = max(int(t / dt), 1)
        for _ in range(n):
            v += self._gravity(r) * dt
            r += v * dt
        return r, v

    def midcourse_maneuver(self, t, command):
        cmd = np.clip(np.asarray(command, dtype=float), -self.midcourse_accel_limit, self.midcourse_accel_limit)
        self.v += cmd * t
        return self.ballistic(t)

    def state_at(self, t):
        return self.ballistic(t)
