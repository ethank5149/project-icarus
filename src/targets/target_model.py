import numpy as np


class TargetModel:
    def __init__(self, mass=500.0, area=0.2, cd=0.3, initial_position=None, initial_velocity=None):
        self.mass = mass
        self.area = area
        self.cd = cd
        self.r = np.asarray(initial_position if initial_position is not None else [0.0, 0.0, 0.0], dtype=float)
        self.v = np.asarray(initial_velocity if initial_velocity is not None else [0.0, 0.0, 0.0], dtype=float)
        self.midcourse_accel_limit = 5.0

    def ballistic(self, t, dt=0.1):
        n_steps = int(t / dt)
        r = self.r.copy()
        v = self.v.copy()
        for _ in range(n_steps):
            v -= 9.81 * dt * np.array([0.0, 0.0, 1.0])
            r += v * dt
        return r, v

    def midcourse_maneuver(self, t, command):
        cmd = np.clip(np.asarray(command, dtype=float), -self.midcourse_accel_limit, self.midcourse_accel_limit)
        self.v += cmd * t
        return self.ballistic(t)

    def state_at(self, t):
        return self.ballistic(t)
