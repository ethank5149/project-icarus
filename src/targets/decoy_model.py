import numpy as np


class DecoyModel:
    def __init__(self, mass=50.0, area=0.1, cd=0.4, radar_rcs_bias=0.8, ir_bias=0.5, release_altitude=50e3):
        self.mass = mass
        self.area = area
        self.cd = cd
        self.radar_rcs_bias = radar_rcs_bias
        self.ir_bias = ir_bias
        self.release_altitude = release_altitude
        self.active = False
        self.release_time = 0.0

    def release(self, t):
        self.active = True
        self.release_time = t

    def state_at(self, t):
        if not self.active:
            return None
        dt = max(t - self.release_time, 0.0)
        r = np.array([100000.0 + 50.0 * dt, 0.0, 50000.0 - 100.0 * dt])
        v = np.array([50.0, 0.0, -100.0])
        return r, v
