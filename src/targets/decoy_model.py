import numpy as np

MU_EARTH = 3.986004418e14
R_EARTH = 6371e3


class DecoyModel:
    def __init__(self, mass=50.0, area=0.1, cd=0.5, radar_rcs_bias=0.8, ir_bias=0.5,
                 release_altitude=50e3, ballistic_coeff=None):
        self.mass = mass
        self.area = area
        self.cd = cd
        # Lower ballistic coefficient than an RV by default (lighter/fluffier).
        self.ballistic_coeff = ballistic_coeff if ballistic_coeff is not None else mass / (cd * area)
        self.radar_rcs_bias = radar_rcs_bias
        self.ir_bias = ir_bias
        self.release_altitude = release_altitude
        self.active = False
        self.release_time = 0.0
        self.r = np.array([100000.0, 0.0, 50000.0], dtype=float)
        self.v = np.array([50.0, 0.0, -100.0], dtype=float)

    def release(self, t, r=None, v=None):
        self.active = True
        self.release_time = t
        if r is not None:
            self.r = np.asarray(r, dtype=float)
        if v is not None:
            self.v = np.asarray(v, dtype=float)

    def _gravity(self, r):
        rm = np.linalg.norm(r)
        if rm < 1e-6:
            return np.zeros(3)
        a = -MU_EARTH / rm**3 * r
        alt = rm - R_EARTH
        if alt < 150e3:
            rho = 1.225 * np.exp(-alt / 8500.0)
            vmag = np.linalg.norm(self.v)
            a_drag = -0.5 * rho * vmag * self.cd * self.area / self.mass * self.v
            a = a + a_drag
        return a

    def state_at(self, t):
        if not self.active:
            return None
        dt = 0.1
        tc = max(t - self.release_time, 0.0)
        n = max(int(tc / dt), 1)
        r = self.r.copy()
        v = self.v.copy()
        for _ in range(n):
            a = self._gravity(r)
            v = v + a * dt
            r = r + v * dt
        return r, v
