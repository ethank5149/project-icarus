import numpy as np


R_AIR = 287.05
GAMMA = 1.4
G0 = 9.80665
T0 = 288.15
P0 = 101325.0
RHO0 = 1.225

_USSA76_LAYERS = [
    (0.0, 11e3, -0.0065, T0, P0),
    (11e3, 20e3, 0.0, 216.65, 22632.0),
    (20e3, 32e3, 0.001, 216.65, 5474.9),
    (32e3, 47e3, 0.0028, 228.65, 868.02),
    (47e3, 51e3, 0.0, 270.65, 110.91),
    (51e3, 71e3, -0.0028, 270.65, 66.939),
    (71e3, 86e3, -0.002, 214.65, 3.9964),
    (86e3, 100e3, 0.0, 186.95, 0.37338),
]


def _ussa76_density(h):
    h = np.asarray(h, dtype=float)
    rho = np.zeros_like(h, dtype=float)
    for idx, (h_bottom, h_top, lapse, T_bottom, P_bottom) in enumerate(_USSA76_LAYERS):
        mask = (h >= h_bottom) & (h < h_top) if idx < len(_USSA76_LAYERS) - 1 else (h >= h_bottom) & (h <= h_top)
        if not np.any(mask):
            continue
        h_sub = h[mask]
        if lapse == 0:
            T = T_bottom * np.ones_like(h_sub)
            P = P_bottom * np.exp(-G0 * (h_sub - h_bottom) / (R_AIR * T_bottom))
        else:
            T = T_bottom + lapse * (h_sub - h_bottom)
            ratio = np.where(np.abs(T_bottom) > 1e-12, T / T_bottom, 1.0)
            P = P_bottom * np.where(ratio > 0, ratio ** (-G0 / (R_AIR * lapse)), 0.0)
        rho[mask] = P / (R_AIR * T)
    return rho


def _ussa76_temperature(h):
    h = np.asarray(h, dtype=float)
    T = np.zeros_like(h, dtype=float)
    for h_bottom, h_top, lapse, T_bottom, _ in _USSA76_LAYERS:
        if (h_bottom, h_top) == (86e3, 100e3):
            mask = (h >= h_bottom) & (h <= h_top)
        else:
            mask = (h >= h_bottom) & (h < h_top)
        if not np.any(mask):
            continue
        T[mask] = T_bottom + lapse * (h[mask] - h_bottom)
    return T


def _ussa76_pressure(h):
    h = np.asarray(h, dtype=float)
    P = np.zeros_like(h, dtype=float)
    for h_bottom, h_top, lapse, T_bottom, P_bottom in _USSA76_LAYERS:
        if (h_bottom, h_top) == (86e3, 100e3):
            mask = (h >= h_bottom) & (h <= h_top)
        else:
            mask = (h >= h_bottom) & (h < h_top)
        if not np.any(mask):
            continue
        h_sub = h[mask]
        if lapse == 0:
            P[mask] = P_bottom * np.exp(-G0 * (h_sub - h_bottom) / (R_AIR * T_bottom))
        else:
            T_local = T_bottom + lapse * (h_sub - h_bottom)
            ratio = np.where(np.abs(T_bottom) > 1e-12, T_local / T_bottom, 1.0)
            P[mask] = P_bottom * np.where(ratio > 0, ratio ** (-G0 / (R_AIR * lapse)), 0.0)
    return P


class EndoAtmosphere:
    def __init__(self, rho0=RHO0, h0=0.0, T0_=T0, gamma=GAMMA, R_air=R_AIR):
        self.rho0 = rho0
        self.h0 = h0
        self.T0_ = T0_
        self.gamma = gamma
        self.R_air = R_air

    def density(self, h):
        h = np.asarray(h, dtype=float)
        return _ussa76_density(h)

    def temperature(self, h):
        h = np.asarray(h, dtype=float)
        return _ussa76_temperature(h)

    def pressure(self, h):
        h = np.asarray(h, dtype=float)
        return _ussa76_pressure(h)

    def speed_of_sound(self, h):
        T = self.temperature(h)
        return np.sqrt(self.gamma * self.R_air * np.maximum(T, 150.0))

    def dynamic_viscosity(self, h):
        T = self.temperature(h)
        return 1.458e-6 * (T ** 1.5) / (T + 110.4)


class ExoAtmosphere:
    """Realistic thermosphere above 100 km.

    Temperature relaxes from a ~200 K base near the turbopause up toward a
    solar-activity-dependent exospheric temperature (default 500 K), and
    density follows an exponential profile with scale height ~50 km.
    """

    def __init__(self, T_inf=500.0, T_base=200.0, scale_height=50e3, rho0=5.297e-7, h0=100e3):
        self.T_inf = T_inf
        self.T_base = T_base
        self.scale_height = scale_height
        self.rho0 = rho0
        self.h0 = h0

    def density(self, h):
        h = np.asarray(h, dtype=float)
        return self.rho0 * np.exp(-np.maximum(h - self.h0, 0.0) / self.scale_height)

    def temperature(self, h):
        h = np.asarray(h, dtype=float)
        hgt = np.maximum(h - self.h0, 0.0)
        frac = 1.0 - np.exp(-np.clip(hgt / 150e3, 0.0, 10.0))
        return self.T_base + (self.T_inf - self.T_base) * frac

    def pressure(self, h):
        rho = self.density(h)
        T = self.temperature(h)
        return rho * R_AIR * T

    def speed_of_sound(self, h):
        T = self.temperature(h)
        return np.sqrt(GAMMA * R_AIR * np.maximum(T, 150.0))

    def dynamic_viscosity(self, h):
        T = self.temperature(h)
        return 1.458e-6 * (T ** 1.5) / (T + 110.4)


class Atmosphere:
    def __init__(self, boundary_alt=100e3, taper_width=5e3):
        self.boundary_alt = boundary_alt
        self.taper_width = taper_width
        self.exo = ExoAtmosphere()
        self.endo = EndoAtmosphere()

    def density(self, h):
        return self._regime_value(h, "density")

    def temperature(self, h):
        return self._regime_value(h, "temperature")

    def pressure(self, h):
        return self._regime_value(h, "pressure")

    def speed_of_sound(self, h):
        return self._regime_value(h, "speed_of_sound")

    def dynamic_viscosity(self, h):
        return self._regime_value(h, "dynamic_viscosity")

    def _regime_value(self, h, attr):
        h = np.asarray(h, dtype=float)
        endo_val = getattr(self.endo, attr)(h)
        exo_val = getattr(self.exo, attr)(h)

        taper_low = self.boundary_alt - self.taper_width
        taper_high = self.boundary_alt + self.taper_width

        blend = np.clip((h - taper_low) / (taper_high - taper_low), 0.0, 1.0)
        blend = 0.5 * (1.0 - np.cos(np.pi * blend))

        return endo_val * (1.0 - blend) + exo_val * blend

    def is_endo(self, h):
        h = np.asarray(h, dtype=float)
        return h < self.boundary_alt

    def regime_mask(self, h):
        h = np.asarray(h, dtype=float)
        return np.where(h < self.boundary_alt, 0, 1)
