import numpy as np


class ExoAtmosphere:
    """Newtonian, zero-drag atmosphere above 100 km."""

    def __init__(self, rho0=0.0, h0=100e3, scale_height=0.0):
        self.rho0 = rho0
        self.h0 = h0
        self.scale_height = scale_height

    def density(self, h):
        return np.zeros_like(h, dtype=float)

    def temperature(self, h):
        return np.full_like(h, 200.0, dtype=float)

    def pressure(self, h):
        return np.zeros_like(h, dtype=float)

    def speed_of_sound(self, h):
        return np.full_like(h, 300.0, dtype=float)

    def dynamic_viscosity(self, h):
        return np.zeros_like(h, dtype=float)


class EndoAtmosphere:
    """Exponential atmosphere below 100 km with US Standard Atmosphere 1976 parameters."""

    def __init__(self, rho0=1.225, h0=0.0, scale_height=8500.0, T0=288.15, gamma=1.4, R_air=287.05):
        self.rho0 = rho0
        self.h0 = h0
        self.scale_height = scale_height
        self.T0 = T0
        self.gamma = gamma
        self.R_air = R_air

    def density(self, h):
        return self.rho0 * np.exp(-(h - self.h0) / self.scale_height)

    def temperature(self, h):
        return self.T0 - 0.0065 * h

    def pressure(self, h):
        T = self.temperature(h)
        return self.rho0 * self.R_air * T * np.exp(-(h - self.h0) / self.scale_height)

    def speed_of_sound(self, h):
        T = self.temperature(h)
        return np.sqrt(self.gamma * self.R_air * np.maximum(T, 150.0))

    def dynamic_viscosity(self, h):
        T = self.temperature(h)
        sutherland = 1.458e-6 * (T ** 1.5) / (T + 110.4)
        return sutherland


class Atmosphere:
    """Regime-aware atmosphere with smooth taper around 100 km boundary."""

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
