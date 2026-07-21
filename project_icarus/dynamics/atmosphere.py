import numpy as np

try:  # NRLMSISE-00 is an optional dependency; fall back to the analytic thermosphere.
    from nrlmsise00 import msise_model as _msise_model
    _HAVE_NRLMSISE = True
except Exception:  # pragma: no cover - environment dependent
    _msise_model = None
    _HAVE_NRLMSISE = False


R_AIR = 287.05
GAMMA = 1.4
G0 = 9.80665
T0 = 288.15
P0 = 101325.0
RHO0 = 1.225

# Species molar masses [g/mol] for converting NRLMSISE number densities [cm^-3]
# to mass densities [kg/m^3].
_AVOGADRO = 6.02214076e23
_SPECIES_MOLAR = {
    "He": 4.002602,
    "O": 15.999,
    "N2": 28.0134,
    "O2": 31.9988,
    "Ar": 39.948,
    "H": 1.00794,
    "N": 14.0067,
    "AnO": 15.999,
}

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

_USSA76_BOTTOMS = np.array([l[0] for l in _USSA76_LAYERS])
_USSA76_TOPS = np.array([l[1] for l in _USSA76_LAYERS])
_USSA76_LAPSES = np.array([l[2] for l in _USSA76_LAYERS])
_USSA76_T_BOTTOMS = np.array([l[3] for l in _USSA76_LAYERS])
_USSA76_P_BOTTOMS = np.array([l[4] for l in _USSA76_LAYERS])


def _layer_index(h):
    h = np.asarray(h, dtype=float)
    idx = np.searchsorted(_USSA76_BOTTOMS, h, side='right') - 1
    return np.clip(idx, 0, len(_USSA76_LAYERS) - 1)


def _ussa76_density(h):
    h = np.asarray(h, dtype=float)
    idx = _layer_index(h)
    h_bottom = _USSA76_BOTTOMS[idx]
    lapse = _USSA76_LAPSES[idx]
    T_bottom = _USSA76_T_BOTTOMS[idx]
    P_bottom = _USSA76_P_BOTTOMS[idx]
    T = T_bottom + lapse * (h - h_bottom)
    T = np.maximum(T, 0.0)
    ratio = np.where(np.abs(T_bottom) > 1e-12, T / T_bottom, 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(
            np.abs(lapse) < 1e-12,
            P_bottom * np.exp(-G0 * (h - h_bottom) / (R_AIR * T_bottom)),
            P_bottom * np.where(ratio > 0, ratio ** (-G0 / (R_AIR * lapse)), 0.0),
        )
    return P / (R_AIR * T)


def _ussa76_temperature(h):
    h = np.asarray(h, dtype=float)
    idx = _layer_index(h)
    T_bottom = _USSA76_T_BOTTOMS[idx]
    lapse = _USSA76_LAPSES[idx]
    h_bottom = _USSA76_BOTTOMS[idx]
    return T_bottom + lapse * (h - h_bottom)


def _ussa76_pressure(h):
    h = np.asarray(h, dtype=float)
    idx = _layer_index(h)
    h_bottom = _USSA76_BOTTOMS[idx]
    lapse = _USSA76_LAPSES[idx]
    T_bottom = _USSA76_T_BOTTOMS[idx]
    P_bottom = _USSA76_P_BOTTOMS[idx]
    T = T_bottom + lapse * (h - h_bottom)
    ratio = np.where(np.abs(T_bottom) > 1e-12, T / T_bottom, 1.0)
    return P_bottom * np.where(ratio > 0, ratio ** (-G0 / (R_AIR * lapse)), 0.0)


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


class NRLMSISEExo:
    """NRLMSISE-00 thermosphere/exosphere model.

    A physically-based empirical atmosphere driven by the 81-day mean and
    previous-day F10.7 solar radio flux and the geomagnetic ap index, as well
    as the date/time, latitude, longitude and altitude. Requires the optional
    ``nrlmsise00`` package; if it is unavailable the analytic :class:`ExoAtmosphere`
    is used instead.
    """

    def __init__(self, f107a=150.0, f107=150.0, ap=4.0,
                 time=None, lat=0.0, lon=0.0):
        self.f107a = f107a
        self.f107 = f107
        self.ap = ap
        self.time = time
        self.lat = lat
        self.lon = lon

    def _msise(self, h):
        from datetime import datetime
        time = self.time if self.time is not None else datetime(2000, 3, 21, 12, 0, 0)
        alt_km = float(np.asarray(h, dtype=float))
        d, t = _msise_model(
            time, alt_km, self.lat, self.lon,
            self.f107a, self.f107, self.ap, method="gtd7d",
        )
        # d[5] = total mass density [g/cm^3] (gtd7d includes anomalous O).
        rho_g_cm3 = float(d[5])
        rho = rho_g_cm3 * 1.0e3  # g/cm^3 -> kg/m^3
        T = float(t[1])          # temperature at altitude [K]
        Texo = float(t[0])       # exospheric temperature [K]
        return rho, T, Texo

    def density(self, h):
        h = np.asarray(h, dtype=float)
        out = np.empty_like(h, dtype=float)
        for i in range(h.shape[0]):
            out[i] = self._msise(h[i])[0]
        return out

    def temperature(self, h):
        h = np.asarray(h, dtype=float)
        out = np.empty_like(h, dtype=float)
        for i in range(h.shape[0]):
            out[i] = self._msise(h[i])[1]
        return out

    def exospheric_temperature(self, h):
        h = np.asarray(h, dtype=float)
        out = np.empty_like(h, dtype=float)
        for i in range(h.shape[0]):
            out[i] = self._msise(h[i])[2]
        return out

    def pressure(self, h):
        rho = self.density(h)
        T = self.temperature(h)
        M_mean = 28.0e-3  # kg/mol, mean molar mass of the thermosphere
        return rho / M_mean * R_UNIVERSAL * np.maximum(T, 50.0)

    def speed_of_sound(self, h):
        T = self.temperature(h)
        return np.sqrt(GAMMA * R_AIR * np.maximum(T, 150.0))

    def dynamic_viscosity(self, h):
        T = self.temperature(h)
        return 1.458e-6 * (T ** 1.5) / (T + 110.4)


R_UNIVERSAL = 8.314462618


class Atmosphere:
    def __init__(self, boundary_alt=100e3, taper_width=5e3, exo_model=None):
        self.boundary_alt = boundary_alt
        self.taper_width = taper_width
        self.endo = EndoAtmosphere()
        if exo_model is not None:
            self.exo = exo_model
        elif _HAVE_NRLMSISE:
            self.exo = NRLMSISEExo()
        else:
            self.exo = ExoAtmosphere()
        self.uses_nrlmsise = isinstance(self.exo, NRLMSISEExo)

    def set_exo_solar_geomagnetic(self, f107a=None, f107=None, ap=None,
                                  time=None, lat=None, lon=None):
        """Update the exo-atmospheric drivers (solar flux, geomagnetism, geography)."""
        if not isinstance(self.exo, NRLMSISEExo):
            return
        if f107a is not None:
            self.exo.f107a = f107a
        if f107 is not None:
            self.exo.f107 = f107
        if ap is not None:
            self.exo.ap = ap
        if time is not None:
            self.exo.time = time
        if lat is not None:
            self.exo.lat = lat
        if lon is not None:
            self.exo.lon = lon

    def density(self, h):
        h = np.asarray(h, dtype=float)
        return self._regime_value(h, "density")

    def density_scalar(self, h):
        return self._regime_value_scalar(h, "density")

    def temperature(self, h):
        h = np.asarray(h, dtype=float)
        return self._regime_value(h, "temperature")

    def temperature_scalar(self, h):
        return self._regime_value_scalar(h, "temperature")

    def pressure(self, h):
        h = np.asarray(h, dtype=float)
        return self._regime_value(h, "pressure")

    def pressure_scalar(self, h):
        return self._regime_value_scalar(h, "pressure")

    def speed_of_sound(self, h):
        h = np.asarray(h, dtype=float)
        return self._regime_value(h, "speed_of_sound")

    def speed_of_sound_scalar(self, h):
        return self._regime_value_scalar(h, "speed_of_sound")

    def dynamic_viscosity(self, h):
        h = np.asarray(h, dtype=float)
        return self._regime_value(h, "dynamic_viscosity")

    def dynamic_viscosity_scalar(self, h):
        return self._regime_value_scalar(h, "dynamic_viscosity")

    def _regime_value(self, h, attr):
        h = np.asarray(h, dtype=float)
        endo_val = getattr(self.endo, attr)(h)
        exo_val = getattr(self.exo, attr)(h)

        taper_low = self.boundary_alt - self.taper_width
        taper_high = self.boundary_alt + self.taper_width

        blend = np.clip((h - taper_low) / (taper_high - taper_low), 0.0, 1.0)
        blend = 0.5 * (1.0 - np.cos(np.pi * blend))

        return endo_val * (1.0 - blend) + exo_val * blend

    def _regime_value_scalar(self, h, attr):
        h = float(h)
        endo_val = float(getattr(self.endo, attr)(np.array([h]))[0])
        exo_val = float(getattr(self.exo, attr)(np.array([h]))[0])

        taper_low = self.boundary_alt - self.taper_width
        taper_high = self.boundary_alt + self.taper_width

        blend = max(0.0, min(1.0, (h - taper_low) / (taper_high - taper_low)))
        blend = 0.5 * (1.0 - np.cos(np.pi * blend))

        return endo_val * (1.0 - blend) + exo_val * blend

    def is_endo(self, h):
        h = np.asarray(h, dtype=float)
        return h < self.boundary_alt

    def regime_mask(self, h):
        h = np.asarray(h, dtype=float)
        return np.where(h < self.boundary_alt, 0, 1)
