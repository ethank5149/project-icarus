"""Boost-phase guidance laws for offensive ICBM and defensive interceptor ascent.

Provides:
- ``BoostGuidance``        : original q-triggered pitch-over (interceptor-centric)
- ``EnergyPitchSchedule``  : energy-based elevation schedule for ICBM boost
- ``PIPGuidance``          : Predictive Impact Point closed-loop guidance
"""

import numpy as np


MU_EARTH = 3.986004418e14
R_EARTH = 6371e3
G0 = 9.80665


class BoostGuidance:
    """Original q-triggered pitch-over guidance.

    Used by the defensive interceptor boost phase.  Triggers a fixed
    pitch-over angle once dynamic pressure exceeds ``pitch_over_q``.
    """

    def __init__(self, pitch_over_q=15000.0, pitch_over_angle=np.radians(5.0),
                 gimbal_limits=(np.radians(15), np.radians(15)), mu=MU_EARTH,
                 r0=R_EARTH, g0=G0):
        self.pitch_over_q = pitch_over_q
        self.pitch_over_angle = pitch_over_angle
        self.gimbal_limits = np.asarray(gimbal_limits, dtype=float)
        self.mu = mu
        self.r0 = r0
        self.g0 = g0
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

    def gravity_turn_angle(self, v0_mag, r0=None):
        """Pitch-over (flight-path) angle for a gravity-turn to desired apogee.

        theta0 = arcsin[1 / (1 + v0^2 / (g0 * r0))] (flat-Earth energy estimate).
        """
        r0 = r0 if r0 is not None else self.r0
        denom = 1.0 + v0_mag**2 / max(self.g0 * r0, 1e-6)
        return np.arcsin(np.clip(1.0 / denom, -1.0, 1.0))

    def pitch_rate(self, lift_accel, mass, gamma, speed):
        """Gravity-turn flight-path rate: gamma_dot = (L/m) - g*cos(gamma)/V."""
        g_local = self.mu / max(self.r0**2, 1e-6)
        return (lift_accel / max(mass, 1e-6)) - g_local * np.cos(gamma) / max(speed, 1e-6)


class EnergyPitchSchedule:
    """Energy-based pitch schedule for ICBM boost phase.

    Computes a desired elevation angle as a function of specific orbital
    energy and azimuth, automatically adapting to staging and dispersions
    without fixed time-based schedules.

    Parameters
    ----------
    el_table : list of (eps_norm, el_deg)
        Piecewise-linear table mapping normalized specific energy
        ``eps_norm = (eps - eps_min) / (eps_max - eps_min)`` to desired
        elevation angle in degrees.
    az_target : float
        Great-circle azimuth to target (degrees, 0=N CW).
    """

    def __init__(self, el_table=None, az_target=0.0):
        if el_table is None:
            # Default RS-28-like schedule: high pitch early, reducing to ~28°
            el_table = [
                (0.0, 55.0),
                (0.2, 50.0),
                (0.5, 40.0),
                (0.8, 32.0),
                (1.0, 28.0),
            ]
        self.el_table = sorted(el_table, key=lambda x: x[0])
        self.az_target = az_target

    def desired_elevation(self, r, v):
        """Return desired elevation angle (rad) above local horizon."""
        rmag = np.linalg.norm(r)
        eps = 0.5 * np.dot(v, v) - MU_EARTH / max(rmag, 1e-6)
        eps_min = -MU_EARTH / max(rmag, 1e-6)
        eps_max = 0.5 * (7500.0 ** 2) - MU_EARTH / max(rmag, 1e-6)
        eps_norm = (eps - eps_min) / max(eps_max - eps_min, 1e-9)
        eps_norm = np.clip(eps_norm, 0.0, 1.0)

        # Linear interpolation over the table
        for (e0, el0), (e1, el1) in zip(self.el_table, self.el_table[1:]):
            if eps_norm <= e1:
                frac = (eps_norm - e0) / max(e1 - e0, 1e-9)
                return np.radians(el0 + frac * (el1 - el0))
        return np.radians(self.el_table[-1][1])

    def thrust_direction(self, r, v):
        """Return unit thrust direction vector in inertial frame."""
        lat, lon, _ = _ecef_to_geodetic(r)
        east, north, up = _enu_basis(lat, lon)

        az = np.radians(self.az_target)
        gc_horiz = np.cos(az) * north + np.sin(az) * east

        el = self.desired_elevation(r, v)
        el = np.clip(el, np.radians(15.0), np.radians(75.0))
        return np.cos(el) * gc_horiz + np.sin(el) * up


class PIPGuidance:
    """Predictive Impact Point (PIP) closed-loop guidance.

    At each timestep, computes the predicted impact point of the current
    velocity vector (assuming ballistic coast from current state) and
    adjusts the thrust elevation to reduce the cross-range error.

    Parameters
    ----------
    Kp : float
        Proportional gain on cross-range error (rad/m).
    max_correction : float
        Maximum elevation correction (rad).
    """

    def __init__(self, Kp=5e-5, max_correction=np.radians(15.0)):
        self.Kp = Kp
        self.max_correction = max_correction

    def correction(self, r, v, target_r, dt_predict=300.0):
        """Return elevation correction (rad) to reduce miss distance.

        Parameters
        ----------
        r, v : ndarray
            Current position and velocity.
        target_r : ndarray
            Target position.
        dt_predict : float
            Look-ahead time for ballistic prediction (s).
        """
        # Propagate current state ballistically to estimate impact point
        r_pip = self._predict_impact(r, v, dt_predict)
        error = r_pip - target_r

        lat, lon, _ = _ecef_to_geodetic(r)
        east, north, up = _enu_basis(lat, lon)
        gc_horiz = error - np.dot(error, up) * up
        gc_horiz_norm = np.linalg.norm(gc_horiz)
        if gc_horiz_norm > 1e-6:
            gc_horiz = gc_horiz / gc_horiz_norm
        else:
            return 0.0

        cross_range = np.dot(error, gc_horiz)
        correction = -self.Kp * cross_range
        return float(np.clip(correction, -self.max_correction, self.max_correction))

    @staticmethod
    def _predict_impact(r, v, dt):
        """Simple ballistic prediction: constant velocity over dt."""
        r_pred = r + v * dt
        return r_pred


# ---------------------------------------------------------------------------
# Geodetic helpers (duplicated from target_factory to avoid circular imports)
# ---------------------------------------------------------------------------

def _ecef_to_geodetic(r):
    """Convert ECEF to WGS84 geodetic coordinates."""
    x, y, z = np.asarray(r, dtype=float)
    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x**2 + y**2)
    if p < 1e-6:
        lat = 90.0 if z > 0 else -90.0
        return float(lat), float(lon), float(abs(z) - (_WGS84_A * (1.0 - 1.0 / 298.257223563)))
    b = _WGS84_A * (1.0 - 1.0 / 298.257223563)
    e_prime2 = (_WGS84_A**2 - b**2) / b**2
    theta = np.arctan2(z * _WGS84_A, p * b)
    lat = np.arctan2(
        z + e_prime2 * b * np.sin(theta)**3,
        p - ((_WGS84_A**2 - b**2) / _WGS84_A) * np.cos(theta)**3,
    )
    sin_lat = np.sin(lat)
    N = _WGS84_A / np.sqrt(1.0 - ((_WGS84_A**2 - b**2) / _WGS84_A**2) * sin_lat**2)
    alt = p / np.cos(lat) - N
    return float(np.degrees(lat)), float(lon), float(alt)


def _enu_basis(lat_deg, lon_deg):
    """Return local East/North/Up unit vectors (ECEF) at a geodetic point."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return east, north, up


_WGS84_A = 6378137.0
