from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Protocol, runtime_checkable
import numpy as np
from scipy.integrate import solve_ivp

from ..dynamics.eom_6dof import EOM6DOF
from ..guidance.terminal_guidance import TerminalGuidance
from ..aero.aero_analytical import blended_aero
from ..dynamics.atmosphere import Atmosphere
from ..dynamics.gravity import gravity_inertial
from ..dynamics.thrust import StageSpec, MultiStageThrustModel, StageSeparation

try:
    from ..dynamics.gravity import gravity_inertial_jit, GRAVITY_JIT_AVAILABLE
except Exception:
    GRAVITY_JIT_AVAILABLE = False


MU_EARTH = 3.986004418e14
R_EARTH = 6371e3
J2_EARTH = 1.08263e-3

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = (_WGS84_A**2 - _WGS84_B**2) / _WGS84_A**2


def _wgs84_radius(r):
    """Compute the WGS84 ellipsoid radius at the given ECEF position."""
    x, y, z = np.asarray(r, dtype=float)
    p = np.sqrt(x**2 + y**2)
    if p < 1e-6:
        return _WGS84_B if z > 0 else _WGS84_B
    lat = np.arctan2(z, p)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat)**2)
    return N * np.sqrt(np.cos(lat)**2 + ((1.0 - _WGS84_E2) * np.sin(lat))**2)


def _ground_altitude(r):
    """Compute altitude above actual ground surface (meters)."""
    x, y, z = np.asarray(r, dtype=float)
    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x * x + y * y)
    if p < 1e-6:
        lat = 90.0 if z > 0 else -90.0
        eccen_n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(np.radians(lat)) ** 2)
        alt_ellip = float(abs(z) - eccen_n * (1.0 - _WGS84_E2))
    else:
        b = _WGS84_B
        e_prime2 = _WGS84_E2 / (1.0 - _WGS84_E2)
        theta = np.arctan2(z * _WGS84_A, p * b)
        lat = np.degrees(np.arctan2(
            z + e_prime2 * b * np.sin(theta) ** 3,
            p - _WGS84_E2 * _WGS84_A * np.cos(theta) ** 3,
        ))
        sin_lat = np.sin(np.radians(lat))
        N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat ** 2)
        alt_ellip = float(p / np.cos(np.radians(lat)) - N)
    from ..reference.surface_elevation import get_surface_elevation
    elev = float(get_surface_elevation(float(lat), float(lon)))
    return alt_ellip - elev


def _two_body_accel(r, use_j2=True, use_j3=False, use_j4=False,
                     use_high_order=False, use_third_body=False, use_tides=False,
                     max_degree=10, t=0.0):
    """Central + zonal + third-body + tidal acceleration (thin wrapper over gravity_inertial)."""
    if GRAVITY_JIT_AVAILABLE and not (use_j3 or use_j4 or use_third_body or use_tides):
        return gravity_inertial_jit(
            r,
            use_j2,
            J2_EARTH,
            -2.532e-6,
            -1.610e-6,
            use_high_order,
            False,
            False,
            t,
            False,
            False,
            max_degree,
        )
    return gravity_inertial(
        r,
        use_j2=use_j2,
        j2=J2_EARTH,
        j3=-2.532e-6,
        j4=-1.610e-6,
        use_high_order=use_high_order,
        use_third_body=use_third_body,
        use_tides=use_tides,
        t=t,
        use_j3=use_j3,
        use_j4=use_j4,
        max_degree=max_degree,
    )


def _rk4_step(r, v, dt, accel_func):
    k1_v = accel_func(r, v)
    k1_r = v
    k2_v = accel_func(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v)
    k2_r = v + 0.5 * dt * k1_v
    k3_v = accel_func(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v)
    k3_r = v + 0.5 * dt * k2_v
    k4_v = accel_func(r + dt * k3_r, v + dt * k3_v)
    k4_r = v + dt * k3_v
    r_new = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return r_new, v_new


try:
    from numba import njit

    @njit
    def _propagate_ballistic_jit(r0, v0, dt, n_steps, use_j2, j2, j3, j4,
                                 use_high_order, use_third_body, use_tides, max_degree, t,
                                 use_j3_flag, use_j4_flag):
        r = r0.copy()
        v = v0.copy()
        for _ in range(n_steps):
            k1_v = gravity_inertial_jit(r, use_j2, j2, j3, j4, use_high_order,
                                        use_third_body, use_tides, t, use_j3_flag, use_j4_flag, max_degree)
            k1_r = v
            k2_v = gravity_inertial_jit(r + 0.5*dt*k1_r, use_j2, j2, j3, j4, use_high_order,
                                        use_third_body, use_tides, t, use_j3_flag, use_j4_flag, max_degree)
            k2_r = v + 0.5*dt*k1_v
            k3_v = gravity_inertial_jit(r + 0.5*dt*k2_r, use_j2, j2, j3, j4, use_high_order,
                                        use_third_body, use_tides, t, use_j3_flag, use_j4_flag, max_degree)
            k3_r = v + 0.5*dt*k2_v
            k4_v = gravity_inertial_jit(r + dt*k3_r, use_j2, j2, j3, j4, use_high_order,
                                        use_third_body, use_tides, t, use_j3_flag, use_j4_flag, max_degree)
            k4_r = v + dt*k3_v
            r = r + (dt/6.0)*(k1_r + 2.0*k2_r + 2.0*k3_r + k4_r)
            v = v + (dt/6.0)*(k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)
        return r, v

    @njit
    def _propagate_suppressed_jit(r0, v0, dt, n_steps, use_j2, j2, j3, j4,
                                  use_high_order, use_third_body, use_tides, max_degree, t,
                                  use_j3_flag, use_j4_flag,
                                  maneuver_mag, maneuver_dir, maneuver_interval):
        r = r0.copy()
        v = v0.copy()
        applied = np.zeros(1000, dtype=np.bool_)
        for k in range(n_steps):
            tc = (k + 1) * dt
            k1_v = gravity_inertial_jit(r, use_j2, j2, j3, j4, use_high_order,
                                        use_third_body, use_tides, t, use_j3_flag, use_j4_flag, max_degree)
            k1_r = v
            k2_v = gravity_inertial_jit(r + 0.5*dt*k1_r, use_j2, j2, j3, j4, use_high_order,
                                        use_third_body, use_tides, t, use_j3_flag, use_j4_flag, max_degree)
            k2_r = v + 0.5*dt*k1_v
            k3_v = gravity_inertial_jit(r + 0.5*dt*k2_r, use_j2, j2, j3, j4, use_high_order,
                                        use_third_body, use_tides, t, use_j3_flag, use_j4_flag, max_degree)
            k3_r = v + 0.5*dt*k2_v
            k4_v = gravity_inertial_jit(r + dt*k3_r, use_j2, j2, j3, j4, use_high_order,
                                        use_third_body, use_tides, t, use_j3_flag, use_j4_flag, max_degree)
            k4_r = v + dt*k3_v
            r = r + (dt/6.0)*(k1_r + 2.0*k2_r + 2.0*k3_r + k4_r)
            v = v + (dt/6.0)*(k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)
            if tc > 60.0:
                m = int(tc / maneuver_interval)
                if m % 2 == 0 and not applied[m]:
                    v = v + maneuver_mag * maneuver_dir
                    applied[m] = True
        return r, v

    NUMBA_PROPAGATORS_AVAILABLE = True
except Exception:
    NUMBA_PROPAGATORS_AVAILABLE = False

def _ecef_to_geodetic(r):
    """Convert ECEF to WGS84 geodetic coordinates."""
    x, y, z = np.asarray(r, dtype=float)
    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x**2 + y**2)
    if p < 1e-6:
        lat = 90.0 if z > 0 else -90.0
        return float(lat), float(lon), float(abs(z) - _WGS84_B)
    b = _WGS84_B
    e_prime2 = _WGS84_E2 / (1.0 - _WGS84_E2)
    theta = np.arctan2(z * _WGS84_A, p * b)
    lat = np.arctan2(
        z + e_prime2 * b * np.sin(theta)**3,
        p - _WGS84_E2 * _WGS84_A * np.cos(theta)**3,
    )
    sin_lat = np.sin(lat)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
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


def _geodetic_to_ecef_simple(lat_deg, lon_deg, alt_m=0.0):
    """Convert geodetic to ECEF using WGS84 ellipsoid."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat**2)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z])


def _destination_point(lat_deg, lon_deg, az_deg, dist_deg):
    """Spherical destination point given start, azimuth (deg, 0=N CW), arc deg."""
    lat1 = np.radians(lat_deg)
    lon1 = np.radians(lon_deg)
    az = np.radians(az_deg)
    d = np.radians(dist_deg)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(az))
    lon2 = lon1 + np.arctan2(np.sin(az) * np.sin(d) * np.cos(lat1),
                              np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return float(np.degrees(lat2)), float(np.degrees(lon2))
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(az)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(az) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.degrees(lat2)), float(np.degrees(lon2))


@dataclass
class GuidedThreatConfig:
    """Vehicle + guidance parameters for a real closed-loop (PN-guided) threat.

    Reuses the repo's existing EOM6DOF + TerminalGuidance laws rather than the
    ad-hoc steering term used by the analytic scenarios.
    """

    mass: float = 1200.0
    inertia: Any = field(default_factory=lambda: np.diag([150.0, 300.0, 300.0]))
    area: float = 0.2
    ref_length: float = 1.5
    boundary_alt: float = 100e3
    taper_width: float = 5e3
    use_j2: bool = True
    accel_limit: float = 80.0
    N: float = 4.0


def _guided_surrogate(mach, alpha, beta, alt):
    return blended_aero(mach, alpha, beta, alt)[:3]


def simulate_guided_threat(
    r0: np.ndarray,
    v0: np.ndarray,
    aim_point: np.ndarray,
    vehicle: Optional[GuidedThreatConfig] = None,
    guidance_law: Optional[TerminalGuidance] = None,
    t0: float = 0.0,
    t_end: float = 600.0,
):
    """Closed-loop PN-guided propagation of a threat toward ``aim_point``.

    Integrates the full 6-DOF EOM under aerodynamic + gravity loads, with a
    proportional-navigation terminal-guidance law homing on the (ground,
    stationary) aim point. Returns the sampled times and 6-vector states
    ``[r(3), v(3)]`` so callers can interpolate at arbitrary query times.

    The closed-loop RHS mirrors ``sim.runner._closed_loop_rhs`` but with the
    threat as the controlled body and the defended aim point as the PN target.
    """
    if vehicle is None:
        vehicle = GuidedThreatConfig()
    if guidance_law is None:
        guidance_law = TerminalGuidance(N=vehicle.N, accel_limit=vehicle.accel_limit)

    eom = EOM6DOF(
        mass=vehicle.mass,
        inertia=vehicle.inertia,
        area=vehicle.area,
        ref_length=vehicle.ref_length,
        boundary_alt=vehicle.boundary_alt,
        taper_width=vehicle.taper_width,
        use_j2=vehicle.use_j2,
    )

    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    omega0 = np.zeros(3)
    m0 = vehicle.mass
    y0 = np.concatenate([r0, v0, q0, omega0, [m0]])

    aim = np.asarray(aim_point, dtype=float)

    def rhs(t, y):
        r = y[:3]
        v = y[3:6]
        q = y[6:10]
        omega = y[10:13]
        m = y[13]
        q = q / max(np.linalg.norm(q), 1e-12)

        eom_state = {"r": r, "v": v, "q": q, "omega": omega, "m": m}
        target_state = np.concatenate([aim, np.zeros(3)])
        accel_cmd = guidance_law.commanded_accel(
            t, eom_state, target_state, disable_fov=True,
        )

        derivs = eom.compute(t, eom_state, _guided_surrogate)
        dr_dt = derivs["r"]
        dv_dt = derivs["v"] + accel_cmd * m / max(m, 1e-6)
        return np.concatenate([
            dr_dt, dv_dt, derivs["q"], derivs["omega"], [derivs["m"]],
        ])

    # Ground-impact terminal event: stop the integration the moment the RV
    # reaches the surface (geodetic altitude <= 0). The adaptive Runge-Kutta
    # stepper (RK45) takes large steps in vacuum and refines them automatically
    # only inside the atmosphere, so no hand-rolled step-count guard is needed.
    def _ground_event(t, y):
        return _ground_altitude(y[:3])

    _ground_event.terminal = True
    _ground_event.direction = -1

    sol = solve_ivp(
        rhs, (t0, t_end), y0,
        method="RK45",
        rtol=1e-6, atol=1e-9,
        max_step=5.0,
        dense_output=True,
        events=_ground_event,
        first_step=0.05,
    )

    # Sample the dense interpolant on a uniform grid for callers that linearly
    # interpolate the returned arrays. The grid is coarse (2 s) because the
    # interpolant is accurate; the ground event caps the final sample.
    if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
        t_end_eff = float(sol.t_events[0][0])
    else:
        t_end_eff = float(sol.t[-1])

    if t_end_eff <= t0:
        sol_t = np.array([t0, t0 + 1e-3])
    else:
        sol_t = np.linspace(t0, t_end_eff, max(int((t_end_eff - t0) / 2.0) + 1, 2))

    try:
        grid = sol.sol(sol_t)
        six_vec = np.vstack([grid[:3, :], grid[3:6, :]])
    except Exception:
        idx = np.searchsorted(sol.t, sol_t)
        idx = np.clip(idx, 0, len(sol.t) - 1)
        six_vec = np.vstack([sol.y[:3, idx], sol.y[3:6, idx]])
    return sol_t, six_vec


@runtime_checkable
class TargetScenario(Protocol):
    target_type: str
    metadata: Dict[str, Any]

    def propagate(self, t: float) -> np.ndarray: ...
    def propagate_batch(self, times: np.ndarray) -> np.ndarray: ...


@dataclass
class BallisticScenario:
    target_type: str = "ballistic"
    r0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    use_j2: bool = True
    use_j3: bool = False
    use_j4: bool = False
    use_high_order: bool = False
    use_third_body: bool = False
    use_tides: bool = False
    max_degree: int = 10
    adaptive: bool = False
    ballistic_coeff: float = 100.0
    cd: float = 0.3
    area: float = 0.02
    mass: float = 1000.0
    geometry_key: str = "rs28_sarmat"
    atmosphere: Optional[Atmosphere] = None
    wind_model: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _wind_cache_key: Any = field(default=None, repr=False, compare=False)
    _wind_cache_val: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.atmosphere is None:
            self.atmosphere = Atmosphere()

    def _wind_accel(self, r, v, t, dt):
        if self.wind_model is None:
            return np.zeros(3)
        try:
            lat, lon, alt = _ecef_to_geodetic(r)
        except Exception:
            return np.zeros(3)
        if alt > 150e3 or alt < 0.0:
            return np.zeros(3)
        key = (round(float(lat), 4), round(float(lon), 4), round(float(alt), 1), round(float(t), 2))
        if getattr(self, '_wind_cache_key', None) != key:
            U0 = np.asarray(self.wind_model.wind(float(lat), float(lon), float(alt), float(t)), dtype=float)
            dU_dt = np.zeros(3)
            if dt > 1e-6:
                U1 = np.asarray(self.wind_model.wind(float(lat), float(lon), float(alt), float(t) + dt), dtype=float)
                dU_dt = (U1 - U0) / dt
            eps = 1e-4
            U_lat = np.asarray(self.wind_model.wind(float(lat) + eps, float(lon), float(alt), float(t)), dtype=float)
            U_lon = np.asarray(self.wind_model.wind(float(lat), float(lon) + eps, float(alt), float(t)), dtype=float)
            deg2m_lat = 111_132.92
            deg2m_lon = 111_132.92 * np.cos(np.radians(float(lat)))
            a_wind = np.zeros(3)
            for i in range(3):
                a_wind[i] = dU_dt[i] + v[0] * (U_lon[i] - U0[i]) / (deg2m_lon * eps) + v[1] * (U_lat[i] - U0[i]) / (deg2m_lat * eps)
            self._wind_cache_key = key
            self._wind_cache_val = a_wind.tolist()
        return np.array(self._wind_cache_val)

    def _accel(self, r, v, t=None, dt=0.5):
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
        vmag = np.linalg.norm(v)
        a_grav = _two_body_accel(r, use_j2=self.use_j2, use_j3=self.use_j3,
                                 use_j4=self.use_j4, use_high_order=self.use_high_order,
                                 use_third_body=self.use_third_body, use_tides=self.use_tides,
                                 max_degree=self.max_degree)
        a_drag = np.zeros(3)
        alt = _ground_altitude(r)
        if 0.0 < alt < 100e3 and vmag > 1e-6:
            rho = self.atmosphere.density_scalar(alt)
            q_dyn = 0.5 * rho * vmag**2
            a_drag = -q_dyn * self.cd * self.area / self.mass * (v / vmag)
        a = a_grav + a_drag
        if t is not None:
            a = a + self._wind_accel(r, v, t, dt)
        return a

    def propagate(self, t: float) -> np.ndarray:
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        y0 = np.concatenate([r, v])

        def rhs(ti, y):
            r, v = y[:3], y[3:]
            a = self._accel(r, v, t=ti, dt=0.5)
            return np.concatenate([v, a])

        def ground_event(ti, y):
            return _ground_altitude(y[:3])
        ground_event.terminal = True
        ground_event.direction = -1

        sol = solve_ivp(rhs, (0.0, t), y0, method="RK45",
                        rtol=1e-6, atol=1e-9, max_step=5.0,
                        events=ground_event, dense_output=True)
        if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
            t_end_eff = float(sol.t_events[0][0])
        else:
            t_end_eff = float(sol.t[-1])
        if t_end_eff <= 0.0:
            return np.concatenate([r, v])
        return sol.sol(t_end_eff)[:6]

    def propagate_batch(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        return np.array([self.propagate(ti) for ti in times])


def _kepler_propagate(r0, v0, t, mu):
    """Universal-variable Kepler propagation of a two-body state to time ``t``."""
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    rmag = np.linalg.norm(r0)
    vmag = np.linalg.norm(v0)
    if rmag < 1e-9 or abs(t) < 1e-12:
        return np.concatenate([r0, v0])

    energy = vmag**2 / 2.0 - mu / rmag
    a = -mu / (2.0 * energy)
    sqrt_mu = np.sqrt(mu)
    rv = np.dot(r0, v0)
    # Initial guess for universal anomaly chi.
    if a > 0:
        chi = sqrt_mu * t / a
    else:
        chi = np.sign(t) * np.sqrt(-a) * np.log(
            max(1.0 + abs(t) * np.sqrt(-energy) / rmag, 1e-12)
        )

    psi = chi**2 / a
    c2 = (1.0 - np.cos(np.sqrt(psi))) / psi if psi > 1e-9 else 0.5
    c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / np.power(psi, 1.5) if psi > 1e-9 else 1.0 / 6.0

    for _ in range(100):
        r = (r0.dot(r0) / rmag + chi * (rv / sqrt_mu + chi * c2) +
             a * (1.0 - psi * c3))
        if abs(r) < 1e-9:
            break
        chi_next = chi + (sqrt_mu * t - chi**3 * c3 - rv / sqrt_mu * chi**2 * c2 -
                          r0.dot(r0) / rmag * chi * (1.0 - psi * c3)) / r
        # Derivative of c2/c3 w.r.t. psi for Newton correction.
        if psi > 1e-9:
            c2 = (1.0 - np.cos(np.sqrt(psi))) / psi
            c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / np.power(psi, 1.5)
        else:
            c2, c3 = 0.5, 1.0 / 6.0
        dpsi = 2.0 * chi / a * (chi_next - chi)
        c2 = (1.0 - np.cos(np.sqrt(psi + dpsi))) / (psi + dpsi) if (psi + dpsi) > 1e-9 else 0.5
        c3 = (np.sqrt(psi + dpsi) - np.sin(np.sqrt(psi + dpsi))) / np.power(psi + dpsi, 1.5) if (psi + dpsi) > 1e-9 else 1.0 / 6.0
        if abs(chi_next - chi) < 1e-9:
            chi = chi_next
            break
        chi = chi_next

    f = 1.0 - chi**2 / rmag * c2
    g = t - chi**3 / sqrt_mu * c3
    r_new = f * r0 + g * v0
    rmag_new = np.linalg.norm(r_new)
    f_dot = sqrt_mu / (rmag * rmag_new) * chi * (psi * c3 - 1.0)
    g_dot = 1.0 - chi**2 / rmag_new * c2
    v_new = f_dot * r0 + g_dot * v0
    return np.concatenate([r_new, v_new])

    @classmethod
    def from_range(cls, range_km: float, launch_az_deg: float = 0.0, launch_el_deg: float = 45.0):
        az = np.radians(launch_az_deg)
        el = np.radians(launch_el_deg)
        speed = np.sqrt(2 * MU_EARTH / (R_EARTH + 0.0))
        v0 = speed * np.array([
            np.cos(el) * np.sin(az),
            np.cos(el) * np.cos(az),
            np.sin(el),
        ])
        r0 = np.array([R_EARTH, 0.0, 0.0])
        return cls(r0=r0, v0=v0)


@dataclass
class FOBSScenario:
    target_type: str = "fobs"
    r0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    apoapsis_km: float = 200.0
    inclination_deg: float = 0.0
    reentry_corridor_deg: float = 3.0
    thrust: float = 0.0
    _boost_duration: float = 120.0
    _coast_duration: float = 600.0
    _reentry_alt: float = 100e3
    deorbit_dv: float = 1.0
    vehicle: Any = None
    guidance: Any = None
    geometry_key: str = "fobs_payload"
    use_j2: bool = True
    use_j3: bool = False
    use_j4: bool = False
    use_high_order: bool = False
    use_third_body: bool = False
    use_tides: bool = False
    max_degree: int = 10
    wind_model: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cache: Any = field(default=None, repr=False)
    _wind_cache_key: Any = field(default=None, repr=False, compare=False)
    _wind_cache_val: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        rmag = np.linalg.norm(self.r0)
        if rmag > 1e-6 and np.linalg.norm(self.v0) > 1e-6:
            lat0, lon0, _ = _ecef_to_geodetic(self.r0)
            v_unit = self.v0 / np.linalg.norm(self.v0)
            east = np.array([-np.sin(np.radians(lon0)), np.cos(np.radians(lon0)), 0.0])
            north = np.array([
                -np.sin(np.radians(lat0)) * np.cos(np.radians(lon0)),
                -np.sin(np.radians(lat0)) * np.sin(np.radians(lon0)),
                np.cos(np.radians(lat0)),
            ])
            az = np.degrees(np.arctan2(np.dot(v_unit, east), np.dot(v_unit, north))) % 360.0
            coast_arc_deg = (self._coast_duration * np.linalg.norm(self.v0) / rmag) / np.pi * 180.0
            aim_lat, aim_lon = _destination_point(lat0, lon0, az, coast_arc_deg * 0.5)
            self._aim_point = _geodetic_to_ecef_simple(aim_lat, aim_lon, 0.0)
        else:
            self._aim_point = np.array([R_EARTH + self._reentry_alt, 0.0, 0.0])
        self._cache = None

    def _wind_accel(self, r, v, t, dt):
        if self.wind_model is None:
            return np.zeros(3)
        try:
            lat, lon, alt = _ecef_to_geodetic(r)
        except Exception:
            return np.zeros(3)
        if alt > 150e3 or alt < 0.0:
            return np.zeros(3)
        key = (round(float(lat), 4), round(float(lon), 4), round(float(alt), 1), round(float(t), 2))
        if getattr(self, '_wind_cache_key', None) != key:
            U0 = np.asarray(self.wind_model.wind(float(lat), float(lon), float(alt), float(t)), dtype=float)
            dU_dt = np.zeros(3)
            if dt > 1e-6:
                U1 = np.asarray(self.wind_model.wind(float(lat), float(lon), float(alt), float(t) + dt), dtype=float)
                dU_dt = (U1 - U0) / dt
            eps = 1e-4
            U_lat = np.asarray(self.wind_model.wind(float(lat) + eps, float(lon), float(alt), float(t)), dtype=float)
            U_lon = np.asarray(self.wind_model.wind(float(lat), float(lon) + eps, float(alt), float(t)), dtype=float)
            deg2m_lat = 111_132.92
            deg2m_lon = 111_132.92 * np.cos(np.radians(float(lat)))
            a_wind = np.zeros(3)
            for i in range(3):
                a_wind[i] = dU_dt[i] + v[0] * (U_lat[i] - U0[i]) / (deg2m_lat * eps) + v[1] * (U_lon[i] - U0[i]) / (deg2m_lon * eps)
            self._wind_cache_key = key
            self._wind_cache_val = a_wind.tolist()
        return np.array(self._wind_cache_val)

    def _orbital_speed(self, r):
        return np.sqrt(MU_EARTH / max(r, 1e-6))

    def _full_trajectory(self):
        if self._cache is not None:
            return self._cache

        # ------------------------------------------------------------------
        # Phase 1: Boost to parking orbit altitude
        # ------------------------------------------------------------------
        if abs(self.thrust) > 1e-6 and self._boost_duration > 0:
            dt_boost = 0.05
            boost_t = np.arange(0.0, self._boost_duration + dt_boost, dt_boost)
            boost_states = []
            r = self.r0.astype(float).copy()
            v = self.v0.astype(float).copy()
            for ti_idx, ti in enumerate(boost_t):
                boost_states.append(np.concatenate([r.copy(), v.copy()]))
                a = _two_body_accel(r, use_j2=self.use_j2, use_j3=self.use_j3,
                                    use_j4=self.use_j4, use_high_order=self.use_high_order,
                                    use_third_body=self.use_third_body, use_tides=self.use_tides,
                                    max_degree=self.max_degree) + np.array([self.thrust, 0.0, 0.0])
                if self.wind_model is not None and _ground_altitude(r) < 150e3:
                    a = a + self._wind_accel(r, v, ti, dt_boost)
                v = v + a * dt_boost
                r = r + v * dt_boost
            boost_states.append(np.concatenate([r.copy(), v.copy()]))
            r_coast = boost_states[-1][:3].copy()
            v_coast = boost_states[-1][3:].copy()
            t0_coast = self._boost_duration
        else:
            boost_states = [np.concatenate([self.r0.astype(float).copy(), self.v0.astype(float).copy()])]
            r_coast = self.r0.astype(float).copy()
            v_coast = self.v0.astype(float).copy()
            t0_coast = 0.0
            boost_t = np.array([0.0])

        # ------------------------------------------------------------------
        # Phase 2: Coast in parking orbit (circular LEO)
        # ------------------------------------------------------------------
        r_i = np.linalg.norm(r_coast)
        v_circular = np.sqrt(MU_EARTH / r_i)
        v_coast_dir = v_coast / max(np.linalg.norm(v_coast), 1e-12)
        v_coast = v_coast_dir * v_circular

        y0_coast = np.concatenate([r_coast, v_coast])

        def rhs_coast(ti, y):
            r, v = y[:3], y[3:]
            a = _two_body_accel(r, use_j2=self.use_j2, use_j3=self.use_j3,
                                use_j4=self.use_j4, use_high_order=self.use_high_order,
                                use_third_body=self.use_third_body, use_tides=self.use_tides,
                                max_degree=self.max_degree)
            if self.wind_model is not None and _ground_altitude(r) < 150e3:
                a = a + self._wind_accel(r, v, ti, dt_coast)
            return np.concatenate([v, a])

        dt_coast = 0.05
        coast_t_eval = np.arange(0.0, self._coast_duration + dt_coast, dt_coast)
        sol_coast = solve_ivp(rhs_coast, (0.0, self._coast_duration), y0_coast,
                              method="RK45", t_eval=coast_t_eval,
                              rtol=1e-9, atol=1e-12)
        coast_states = sol_coast.y.T if sol_coast.y.size > 0 else np.array([y0_coast])

        # ------------------------------------------------------------------
        # Phase 3: Deorbit burn (retrograde impulse at apogee)
        # ------------------------------------------------------------------
        r_deorbit = coast_states[-1][:3].copy()
        v_deorbit_circular = coast_states[-1][3:].copy()
        v_deorbit_dir = v_deorbit_circular / max(np.linalg.norm(v_deorbit_circular), 1e-12)

        r_a = np.linalg.norm(r_deorbit)
        r_p = R_EARTH + 80e3
        a_t = (r_a + r_p) / 2.0
        v_a = np.sqrt(MU_EARTH * (2.0 / r_a - 1.0 / a_t))
        delta_v = v_circular - v_a

        retrograde_dir = -v_deorbit_dir
        to_aim = self._aim_point - r_deorbit
        to_aim_unit = to_aim / max(np.linalg.norm(to_aim), 1e-6)

        orbital_plane_normal = np.cross(r_deorbit, v_deorbit_circular)
        orbital_plane_normal = orbital_plane_normal / max(np.linalg.norm(orbital_plane_normal), 1e-12)
        to_aim_in_plane = to_aim_unit - np.dot(to_aim_unit, orbital_plane_normal) * orbital_plane_normal
        to_aim_in_plane_norm = np.linalg.norm(to_aim_in_plane)
        if to_aim_in_plane_norm > 1e-6:
            to_aim_in_plane = to_aim_in_plane / to_aim_in_plane_norm
        else:
            to_aim_in_plane = retrograde_dir

        burn_dir = 0.6 * retrograde_dir + 0.4 * to_aim_in_plane
        burn_dir = burn_dir / max(np.linalg.norm(burn_dir), 1e-12)
        v_deorbit = v_deorbit_circular + burn_dir * delta_v * self.deorbit_dv

        # Entry interface conditions
        r_EI = R_EARTH + 100e3
        v_EI = np.sqrt(v_a**2 + 2.0 * MU_EARTH * (1.0 / r_EI - 1.0 / r_a))
        cos_gamma_EI = (r_a * v_a) / (r_EI * v_EI)
        gamma_EI = -np.arccos(np.clip(cos_gamma_EI, -1.0, 1.0))

        # ------------------------------------------------------------------
        # Phase 4: Guided reentry using PN guidance
        # ------------------------------------------------------------------
        guided_times, guided_states = simulate_guided_threat(
            r_deorbit, v_deorbit, self._aim_point,
            vehicle=self.vehicle, guidance_law=self.guidance,
            t0=t0_coast + self._coast_duration,
            t_end=self.metadata.get("engagement_end", 1200.0),
        )

        all_t = np.concatenate([boost_t, t0_coast + coast_t_eval, guided_times])
        all_s = np.concatenate([
            np.array(boost_states),
            coast_states,
            guided_states.T,
        ], axis=0)
        self._cache = (all_t, all_s)
        return self._cache

    def propagate(self, t: float) -> np.ndarray:
        all_t, all_s = self._full_trajectory()
        if t >= all_t[-1]:
            return all_s[-1]
        idx = int(np.searchsorted(all_t, t))
        idx = min(max(idx, 1), len(all_t) - 1)
        t0, t1 = all_t[idx - 1], all_t[idx]
        if abs(t1 - t0) < 1e-9:
            return all_s[idx]
        frac = (t - t0) / (t1 - t0)
        return (1.0 - frac) * all_s[idx - 1] + frac * all_s[idx]

    def propagate_batch(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        all_t, all_s = self._full_trajectory()
        idx = np.searchsorted(all_t, times)
        idx = np.clip(idx, 1, len(all_t) - 1)
        t0 = all_t[idx - 1]
        t1 = all_t[idx]
        frac = np.where(np.abs(t1 - t0) < 1e-9, 0.0, (times - t0) / (t1 - t0))
        return (1.0 - frac)[:, None] * all_s[idx - 1] + frac[:, None] * all_s[idx]

    @classmethod
    def from_orbital_params(cls, apoapsis_km: float, inclination_deg: float, launch_site_alt_km: float = 0.0):
        r0 = np.array([R_EARTH + launch_site_alt_km * 1e3, 0.0, 0.0])
        inc = np.radians(inclination_deg)
        r_apo = R_EARTH + apoapsis_km * 1e3
        v0_mag = np.sqrt(MU_EARTH * (2.0 / np.linalg.norm(r0) - 1.0 / r_apo))
        v0 = v0_mag * np.array([0.0, np.cos(inc), np.sin(inc)])
        return cls(r0=r0, v0=v0, apoapsis_km=apoapsis_km, inclination_deg=inclination_deg)


@dataclass
class HGVScenario:
    target_type: str = "hgp"
    r0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    max_alt_km: float = 80.0
    lateral_range_km: float = 2000.0
    ballistic_coeff: float = 100.0
    cd: float = 0.05
    cl: float = 0.3
    skip_threshold_deg: float = 2.0
    use_j2: bool = True
    use_j3: bool = False
    use_j4: bool = False
    use_high_order: bool = False
    use_third_body: bool = False
    use_tides: bool = False
    max_degree: int = 10
    adaptive: bool = False
    atmosphere: Optional[Atmosphere] = None
    wind_model: Optional[Any] = None
    geometry_key: str = "avangard"
    metadata: Dict[str, Any] = field(default_factory=dict)
    _wind_cache_key: Any = field(default=None, repr=False, compare=False)
    _wind_cache_val: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.atmosphere is None:
            self.atmosphere = Atmosphere()

    def _wind_accel(self, r, v, t, dt):
        if self.wind_model is None:
            return np.zeros(3)
        try:
            lat, lon, alt = _ecef_to_geodetic(r)
        except Exception:
            return np.zeros(3)
        if alt > 150e3 or alt < 0.0:
            return np.zeros(3)
        key = (round(float(lat), 4), round(float(lon), 4), round(float(alt), 1), round(float(t), 2))
        if getattr(self, '_wind_cache_key', None) != key:
            U0 = np.asarray(self.wind_model.wind(float(lat), float(lon), float(alt), float(t)), dtype=float)
            dU_dt = np.zeros(3)
            if dt > 1e-6:
                U1 = np.asarray(self.wind_model.wind(float(lat), float(lon), float(alt), float(t) + dt), dtype=float)
                dU_dt = (U1 - U0) / dt
            eps = 1e-4
            U_lat = np.asarray(self.wind_model.wind(float(lat) + eps, float(lon), float(alt), float(t)), dtype=float)
            U_lon = np.asarray(self.wind_model.wind(float(lat), float(lon) + eps, float(alt), float(t)), dtype=float)
            deg2m_lat = 111_132.92
            deg2m_lon = 111_132.92 * np.cos(np.radians(float(lat)))
            a_wind = np.zeros(3)
            for i in range(3):
                a_wind[i] = dU_dt[i] + v[0] * (U_lat[i] - U0[i]) / (deg2m_lat * eps) + v[1] * (U_lon[i] - U0[i]) / (deg2m_lon * eps)
            self._wind_cache_key = key
            self._wind_cache_val = a_wind.tolist()
        return np.array(self._wind_cache_val)

    def _accel(self, r, v, t=None, dt=0.5):
        vmag = np.linalg.norm(v)
        if vmag < 1e-6:
            a_grav = _two_body_accel(r, use_j2=self.use_j2, use_j3=self.use_j3,
                                     use_j4=self.use_j4, use_high_order=self.use_high_order,
                                     use_third_body=self.use_third_body, use_tides=self.use_tides,
                                     max_degree=self.max_degree)
            if t is not None:
                a_grav = a_grav + self._wind_accel(r, v, t, dt)
            return a_grav
        g_unit = -r / np.linalg.norm(r)
        v_unit = v / vmag
        lift_dir = g_unit - np.dot(g_unit, v_unit) * v_unit
        lift_norm = np.linalg.norm(lift_dir)
        if lift_norm > 1e-6:
            lift_dir = lift_dir / lift_norm
        else:
            lift_dir = np.zeros(3)
        a_grav = _two_body_accel(r, use_j2=self.use_j2, use_j3=self.use_j3,
                                 use_j4=self.use_j4, use_high_order=self.use_high_order,
                                 use_third_body=self.use_third_body, use_tides=self.use_tides,
                                 max_degree=self.max_degree)
        alt = _ground_altitude(r)
        a_drag = np.zeros(3)
        a_lift = np.zeros(3)
        if 0.0 < alt < 150e3:
            rho = self.atmosphere.density_scalar(alt)
            q_dyn = 0.5 * rho * vmag**2
            a_drag = -q_dyn * self.cd / self.ballistic_coeff * v_unit
            gamma = np.arcsin(np.clip(v_unit[2], -1.0, 1.0))
            if np.degrees(gamma) > self.skip_threshold_deg:
                a_lift = q_dyn * self.cl / self.ballistic_coeff * lift_dir
        a = a_grav + a_drag + a_lift
        if t is not None:
            a = a + self._wind_accel(r, v, t, dt)
        return a

    def propagate(self, t: float) -> np.ndarray:
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        y0 = np.concatenate([r, v])

        def rhs(ti, y):
            r, v = y[:3], y[3:]
            a = self._accel(r, v, t=ti, dt=0.5)
            return np.concatenate([v, a])

        def ground_event(ti, y):
            return _ground_altitude(y[:3])
        ground_event.terminal = True
        ground_event.direction = -1

        sol = solve_ivp(rhs, (0.0, t), y0, method="RK45",
                        rtol=1e-6, atol=1e-9, max_step=5.0,
                        events=ground_event, dense_output=True)
        if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
            t_end_eff = float(sol.t_events[0][0])
        else:
            t_end_eff = float(sol.t[-1])
        if t_end_eff <= 0.0:
            return np.concatenate([r, v])
        return sol.sol(t_end_eff)[:6]

    def propagate_batch(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        t_max = float(np.max(times)) if times.size > 0 else 0.0
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        y0 = np.concatenate([r, v])

        def rhs(ti, y):
            r, v = y[:3], y[3:]
            a = self._accel(r, v)
            return np.concatenate([v, a])

        def ground_event(ti, y):
            return _ground_altitude(y[:3])
        ground_event.terminal = True
        ground_event.direction = -1

        sol = solve_ivp(rhs, (0.0, t_max), y0, method="RK45",
                        rtol=1e-6, atol=1e-9, max_step=5.0,
                        events=ground_event, dense_output=True)
        if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
            t_ground = float(sol.t_events[0][0])
        else:
            t_ground = float(sol.t[-1])

        t_clamped = np.clip(times, 0.0, t_ground)
        states = sol.sol(t_clamped)[:6, :].T
        return states

    def _propagate_adaptive(self, t: float) -> np.ndarray:
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        y0 = np.concatenate([r, v])

        def rhs(ti, y):
            r, v = y[:3], y[3:]
            a = self._accel(r, v)
            return np.concatenate([v, a])

        def ground_event(ti, y):
            return _ground_altitude(y[:3])
        ground_event.terminal = True
        ground_event.direction = -1

        sol = solve_ivp(rhs, (0.0, t), y0, method="RK45",
                        rtol=1e-6, atol=1e-9, max_step=5.0,
                        events=ground_event, dense_output=True)
        if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
            t_end_eff = float(sol.t_events[0][0])
        else:
            t_end_eff = float(sol.t[-1])
        if t_end_eff <= 0.0:
            return np.concatenate([r, v])
        return sol.sol(t_end_eff)[:6]

    @classmethod
    def from_params(cls, max_alt_km: float, lateral_range_km: float, speed_mach: float = 10.0):
        r0 = np.array([R_EARTH + max_alt_km * 1e3, 0.0, 0.0])
        v_mag = speed_mach * 300.0
        v0 = np.array([0.0, v_mag, 0.0])
        return cls(r0=r0, v0=v0, max_alt_km=max_alt_km, lateral_range_km=lateral_range_km)


@dataclass
class SuppressedScenario:
    target_type: str = "suppressed"
    r0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dip_alt_km: float = 50.0
    midcourse_maneuver_mag: float = 5.0
    maneuver_interval: float = 60.0
    use_j2: bool = True
    use_j3: bool = False
    use_j4: bool = False
    use_high_order: bool = False
    use_third_body: bool = False
    use_tides: bool = False
    max_degree: int = 10
    adaptive: bool = False
    atmosphere: Optional[Atmosphere] = None
    wind_model: Optional[Any] = None
    geometry_key: str = "rs28_sarmat"
    metadata: Dict[str, Any] = field(default_factory=dict)
    _wind_cache_key: Any = field(default=None, repr=False, compare=False)
    _wind_cache_val: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.atmosphere is None:
            self.atmosphere = Atmosphere()
        vmag = np.linalg.norm(self.v0)
        if vmag > 1e-6:
            horiz = self.v0.copy()
            horiz[2] = 0.0
            hmag = np.linalg.norm(horiz)
            self._maneuver_dir = (horiz / hmag) if hmag > 1e-6 else np.array([1.0, 0.0, 0.0])
        else:
            self._maneuver_dir = np.array([1.0, 0.0, 0.0])

    def _wind_accel(self, r, v, t, dt):
        if self.wind_model is None:
            return np.zeros(3)
        try:
            lat, lon, alt = _ecef_to_geodetic(r)
        except Exception:
            return np.zeros(3)
        if alt > 150e3 or alt < 0.0:
            return np.zeros(3)
        key = (round(float(lat), 4), round(float(lon), 4), round(float(alt), 1), round(float(t), 2))
        if getattr(self, '_wind_cache_key', None) != key:
            U0 = np.asarray(self.wind_model.wind(float(lat), float(lon), float(alt), float(t)), dtype=float)
            dU_dt = np.zeros(3)
            if dt > 1e-6:
                U1 = np.asarray(self.wind_model.wind(float(lat), float(lon), float(alt), float(t) + dt), dtype=float)
                dU_dt = (U1 - U0) / dt
            eps = 1e-4
            U_lat = np.asarray(self.wind_model.wind(float(lat) + eps, float(lon), float(alt), float(t)), dtype=float)
            U_lon = np.asarray(self.wind_model.wind(float(lat), float(lon) + eps, float(alt), float(t)), dtype=float)
            deg2m_lat = 111_132.92
            deg2m_lon = 111_132.92 * np.cos(np.radians(float(lat)))
            a_wind = np.zeros(3)
            for i in range(3):
                a_wind[i] = dU_dt[i] + v[0] * (U_lon[i] - U0[i]) / (deg2m_lon * eps) + v[1] * (U_lat[i] - U0[i]) / (deg2m_lat * eps)
            self._wind_cache_key = key
            self._wind_cache_val = a_wind.tolist()
        return np.array(self._wind_cache_val)

    def _accel(self, r, v, t=None, dt=0.5):
        a = _two_body_accel(r, use_j2=self.use_j2, use_j3=self.use_j3,
                            use_j4=self.use_j4, use_high_order=self.use_high_order,
                            use_third_body=self.use_third_body, use_tides=self.use_tides,
                            max_degree=self.max_degree)
        if t is not None:
            a = a + self._wind_accel(r, v, t, dt)
        return a

    def propagate(self, t: float) -> np.ndarray:
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        y0 = np.concatenate([r, v])

        def _ground_event(ti, y):
            return _ground_altitude(y[:3])
        _ground_event.terminal = True
        _ground_event.direction = -1

        def _rhs(ti, y):
            r, v = y[:3], y[3:]
            a = self._accel(r, v, t=ti, dt=0.5)
            return np.concatenate([v, a])

        maneuver_times = []
        if t > 60.0:
            dt = 0.5
            for k in range(int(t / dt)):
                tc = (k + 1) * dt
                if tc > 60.0:
                    m = int(tc / self.maneuver_interval)
                    if m % 2 == 0:
                        maneuver_times.append(tc)
            maneuver_times = np.array(maneuver_times)
            maneuver_times = maneuver_times[maneuver_times <= t]

        current_y = y0.copy()
        t_current = 0.0

        for maneuver_t in maneuver_times:
            if maneuver_t > t:
                break
            alt_before = _ground_altitude(current_y[:3])
            if alt_before < 100e3:
                break
            sol = solve_ivp(
                _rhs,
                (t_current, maneuver_t),
                current_y,
                method="RK45",
                rtol=1e-6, atol=1e-9,
                dense_output=True,
            )
            current_y = sol.y[:, -1].copy()
            v_curr = current_y[3:6]
            v_mag = np.linalg.norm(v_curr)
            if v_mag > 1e-6:
                v_dir = v_curr / v_mag
                r_curr = current_y[:3]
                r_mag = np.linalg.norm(r_curr)
                if r_mag > 1e-6:
                    r_dir = r_curr / r_mag
                    lateral = np.cross(r_dir, v_dir)
                    lateral_mag = np.linalg.norm(lateral)
                    if lateral_mag > 1e-6:
                        lateral = lateral / lateral_mag
                        current_y[3:6] = v_curr + self.midcourse_maneuver_mag * lateral
            t_current = maneuver_t

        sol = solve_ivp(
            _rhs,
            (t_current, t),
            current_y,
            method="RK45",
            rtol=1e-6, atol=1e-9,
            events=[_ground_event],
            dense_output=True,
        )
        if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
            t_end_eff = float(sol.t_events[0][0])
        else:
            t_end_eff = float(sol.t[-1])
        if t_end_eff <= 0.0:
            return np.concatenate([r, v])
        return sol.sol(t_end_eff)[:6]

    def propagate_batch(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        return np.array([self.propagate(ti) for ti in times])

    def _propagate_adaptive(self, t: float) -> np.ndarray:
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        y0 = np.concatenate([r, v])

        def rhs(ti, y):
            r, v = y[:3], y[3:]
            a = self._accel(r, v)
            return np.concatenate([v, a])

        def ground_event(ti, y):
            return _ground_altitude(y[:3])
        ground_event.terminal = True
        ground_event.direction = -1

        sol = solve_ivp(rhs, (0.0, t), y0, method="RK45",
                        rtol=1e-6, atol=1e-9, max_step=5.0,
                        events=ground_event, dense_output=True)
        if sol.t_events[0] is not None and len(sol.t_events[0]) > 0:
            t_end_eff = float(sol.t_events[0][0])
        else:
            t_end_eff = float(sol.t[-1])
        if t_end_eff <= 0.0:
            return np.concatenate([r, v])
        return sol.sol(t_end_eff)[:6]


@dataclass
class SwarmScenario:
    target_type: str = "swarm"
    bus_r0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bus_v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_payloads: int = 3
    spread_deg: float = 2.0
    bus_geometry_key: str = "swarm_bus"
    payload_geometry_keys: List[str] = field(default_factory=lambda: ["rs28_sarmat"])
    wind_model: Optional[Any] = None
    geometry_key: str = "swarm_bus"
    metadata: Dict[str, Any] = field(default_factory=dict)
    _payloads: List = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not self._payloads:
            self._payloads = []
            for i in range(self.n_payloads):
                spread = np.radians(self.spread_deg)
                dr = np.array([
                    np.cos(i * spread),
                    np.sin(i * spread),
                    np.sin(i * spread * 0.5),
                ]) * 100.0
                dv = np.array([
                    -np.sin(i * spread),
                    np.cos(i * spread),
                    0.1 * np.sin(i * spread),
                ]) * 10.0
                geom_key = self.payload_geometry_keys[i % len(self.payload_geometry_keys)]
                payload = BallisticScenario(r0=self.bus_r0 + dr, v0=self.bus_v0 + dv, geometry_key=geom_key)
                payload.wind_model = self.wind_model
                self._payloads.append(payload)

    def propagate(self, t: float) -> np.ndarray:
        return self._payloads[0].propagate(t)

    def propagate_batch(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        return self._payloads[0].propagate_batch(times)

    def payload_states(self, t: float) -> List[np.ndarray]:
        return [p.propagate(t) for p in self._payloads]

    def payload_states_batch(self, times: np.ndarray) -> List[np.ndarray]:
        times = np.asarray(times, dtype=float)
        return [p.propagate_batch(times) for p in self._payloads]

    @classmethod
    def from_params(cls, n_payloads: int, spread_deg: float, range_km: float):
        r0 = np.array([R_EARTH, 0.0, 0.0])
        speed = np.sqrt(2 * MU_EARTH / (R_EARTH + 0.0))
        v0 = np.array([0.0, speed, 0.0])
        return cls(bus_r0=r0, bus_v0=v0, n_payloads=n_payloads, spread_deg=spread_deg)


@dataclass
class ThreatSignatureLibrary:
    """OSINT-approximate threat signatures for discrimination training (2C.2).

    Each entry is a labelled 4-feature sample
    ``[RCS_bias, IR_flux, Doppler_width, micro_motion_flag]`` for an RV (+1) or
    decoy (0). These are illustrative research defaults, NOT controlled data,
    and are consumed by ``guidance.seeker.DiscriminationModel.calibrate``.
    """

    rv_samples: List[np.ndarray] = field(default_factory=list)
    decoy_samples: List[np.ndarray] = field(default_factory=list)

    @classmethod
    def default(cls, n: int = 40, seed: int = 0) -> "ThreatSignatureLibrary":
        rng = np.random.default_rng(seed)
        # RVs: bright, high Doppler, strong micro-motion.
        rv = np.column_stack([
            rng.normal(0.4, 0.15, n),
            rng.normal(1.6, 0.20, n),
            rng.normal(75.0, 8.0, n),
            rng.choice([0.0, 1.0], n, p=[0.2, 0.8]),
        ])
        # Decoys: dim, low Doppler, often no micro-motion.
        decoy = np.column_stack([
            rng.normal(-0.4, 0.12, n),
            rng.normal(0.6, 0.18, n),
            rng.normal(35.0, 7.0, n),
            rng.choice([0.0, 1.0], n, p=[0.85, 0.15]),
        ])
        return cls(rv_samples=[r for r in rv], decoy_samples=[r for r in decoy])

    def labelled_matrix(self) -> tuple:
        """Return ``(features_matrix, labels)`` for calibration."""
        X = np.array(self.rv_samples + self.decoy_samples, dtype=float)
        y = np.array([1] * len(self.rv_samples) + [0] * len(self.decoy_samples))
        return X, y


@dataclass
class DecoyThreatScenario:
    """Threat with one or more decoys released alongside the RV (2C.1).

    ``propagate`` returns the RV 6-vector; ``decoy_states`` returns the live
    decoy states (or None before release). The interceptor's seeker feeds the
    decoy feature vectors into ``DiscriminationModel`` to select the RV.
    """

    target_type: str = "decoy_threat"
    rv: Any = field(default_factory=lambda: BallisticScenario())
    decoys: List[Any] = field(default_factory=list)
    release_t: float = 200.0
    wind_model: Optional[Any] = None
    geometry_key: str = "rs28_sarmat"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from ..targets.decoy_model import DecoyModel
        built = []
        for d in self.decoys:
            if isinstance(d, DecoyModel):
                built.append(d)
            elif isinstance(d, dict):
                built.append(DecoyModel(**d))
        self.decoys = built
        if self.wind_model is not None and hasattr(self.rv, 'wind_model'):
            self.rv.wind_model = self.wind_model

    def propagate(self, t: float) -> np.ndarray:
        return self.rv.propagate(t)

    def propagate_batch(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        if hasattr(self.rv, "propagate_batch"):
            return self.rv.propagate_batch(times)
        return np.array([self.rv.propagate(ti) for ti in times])

    def _release_decoys(self, t: float):
        if t >= self.release_t:
            rv_state = self.rv.propagate(t)
            for d in self.decoys:
                if not d.active:
                    d.release(t, r=rv_state[:3].copy(), v=rv_state[3:].copy())

    def decoy_states(self, t: float) -> List[Optional[np.ndarray]]:
        self._release_decoys(t)
        return [d.state_at(t) for d in self.decoys]

    def decoy_features(self, t: float, seed: int = 0) -> List[np.ndarray]:
        self._release_decoys(t)
        rng = np.random.default_rng(seed)
        return [d.discrimination_features(rng) for d in self.decoys]


@dataclass
class CruiseMissileScenario:
    """Terrain-following cruise missile with boost + cruise + terminal dive (2C.1).

    The trajectory is precomputed and cached on first access, so ``propagate``
    is O(log N) for arbitrary query times.  The model is intentionally
    lightweight (3-DOF point-mass) but captures the three canonical phases:
    (1) rocket boost to cruise altitude/speed, (2) great-circle terrain-following
    cruise at low altitude driven by a simplified turbofan model, and (3) a
    steep terminal dive when the missile is within ``terminal_range_km`` of the
    defended aim point.
    """

    target_type: str = "cruise"
    r0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cruise_alt_m: float = 100.0
    cruise_speed_mach: float = 0.8
    boost_duration: float = 20.0
    boost_thrust: float = 4000.0
    mass_initial: float = 1500.0
    mass_final: float = 1000.0
    isp: float = 3000.0
    terminal_dive_angle_deg: float = 45.0
    terminal_range_km: float = 50.0
    use_j2: bool = True
    use_j3: bool = False
    use_j4: bool = False
    use_high_order: bool = False
    use_third_body: bool = False
    use_tides: bool = False
    max_degree: int = 10
    wind_model: Optional[Any] = None
    geometry_key: str = "yj18"
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cache: Any = field(default=None, repr=False)
    _wind_cache_key: Any = field(default=None, repr=False, compare=False)
    _wind_cache_val: Any = field(default=None, repr=False, compare=False)

    def _wind_accel(self, r, v, t, dt):
        if self.wind_model is None:
            return np.zeros(3)
        try:
            lat, lon, alt = _ecef_to_geodetic(r)
        except Exception:
            return np.zeros(3)
        if alt > 150e3 or alt < 0.0:
            return np.zeros(3)
        key = (round(float(lat), 4), round(float(lon), 4), round(float(alt), 1), round(float(t), 2))
        if getattr(self, '_wind_cache_key', None) != key:
            U0 = np.asarray(self.wind_model.wind(float(lat), float(lon), float(alt), float(t)), dtype=float)
            dU_dt = np.zeros(3)
            if dt > 1e-6:
                U1 = np.asarray(self.wind_model.wind(float(lat), float(lon), float(alt), float(t) + dt), dtype=float)
                dU_dt = (U1 - U0) / dt
            eps = 1e-4
            U_lat = np.asarray(self.wind_model.wind(float(lat) + eps, float(lon), float(alt), float(t)), dtype=float)
            U_lon = np.asarray(self.wind_model.wind(float(lat), float(lon) + eps, float(alt), float(t)), dtype=float)
            deg2m_lat = 111_132.92
            deg2m_lon = 111_132.92 * np.cos(np.radians(float(lat)))
            a_wind = np.zeros(3)
            for i in range(3):
                a_wind[i] = dU_dt[i] + v[0] * (U_lat[i] - U0[i]) / (deg2m_lat * eps) + v[1] * (U_lon[i] - U0[i]) / (deg2m_lon * eps)
            self._wind_cache_key = key
            self._wind_cache_val = a_wind.tolist()
        return np.array(self._wind_cache_val)

    def _compute_trajectory(self):
        if self._cache is not None:
            return self._cache

        dt = 0.5
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        m = self.mass_initial
        vmag = np.linalg.norm(v)
        v_unit = v / max(vmag, 1e-9)
        cruise_speed = self.cruise_speed_mach * 340.0
        atmo = Atmosphere()

        def boost_accel(r, v, ti):
            a_thrust = self.boost_thrust / max(m, 1e-6) * v_unit
            a_grav = _two_body_accel(r, use_j2=self.use_j2, use_j3=self.use_j3,
                                     use_j4=self.use_j4, use_high_order=self.use_high_order,
                                     use_third_body=self.use_third_body, use_tides=self.use_tides,
                                     max_degree=self.max_degree)
            alt = _ground_altitude(r)
            rho = atmo.density_scalar(max(alt, 0.0))
            q_dyn = 0.5 * rho * vmag**2
            a_drag = -q_dyn * 0.3 / max(m, 1e-6) * v_unit
            a = a_grav + a_thrust + a_drag
            if self.wind_model is not None:
                a = a + self._wind_accel(r, v, ti, dt)
            return a

        # --- Boost phase: rocket motor ---
        t_boost = np.arange(0.0, self.boost_duration + dt, dt)
        boost_states = []
        for ti in t_boost:
            boost_states.append(np.concatenate([r.copy(), v.copy()]))
            r, v = _rk4_step(r, v, dt, lambda ri, vi: boost_accel(ri, vi, ti))
            mdot = self.boost_thrust / (self.isp * 9.81)
            m = max(m - mdot * dt, self.mass_final)
            vmag = np.linalg.norm(v)
            v_unit = v / max(vmag, 1e-9)
        boost_states.append(np.concatenate([r.copy(), v.copy()]))

        # --- Transition to cruise: align with great-circle path ---
        cruise_dir = v / max(np.linalg.norm(v), 1e-9)
        cruise_v = cruise_dir * cruise_speed
        r_cruise = r.copy()
        r_cruise = r_cruise / np.linalg.norm(r_cruise) * (R_EARTH + self.cruise_alt_m)

        # --- Cruise phase: great-circle at constant altitude ---
        cruise_t = np.arange(0.0, 600.0, dt)
        cruise_states = []
        for ti in cruise_t:
            cruise_states.append(np.concatenate([r_cruise.copy(), cruise_v.copy()]))
            a_grav = _two_body_accel(r_cruise, use_j2=self.use_j2, use_j3=self.use_j3,
                                     use_j4=self.use_j4, use_high_order=self.use_high_order,
                                     use_third_body=self.use_third_body, use_tides=self.use_tides,
                                     max_degree=self.max_degree)
            a = a_grav
            if self.wind_model is not None:
                a = a + self._wind_accel(r_cruise, cruise_v, ti, dt)
            r_cruise = r_cruise + cruise_v * dt + 0.5 * a * dt * dt
            r_cruise = r_cruise / np.linalg.norm(r_cruise) * (R_EARTH + self.cruise_alt_m)
            radial = r_cruise / np.linalg.norm(r_cruise)
            cruise_v = cruise_v - np.dot(cruise_v, radial) * radial
            vnorm = np.linalg.norm(cruise_v)
            if vnorm > 1e-6:
                cruise_v = cruise_v / vnorm * cruise_speed

        # --- Terminal dive ---
        dive_angle = np.radians(self.terminal_dive_angle_deg)
        dive_t = np.arange(0.0, 60.0, dt)
        dive_states = []

        def dive_accel(r, v, ti):
            radial = r / max(np.linalg.norm(r), 1e-9)
            horiz = v - np.dot(v, radial) * radial
            hnorm = np.linalg.norm(horiz)
            if hnorm > 1e-6:
                horiz = horiz / hnorm
            else:
                horiz = np.cross(radial, np.array([0.0, 0.0, 1.0]))
                hnorm = np.linalg.norm(horiz)
                if hnorm > 1e-6:
                    horiz = horiz / hnorm
                else:
                    horiz = np.array([1.0, 0.0, 0.0])
            dive_v = horiz * np.cos(dive_angle) + radial * (-np.sin(dive_angle))
            dive_v = dive_v / max(np.linalg.norm(dive_v), 1e-9) * cruise_speed
            a_grav = _two_body_accel(r, use_j2=self.use_j2, use_j3=self.use_j3,
                                     use_j4=self.use_j4, use_high_order=self.use_high_order,
                                     use_third_body=self.use_third_body, use_tides=self.use_tides,
                                     max_degree=self.max_degree)
            a = a_grav + dive_v * 3.0
            if self.wind_model is not None:
                a = a + self._wind_accel(r, v, ti, dt)
            return a

        for ti in dive_t:
            dive_states.append(np.concatenate([r_cruise.copy(), cruise_v.copy()]))
            r_cruise, cruise_v = _rk4_step(r_cruise, cruise_v, dt, lambda ri, vi: dive_accel(ri, vi, ti))

        all_t = np.concatenate([
            t_boost,
            self.boost_duration + cruise_t,
            self.boost_duration + 600.0 + dive_t,
        ])
        all_s = np.array(boost_states + cruise_states + dive_states)
        self._cache = (all_t, all_s)
        return self._cache

    def propagate(self, t: float) -> np.ndarray:
        all_t, all_s = self._compute_trajectory()
        idx = int(np.searchsorted(all_t, t))
        idx = min(max(idx, 1), len(all_t) - 1)
        t0, t1 = all_t[idx - 1], all_t[idx]
        if abs(t1 - t0) < 1e-9:
            return all_s[idx]
        frac = (t - t0) / (t1 - t0)
        return (1.0 - frac) * all_s[idx - 1] + frac * all_s[idx]

    def propagate_batch(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        all_t, all_s = self._compute_trajectory()
        idx = np.searchsorted(all_t, times)
        idx = np.clip(idx, 1, len(all_t) - 1)
        t0 = all_t[idx - 1]
        t1 = all_t[idx]
        frac = np.where(np.abs(t1 - t0) < 1e-9, 0.0, (times - t0) / (t1 - t0))
        return (1.0 - frac)[:, None] * all_s[idx - 1] + frac[:, None] * all_s[idx]

    @classmethod
    def from_params(cls, launch_alt_km: float = 0.0, range_km: float = 500.0,
                    speed_mach: float = 0.8):
        r0 = np.array([R_EARTH + launch_alt_km * 1e3, 0.0, 0.0])
        v0 = np.array([0.0, speed_mach * 340.0, 0.0])
        return cls(r0=r0, v0=v0, cruise_speed_mach=speed_mach)


@dataclass
class SarmatScenario(FOBSScenario):
    """RS-28 Sarmat-specific threat profile.

    Extends :class:`FOBSScenario` with:
    - Shortened/mortar boost phase with TVC guidance corrections
    - PBB warhead/decoy release events
    - Midcourse trajectory alterations
    - Terminal PN-guided reentry
    """

    target_type: str = "sarmat"
    geometry_key: str = "rs28_sarmat"
    boost_profile: str = "shortened"  # "standard" | "shortened" | "mortar"
    boost_duration_s: float = 342.0
    thrust_profile: Optional[List[Tuple[float, float]]] = None
    mortar_pop_height_m: float = 20.0
    ir_signature_scale: float = 0.3
    midcourse_profile: str = "standard"  # "standard" | "depressed" | "fobs-loop"
    midcourse_alteration_delta_v: float = 0.0
    midcourse_alteration_times: Optional[np.ndarray] = None
    pbb_warheads: int = 10
    pbb_decoys: int = 50
    pbb_release_alt_km: float = 120.0
    decoy_release_times: Optional[np.ndarray] = None

    _warheads: List = field(default_factory=list, repr=False)
    _decoys: List = field(default_factory=list, repr=False)
    _pbb_released: bool = False
    _boost_end_state: Optional[np.ndarray] = field(default=None, repr=False)
    initial_thrust_dir: Optional[np.ndarray] = field(default=None, repr=False)
    atmosphere: Optional[Atmosphere] = None
    cd: float = 0.3
    area: float = 7.07
    mass: float = 208100.0

    # DC defended target (actual aim point for Sarmat)
    _TARGET_LAT: float = 38.90
    _TARGET_LON: float = -77.04
    _TARGET_ALT: float = 31.0

    # RS-28 Sarmat stage breakdown from authoritative sources:
    # - Stage 1: PDU-99 derived from RD-274 (~4,952 kN)
    # - Stage 2: RD-250 class main + verniers (~1,200 kN estimated)
    # - Stage 3: Four RS-99 engines, "over 100 tons thrust" (~1,000+ kN)
    # Refs: Astronautica RD-274 (4,952 kN), Missile Defense Advocacy (PDU-99),
    #       Army Recognition ("four engines ... over 100 tons thrust")
    _SARMAT_STAGES = [
        {"t_start": 0.0,   "t_end": 90.0,  "thrust": 5.0e6,   "m_dot": 1430.0},
        {"t_start": 90.0,  "t_end": 270.0, "thrust": 1.2e6,   "m_dot": 247.0},
        {"t_start": 270.0, "t_end": 360.0, "thrust": 1.0e6,   "m_dot": 88.0},
    ]

    def __post_init__(self):
        super().__post_init__()
        self._pbb_released = False
        self._warheads = []
        self._decoys = []
        self._boost_end_state = None
        self._initial_thrust_dir = None
        if self.atmosphere is None:
            self.atmosphere = Atmosphere()
        if self.decoy_release_times is None:
            spread = np.linspace(0.0, 60.0, self.pbb_decoys)
            self.decoy_release_times = spread
        
        if self.initial_thrust_dir is not None:
            self._initial_thrust_dir = self.initial_thrust_dir / np.linalg.norm(self.initial_thrust_dir)
        
        # Override aim point to exact defended target coordinates
        if hasattr(self, 'r0'):
            self._aim_point = _geodetic_to_ecef_simple(
                self._TARGET_LAT, self._TARGET_LON, self._TARGET_ALT
            )
        
        self._era5_loaded = False
        self._era5_interpolator = None

    def _init_icbm_guidance(self):
        from ..guidance.precomputed_trajectory import SarmatTrajectoryLibrary

        self._target_ecef = _geodetic_to_ecef_simple(
            self._TARGET_LAT, self._TARGET_LON, self._TARGET_ALT
        )

        library = SarmatTrajectoryLibrary(launch_ecef=self.r0)
        self._profile = library.get_or_compute(
            target_ecef=self._target_ecef,
            profile_name="standard",
            stages=self._SARMAT_STAGES,
            mass_initial=self._initial_mass(),
            cd=self.cd,
            area=self.area,
        )

    def _ensure_era5(self):
        if not self._era5_loaded and self.wind_model is None:
            self._era5_loaded = True
            try:
                era5_root = "/mnt/user/public/project-icarus/reference/ERA5"
                import glob
                gribs = sorted(glob.glob(f"{era5_root}/era5_global_native_025_2015_*.grib"))
                if gribs:
                    from ..reference.era5 import ERA5Interpolator
                    self._era5_interpolator = ERA5Interpolator(gribs)
            except Exception:
                pass

    def _wind_accel(self, r, v, t, dt):
        self._ensure_era5()
        if self._era5_interpolator is not None:
            wind = self._era5_interpolator
            try:
                lat, lon, alt = _ecef_to_geodetic(r)
                U = np.asarray(wind(float(lat), float(lon), float(alt), float(t)), dtype=float)
                dU_dt = np.zeros(3)
                if dt > 1e-6:
                    U1 = np.asarray(wind(float(lat), float(lon), float(alt), float(t) + dt), dtype=float)
                    dU_dt = (U1 - U) / dt
                eps = 1e-4
                U_lat = np.asarray(wind(float(lat) + eps, float(lon), float(alt), float(t)), dtype=float)
                U_lon = np.asarray(wind(float(lat), float(lon) + eps, float(alt), float(t)), dtype=float)
                deg2m_lat = 111_132.92
                deg2m_lon = 111_132.92 * np.cos(np.radians(float(lat)))
                a_wind = np.zeros(3)
                for i in range(3):
                    a_wind[i] = dU_dt[i] + v[0] * (U_lon[i] - U[i]) / (deg2m_lon * eps) + v[1] * (U_lat[i] - U[i]) / (deg2m_lat * eps)
                return a_wind
            except Exception:
                pass
        return np.zeros(3)

    def _initial_mass(self) -> float:
        return 208100.0

    def _current_thrust_dir(self, t, r, v):
        if not hasattr(self, '_profile'):
            self._init_icbm_guidance()
        return self._profile.thrust_direction(t, r, v)

    def _boost_guidance_correction(self, t, r, v):
        """Deprecated: guidance now applied directly via _current_thrust_dir."""
        return np.zeros(3)

    def _rhs_full(self, t, y):
        """Full 3-DOF RHS with multi-stage thrust and mass tracking."""
        r, v = y[:3], y[3:]
        m = y[6] if len(y) > 6 else self.mass

        # Multi-stage thrust magnitude
        T_mag = 0.0
        dm_dt = 0.0
        if self._thrust_model is not None:
            try:
                state = {"r": r, "v": v, "q": np.array([1.0, 0.0, 0.0, 0.0]),
                         "omega": np.zeros(3), "m": m}
                T_mag = float(self._thrust_model.thrust(t, state))
                dm_dt = float(self._thrust_model.mass_rate(t, state))
            except Exception:
                pass

        # Thrust direction in inertial frame
        thrust_dir = self._current_thrust_dir(t, r, v)
        a_thrust = (T_mag / max(m, 1e-6)) * thrust_dir

        a_grav = _two_body_accel(r, use_j2=self.use_j2, use_j3=self.use_j3,
                                 use_j4=self.use_j4, use_high_order=self.use_high_order,
                                 use_third_body=self.use_third_body, use_tides=self.use_tides,
                                 max_degree=self.max_degree, t=t)
        a_drag = np.zeros(3)
        alt = _ground_altitude(r)
        if 0.0 < alt < 100e3:
            rho = self.atmosphere.density_scalar(alt)
            vmag = np.linalg.norm(v)
            if vmag > 1e-6:
                q = 0.5 * rho * vmag**2
                a_drag = -q * self.cd * self.area / m * (v / vmag)

        # Boost-phase guidance: correct trajectory toward target during boost
        a_guidance = np.zeros(3)
        if self.boost_profile == "shortened" and t < self.boost_duration_s:
            a_guidance = self._boost_guidance_correction(t, r, v, T_mag=T_mag, m=m)

        a_total = a_grav + a_drag + a_thrust
        if self.wind_model is not None and 0 < alt < 150e3:
            a_total = a_total + self._wind_accel(r, v, t, 0.5)
        return np.concatenate([v, a_total, [dm_dt]])

    def _thrust_accel(self, t):
        if self.thrust_profile is not None:
            for t_start, t_end, T in self.thrust_profile:
                if t_start <= t < t_end:
                    return np.array([T, 0.0, 0.0])
        return np.zeros(3)

    def _ground_event(self, t, y):
        return _ground_altitude(y[:3])
    _ground_event.terminal = True
    _ground_event.direction = -1

    def _boost_end_event(self, t, y):
        return t - self.boost_duration_s
    _boost_end_event.terminal = True
    _boost_end_event.direction = 0

    def _integrate_full(self):
        """Manual RK4 integration with explicit RS-28 stage thrust/mass."""
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        m = self._initial_mass()
        t_current = 0.0
        hit_ground = False
        all_times = [0.0]
        all_states = [np.concatenate([r.copy(), v.copy(), [m]])]

        while not hit_ground and t_current < 4000.0:
            alt = _ground_altitude(r)
            if alt < -5.0 and t_current > 1.0:
                hit_ground = True
                break

            dt = 0.05 if t_current < 200.0 else 0.5

            def rhs_tmp(rr, vv, mm, tt):
                T_mag = 0.0
                dm_dt_val = 0.0
                for stage in self._SARMAT_STAGES:
                    if stage["t_start"] <= tt < stage["t_end"]:
                        T_mag = stage["thrust"]
                        dm_dt_val = -stage["m_dot"]
                        break

                thrust_dir = self._current_thrust_dir(tt, rr, vv)
                a_thrust = (T_mag / max(mm, 1e-6)) * thrust_dir

                a_grav = _two_body_accel(rr, use_j2=self.use_j2, use_j3=self.use_j3,
                                         use_j4=self.use_j4, use_high_order=self.use_high_order,
                                         use_third_body=self.use_third_body, use_tides=self.use_tides,
                                         max_degree=self.max_degree, t=tt)
                a_drag = np.zeros(3)
                alt = _ground_altitude(rr)
                if 0.0 < alt < 100e3:
                    rho = self.atmosphere.density_scalar(alt)
                    vmag = np.linalg.norm(vv)
                    if vmag > 1e-6:
                        q = 0.5 * rho * vmag**2
                        a_drag = -q * self.cd * self.area / mm * (vv / vmag)

                a_total = a_grav + a_drag + a_thrust
                if self.wind_model is not None and 0 < alt < 150e3:
                    a_total = a_total + self._wind_accel(rr, vv, tt, dt)
                return np.concatenate([vv, a_total, [dm_dt_val]])

            y = np.concatenate([r, v, [m]])
            k1 = rhs_tmp(r, v, m, t_current)
            k2 = rhs_tmp(r + 0.5*dt*k1[:3], v + 0.5*dt*k1[3:6], m + 0.5*dt*k1[6], t_current + 0.5*dt)
            k3 = rhs_tmp(r + 0.5*dt*k2[:3], v + 0.5*dt*k2[3:6], m + 0.5*dt*k2[6], t_current + 0.5*dt)
            k4 = rhs_tmp(r + dt*k3[:3], v + dt*k3[3:6], m + dt*k3[6], t_current + dt)

            r = r + (dt/6.0) * (k1[:3] + 2*k2[:3] + 2*k3[:3] + k4[:3])
            v = v + (dt/6.0) * (k1[3:6] + 2*k2[3:6] + 2*k3[3:6] + k4[3:6])
            m = m + (dt/6.0) * (k1[6] + 2*k2[6] + 2*k3[6] + k4[6])
            t_current += dt

            if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)) or not np.isfinite(m):
                break
            if m < 1e-3:
                m = 0.0
                break

            all_times.append(t_current)
            all_states.append(np.concatenate([r.copy(), v.copy(), [m]]))

        arr = np.array(all_states)
        if arr.shape[0] == 0:
            return np.array([0.0]), np.array([np.concatenate([self.r0, self.v0, [self._initial_mass()]])])
        return np.array(all_times), arr

    def propagate(self, t: float) -> np.ndarray:
        if hasattr(self, '_cache') and self._cache is not None:
            all_t, all_s = self._cache
            if t >= all_t[-1]:
                return all_s[-1, :6]
            idx = int(np.searchsorted(all_t, t))
            idx = min(max(idx, 1), len(all_t) - 1)
            t0, t1 = all_t[idx-1], all_t[idx]
            if abs(t1 - t0) < 1e-12:
                return all_s[idx, :6]
            alpha = (t - t0) / (t1 - t0)
            return (1 - alpha) * all_s[idx-1, :6] + alpha * all_s[idx, :6]

        all_t, all_s = self._integrate_full()
        self._cache = (all_t, all_s)
        if t >= all_t[-1]:
            return all_s[-1, :6]
        idx = int(np.searchsorted(all_t, t))
        idx = min(max(idx, 1), len(all_t) - 1)
        t0, t1 = all_t[idx-1], all_t[idx]
        if abs(t1 - t0) < 1e-12:
            return all_s[idx, :6]
        alpha = (t - t0) / (t1 - t0)
        return (1 - alpha) * all_s[idx-1, :6] + alpha * all_s[idx, :6]

    def propagate_batch(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        return np.array([self.propagate(ti) for ti in times])

    def _release_pbb(self, r_bus, v_bus, coast_states, coast_times):
        """Populate lightweight warhead and decoy scenarios from bus state."""
        self._pbb_released = True
        bus_alt = np.linalg.norm(r_bus) - R_EARTH
        if bus_alt < self.pbb_release_alt_km * 1e3:
            return
        n = max(self.pbb_warheads, 1)
        spread_deg = 2.0
        for i in range(n):
            angle = np.radians(i * spread_deg)
            dr = np.array([np.cos(angle), np.sin(angle), 0.0]) * 50.0
            dv = np.array([-np.sin(angle), np.cos(angle), 0.0]) * 5.0
            self._warheads.append(BallisticScenario(
                r0=r_bus + dr, v0=v_bus + dv,
                geometry_key="threat_rv",
                mass=800.0, area=0.02, cd=0.3,
                use_j2=self.use_j2,
            ))
        for i in range(max(self.pbb_decoys, 0)):
            angle = np.radians(i * 3.0 + 0.5)
            dr = np.array([np.cos(angle), np.sin(angle), 0.1 * np.sin(angle)]) * 80.0
            dv = np.array([-np.sin(angle), np.cos(angle), 0.05 * np.cos(angle)]) * 8.0
            self._decoys.append(BallisticScenario(
                r0=r_bus + dr, v0=v_bus + dv,
                geometry_key="signature_balloon_decoy",
                mass=30.0, area=0.5, cd=1.2,
                use_j2=self.use_j2,
            ))

    def warhead_states(self, t: float) -> List[np.ndarray]:
        if not self._pbb_released:
            self._full_trajectory()
        return [w.propagate(t) for w in self._warheads]

    def decoy_states(self, t: float) -> List[np.ndarray]:
        if not self._pbb_released:
            self._full_trajectory()
        return [d.propagate(t) for d in self._decoys]

    def discrimination_features(self, t: float, seed: int = 0) -> List[np.ndarray]:
        rng = np.random.default_rng(seed)
        feats = []
        for w in self._warheads:
            feats.append(np.array([1.0, 1.0, 45.0, 1.0]) + rng.normal(0.0, 0.1, 4))
        for d in self._decoys:
            feats.append(np.array([0.4, 0.6, 35.0, 0.2]) + rng.normal(0.0, 0.1, 4))
        return feats


@dataclass
class EngagementScenario:
    """Scenario definition for an engagement."""
    name: str = "Default"
    interceptor_launch_site: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_launch_site: np.ndarray = field(default_factory=lambda: np.zeros(3))
    threat_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    engagement_start: float = 0.0
    engagement_end: float = 300.0
    sensor_noise: float = 0.01
    params: Dict[str, Any] = field(default_factory=dict)
