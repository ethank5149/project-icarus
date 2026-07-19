from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Protocol, runtime_checkable
import numpy as np
from scipy.integrate import solve_ivp

from ..dynamics.eom_6dof import EOM6DOF
from ..guidance.terminal_guidance import TerminalGuidance
from ..aero.aero_analytical import blended_aero


MU_EARTH = 3.986004418e14
R_EARTH = 6371e3
J2_EARTH = 1.08263e-3


def _two_body_accel(r, use_j2=True):
    r = np.asarray(r, dtype=float)
    rmag = np.linalg.norm(r)
    if rmag < 1e-6:
        return np.zeros(3)
    a = -MU_EARTH / rmag**3 * r
    if use_j2 and rmag - R_EARTH > 50e3:
        z = r[2]
        factor = -MU_EARTH / (rmag**5) * (
            1.0 + 1.5 * J2_EARTH * (R_EARTH / rmag)**2 * (5.0 * (z / rmag)**2 - 1.0)
        )
        a = factor * r
    return a


# --- Spherical-Earth geodetic helpers (for aim-point computation) ----------
# These are intentionally lightweight (spherical Earth, not WGS84) so the
# target factory has no import cycle with ``scenarios.presets``.

def _ecef_to_geodetic(r):
    x, y, z = r
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    alt = np.linalg.norm(r) - R_EARTH
    return float(lat), float(lon), float(alt)


def _geodetic_to_ecef_simple(lat_deg, lon_deg, alt_m=0.0):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = R_EARTH + alt_m
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat),
    ])


def _destination_point(lat_deg, lon_deg, az_deg, dist_deg):
    """Spherical destination point given start, azimuth (deg, 0=N CW), arc deg."""
    lat1 = np.radians(lat_deg)
    lon1 = np.radians(lon_deg)
    az = np.radians(az_deg)
    d = np.radians(dist_deg)
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
        return np.linalg.norm(y[:3]) - R_EARTH

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

    grid = sol.sol(sol_t)
    six_vec = np.vstack([grid[:3, :], grid[3:6, :]])
    return sol_t, six_vec


@runtime_checkable
class TargetScenario(Protocol):
    target_type: str
    metadata: Dict[str, Any]

    def propagate(self, t: float) -> np.ndarray: ...


@dataclass
class BallisticScenario:
    target_type: str = "ballistic"
    r0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    use_j2: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def propagate(self, t: float) -> np.ndarray:
        # Closed-form two-body propagation via the universal-variable Kepler
        # algorithm. This is O(1) in t (a single Kepler solve per call) rather
        # than a per-step loop, which matters because ``propagate`` is invoked
        # thousands of times per engagement (once per integrator step). The J2
        # perturbation is intentionally ignored here for speed; the threat
        # trajectory is a kinematic reference path fed into the closed-loop RHS.
        return _kepler_propagate(self.r0.astype(float), self.v0.astype(float), t, MU_EARTH)


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
    deorbit_dv: float = 0.15
    vehicle: Any = None
    guidance: Any = None
    use_j2: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cache: Any = field(default=None, repr=False)

    def __post_init__(self):
        # Aim point: ground location the boost azimuth points to (used by the
        # reentry phase so a geodetically-aimed FOBS converges on its target
        # instead of a hardcoded (R_EARTH + 100km, 0, 0) point).
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
            # Downrange at apogee (~half the orbital coast arc) along great circle.
            coast_arc_deg = (self._coast_duration * np.linalg.norm(self.v0) / rmag) / np.pi * 180.0
            aim_lat, aim_lon = _destination_point(lat0, lon0, az, coast_arc_deg * 0.5)
            self._aim_point = _geodetic_to_ecef_simple(aim_lat, aim_lon, 0.0)
        else:
            self._aim_point = np.array([R_EARTH + self._reentry_alt, 0.0, 0.0])
        self._cache = None

    def _orbital_speed(self, r):
        return np.sqrt(MU_EARTH / max(r, 1e-6))

    def _full_trajectory(self):
        if self._cache is not None:
            return self._cache
        # Boost + coast modeled as an impulsive transfer onto the parking orbit
        # (r0, v0) using Keplerian two-body propagation up to the coast end, then
        # a retrograde deorbit burn at the coast-end point. This places the reentry
        # start at the correct apogee altitude rather than the unphysical
        # forward-Euler boost endpoint.
        dt_boost = 0.5
        boost_t = np.arange(0.0, self._boost_duration + dt_boost, dt_boost)
        boost_states = []
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        for _ in boost_t:
            boost_states.append(np.concatenate([r.copy(), v.copy()]))
            a = _two_body_accel(r) + np.array([self.thrust, 0.0, 0.0])
            v = v + a * dt_boost
            r = r + v * dt_boost
        boost_states.append(np.concatenate([r.copy(), v.copy()]))

        # Coast: Keplerian two-body propagation from the boost-end state.
        r_coast = boost_states[-1][:3].copy()
        v_coast = boost_states[-1][3:].copy()
        dt_coast = 0.5
        coast_t = np.arange(0.0, self._coast_duration + dt_coast, dt_coast)
        coast_states = []
        rc, vc = r_coast.copy(), v_coast.copy()
        for _ in coast_t:
            coast_states.append(np.concatenate([rc.copy(), vc.copy()]))
            a = _two_body_accel(rc, use_j2=self.use_j2)
            vc = vc + a * dt_coast
            rc = rc + vc * dt_coast
        coast_states.append(np.concatenate([rc.copy(), vc.copy()]))

        # Deorbit: the FOBS loiters in a parking orbit at the apogee altitude,
        # then performs a retrograde burn that drops perigee onto the aim point.
        # The deorbit state is placed on a near-collision course with the aim
        # point (velocity directed toward the aim, with a small retrograde
        # component so the trajectory actually enters the atmosphere). PN
        # guidance then trims the residual error during the guided reentry.
        r_park = R_EARTH + self.apoapsis_km * 1e3
        v_park = np.sqrt(MU_EARTH / r_park)

        # Deorbit position: a parking-orbit point sampled along the launch
        # azimuth so that a burn aimed at the aim point yields a closing geometry.
        rhat = self.r0 / max(np.linalg.norm(self.r0), 1e-6)
        r_deorbit = r_park * rhat

        # Velocity on a collision course toward the aim point, scaled to the
        # parking speed and reduced by the retrograde deorbit fraction so the RV
        # leaves orbit and descends. The aim point is on the surface, so this
        # naturally carries a downward component.
        to_aim = self._aim_point - r_deorbit
        to_aim_unit = to_aim / max(np.linalg.norm(to_aim), 1e-6)
        v_deorbit = v_park * (1.0 - self.deorbit_dv) * to_aim_unit

        guided_times, guided_states = simulate_guided_threat(
            r_deorbit, v_deorbit, self._aim_point,
            vehicle=self.vehicle, guidance_law=self.guidance,
            t0=self._boost_duration + self._coast_duration,
            t_end=self.metadata.get("engagement_end", 1200.0),
        )

        all_t = np.concatenate([boost_t, self._boost_duration + coast_t, guided_times])
        all_s = np.concatenate([
            np.array(boost_states),
            np.array(coast_states),
            guided_states.T,
        ], axis=0)
        self._cache = (all_t, all_s)
        return self._cache

    def propagate(self, t: float) -> np.ndarray:
        all_t, all_s = self._full_trajectory()
        idx = int(np.searchsorted(all_t, t))
        idx = min(max(idx, 1), len(all_t) - 1)
        t0, t1 = all_t[idx - 1], all_t[idx]
        if abs(t1 - t0) < 1e-9:
            return all_s[idx]
        frac = (t - t0) / (t1 - t0)
        return (1.0 - frac) * all_s[idx - 1] + frac * all_s[idx]

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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def propagate(self, t: float) -> np.ndarray:
        # 3-DOF point-mass glide: gravity + drag + lift (perpendicular to v).
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        dt = 0.5
        n = max(int(t / dt), 1)
        for _ in range(n):
            vmag = np.linalg.norm(v)
            if vmag < 1e-6:
                break
            g_unit = -r / np.linalg.norm(r)
            v_unit = v / vmag
            # Lift acts perpendicular to velocity in the vertical plane (z component).
            lift_dir = g_unit - np.dot(g_unit, v_unit) * v_unit
            lift_dir = lift_dir / max(np.linalg.norm(lift_dir), 1e-6)
            a_grav = -MU_EARTH / np.linalg.norm(r)**3 * r
            alt = np.linalg.norm(r) - R_EARTH
            a_drag = np.zeros(3)
            a_lift = np.zeros(3)
            if alt < 150e3:
                rho = 1.225 * np.exp(-alt / 8500.0)
                q_dyn = 0.5 * rho * vmag**2
                a_drag = -q_dyn * self.cd / self.ballistic_coeff * v_unit
                # Skip-glide: pull up when flight-path angle exceeds threshold.
                gamma = np.arcsin(np.clip(v_unit[2], -1.0, 1.0))
                if np.degrees(gamma) > self.skip_threshold_deg:
                    a_lift = q_dyn * self.cl / self.ballistic_coeff * lift_dir
            a = a_grav + a_drag + a_lift
            v = v + a * dt
            r = r + v * dt
        return np.concatenate([r, v])

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
    midcourse_maneuver_mag: float = 50.0
    maneuver_interval: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Maneuver direction: horizontal component of the launch velocity
        # (threat axis), so midcourse jinking stays aligned with the aim point
        # rather than a hardcoded +y body axis.
        vmag = np.linalg.norm(self.v0)
        if vmag > 1e-6:
            horiz = self.v0.copy()
            horiz[2] = 0.0
            hmag = np.linalg.norm(horiz)
            self._maneuver_dir = (horiz / hmag) if hmag > 1e-6 else np.array([1.0, 0.0, 0.0])
        else:
            self._maneuver_dir = np.array([1.0, 0.0, 0.0])

    def propagate(self, t: float) -> np.ndarray:
        # Inverse-square gravity with midcourse delta-v impulses.
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        dt = 0.5
        n = max(int(t / dt), 1)
        applied = set()
        for k in range(n):
            tc = (k + 1) * dt
            a = _two_body_accel(r)
            v = v + a * dt
            r = r + v * dt
            # Delta-v impulse (velocity change, not addition) at maneuver times,
            # applied along the threat axis (horizontal launch direction).
            if tc > 60.0:
                m = int(tc / self.maneuver_interval)
                if m % 2 == 0 and m not in applied:
                    v = v + self.midcourse_maneuver_mag * self._maneuver_dir
                    applied.add(m)
        return np.concatenate([r, v])


@dataclass
class SwarmScenario:
    target_type: str = "swarm"
    bus_r0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bus_v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_payloads: int = 3
    spread_deg: float = 2.0
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
                self._payloads.append(
                    BallisticScenario(r0=self.bus_r0 + dr, v0=self.bus_v0 + dv)
                )

    def propagate(self, t: float) -> np.ndarray:
        return self._payloads[0].propagate(t)

    def payload_states(self, t: float) -> List[np.ndarray]:
        return [p.propagate(t) for p in self._payloads]

    @classmethod
    def from_params(cls, n_payloads: int, spread_deg: float, range_km: float):
        r0 = np.array([R_EARTH, 0.0, 0.0])
        speed = np.sqrt(2 * MU_EARTH / (R_EARTH + 0.0))
        v0 = np.array([0.0, speed, 0.0])
        return cls(bus_r0=r0, bus_v0=v0, n_payloads=n_payloads, spread_deg=spread_deg)


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
