"""Physically realistic ICBM guidance system.

Implements the same guidance architecture used by real ICBMs:

1. Pre-launch targeting (computes azimuth and required burnout state)
2. Inertial Navigation System (INS) - integrates accelerometers/gyros
3. Programmed pitch maneuver (time/energy-based pitch schedule)
4. Gravity turn (thrust aligned with velocity, zero angle of attack)
5. Trajectory correction updates (optional midcourse/terminal)

Reference: US/Russian ICBM guidance fundamentals, "Fundamentals of Ballistic
Missile Trajectories" (GAO/NSIA), and open-literature INS/GNSS guidance.
"""

import numpy as np
from typing import Optional, Tuple

from .lambert import lambert, required_burnout_velocity_lambert

MU_EARTH = 3.986004418e14
R_EARTH = 6371e3
G0 = 9.80665

# Standard gravitational parameter and Earth rotation rate
_OMEGA_EARTH = 7.292115e-5  # rad/s


def _two_body_accel_j2(r: np.ndarray) -> np.ndarray:
    """Two-body gravity with J2 perturbation."""
    r = np.asarray(r, dtype=float)
    rmag = np.linalg.norm(r)
    g = -MU_EARTH / rmag**3 * r
    j2 = 1.08263e-3
    r2 = rmag**2
    r5 = rmag**5
    factor = 1.5 * j2 * (R_EARTH**2) / r5
    g = g + factor * r * (5.0 * r[2]**2 / r2 - 1.0)
    g = g + factor * np.array([r[0], r[1], -3.0 * r[2]])
    return g


def _ecef_to_geodetic(r: np.ndarray) -> Tuple[float, float, float]:
    """Convert ECEF to WGS84 geodetic coordinates."""
    x, y, z = np.asarray(r, dtype=float)
    lon = np.degrees(np.arctan2(y, x))
    p = np.sqrt(x**2 + y**2)
    if p < 1e-6:
        lat = 90.0 if z > 0 else -90.0
        return float(lat), float(lon), float(abs(z) - (6378137.0 * (1.0 - 1.0 / 298.257223563)))
    a = 6378137.0
    b = 6378137.0 * (1.0 - 1.0 / 298.257223563)
    e2 = (a**2 - b**2) / a**2
    e_prime2 = (a**2 - b**2) / b**2
    theta = np.arctan2(z * a, p * b)
    lat = np.arctan2(
        z + e_prime2 * b * np.sin(theta)**3,
        p - e2 * a * np.cos(theta)**3,
    )
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1.0 - e2 * sin_lat**2)
    alt = p / np.cos(lat) - N
    return float(np.degrees(lat)), float(lon), float(alt)


def _enu_basis(lat_deg: float, lon_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local East/North/Up unit vectors (ECEF)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    east = np.array([-sin_lon, cos_lon, 0.0])
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    return east, north, up


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """Convert geodetic to ECEF using WGS84 ellipsoid."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    a = 6378137.0
    e2 = 1.0 / 298.257223563 * (2.0 - 1.0 / 298.257223563)
    N = a / np.sqrt(1.0 - e2 * sin_lat**2)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - e2) + alt_m) * sin_lat
    return np.array([x, y, z])


def _great_circle_azimuth(r0: np.ndarray, r1: np.ndarray) -> float:
    """Compute great-circle initial azimuth from r0 to r1 (degrees, 0=N CW)."""
    lat0, lon0, _ = _ecef_to_geodetic(r0)
    lat1, lon1, _ = _ecef_to_geodetic(r1)
    phi1 = np.radians(lat0)
    phi2 = np.radians(lat1)
    dlam = np.radians(lon1 - lon0)
    az = np.degrees(np.arctan2(
        np.cos(phi2) * np.sin(dlam),
        np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam),
    )) % 360.0
    return float(az)


class InertialNavigationSystem:
    """Simplified strapdown INS for ICBM boost phase.

    Integrates accelerometer and gyro measurements to estimate position,
    velocity, and attitude.  Real ICBM INS uses redundant gyros and
    accelerometers, temperature compensation, and continuous self-alignment.
    This implementation captures the essential physics.
    """

    def __init__(self, r0: np.ndarray, v0: np.ndarray, q0: np.ndarray = None,
                 gyro_bias: np.ndarray = None, accel_bias: np.ndarray = None,
                 use_j2: bool = True):
        """
        Parameters
        ----------
        r0, v0 : ndarray
            Initial position (ECEF) and velocity (ECEF).
        q0 : ndarray, optional
            Initial attitude quaternion (body-to-ECEF). Defaults to identity.
        gyro_bias, accel_bias : ndarray, optional
            Constant sensor biases (rad/s, m/s^2).
        use_j2 : bool
            Whether to include J2 gravity in INS propagation.
        """
        self.r = np.asarray(r0, dtype=float).copy()
        self.v = np.asarray(v0, dtype=float).copy()
        if q0 is None:
            q0 = np.array([1.0, 0.0, 0.0, 0.0])
        self.q = np.asarray(q0, dtype=float).copy()
        self.gyro_bias = np.asarray(gyro_bias, dtype=float) if gyro_bias is not None else np.zeros(3)
        self.accel_bias = np.asarray(accel_bias, dtype=float) if accel_bias is not None else np.zeros(3)
        self.use_j2 = use_j2
        self.time = 0.0

    def state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.r.copy(), self.v.copy(), self.q.copy()

    def _gravity(self, r: np.ndarray) -> np.ndarray:
        """Central + J2 gravity in ECEF."""
        rmag = np.linalg.norm(r)
        g_mag = MU_EARTH / rmag**3
        g = -g_mag * r
        if self.use_j2:
            j2 = 1.08263e-3
            r2 = rmag**2
            r5 = rmag**5
            factor = 1.5 * j2 * (R_EARTH**2) / r5
            g = g + factor * r * (5.0 * (r[2]**2) / r2 - 1.0)
            g = g + factor * np.array([r[0], r[1], -3.0 * r[2]])
        return g

    def _quat_derivative(self, q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
        """Quaternion kinematics: q_dot = 0.5 * q * omega_quat."""
        qw, qx, qy, qz = q
        ox, oy, oz = omega_body
        return 0.5 * np.array([
            -qx * ox - qy * oy - qz * oz,
             qw * ox + qy * oz - qz * oy,
             qw * oy - qx * oz + qz * ox,
             qw * oz + qx * oy - qy * ox,
        ])

    def propagate(self, f_body: np.ndarray, omega_body: np.ndarray, dt: float):
        """Propagate INS one step given body-frame specific force and angular rate.

        Parameters
        ----------
        f_body : ndarray
            Specific force in body frame (m/s^2), includes thrust and aero.
        omega_body : ndarray
            Angular velocity in body frame (rad/s).
        dt : float
            Integration timestep (s).
        """
        self.time += dt
        # Remove biases (INS would calibrate these out, but we add them back
        # as measurement noise analog)
        f_meas = f_body - self.accel_bias
        omega_meas = omega_body - self.gyro_bias

        # Attitude update (simple Euler, RK4 for better accuracy)
        q = self.q
        k1_q = self._quat_derivative(q, omega_meas)
        k2_q = self._quat_derivative(q + 0.5 * dt * k1_q, omega_meas)
        k3_q = self._quat_derivative(q + 0.5 * dt * k2_q, omega_meas)
        k4_q = self._quat_derivative(q + dt * k3_q, omega_meas)
        q_new = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        q_new = q_new / max(np.linalg.norm(q_new), 1e-12)
        self.q = q_new

        # Transform specific force to ECEF
        # C_body_to_ecef from quaternion
        qw, qx, qy, qz = q_new
        C = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
        ])
        f_ecef = C @ f_meas

        # Velocity update (coning + sculling omitted for simplicity)
        g = self._gravity(self.r)
        v_new = self.v + dt * (f_ecef + g)

        # Position update
        r_new = self.r + dt * self.v

        self.r = r_new
        self.v = v_new
        self.q = q_new


class ICBMGuidance:
    """Realistic ICBM boost-phase guidance based on inertial navigation.

    Real ICBM guidance works as follows:

    1. **Pre-launch**: Target coordinates are loaded.  A launch azimuth
       is computed from the great-circle path to the target.  A burnout
       velocity magnitude is estimated from range tables.

    2. **Silo egress (t < 5-10 s)**: Vertical ascent.  The thrust vector
       is aligned with the local vertical (up) to clear the silo.

    3. **Pitch-over (t ~ 5-30 s)**: A pre-programmed pitch schedule
       gradually turns the vehicle from vertical to the desired gravity-turn
       flight path.  This is NOT closed-loop guidance -- it is a
       time/energy-based open-loop program stored in the guidance computer.

    4. **Gravity turn (t > 30 s)**: The thrust vector is aligned with the
       current velocity vector (zero angle of attack).  Gravity naturally
       turns the trajectory toward the target.  The INS estimates position
       and velocity; periodic star sightings correct INS drift.

    5. **Stage separation**: Mass drops and small impulse corrections are
       applied.  The pitch schedule compensates for staging dispersions.

    6. **Bus/Midcourse**: Ballistic coast.  The bus may perform small
       maneuvers for countermeasure deployment.  No active guidance for
       the bus or individual RVs.

    7. **Terminal**: RVs follow pre-calculated ballistic trajectories.
       Advanced RVs may use terminal corrections (radar/optical/inertial).

    This class implements steps 1-4 (boost-phase guidance).  Steps 5-7
    are handled by the scenario or terminal guidance modules.
    """

    def __init__(
        self,
        target_ecef: np.ndarray,
        launch_ecef: np.ndarray,
        burnout_vmag: float = 6200.0,
        pitch_over_start: float = 5.0,
        initial_elevation: float = np.radians(55.0),
        use_j2: bool = True,
    ):
        """
        Parameters
        ----------
        target_ecef : ndarray
            Target position in ECEF (m).
        launch_ecef : ndarray
            Launch position in ECEF (m).
        burnout_vmag : float
            Desired velocity magnitude at thrust end (m/s).
        pitch_over_start : float
            Time after launch when pitch-over begins (s).
        initial_elevation : float
            Launch elevation (rad) above local horizontal after silo egress.
        use_j2 : bool
            Whether guidance computations include J2 perturbations.
        """
        self.target = np.asarray(target_ecef, dtype=float)
        self.launch = np.asarray(launch_ecef, dtype=float)
        self.burnout_vmag = burnout_vmag
        self.pitch_over_start = pitch_over_start
        self.initial_elevation = initial_elevation
        self.use_j2 = use_j2

        self.azimuth = np.radians(_great_circle_azimuth(self.launch, self.target))

        lat0, lon0, _ = _ecef_to_geodetic(self.launch)
        self._east, self._north, self._up = _enu_basis(lat0, lon0)

        self._gc_horiz = (
            np.cos(self.azimuth) * self._north + np.sin(self.azimuth) * self._east
        )
        gc_norm = np.linalg.norm(self._gc_horiz)
        if gc_norm > 1e-9:
            self._gc_horiz = self._gc_horiz / gc_norm
        else:
            self._gc_horiz = np.array([0.0, 0.0, 0.0])

        range_km = np.linalg.norm(self.target - self.launch) / 1000.0
        self._final_gamma = np.radians(np.clip(20.0 - 0.3 * (range_km - 7000.0) / 1000.0, 12.0, 25.0))

        self.ins: Optional[InertialNavigationSystem] = None

        # Star-sighting correction state

    def reset(self, launch_ecef: np.ndarray = None):
        if launch_ecef is not None:
            self.launch = np.asarray(launch_ecef, dtype=float)
            lat0, lon0, _ = _ecef_to_geodetic(self.launch)
            self._east, self._north, self._up = _enu_basis(lat0, lon0)
            self._gc_horiz = (
                np.cos(self.azimuth) * self._north + np.sin(self.azimuth) * self._east
            )
            gc_norm = np.linalg.norm(self._gc_horiz)
            if gc_norm > 1e-9:
                self._gc_horiz = self._gc_horiz / gc_norm
        if self.ins is not None:
            self.ins.r = self.launch.copy()
            self.ins.v = np.zeros(3)
            self.ins.time = 0.0

    def _gravity(self, r: np.ndarray) -> np.ndarray:
        rmag = np.linalg.norm(r)
        g = -MU_EARTH / rmag**3 * r
        if self.use_j2:
            j2 = 1.08263e-3
            r2 = rmag**2
            r5 = rmag**5
            factor = 1.5 * j2 * (R_EARTH**2) / r5
            g = g + factor * r * (5.0 * (r[2]**2) / r2 - 1.0)
            g = g + factor * np.array([r[0], r[1], -3.0 * r[2]])
        return g

    def _desired_flight_path_angle(self, r: np.ndarray, v: np.ndarray, t: float) -> float:
        """Compute the desired flight path angle (rad above local horizontal).

        Uses a pre-programmed energy-based pitch schedule optimized for
        suborbital ICBM trajectories. The schedule transitions from vertical
        to near-horizontal rapidly to maximize horizontal velocity at burnout.
        """
        vmag = np.linalg.norm(v)
        if vmag < 1e-6:
            return np.pi / 2.0

        lat, lon, _ = _ecef_to_geodetic(r)
        _, _, up = _enu_basis(lat, lon)
        v_dir = v / vmag

        if t < self.pitch_over_start:
            return np.pi / 2.0

        rmag = np.linalg.norm(r)
        eps = 0.5 * vmag**2 - MU_EARTH / max(rmag, 1e-6)
        eps_min = -MU_EARTH / max(rmag, 1e-6)
        eps_max = 0.5 * self.burnout_vmag**2 - MU_EARTH / max(rmag, 1e-6)
        eps_norm = (eps - eps_min) / max(eps_max - eps_min, 1e-9)
        eps_norm = float(np.clip(eps_norm, 0.0, 1.0))

        if eps_norm < 0.001:
            return np.pi / 2.0
        elif eps_norm < 0.05:
            frac = (eps_norm - 0.001) / 0.049
            frac = 0.5 * (1.0 - np.cos(np.pi * frac))
            return float(self.initial_elevation - frac * (self.initial_elevation - np.radians(25.0)))
        elif eps_norm < 0.40:
            frac = (eps_norm - 0.05) / 0.35
            return float(np.radians(25.0) - frac * np.radians(13.0))
        else:
            return float(np.arcsin(np.clip(np.dot(v_dir, up), -1.0, 1.0)))

    def thrust_direction(self, r: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        """Compute the commanded thrust direction vector in ECEF.

        Uses the pre-computed great-circle azimuth and an energy-based pitch
        schedule that transitions from vertical to the gravity-turn flight
        path. This matches how real ICBMs operate: azimuth computed at
        launch, pitch schedule energy-based.
        """
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
        lat, lon, _ = _ecef_to_geodetic(r)
        _, _, up = _enu_basis(lat, lon)

        gc_horiz = self._gc_horiz

        gamma = self._desired_flight_path_angle(r, v, t)

        thrust_dir = np.cos(gamma) * gc_horiz + np.sin(gamma) * up

        norm = np.linalg.norm(thrust_dir)
        if norm > 1e-9:
            thrust_dir = thrust_dir / norm
        else:
            thrust_dir = up.copy()

        return thrust_dir

    def azimuth_error(self, r: np.ndarray, v: np.ndarray) -> float:
        """Compute cross-range error from desired great-circle azimuth (rad)."""
        lat, lon, _ = _ecef_to_geodetic(r)
        _, _, up = _enu_basis(lat, lon)
        vmag = np.linalg.norm(v)
        if vmag < 1e-6:
            return 0.0
        v_dir = v / vmag
        v_horiz = v_dir - np.dot(v_dir, up) * up
        v_horiz_norm = np.linalg.norm(v_horiz)
        if v_horiz_norm < 1e-6:
            return 0.0
        v_horiz = v_horiz / v_horiz_norm
        # Cross-range is the sine of the angle between velocity azimuth and desired azimuth
        cross = np.dot(v_horiz, self._gc_horiz)
        return float(np.arccos(np.clip(cross, -1.0, 1.0)))


class BurnoutTargeting:
    """Computes required burnout velocity for suborbital ICBM trajectory.

    For suborbital ICBMs, the trajectory consists of:
    1. Powered boost phase (controlled by guidance)
    2. Ballistic coast phase (free-fall Keplerian orbit from burnout to impact)

    This class solves the inverse problem: given launch site, target, and
    time of flight, compute the burnout velocity that makes the coasting
    trajectory intersect the target.

    Uses Lambert's problem to determine the required velocity vector at
    engine cutoff. At burnout, the missile's position and the target position
    define a boundary-value problem: find the velocity v1 at r1 that reaches
    r2 in the given time of flight under gravitational influence.
    """

    def __init__(self, mu: float = MU_EARTH):
        self.mu = mu

    def required_burnout_velocity(
        self,
        r_burnout: np.ndarray,
        r_target: np.ndarray,
        tof_coast: float,
    ) -> np.ndarray:
        """Find burnout velocity vector for suborbital trajectory to target.

        Uses Lambert's problem to compute the required velocity at burnout
        that, when coasted under J2-perturbed gravity, reaches the target.

        Returns velocity vector in ECEF.
        """
        r_burnout = np.asarray(r_burnout, dtype=float)
        r_target = np.asarray(r_target, dtype=float)
        tof_coast_ref = tof_coast
        dt_step = 1.0

        def propagate_coast(v_vec):
            r = r_burnout.copy()
            v = v_vec.copy()
            t = 0.0
            while t < tof_coast_ref:
                alt = np.linalg.norm(r) - R_EARTH
                if alt < 0.0 and t > 1.0:
                    break
                a1 = _two_body_accel_j2(r)
                k1r, k1v = v, a1
                k2r = v + 0.5 * dt_step * k1v
                k2v = _two_body_accel_j2(r + 0.5 * dt_step * k1r)
                k3r = v + 0.5 * dt_step * k2v
                k3v = _two_body_accel_j2(r + 0.5 * dt_step * k2r)
                k4r = v + dt_step * k3v
                k4v = _two_body_accel_j2(r + dt_step * k3r)
                r = r + (dt_step / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
                v = v + (dt_step / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
                t += dt_step
                if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
                    break
            return r

        from .lambert import lambert_with_correction
        return lambert_with_correction(
            r_burnout, r_target, tof_coast, self.mu,
            propagate_fn=propagate_coast,
        )

    def _coast_rhs(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)
        g = _two_body_accel_j2(r)
        return np.concatenate([v, g])
