from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Protocol, runtime_checkable
import numpy as np


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
        # Keplerian two-body propagation (forward Euler, fine for short arcs).
        r = self.r0.astype(float).copy()
        v = self.v0.astype(float).copy()
        dt = 0.5
        n = max(int(t / dt), 1)
        for _ in range(n):
            a = _two_body_accel(r, use_j2=self.use_j2)
            v = v + a * dt
            r = r + v * dt
        return np.concatenate([r, v])

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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _orbital_speed(self, r):
        return np.sqrt(MU_EARTH / max(r, 1e-6))

    def propagate(self, t: float) -> np.ndarray:
        if t < self._boost_duration:
            # Boost: constant thrust + inverse-square gravity.
            r = self.r0.astype(float).copy()
            v = self.v0.astype(float).copy()
            dt = 0.5
            for _ in range(max(int(t / dt), 1)):
                a = _two_body_accel(r) + np.array([self.thrust, 0.0, 0.0])
                v = v + a * dt
                r = r + v * dt
        elif t < self._boost_duration + self._coast_duration:
            # Coast: 2-body orbital motion.
            tc = t - self._boost_duration
            r = self.r0 + self.v0 * self._boost_duration
            v_mag = self._orbital_speed(np.linalg.norm(r))
            v = v_mag * self.v0 / max(np.linalg.norm(self.v0), 1e-6)
            r = r + v * tc
        else:
            # Reentry: patched-conic with atmospheric drag (exponential model).
            tc = t - self._boost_duration - self._coast_duration
            r = np.array([R_EARTH + self._reentry_alt, 0.0, 0.0])
            v = -self._orbital_speed(R_EARTH) * np.array([0.0, 0.0, 1.0])
            dt = 0.5
            pos = r.astype(float).copy()
            vel = v.astype(float).copy()
            for _ in range(max(int(tc / dt), 1)):
                rm = np.linalg.norm(pos)
                a = _two_body_accel(pos)
                alt = rm - R_EARTH
                if alt < 150e3:
                    rho = 1.225 * np.exp(-alt / 8500.0)
                    q_dyn = 0.5 * rho * np.dot(vel, vel)
                    a = a - q_dyn * 0.3 / 100.0 * vel / max(np.linalg.norm(vel), 1e-6)
                vel = vel + a * dt
                pos = pos + vel * dt
            r = pos
            v = vel
        return np.concatenate([r, v])

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
            # Delta-v impulse (velocity change, not addition) at maneuver times.
            if tc > 60.0:
                m = int(tc / self.maneuver_interval)
                if m % 2 == 0 and m not in applied:
                    v = v + self.midcourse_maneuver_mag * np.array([0.0, 1.0, 0.0])
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
