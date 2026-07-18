from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any, Protocol, runtime_checkable
import numpy as np


MU_EARTH = 3.986004418e14
R_EARTH = 6371e3


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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def propagate(self, t: float) -> np.ndarray:
        r = self.r0 + self.v0 * t - 0.5 * np.array([0.0, 0.0, 9.81]) * t**2
        v = self.v0 - np.array([0.0, 0.0, 9.81]) * t
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
    _boost_duration: float = 120.0
    _coast_duration: float = 600.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _orbital_speed(self, r):
        return np.sqrt(MU_EARTH / max(r, 1e-6))

    def propagate(self, t: float) -> np.ndarray:
        if t < self._boost_duration:
            r = self.r0 + self.v0 * t
            v = self.v0
        elif t < self._boost_duration + self._coast_duration:
            tc = t - self._boost_duration
            r = self.r0 + self.v0 * self._boost_duration
            v_mag = self._orbital_speed(np.linalg.norm(r))
            v = v_mag * self.v0 / max(np.linalg.norm(self.v0), 1e-6)
            r = r + v * tc
        else:
            tc = t - self._boost_duration - self._coast_duration
            r = np.array([R_EARTH + 0.0, 0.0, 0.0])
            v = -self._orbital_speed(R_EARTH) * np.array([0.0, 0.0, 1.0])
            r = r + v * tc
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
    skip_frequency: float = 0.05
    skip_amplitude: float = 5000.0
    ballistic_coeff: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def propagate(self, t: float) -> np.ndarray:
        r = self.r0 + self.v0 * t
        lift_up = self.skip_amplitude * np.sin(2 * np.pi * self.skip_frequency * t)
        r[2] += lift_up
        v = self.v0.copy()
        v[2] += self.skip_amplitude * 2 * np.pi * self.skip_frequency * np.cos(2 * np.pi * self.skip_frequency * t)
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
        r = self.r0 + self.v0 * t - 0.5 * np.array([0.0, 0.0, 9.81]) * t**2
        v = self.v0 - np.array([0.0, 0.0, 9.81]) * t
        if t > 60.0 and int(t / self.maneuver_interval) % 2 == 0:
            v += self.midcourse_maneuver_mag * np.array([0.0, 1.0, 0.0])
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
