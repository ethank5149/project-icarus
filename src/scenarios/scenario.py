from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from ..dynamics.gravity import R_EARTH


@dataclass
class EngagementScenario:
    """Scenario definition for an engagement."""
    name: str = "Default"
    interceptor_launch_site: np.ndarray = field(default_factory=lambda: np.array([R_EARTH, 0.0, 0.0]))
    target_launch_site: np.ndarray = field(default_factory=lambda: np.array([R_EARTH, 0.0, 0.0]))
    threat_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    engagement_start: float = 0.0
    engagement_end: float = 300.0
    sensor_noise: float = 0.01
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmScenario(EngagementScenario):
    name: str = "Swarm"
    n_payloads: int = 3
    spread_deg: float = 2.0
    bus_r0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bus_v0: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class SuppressedScenario(EngagementScenario):
    name: str = "Suppressed"
    dip_alt_km: float = 50.0
    midcourse_maneuver_mag: float = 50.0
    maneuver_interval: float = 30.0
