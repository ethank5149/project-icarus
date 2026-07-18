from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


@dataclass
class InterceptorConfig:
    """Configuration for an interceptor system."""
    name: str = "Generic Interceptor"
    mass: float = 1000.0
    area: float = 0.1
    ref_length: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.diag([100.0, 200.0, 300.0]))
    kill_radius: float = 0.5
    kill_mechanism: str = "hit_to_kill"
    thrust_profile: Optional[Callable[[float], float]] = None
    mass_flow: float = 50.0
    mkv_mass: float = 15.0
    mkv_divert_impulse: float = 85.0
    gimbal_limits: tuple = (np.radians(15), np.radians(15))
    accel_limit: float = 150.0
    seeker_snr: float = 20.0
    noise_std: float = 0.01


@dataclass
class GuidanceConfig:
    """Guidance law parameters."""
    name: str = "Standard"
    boost_pitch_over_q: float = 15000.0
    boost_pitch_over_angle: float = np.radians(5.0)
    midcourse_n: float = 5.0
    midcourse_accel_limit: float = 50.0
    terminal_n: float = 4.0
    terminal_accel_limit: float = 150.0
    terminal_kill_radius: float = 0.5
    terminal_mechanism: str = "hit_to_kill"
