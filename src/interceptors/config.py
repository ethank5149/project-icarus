from dataclasses import dataclass, field
from typing import Callable, Optional, List, Any
import numpy as np

from ..dynamics.thrust import StageSeparation, MultiStageThrustModel, StageSpec


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
    # --- Multi-stage support (Phase 1B) ---
    # When ``stages`` is provided, the runner builds a MultiStageThrustModel and
    # injects per-stage separations into the integration loop. ``thrust_profile``
    # (scalar) remains the simple single-stage fallback.
    stages: Optional[List[StageSpec]] = None
    sep_impulses: Optional[List[np.ndarray]] = None
    # Dry mass used by ThrustCutoffEvent to end the boost phase.
    dry_mass: float = -1.0
    peak_thrust: float = 0.0

    def __post_init__(self):
        if self.dry_mass is not None and self.dry_mass < 0 and self.stages:
            self.dry_mass = sum(s.dry_mass for s in self.stages)
        if self.peak_thrust <= 0.0 and self.stages:
            self.peak_thrust = max(
                float(s.thrust(0.0)) for s in self.stages
            )
        if self.peak_thrust <= 0.0 and self.thrust_profile is not None:
            # Sample a nominal peak for the scalar profile.
            self.peak_thrust = max(
                float(self.thrust_profile(t)) for t in (0.0, 0.5, 1.0, 2.0)
            )

    # --- Wiring accessed by the runner ------------------------------------
    @property
    def _thrust_callable(self) -> Optional[Callable]:
        if self.stages:
            model = MultiStageThrustModel(self.stages, self.sep_impulses)
            return model.thrust
        return self.thrust_profile

    @property
    def _separations(self) -> List[Any]:
        if self.stages:
            model = MultiStageThrustModel(self.stages, self.sep_impulses)
            return [s for s in model.separations if s is not None]
        return []


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
