from dataclasses import dataclass, field
from typing import Callable, Optional, List, Any
import numpy as np

from ..dynamics.thrust import StageSeparation, MultiStageThrustModel, StageSpec, MKVSystem


@dataclass
class InterceptorConfig:
    """Configuration for an interceptor system."""
    name: str = "Generic Interceptor"
    mass: float = 1000.0
    area: float = 0.1
    ref_length: float = 1.0
    geometry_key: str = "generic"
    reference_area_override: float = 0.0
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

    @property
    def _mkv(self) -> Optional[Any]:
        """Optional multi-KV payload (Phase 1A.3), wired only when the vehicle
        declares a non-zero KV mass. The runner ejects it at terminal phase."""
        if self.mkv_mass and self.mkv_mass > 0.0:
            return MKVSystem(
                kv_mass=self.mkv_mass,
                v_rel=1.5,
                divert_thrust=self.mkv_divert_impulse,
            )
        return None


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
    seeker_mode: str = "radar"  # "radar" | "semi_active_radar" | "ir"
    seeker_fov_deg: float = 60.0
    seeker_range_max: float = 50e3
    seeker_snr_db: float = 20.0
    seeker_noise_seed: int = 0
    ukf_enabled: bool = True
    # --- 2B.2 Guidance backend (terminal phase) ---
    # "pn"          : classic proportional navigation (default)
    # "apn"         : augmented PN (gravity-compensated target acceleration bias)
    # "zem"         : zero-effort-miss / impact-angle guidance
    # "sdre_mpc"    : SDRE-based MPC-lite (finite-horizon LQR on linearized EOM)
    terminal_guidance_law: str = "pn"
    apn_gravity_comp: bool = True
    zem_horizon: float = 5.0
    sdre_q_pos: float = 1.0
    sdre_q_vel: float = 0.1
    sdre_r_accel: float = 1.0
