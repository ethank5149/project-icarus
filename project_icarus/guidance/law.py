from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

from ..interceptors.config import InterceptorConfig, GuidanceConfig
from ..guidance.boost_guidance import BoostGuidance
from ..guidance.midcourse_guidance import MidcourseGuidance
from ..guidance.terminal_guidance import TerminalGuidance
from ..guidance.seeker import SeekerModel, SeekerConfig, DiscriminationModel
from ..guidance.tracker import TargetTracker, TrackerConfig
from ..guidance.autopilot import Autopilot, AutopilotConfig
from ..scenarios.target_factory import ThreatSignatureLibrary


@dataclass
class GuidanceLaw:
    """Wrapper for guidance laws with configured parameters."""
    config: Optional[GuidanceConfig] = None

    def __post_init__(self):
        if self.config is None:
            self.config = GuidanceConfig()

        try:
            from ..dynamics.gravity import gravity_inertial
        except Exception:
            gravity_inertial = None

        tracker_cfg = TrackerConfig(
            dt=getattr(self.config, "tracker_dt", 1.0),
            q_pos=getattr(self.config, "tracker_q_pos", 100.0),
            q_vel=getattr(self.config, "tracker_q_vel", 10.0),
            q_accel=getattr(self.config, "tracker_q_accel", 1.0),
            sigma_meas=getattr(self.config, "tracker_sigma_meas", 10.0),
        )
        self.tracker = TargetTracker(tracker_cfg)

        auto_cfg = AutopilotConfig(
            omega_n=getattr(self.config, "autopilot_omega_n", 100.0),
            damping=getattr(self.config, "autopilot_damping", 0.7),
            accel_rate_limit=getattr(self.config, "autopilot_rate_limit", 2000.0),
            accel_limit=self.config.terminal_accel_limit,
            gimbal_limit_deg=getattr(self.config, "autopilot_gimbal_limit_deg", 30.0),
            gimbal_rate_limit_deg_s=getattr(self.config, "autopilot_gimbal_rate_limit_deg_s", 200.0),
            use_gimbal_limit=getattr(self.config, "autopilot_use_gimbal_limit", True),
        )
        self.autopilot = Autopilot(auto_cfg)

        self.boost = BoostGuidance(
            pitch_over_q=self.config.boost_pitch_over_q,
            pitch_over_angle=self.config.boost_pitch_over_angle,
            gimbal_limits=np.radians(15),
        )
        self.midcourse = MidcourseGuidance(
            N=self.config.midcourse_n,
            accel_limit=self.config.midcourse_accel_limit,
            use_cython=True,
            tracker=self.tracker,
        )
        self.terminal = TerminalGuidance(
            N=self.config.terminal_n,
            accel_limit=self.config.terminal_accel_limit,
            kill_radius=self.config.terminal_kill_radius,
            mechanism=self.config.terminal_mechanism,
            noise_std=0.01,
            law=self.config.terminal_guidance_law,
            zem_horizon=self.config.zem_horizon,
            sdre_q_pos=self.config.sdre_q_pos,
            sdre_q_vel=self.config.sdre_q_vel,
            sdre_r_accel=self.config.sdre_r_accel,
            use_cython=True,
            tracker=self.tracker,
            gravity_model=gravity_inertial,
        )
        if self.config.ukf_enabled:
            seeker_cfg = SeekerConfig(
                mode=self.config.seeker_mode,
                fov=math.radians(self.config.seeker_fov_deg),
                range_max=self.config.seeker_range_max,
                snr_db=self.config.seeker_snr_db,
                noise_seed=self.config.seeker_noise_seed,
            )
            self.seeker = SeekerModel(seeker_cfg)
        else:
            self.seeker = None
        # Calibrated RV-vs-decoy discriminator (2C.2), trained from the default
        # OSINT-approximate threat signature library.
        lib = ThreatSignatureLibrary.default()
        X, y = lib.labelled_matrix()
        self.discriminator = DiscriminationModel().calibrate(X, y)

    def discriminate_target(self, features: np.ndarray) -> bool:
        """Return True if ``features`` is more likely an RV than a decoy."""
        return self.discriminator.is_rv(np.asarray(features, dtype=float))

    @classmethod
    def from_dict(cls, d: dict) -> "GuidanceLaw":
        return cls(config=GuidanceConfig(**d))

    @classmethod
    def from_config(cls, config: GuidanceConfig) -> "GuidanceLaw":
        return cls(config=config)
