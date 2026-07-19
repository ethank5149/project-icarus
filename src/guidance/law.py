from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

from ..interceptors.config import InterceptorConfig, GuidanceConfig
from ..guidance.boost_guidance import BoostGuidance
from ..guidance.midcourse_guidance import MidcourseGuidance
from ..guidance.terminal_guidance import TerminalGuidance
from ..guidance.seeker import SeekerModel, SeekerConfig, DiscriminationModel
from ..scenarios.target_factory import ThreatSignatureLibrary


@dataclass
class GuidanceLaw:
    """Wrapper for guidance laws with configured parameters."""
    config: Optional[GuidanceConfig] = None

    def __post_init__(self):
        if self.config is None:
            self.config = GuidanceConfig()

        self.boost = BoostGuidance(
            pitch_over_q=self.config.boost_pitch_over_q,
            pitch_over_angle=self.config.boost_pitch_over_angle,
            gimbal_limits=np.radians(15),
        )
        self.midcourse = MidcourseGuidance(
            N=self.config.midcourse_n,
            accel_limit=self.config.midcourse_accel_limit,
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
