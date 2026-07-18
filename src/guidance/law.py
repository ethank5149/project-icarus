from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..interceptors.config import InterceptorConfig, GuidanceConfig
from ..guidance.boost_guidance import BoostGuidance
from ..guidance.midcourse_guidance import MidcourseGuidance
from ..guidance.terminal_guidance import TerminalGuidance


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
        )

    @classmethod
    def from_dict(cls, d: dict) -> "GuidanceLaw":
        return cls(config=GuidanceConfig(**d))

    @classmethod
    def from_config(cls, config: GuidanceConfig) -> "GuidanceLaw":
        return cls(config=config)
