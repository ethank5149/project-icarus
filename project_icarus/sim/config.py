from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


logger = logging.getLogger("icarus")


@dataclass
class SimConfig:
    """Global, reproducible simulation parameters.

    Centralizes integrator tolerances, phase-transition safety bounds,
    Monte-Carlo trial defaults, and a seedable RNG so that engagements are
    reproducible across runs.
    """

    # --- Integrator ---
    rtol: float = 1e-6
    atol: float = 1e-9
    max_step: float = 0.5
    # Integrator backend selector (Phase 3.3). "rk45" is the historical explicit
    # Runge-Kutta; "dop853" uses scipy's adaptive 8th-order Dormand-Prince for
    # stiff/hypersonic endo phases. Both share the same event-driven RHS.
    integrator: str = "rk45"
    # Hard safety cap on integration time (seconds) even if no event fires.
    t_max: float = 1800.0

    # --- Numerical safeguards (Phase 3.4) ---
    # Minimum allowable vehicle mass (kg); the RHS clamps mass above this floor so
    # a (near) dry stage cannot drive 1/mass to infinity and stiffen the EOM.
    mass_floor: float = 1e-3
    # When True, Monte-Carlo trials producing non-finite (NaN/Inf) miss distances
    # are rejected and logged rather than silently polluting kill statistics.
    reject_nonfinite: bool = True

    # --- Event-driven phase transitions ---
    # Boost -> midcourse when thrust drops below this fraction of peak OR mass
    # reaches dry mass.
    thrust_cutoff_frac: float = 1e-3
    # Midcourse -> terminal when altitude drops below this (m) ...
    reentry_alt: float = 100e3
    # ... OR range-to-target drops below this (m) ...
    terminal_range: float = 50e3
    # ... OR velocity (m/s) indicates final homing regime.
    terminal_speed: float = 4000.0

    # --- Monte Carlo ---
    n_trials: int = 8
    perturbations: dict = field(default_factory=lambda: {
        "position_sigma": 10.0,
        "velocity_sigma": 1.0,
        "mass_sigma": 1.0,
    })
    seed: int = 12345

    # --- Wind / ERA5 ---
    use_wind: bool = False
    wind_model: Optional[Any] = field(default=None)

    # --- Logging ---
    log_level: int = logging.INFO

    _rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def __post_init__(self):
        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        return self._rng

    def reseed(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def default(cls) -> "SimConfig":
        return cls()


# Module-level singleton used by the runner when no explicit config is passed.
_DEFAULT_CONFIG = SimConfig()


def get_config() -> SimConfig:
    return _DEFAULT_CONFIG
