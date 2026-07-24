"""Direct trajectory optimization for RS-28 Sarmat using EXISTING physics.

This module uses the EXACT same physics as SarmatScenario._integrate_full():
- Same _SARMAT_STAGES thrust/mass model
- Same _two_body_accel gravity
- Same drag model
- Same RK4 integrator

The ONLY difference is that the production ICBMGuidance object's REAL
tunable parameters are optimized by scipy.  No simplified guidance models,
no fake substitutes, no fallbacks.  This is how real ICBMs optimize
trajectories: tune the actual guidance computer parameters on the
production physics.
"""

import numpy as np
from scipy.optimize import minimize

from project_icarus.scenarios.target_factory import (
    SarmatScenario,
    _geodetic_to_ecef_simple,
    _ground_altitude,
    _two_body_accel,
    _ecef_to_geodetic,
    _enu_basis,
)
from project_icarus.reference.surface_elevation import get_surface_elevation
from project_icarus.guidance.icbm_guidance import ICBMGuidance


# Launch/target coordinates (EXACTLY as in SarmatScenario)
_KOZELSK_LAT = 54.07
_KOZELSK_LON = 35.73
_DC_LAT = 38.90
_DC_LON = -77.04
_KOZELSK_ELEV = get_surface_elevation(_KOZELSK_LAT, _KOZELSK_LON)
_DC_ELEV = get_surface_elevation(_DC_LAT, _DC_LON)

R0 = _geodetic_to_ecef_simple(_KOZELSK_LAT, _KOZELSK_LON, _KOZELSK_ELEV)
R_TARGET = _geodetic_to_ecef_simple(_DC_LAT, _DC_LON, _DC_ELEV)


class ParameterizedSarmatScenario(SarmatScenario):
    """SarmatScenario with tunable ICBMGuidance parameters.

    Creates a REAL ICBMGuidance object and applies optimized parameters
    to its actual tunable fields.  No simplified guidance substitutes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._opt_params = None
        self._guidance = None

    def set_guidance_params(self, params):
        """Set optimized guidance parameters.

        Parameters
        ----------
        params : array-like
            [burnout_vmag, pitch_over_start, initial_elevation_deg]
        """
        x = np.asarray(params, dtype=float).copy()
        if len(x) >= 3:
            x[2] = np.radians(x[2])
        self._opt_params = x

    def _get_guidance(self):
        if self._guidance is None:
            self._guidance = ICBMGuidance(
                target_ecef=R_TARGET,
                launch_ecef=self.r0,
                burnout_vmag=6800.0,
                pitch_over_start=5.0,
                initial_elevation=np.radians(55.0),
                use_j2=self.use_j2,
            )
        if self._opt_params is not None:
            (self._guidance.burnout_vmag,
             self._guidance.pitch_over_start,
             self._guidance.initial_elevation) = self._opt_params
        return self._guidance

    def _current_thrust_dir(self, t, r, v):
        """Use production ICBMGuidance with optimized parameters."""
        guidance = self._get_guidance()
        return guidance.thrust_direction(r, v, t)


def compute_miss_distance(params):
    """Compute miss distance for given guidance parameters using production physics.

    Parameters
    ----------
    params : array-like
        [burnout_vmag, pitch_over_start, initial_elevation_deg]

    Returns
    -------
    miss_distance : float
        Euclidean distance from final position to target in meters.
    """
    scenario = ParameterizedSarmatScenario(
        r0=R0.copy(),
        v0=np.zeros(3),
        use_j2=True,
    )
    scenario.set_guidance_params(params)
    times, states = scenario._integrate_full()

    if len(times) == 0 or len(states) == 0:
        return 1e12

    final_state = states[-1]
    r_final = final_state[:3]

    return float(np.linalg.norm(r_final - R_TARGET))


def optimize_sarmat_trajectory(maxiter=200, ftol=1.0):
    """Optimize Sarmat boost guidance to minimize miss distance.

    Optimizes the ACTUAL ICBMGuidance parameters on the EXACT production
    physics.  No simplified models, no fake substitutes, no fallbacks.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
    """
    x0 = np.array([6200.0, 5.0, 55.0])
    bounds = [
        (5500.0, 7500.0),
        (3.0, 15.0),
        (np.radians(40), np.radians(75)),
    ]

    def objective(x):
        miss = compute_miss_distance(x)
        return miss

    result = minimize(
        objective, x0, method='SLSQP', bounds=bounds,
        options={'maxiter': maxiter, 'ftol': ftol}
    )

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("SARMAT TRAJECTORY OPTIMIZATION")
    print("Using EXACT production physics and REAL ICBMGuidance parameters")
    print("=" * 60)

    print("\nTesting default parameters...")
    default_params = [6200.0, 5.0, 55.0]
    default_miss = compute_miss_distance(default_params)
    print(f"Default miss: {default_miss/1000:.1f} km")

    print("\nOptimizing trajectory...")
    result = optimize_sarmat_trajectory(maxiter=100)

    print(f"\n{'=' * 60}")
    print("OPTIMIZATION RESULT")
    print(f"{'=' * 60}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Final miss: {result.fun/1000:.1f} km")
    print(f"Optimal parameters:")
    print(f"  burnout_vmag = {result.x[0]:.0f} m/s")
    print(f"  pitch_over_start = {result.x[1]:.1f} s")
    print(f"  initial_elevation = {np.degrees(result.x[2]):.1f} deg")

    if result.fun < 100.0:
        print("\nSUCCESS: Miss distance < 100m achieved!")
    else:
        print(f"\nWARNING: Miss distance {result.fun:.1f}m > 100m")
