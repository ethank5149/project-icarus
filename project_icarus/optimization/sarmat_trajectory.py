"""Dymos-based RS-28 Sarmat offensive trajectory optimization.

Single-phase boost optimization that minimizes terminal miss distance.
The optimizer tunes the real ``ICBMGuidance`` pitch schedule parameters
exposed as Dymos parameters.
"""

import numpy as np
import openmdao.api as om
import dymos as dm
from .phases.sarmat_boost_phase import SarmatBoostODE

# Kozelsk launch site (ECEF, surface elevation ~175 m)
_KOZELSK_LAT = 54.07
_KOZELSK_LON = 35.73
# DC defended aim point (ECEF, surface elevation ~31 m)
_DC_LAT = 38.90
_DC_LON = -77.04

# Target ECEF position (computed once at module load for consistency)
from ..scenarios.target_factory import _geodetic_to_ecef_simple
from ..reference.surface_elevation import get_surface_elevation

_KOZELSK_ELEV = get_surface_elevation(_KOZELSK_LAT, _KOZELSK_LON)
_DC_ELEV = get_surface_elevation(_DC_LAT, _DC_LON)

R0_KOZELSK = _geodetic_to_ecef_simple(_KOZELSK_LAT, _KOZELSK_LON, _KOZELSK_ELEV)
R_TARGET_DC = _geodetic_to_ecef_simple(_DC_LAT, _DC_LON, _DC_ELEV)


def build_sarmat_trajectory_problem(num_segments=20, order=5, maxiter=200):
    """Assemble the Sarmat boost-phase Dymos trajectory optimization problem.

    This is a single-phase problem that optimizes the boost trajectory to
    minimize the miss distance at burnout.  The optimizer tunes the real
    ``ICBMGuidance`` pitch schedule parameters exposed as Dymos parameters.

    Parameters
    ----------
    num_segments : int
        Number of Radau segments per phase.  Reduce for faster testing.
    order : int
        Polynomial order per segment.  3 is faster; 5 is more accurate.
    maxiter : int
        SLSQP iteration budget for the driver.

    Returns
    -------
    p : openmdao.Problem
        Assembled but unsolved problem.  Call ``p.setup()`` then
        ``dymos.run_problem(p, run_driver=True, simulate=False)``.
    """
    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem("traj", traj, promotes=["*"])

    tx = dm.Radau(num_segments=num_segments, order=order)

    # ------------------------------------------------------------------
    # Phase 1: Boost (staged thrust, energy-based pitch schedule)
    # ------------------------------------------------------------------
    boost = dm.Phase(ode_class=SarmatBoostODE, transcription=tx)
    traj.add_phase("boost", boost)
    boost.options["ode_init_kwargs"] = {
        "geometry_key": "rs28_sarmat",
        "target_lat": _DC_LAT,
        "target_lon": _DC_LON,
    }
    boost.set_time_options(fix_initial=True, duration_bounds=(200.0, 400.0), units="s")

    boost.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                    fix_initial=True, val=R0_KOZELSK)
    boost.add_state("v", rate_source="dv_dt", units="m/s", shape=(3,),
                    fix_initial=True, val=np.array([0.0, 0.0, 0.0]))
    boost.add_state("q", rate_source="dq_dt", units=None, shape=(4,),
                    fix_initial=True, val=np.array([1.0, 0.0, 0.0, 0.0]))
    boost.add_state("omega", rate_source="domega_dt", units="rad/s", shape=(3,),
                    fix_initial=True, val=np.zeros(3))
    boost.add_state("m", rate_source="dm_dt", units="kg",
                    fix_initial=True, val=208100.0)

    # Guidance parameters (optimizer-tunable, wired to real ICBMGuidance)
    boost.add_parameter("pitch_over_start", units="s", val=5.0,
                        opt=True, lower=3.0, upper=15.0)
    boost.add_parameter("initial_elevation", units=None, val=np.radians(55.0),
                        opt=True, lower=np.radians(40), upper=np.radians(75))
    boost.add_parameter("burnout_vmag", units="m/s", val=6200.0,
                        opt=True, lower=5500.0, upper=7500.0)
    boost.add_parameter("target_r", units="m", shape=(3,),
                        val=R_TARGET_DC, opt=False)

    # ------------------------------------------------------------------
    # Objective: minimize final miss distance
    # ------------------------------------------------------------------
    boost.add_objective("miss_distance", scaler=1.0, loc="final")

    p.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
    p.driver.declare_coloring()
    p.driver.opt_settings["maxiter"] = maxiter

    p.setup()

    # Feasible initial guesses
    p.set_val("traj.boost.t_duration", 300.0)
    p.set_val("traj.boost.parameters:pitch_over_start", 5.0)
    p.set_val("traj.boost.parameters:initial_elevation", np.radians(55.0))
    p.set_val("traj.boost.parameters:burnout_vmag", 6200.0)
    p.set_val("traj.boost.parameters:target_r", R_TARGET_DC)

    return p


if __name__ == "__main__":
    p = build_sarmat_trajectory_problem()
    dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
