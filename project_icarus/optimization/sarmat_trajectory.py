"""Dymos-based RS-28 Sarmat offensive trajectory optimization.

Assembles a three-phase Radau transcription problem (boost, midcourse,
terminal) that minimizes the terminal miss distance from Kozelsk to the
defended DC aim point.  The optimizer tunes the staged elevation schedule
(``el_0``, ``el_1``, ``t_cross``) and stage-3 thrust scale (``T3_scale``)
exposed as Dymos parameters on the boost phase.
"""

import numpy as np
import openmdao.api as om
import dymos as dm
from .phases.sarmat_boost_phase import SarmatBoostODE
from .phases.sarmat_midcourse_phase import SarmatMidcourseODE
from .phases.sarmat_terminal_phase import SarmatTerminalODE


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
    """Assemble the Sarmat 3-phase Dymos trajectory optimization problem.

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
    # Phase 1: Boost (staged thrust, parameterized pitch schedule)
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

    # Guidance parameters (optimizer-tunable)
    boost.add_parameter("el_0", units=None, val=np.radians(45.0),
                        opt=True, lower=np.radians(15), upper=np.radians(75))
    boost.add_parameter("el_1", units=None, val=np.radians(25.0),
                        opt=True, lower=np.radians(15), upper=np.radians(75))
    boost.add_parameter("t_cross", units="s", val=100.0,
                        opt=True, lower=5.0, upper=250.0)
    boost.add_parameter("T3_scale", units=None, val=1.0,
                        opt=True, lower=0.5, upper=5.0)
    boost.add_parameter("target_r", units="m", shape=(3,),
                        val=R_TARGET_DC, opt=False)

    # ------------------------------------------------------------------
    # Phase 2: Midcourse (ballistic coast, no thrust)
    # ------------------------------------------------------------------
    midcourse = dm.Phase(ode_class=SarmatMidcourseODE, transcription=tx)
    traj.add_phase("midcourse", midcourse)
    midcourse.options["ode_init_kwargs"] = {"geometry_key": "rs28_sarmat"}
    midcourse.set_time_options(fix_initial=False,
                                duration_bounds=(600.0, 1800.0), units="s")

    midcourse.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                        fix_initial=False, val=np.array([0.0, 200e3, 6471e3]))
    midcourse.add_state("v", rate_source="dv_dt", units="m/s", shape=(3,),
                        fix_initial=False, val=np.array([0.0, 3000.0, 1000.0]))
    midcourse.add_state("q", rate_source="dq_dt", units=None, shape=(4,),
                        fix_initial=False, val=np.array([1.0, 0.0, 0.0, 0.0]))
    midcourse.add_state("omega", rate_source="domega_dt", units="rad/s", shape=(3,),
                        fix_initial=False, val=np.zeros(3))
    midcourse.add_state("m", rate_source="dm_dt", units="kg",
                        fix_initial=False, val=20000.0)

    # ------------------------------------------------------------------
    # Phase 3: Terminal (reentry to ground impact)
    # ------------------------------------------------------------------
    terminal = dm.Phase(ode_class=SarmatTerminalODE, transcription=tx)
    traj.add_phase("terminal", terminal)
    terminal.options["ode_init_kwargs"] = {"geometry_key": "threat_rv"}
    terminal.set_time_options(fix_initial=False,
                              duration_bounds=(300.0, 900.0), units="s")

    terminal.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                       fix_initial=False, val=np.array([100e3, 0.0, 6471e3]))
    terminal.add_state("v", rate_source="dv_dt", units="m/s", shape=(3,),
                       fix_initial=False, val=np.array([-1500.0, 0.0, -500.0]))
    terminal.add_state("q", rate_source="dq_dt", units=None, shape=(4,),
                       fix_initial=False, val=np.array([1.0, 0.0, 0.0, 0.0]))
    terminal.add_state("omega", rate_source="domega_dt", units="rad/s", shape=(3,),
                       fix_initial=False, val=np.zeros(3))
    terminal.add_state("m", rate_source="dm_dt", units="kg",
                       fix_initial=False, val=800.0)

    terminal.add_parameter("target_r", units="m", shape=(3,),
                           val=R_TARGET_DC, opt=False)

    terminal.add_timeseries_output("miss_distance")

    # ------------------------------------------------------------------
    # Link phases by state continuity
    # ------------------------------------------------------------------
    traj.link_phases(phases=["boost", "midcourse"],
                     vars=["r", "v", "q", "omega", "m", "time"])
    traj.link_phases(phases=["midcourse", "terminal"],
                     vars=["r", "v", "q", "omega", "m", "time"])

    # ------------------------------------------------------------------
    # Objective: minimize final miss distance
    # ------------------------------------------------------------------
    terminal.add_objective("miss_distance", scaler=1.0, loc="final")

    p.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
    p.driver.declare_coloring()
    p.driver.opt_settings["maxiter"] = maxiter

    p.setup()

    # Feasible initial guesses for phase durations (midpoints of bounds)
    p.set_val("traj.boost.t_duration", 300.0)
    p.set_val("traj.midcourse.t_duration", 1200.0)
    p.set_val("traj.terminal.t_duration", 600.0)

    return p


if __name__ == "__main__":
    p = build_sarmat_trajectory_problem()
    dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
