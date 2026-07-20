import numpy as np
import openmdao.api as om
import dymos as dm
from .phases.boost_phase import BoostODE
from .phases.midcourse_phase import MidcourseODE
from .phases.terminal_phase import TerminalODE, ClosedLoopTerminalODE


def build_trajectory_problem(interceptor=None, guidance=None, closed_loop=True):
    """Assemble a multi-phase Dymos trajectory optimization problem.

    Uses the modern Dymos 1.15 API: a :class:`dm.Trajectory` holding three
    Radau (Lagrange-ps) phases (boost, midcourse, terminal) linked by state
    continuity. Each phase ODE wraps the shared ``EOM6DOF`` + blended-aero
    surrogate (``boundary_alt`` is fixed via the ODE option).

    Terminal phase
    --------------
    * ``closed_loop=True`` (default) wires the ``EngagementRunner``'s closed-loop
      seeker guidance into the terminal ODE (:class:`ClosedLoopTerminalODE`): the
      interceptor acceleration is computed each node by ``TerminalGuidance`` from
      the relative kinematics, and the optimizer tunes the guidance gains
      (``N``, ``accel_limit``) to directly minimize the final miss distance. This
      is the Phase-7 step "wire EngagementRunner into the Dymos terminal phase as
      the closed-loop ODE so the optimizer directly minimizes miss distance of a
      seeker-guided solution".
    * ``closed_loop=False`` keeps the original open-loop ``accel`` controls and
      minimizes time-to-intercept with a final-position equality constraint.
    """
    """Assemble a multi-phase Dymos trajectory optimization problem.

    Uses the modern Dymos 1.15 API: a :class:`dm.Trajectory` holding three
    Radau (Lagrange-ps) phases (boost, midcourse, terminal) linked by state
    continuity. Each phase ODE wraps the shared ``EOM6DOF`` + blended-aero
    surrogate (``boundary_alt`` is fixed via the ODE option). The terminal
    phase can be wired to the ``EngagementRunner`` ODE for miss-distance
    minimization; here we minimize time as a standalone demonstration.
    """
    if interceptor is None:
        from ..interceptors.config import InterceptorConfig
        interceptor = InterceptorConfig()
    if guidance is None:
        from ..guidance.law import GuidanceLaw
        guidance = GuidanceLaw()

    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem("traj", traj, promotes=["*"])

    tx = dm.Radau(num_segments=10, order=3)

    # --- Boost phase -------------------------------------------------------
    boost = dm.Phase(ode_class=BoostODE, transcription=tx)
    traj.add_phase("boost", boost)
    boost.set_time_options(fix_initial=True, duration_bounds=(20.0, 80.0), units="s")
    boost.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                    fix_initial=True, fix_final=False,
                    val=np.array([0.0, 0.0, 6371e3 + 100e3]))
    boost.add_state("v", rate_source="dv_dt", units="m/s", shape=(3,),
                    fix_initial=True, fix_final=False,
                    val=np.array([0.0, 1200.0, 0.0]))
    boost.add_state("q", rate_source="dq_dt", units=None, shape=(4,),
                    fix_initial=True, fix_final=False,
                    val=np.array([1.0, 0.0, 0.0, 0.0]))
    boost.add_state("omega", rate_source="domega_dt", units="rad/s", shape=(3,),
                    fix_initial=True, fix_final=False,
                    val=np.zeros(3))
    boost.add_state("m", rate_source="dm_dt", units="kg",
                    fix_initial=True, fix_final=False,
                    val=interceptor.mass)
    boost.add_control("thrust", units="N", opt=True, lower=0.0, upper=30000.0,
                      fix_initial=True, fix_final=True, val=25000.0)
    boost.add_control("gimbal_beta", units="rad", opt=True,
                      lower=-0.26, upper=0.26, fix_initial=True, fix_final=True, val=0.0)
    boost.add_control("gimbal_delta", units="rad", opt=True,
                      lower=-0.26, upper=0.26, fix_initial=True, fix_final=True, val=0.0)
    # Boost has no standalone objective; the problem minimizes the
    # terminal-phase miss distance (set below).

    # --- Midcourse phase ---------------------------------------------------
    midcourse = dm.Phase(ode_class=MidcourseODE, transcription=tx)
    traj.add_phase("midcourse", midcourse)
    midcourse.set_time_options(fix_initial=False, duration_bounds=(60.0, 200.0), units="s")
    midcourse.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                        fix_initial=False, fix_final=False,
                        val=np.array([0.0, 200e3, 6471e3]))
    midcourse.add_state("v", rate_source="dv_dt", units="m/s", shape=(3,),
                        fix_initial=False, fix_final=False,
                        val=np.array([0.0, 3000.0, 1000.0]))
    midcourse.add_state("q", rate_source="dq_dt", units=None, shape=(4,),
                        fix_initial=False, fix_final=False,
                        val=np.array([1.0, 0.0, 0.0, 0.0]))
    midcourse.add_state("omega", rate_source="domega_dt", units="rad/s", shape=(3,),
                        fix_initial=False, fix_final=False, val=np.zeros(3))
    midcourse.add_state("m", rate_source="dm_dt", units="kg",
                        fix_initial=False, fix_final=False,
                        val=interceptor.mass * 0.5)
    midcourse.add_control("accel_x", units="m/s**2", opt=True, lower=-50.0, upper=50.0,
                          fix_initial=True, fix_final=True, val=0.0)
    midcourse.add_control("accel_y", units="m/s**2", opt=True, lower=-50.0, upper=50.0,
                          fix_initial=True, fix_final=True, val=0.0)
    midcourse.add_control("accel_z", units="m/s**2", opt=True, lower=-50.0, upper=50.0,
                          fix_initial=True, fix_final=True, val=0.0)

    # --- Terminal phase ----------------------------------------------------
    law = getattr(getattr(guidance, "terminal", None), "law", "pn") if guidance else "pn"
    if closed_loop:
        terminal = dm.Phase(ode_class=ClosedLoopTerminalODE, transcription=tx)
        traj.add_phase("terminal", terminal)
        terminal.options["ode_init_kwargs"] = {"law": law}
        terminal.set_time_options(fix_initial=False, duration_bounds=(10.0, 60.0), units="s")
        # Closed-loop: the final interceptor position is LEFT FREE so the
        # optimizer (tuning N / accel_limit + phase durations) can drive the
        # seeker-guided solution onto the defended aim point; the objective is
        # the final miss distance directly.
        terminal.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                           fix_initial=False, fix_final=False,
                           val=np.array([120e3, 0.0, 6471e3]))
        terminal.add_state("v", rate_source="dv_dt", units="m/s", shape=(3,),
                           fix_initial=False, fix_final=False,
                           val=np.array([-1500.0, 0.0, 0.0]))
        terminal.add_state("q", rate_source="dq_dt", units=None, shape=(4,),
                           fix_initial=False, fix_final=False,
                           val=np.array([1.0, 0.0, 0.0, 0.0]))
        terminal.add_state("omega", rate_source="domega_dt", units="rad/s", shape=(3,),
                           fix_initial=False, fix_final=False, val=np.zeros(3))
        terminal.add_state("m", rate_source="dm_dt", units="kg",
                           fix_initial=False, fix_final=False, val=200.0)
        # Optional open-loop bias retained for differentiability (left at 0).
        terminal.add_control("accel_x", units="m/s**2", opt=True, lower=-20.0, upper=20.0,
                             fix_initial=True, fix_final=True, val=0.0)
        terminal.add_control("accel_y", units="m/s**2", opt=True, lower=-20.0, upper=20.0,
                             fix_initial=True, fix_final=True, val=0.0)
        terminal.add_control("accel_z", units="m/s**2", opt=True, lower=-20.0, upper=20.0,
                             fix_initial=True, fix_final=True, val=0.0)
        # Guidance gains the optimizer tunes to null the miss distance.
        terminal.add_parameter("N", units=None, val=4.0, opt=True,
                               lower=2.0, upper=8.0)
        terminal.add_parameter("accel_limit", units="m/s**2", val=150.0, opt=True,
                               lower=50.0, upper=250.0)
    else:
        terminal = dm.Phase(ode_class=TerminalODE, transcription=tx)
        traj.add_phase("terminal", terminal)
        terminal.set_time_options(fix_initial=False, duration_bounds=(10.0, 60.0), units="s")
        terminal.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                           fix_initial=False, fix_final=True,
                           val=np.array([300e3, 0.0, 6471e3]))
        terminal.add_state("v", rate_source="dv_dt", units="m/s", shape=(3,),
                           fix_initial=False, fix_final=False,
                           val=np.array([-500.0, 0.0, -500.0]))
        terminal.add_state("q", rate_source="dq_dt", units=None, shape=(4,),
                           fix_initial=False, fix_final=False,
                           val=np.array([1.0, 0.0, 0.0, 0.0]))
        terminal.add_state("omega", rate_source="domega_dt", units="rad/s", shape=(3,),
                           fix_initial=False, fix_final=False, val=np.zeros(3))
        terminal.add_state("m", rate_source="dm_dt", units="kg",
                           fix_initial=False, fix_final=False, val=200.0)
        terminal.add_control("accel_x", units="m/s**2", opt=True, lower=-150.0, upper=150.0,
                             fix_initial=True, fix_final=True, val=0.0)
        terminal.add_control("accel_y", units="m/s**2", opt=True, lower=-150.0, upper=150.0,
                             fix_initial=True, fix_final=True, val=0.0)
        terminal.add_control("accel_z", units="m/s**2", opt=True, lower=-150.0, upper=150.0,
                             fix_initial=True, fix_final=True, val=0.0)
    # Aim point (defended location) held fixed across the terminal phase nodes.
    terminal.add_parameter("target_r", units="m", shape=(3,), val=np.array([100e3, 0.0, 6471e3]),
                           opt=False)
    terminal.add_parameter("target_v", units="m/s", shape=(3,), val=np.zeros(3), opt=False)

    # --- Link phases by state continuity ----------------------------------
    traj.link_phases(phases=["boost", "midcourse"],
                     vars=["r", "v", "q", "omega", "m", "time"])
    traj.link_phases(phases=["midcourse", "terminal"],
                     vars=["r", "v", "q", "omega", "m", "time"])

    terminal.add_timeseries_output("miss_distance")
    if closed_loop:
        # Closed-loop: the optimizer directly minimizes the final miss distance
        # of the seeker-guided terminal solution (no final-position constraint
        # needed — the guidance drives r onto target_r by construction).
        terminal.add_objective("miss_distance", scaler=1.0, loc="final")
    else:
        # Terminal-phase objective: minimize time-to-intercept (t_duration)
        # while driving the final interceptor position onto the defended aim
        # point; the miss distance is reported as a timeseries for assessment.
        terminal.add_objective("time_phase", scaler=1.0, loc="final")
        terminal.add_boundary_constraint(
            "r", loc="final", units="m", shape=(3,),
            equals=np.array([100e3, 0.0, 6471e3]),
        )

    p.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
    p.driver.declare_coloring()
    p.driver.opt_settings["maxiter"] = 50

    p.setup()

    # Provide feasible initial phase durations (midpoints of their bounds) so
    # SLSQP starts inside the feasible region instead of stalling on an
    # infeasible t_duration = 1.0 default.
    p.set_val("traj.boost.t_duration", 50.0)
    p.set_val("traj.midcourse.t_duration", 130.0)
    p.set_val("traj.terminal.t_duration", 35.0)

    return p


if __name__ == "__main__":
    p = build_trajectory_problem()
    dm.run_problem(p, run_driver=True, simulate=False)
