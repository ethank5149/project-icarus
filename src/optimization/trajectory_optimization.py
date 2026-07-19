import numpy as np
import openmdao.api as om
import dymos as dm
from .phases.boost_phase import BoostODE
from .phases.midcourse_phase import MidcourseODE
from .phases.terminal_phase import TerminalODE


def build_trajectory_problem(interceptor=None, guidance=None):
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
    boost.add_objective("time_phase", scaler=-1.0, loc="final")

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
    terminal = dm.Phase(ode_class=TerminalODE, transcription=tx)
    traj.add_phase("terminal", terminal)
    terminal.set_time_options(fix_initial=False, duration_bounds=(10.0, 60.0), units="s")
    terminal.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                       fix_initial=False, fix_final=True,
                       val=np.array([100e3, 0.0, 6471e3]))
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

    p.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
    p.driver.declare_coloring()

    # Initial conditions are supplied via ``val=`` in each ``add_state`` /
    # ``add_control`` call above, so Dymos can build its own (internally
    # vectorized) initial guess without post-setup plumbing.
    p.setup()
    return p


if __name__ == "__main__":
    p = build_trajectory_problem()
    dm.run_problem(p, run_driver=True, simulate=False)
