import numpy as np
import openmdao.api as om
import dymos as dm
from .phases.boost_phase import BoostODE
from .phases.midcourse_phase import MidcourseODE
from .phases.terminal_phase import TerminalODE


def build_trajectory_problem(interceptor=None, guidance=None):
    if interceptor is None:
        from ..interceptors.config import InterceptorConfig
        interceptor = InterceptorConfig()
    if guidance is None:
        from ..guidance.law import GuidanceLaw
        guidance = GuidanceLaw()

    p = om.Problem()

    model = p.model
    model.add_subsystem("interceptor", om.IndepVarComp(), promotes=["*"])
    model.interceptor.add_output("mass", val=interceptor.mass)
    model.interceptor.add_output("area", val=interceptor.area)
    model.interceptor.add_output("ref_length", val=interceptor.ref_length)
    model.interceptor.add_output("kill_radius", val=interceptor.kill_radius)
    model.interceptor.add_output("kill_mechanism", val=interceptor.kill_mechanism)

    tx = dm.Radau(num_nodes=10)

    boost = dm.Phase(
        ode_class=BoostODE,
        transcription=tx,
    )
    boost.set_time_options(fix_initial=True, duration_val=60.0)
    boost.add_state("r", rate_source="dr_dt", units="m")
    boost.add_state("v", rate_source="dv_dt", units="m/s")
    boost.add_state("q", rate_source="dq_dt", units=None)
    boost.add_state("omega", rate_source="domega_dt", units="rad/s")
    boost.add_state("m", rate_source="dm_dt", units="kg")

    boost.add_control("thrust", units="N", opt=True, lower=0.0)
    boost.add_control("gimbal_beta", units="rad", opt=True, lower=-0.26, upper=0.26)
    boost.add_control("gimbal_delta", units="rad", opt=True, lower=-0.26, upper=0.26)

    boost.add_input_parameter("boundary_alt", val=100e3, static_target=True)

    boost.add_objective("time", scaler=-1.0)

    model.add_subsystem("boost", boost)

    midcourse = dm.Phase(
        ode_class=MidcourseODE,
        transcription=tx,
    )
    midcourse.set_time_options(fix_initial=False, duration_val=120.0)
    midcourse.add_state("r", rate_source="dr_dt", units="m")
    midcourse.add_state("v", rate_source="dv_dt", units="m/s")
    midcourse.add_state("q", rate_source="dq_dt", units=None)
    midcourse.add_state("omega", rate_source="domega_dt", units="rad/s")
    midcourse.add_state("m", rate_source="dm_dt", units="kg")

    midcourse.add_control("accel_x", units="m/s**2", opt=True, lower=-50.0, upper=50.0)
    midcourse.add_control("accel_y", units="m/s**2", opt=True, lower=-50.0, upper=50.0)
    midcourse.add_control("accel_z", units="m/s**2", opt=True, lower=-50.0, upper=50.0)

    midcourse.add_input_parameter("boundary_alt", val=100e3, static_target=True)

    model.add_subsystem("midcourse", midcourse)

    terminal = dm.Phase(
        ode_class=TerminalODE,
        transcription=tx,
    )
    terminal.set_time_options(fix_initial=False, duration_val=30.0)
    terminal.add_state("r", rate_source="dr_dt", units="m")
    terminal.add_state("v", rate_source="dv_dt", units="m/s")
    terminal.add_state("q", rate_source="dq_dt", units=None)
    terminal.add_state("omega", rate_source="domega_dt", units="rad/s")
    terminal.add_state("m", rate_source="dm_dt", units="kg")

    terminal.add_control("accel_x", units="m/s**2", opt=True, lower=-150.0, upper=150.0)
    terminal.add_control("accel_y", units="m/s**2", opt=True, lower=-150.0, upper=150.0)
    terminal.add_control("accel_z", units="m/s**2", opt=True, lower=-150.0, upper=150.0)

    terminal.add_input_parameter("boundary_alt", val=100e3, static_target=True)
    terminal.add_input_parameter("kill_mechanism", val=interceptor.kill_mechanism, static_target=True)
    terminal.add_input_parameter("kill_radius", val=interceptor.kill_radius, static_target=True)

    model.add_subsystem("terminal", terminal)

    boost.link_phases([midcourse], vars=["r", "v", "q", "omega", "m"], connect=False)
    midcourse.link_phases([terminal], vars=["r", "v", "q", "omega", "m"], connect=False)

    p.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
    p.driver.declare_coloring(show_summary=True)

    p.setup()

    p.set_val("boost.t_initial", 0.0)
    p.set_val("boost.t_duration", 60.0)
    p.set_val("midcourse.t_initial", 60.0)
    p.set_val("midcourse.t_duration", 120.0)
    p.set_val("terminal.t_initial", 180.0)
    p.set_val("terminal.t_duration", 30.0)

    p.set_val("boost.states:r", [0.0, 0.0, 0.0])
    p.set_val("boost.states:v", [0.0, 0.0, 100.0])
    p.set_val("boost.states:q", [1.0, 0.0, 0.0, 0.0])
    p.set_val("boost.states:omega", [0.0, 0.0, 0.0])
    p.set_val("boost.states:m", interceptor.mass)

    p.set_val("terminal.states:r", [100000.0, 0.0, 100000.0])
    p.set_val("terminal.states:v", [-500.0, 0.0, -500.0])
    p.set_val("terminal.states:q", [1.0, 0.0, 0.0, 0.0])
    p.set_val("terminal.states:omega", [0.0, 0.0, 0.0])
    p.set_val("terminal.states:m", 15.0)

    return p


if __name__ == "__main__":
    p = build_trajectory_problem()
    dm.run_problem(p, run_driver=True, simulate=False)
