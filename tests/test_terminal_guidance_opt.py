import numpy as np
import pytest

dymos = pytest.importorskip("dymos")
import warnings

from src.optimization.trajectory_optimization import build_trajectory_problem
from src.optimization.phases.terminal_phase import ClosedLoopTerminalODE, TerminalODE


def _closed_loop_terminal_phase(law="pn", n_seg=6, maxiter=20):
    """Standalone closed-loop terminal phase: optimize guidance gains to null
    the miss distance of a seeker-guided intercept. Returns the solved problem."""
    import openmdao.api as om
    import dymos as dm

    warnings.filterwarnings("ignore")
    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem("traj", traj, promotes=["*"])
    tx = dm.Radau(num_segments=n_seg, order=3)
    term = dm.Phase(ode_class=ClosedLoopTerminalODE, transcription=tx)
    traj.add_phase("term", term)
    term.options["ode_init_kwargs"] = {"law": law}
    term.set_time_options(fix_initial=True, duration_bounds=(10.0, 40.0), units="s")
    # Interceptor starts 20 km in +x from the aim point, closing along -x.
    term.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                   fix_initial=True, fix_final=False, val=np.array([120e3, 0.0, 6471e3]))
    term.add_state("v", rate_source="dv_dt", units="m/s", shape=(3,),
                   fix_initial=True, fix_final=False, val=np.array([-1500.0, 0.0, 0.0]))
    term.add_state("q", rate_source="dq_dt", units=None, shape=(4,),
                   fix_initial=True, fix_final=False, val=np.array([1., 0, 0, 0.]))
    term.add_state("omega", rate_source="domega_dt", units="rad/s", shape=(3,),
                   fix_initial=True, fix_final=False, val=np.zeros(3))
    term.add_state("m", rate_source="dm_dt", units="kg",
                   fix_initial=True, fix_final=False, val=200.0)
    term.add_control("accel_x", units="m/s**2", opt=False, val=0.0)
    term.add_control("accel_y", units="m/s**2", opt=False, val=0.0)
    term.add_control("accel_z", units="m/s**2", opt=False, val=0.0)
    term.add_parameter("target_r", units="m", shape=(3,),
                       val=np.array([100e3, 0.0, 6471e3]), opt=False)
    term.add_parameter("target_v", units="m/s", shape=(3,), val=np.zeros(3), opt=False)
    term.add_parameter("N", units=None, val=4.0, opt=True, lower=2.0, upper=8.0)
    term.add_parameter("accel_limit", units="m/s**2", val=150.0, opt=True,
                       lower=50.0, upper=300.0)
    term.add_timeseries_output("miss_distance")
    term.add_objective("miss_distance", scaler=1.0, loc="final")

    p.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
    p.driver.declare_coloring()
    p.driver.opt_settings["maxiter"] = maxiter
    p.setup()
    p.set_val("traj.term.t_duration", 25.0)
    dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
    return p


class TestClosedLoopTerminal:
    def test_assembles(self):
        # build_trajectory_problem defaults to the closed-loop terminal ODE and
        # a miss-distance objective (Phase-7: wire EngagementRunner guidance in).
        warnings.filterwarnings("ignore")
        p = build_trajectory_problem(closed_loop=True)
        assert p is not None
        p.setup()
        # The terminal phase must be the closed-loop ODE.
        term = p.model.traj.phases.terminal
        assert term.options["ode_class"] is ClosedLoopTerminalODE

    def test_guidance_closes_miss(self):
        # The optimizer tunes N / accel_limit so the closed-loop seeker guidance
        # drives the final miss distance below the initial 20 km offset.
        warnings.filterwarnings("ignore")
        p = _closed_loop_terminal_phase(law="pn", n_seg=6, maxiter=20)
        miss = np.asarray(p.get_val("traj.term.timeseries.miss_distance")).ravel()
        final = float(miss[-1])
        initial = float(miss[0])
        assert np.isfinite(final)
        # Guidance must reduce the miss distance versus the initial offset.
        assert final < 0.8 * initial
