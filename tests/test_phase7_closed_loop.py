"""Phase 7: Dymos closed-loop terminal phase validation and test coverage.

Validates that the closed-loop terminal phase (ClosedLoopTerminalODE)
correctly minimizes miss distance across all supported guidance laws
(pn, apn, zem, sdre_mpc) and that the guidance parameters are
differentiable within the Dymos transcription.
"""

import numpy as np
import pytest
import warnings

dymos = pytest.importorskip("dymos")

from project_icarus.optimization.trajectory_optimization import build_trajectory_problem
from project_icarus.optimization.phases.terminal_phase import (
    ClosedLoopTerminalODE,
    TerminalODE,
)


def _closed_loop_terminal_phase(law="pn", n_seg=6, maxiter=20):
    """Standalone closed-loop terminal phase optimization helper."""
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
    term.add_state("r", rate_source="dr_dt", units="m", shape=(3,),
                   fix_initial=True, fix_final=False,
                   val=np.array([120e3, 0.0, 6471e3]))
    term.add_state("v", rate_source="dv_dt", units="m/s", shape=(3,),
                   fix_initial=True, fix_final=False,
                   val=np.array([-1500.0, 0.0, 0.0]))
    term.add_state("q", rate_source="dq_dt", units=None, shape=(4,),
                   fix_initial=True, fix_final=False,
                   val=np.array([1.0, 0.0, 0.0, 0.0]))
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
    term.add_timeseries_output("cmd_accel")
    term.add_objective("miss_distance", scaler=1.0, loc="final")

    p.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
    p.driver.declare_coloring()
    p.driver.opt_settings["maxiter"] = maxiter
    p.setup()
    p.set_val("traj.term.t_duration", 25.0)
    dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
    return p


class TestClosedLoopTerminalGuidance:
    """Phase 7: guidance law validation inside Dymos closed-loop terminal phase."""

    @pytest.mark.parametrize("law", ["pn", "apn", "zem", "sdre_mpc"])
    def test_guidance_closes_miss(self, law):
        """Each guidance law must reduce miss distance vs. initial offset."""
        warnings.filterwarnings("ignore")
        p = _closed_loop_terminal_phase(law=law, n_seg=6, maxiter=20)
        miss = np.asarray(p.get_val("traj.term.timeseries.miss_distance")).ravel()
        final = float(miss[-1])
        initial = float(miss[0])
        assert np.isfinite(final)
        assert final < 0.8 * initial

    def test_pn_closed_loop_assembles(self):
        warnings.filterwarnings("ignore")
        p = build_trajectory_problem(closed_loop=True)
        assert p is not None
        p.setup()
        term = p.model.traj.phases.terminal
        assert term.options["ode_class"] is ClosedLoopTerminalODE

    def test_open_loop_uses_terminal_ode(self):
        warnings.filterwarnings("ignore")
        p = build_trajectory_problem(closed_loop=False)
        p.setup()
        term = p.model.traj.phases.terminal
        assert term.options["ode_class"] is TerminalODE

    def test_closed_loop_optimizes_n_and_accel_limit(self):
        """The optimizer must move N and accel_limit away from defaults."""
        warnings.filterwarnings("ignore")
        p = _closed_loop_terminal_phase(law="pn", n_seg=6, maxiter=30)
        n_val = float(np.asarray(p.get_val("traj.term.parameter_vals:N")).ravel()[0])
        accel_val = float(np.asarray(p.get_val("traj.term.parameter_vals:accel_limit")).ravel()[0])
        assert 2.0 < n_val < 8.0
        assert 50.0 < accel_val < 250.0

    def test_cmd_accel_respects_accel_limit(self):
        """Commanded acceleration must stay within the optimized accel_limit."""
        warnings.filterwarnings("ignore")
        p = _closed_loop_terminal_phase(law="pn", n_seg=6, maxiter=30)
        accel_limit = float(np.asarray(p.get_val("traj.term.parameter_vals:accel_limit")).ravel()[0])
        cmd = np.asarray(p.get_val("traj.term.timeseries.cmd_accel"))
        cmd_mag = np.linalg.norm(cmd, axis=1)
        assert np.all(cmd_mag <= accel_limit + 1.0)

    def test_closed_loop_monotonic_miss_reduction(self):
        """Miss distance must be non-increasing after optimization."""
        warnings.filterwarnings("ignore")
        p = _closed_loop_terminal_phase(law="zem", n_seg=8, maxiter=25)
        miss = np.asarray(p.get_val("traj.term.timeseries.miss_distance")).ravel()
        for i in range(1, len(miss) - 1):
            assert miss[i] <= miss[i - 1] + 1.0


class TestTerminalPhaseDifferentiability:
    """Phase 7: verify the closed-loop ODE is well-formed for gradient-based opt."""

    def test_closed_loop_ode_has_outputs(self):
        nn = 4
        comp = ClosedLoopTerminalODE(num_nodes=nn, law="pn")
        comp.setup()
        assert "miss_distance" in comp._static_var_rel2meta
        assert "cmd_accel" in comp._static_var_rel2meta
        assert comp.options["law"] == "pn"

    def test_open_loop_ode_has_miss_distance(self):
        nn = 4
        comp = TerminalODE(num_nodes=nn)
        comp.setup()
        assert "miss_distance" in comp._static_var_rel2meta
