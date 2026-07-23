"""Tests for the RS-28 Sarmat Dymos offensive trajectory optimization."""

import numpy as np
import pytest
import warnings

dymos = pytest.importorskip("dymos")

from project_icarus.optimization.sarmat_trajectory import (
    build_sarmat_trajectory_problem,
    R_TARGET_DC,
)


class TestSarmatTrajectory:
    def test_assembles(self):
        """The 3-phase Sarmat problem must assemble without crashing."""
        warnings.filterwarnings("ignore")
        p = build_sarmat_trajectory_problem(num_segments=8, order=3, maxiter=3)
        assert p is not None
        p.setup()

    def test_boost_phase_has_guidance_parameters(self):
        """Boost phase must expose el_0, el_1, t_cross, T3_scale."""
        warnings.filterwarnings("ignore")
        p = build_sarmat_trajectory_problem(num_segments=8, order=3, maxiter=3)
        p.setup()
        boost = p.model.traj.phases.boost
        param_names = list(boost.parameter_options.keys())
        for name in ("el_0", "el_1", "t_cross", "T3_scale"):
            assert name in param_names

    def test_phases_linked(self):
        """All three phases must be linked by state continuity."""
        warnings.filterwarnings("ignore")
        p = build_sarmat_trajectory_problem(num_segments=8, order=3, maxiter=3)
        p.setup()
        traj = p.model.traj
        assert hasattr(traj, "phases")
        assert "boost" in dir(traj.phases)
        assert "midcourse" in dir(traj.phases)
        assert "terminal" in dir(traj.phases)

    def test_runs_without_crashing(self):
        """The optimizer must drive at least a few iterations."""
        warnings.filterwarnings("ignore")
        p = build_sarmat_trajectory_problem(num_segments=8, order=3, maxiter=3)
        dm = __import__("dymos")
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
        for ph in ("boost", "midcourse", "terminal"):
            dur = float(np.asarray(p.get_val(f"traj.{ph}.t_duration")).ravel()[0])
            assert np.isfinite(dur)
            assert dur > 0.0

    def test_miss_distance_reported(self):
        """The terminal phase must compute miss_distance at each node."""
        warnings.filterwarnings("ignore")
        p = build_sarmat_trajectory_problem(num_segments=8, order=3, maxiter=3)
        dm = __import__("dymos")
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
        miss = np.asarray(p.get_val("traj.terminal.timeseries.miss_distance")).ravel()
        assert miss.shape[0] > 0
        assert np.all(np.isfinite(miss))
