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
        """The single-phase Sarmat problem must assemble without crashing."""
        warnings.filterwarnings("ignore")
        p = build_sarmat_trajectory_problem(num_segments=8, order=3, maxiter=3)
        assert p is not None
        p.setup()

    def test_boost_phase_has_guidance_parameters(self):
        """Boost phase must expose pitch_over_start, initial_elevation, burnout_vmag."""
        warnings.filterwarnings("ignore")
        p = build_sarmat_trajectory_problem(num_segments=8, order=3, maxiter=3)
        p.setup()
        boost = p.model.traj.phases.boost
        param_names = list(boost.parameter_options.keys())
        for name in ("pitch_over_start", "initial_elevation", "burnout_vmag"):
            assert name in param_names

    def test_phases_linked(self):
        """The Dymos problem must have the boost phase."""
        warnings.filterwarnings("ignore")
        p = build_sarmat_trajectory_problem(num_segments=8, order=3, maxiter=3)
        p.setup()
        traj = p.model.traj
        assert hasattr(traj, "phases")
        assert "boost" in dir(traj.phases)

    def test_runs_without_crashing(self):
        """The optimizer must drive at least a few iterations."""
        warnings.filterwarnings("ignore")
        p = build_sarmat_trajectory_problem(num_segments=8, order=3, maxiter=3)
        dm = __import__("dymos")
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
        dur = float(np.asarray(p.get_val("traj.boost.t_duration")).ravel()[0])
        assert np.isfinite(dur)
        assert dur > 0.0

    def test_miss_distance_reported(self):
        """The boost phase must compute miss_distance at each node."""
        warnings.filterwarnings("ignore")
        p = build_sarmat_trajectory_problem(num_segments=8, order=3, maxiter=3)
        dm = __import__("dymos")
        dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
        miss = np.asarray(p.get_val("traj.boost.rhs_all.miss_distance")).ravel()
        assert miss.shape[0] > 0
        assert np.all(np.isfinite(miss))
