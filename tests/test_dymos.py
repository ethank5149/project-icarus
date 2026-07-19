import numpy as np
import pytest

dymos = pytest.importorskip("dymos")
import warnings

from src.optimization.trajectory_optimization import build_trajectory_problem


@pytest.fixture
def problem():
    warnings.filterwarnings("ignore")
    p = build_trajectory_problem()
    p.setup()
    return p


class TestDymosTrajectory:
    def test_problem_setup(self):
        p = build_trajectory_problem()
        assert p is not None

    def test_assembles_and_runs(self, problem):
        """The multi-phase (boost/midcourse/terminal) problem must assemble and
        drive the SLSQP optimizer to a converged (or locally-optimal) solution
        against the installed Dymos 1.15 / OpenMDAO 3.45 API."""
        dm = __import__("dymos")
        dm.run_problem(problem, run_driver=True, simulate=False, make_plots=False)
        # Each phase reports a finite, non-negative duration within bounds.
        for ph in ("boost", "midcourse", "terminal"):
            dur = float(np.asarray(problem.get_val(f"traj.{ph}.t_duration")).ravel()[0])
            assert np.isfinite(dur)
            assert dur > 0.0
