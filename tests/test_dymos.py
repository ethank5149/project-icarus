import numpy as np
import pytest

dymos = pytest.importorskip("dymos")
import warnings

from project_icarus.optimization.trajectory_optimization import build_trajectory_problem


@pytest.fixture
def problem():
    warnings.filterwarnings("ignore")
    p = build_trajectory_problem(closed_loop=False)
    p.setup()
    return p


@pytest.mark.slow
class TestDymosTrajectory:
    def test_problem_setup(self):
        p = build_trajectory_problem(closed_loop=False)
        assert p is not None

    def test_assembles_and_runs_without_crashing(self, problem):
        """The multi-phase problem must assemble and drive the optimizer for
        a few iterations without crashing."""
        dm = __import__("dymos")
        dm.run_problem(problem, run_driver=True, simulate=False, make_plots=False)
        for ph in ("boost", "midcourse", "terminal"):
            dur = float(np.asarray(problem.get_val(f"traj.{ph}.t_duration")).ravel()[0])
            assert np.isfinite(dur)
            assert dur > 0.0
